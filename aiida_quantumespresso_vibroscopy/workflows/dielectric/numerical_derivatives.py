# -*- coding: utf-8 -*-
"""Workflow for numerical derivatives. """
from math import pi, sqrt

from aiida import orm
from aiida.engine import WorkChain, calcfunction

import numpy as np
from qe_tools import CONSTANTS

from aiida_quantumespresso_vibroscopy.common import UNITS_FACTORS
from aiida_quantumespresso_vibroscopy.utils.validation import validate_positive


# Local constants
eVinv_to_ang = UNITS_FACTORS.eVinv_to_ang
efield_au_to_si = UNITS_FACTORS.efield_au_to_si
forces_si_to_au = UNITS_FACTORS.forces_si_to_au


def symmetrization(tensor):
    """Symmetrizes a 3x3x3 tensor."""
    sym_tensor = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                sym_tensor[i][j][k] = (1.0 / 6.0) * (
                    tensor[i][j][k]
                    + tensor[i][k][j]
                    + tensor[j][i][k]
                    + tensor[j][k][i]
                    + tensor[k][i][j]
                    + tensor[k][j][i]
                )
    return sym_tensor


def get_central_derivatives_coefficients(accuracy: int, order: int):
    """Return an array with the central derivatives coefficients in 0.

    .. note: non standard format. They are provided as:
        [c1, c-1, c2, c-2, ..., c0] where ci is the coefficient for f(i*step_size)
    """
    if order == 1:
        index = int(accuracy/2)-1

        coefficients = [
            [1 / 2, 0],
            [2 / 3, -1 / 12, 0],
            [3 / 4, -3 / 20, 1 / 60, 0],
            [4 / 5, -1 / 5, 4 / 105, -1 / 280, 0],
        ]

    if order == 2:
        index = int(accuracy/2)-1

        coefficients = [
            [1, -2],
            [4 / 3, -1 / 12, -5 / 2],
            [3 / 2, -3 / 20, 1 / 90, -49 / 18],
            [8 / 5, -1 / 5, 8 / 315, -1 / 560, -205 / 72],
        ]

    return coefficients[index]


def central_derivatives_calculator(
    step_size: float,
    order: int,
    vector_name: str,
    data_0: orm.TrajectoryData,
    **field_data: orm.TrajectoryData
):
    """Calculate the central difference derivatives with a certain displacement of a certain type,
    i.e. forces or polarization. The accuracy of the central finite difference is determined
    by the number of keys in data.

    :param step_size: step size for finite differenciation
    :param order: order of the derivative
    :param vector_name: either `forces` or `electronic_dipole_cartesian_axes`
    :param data: trajectory data for a particular direction in space; it is expected to be given with
        stringed numbers as labels, in the order: [c1, c-1, c2, c-2, ..., c0] where ci is the
        coefficient for f(i*step_size)
    """

    def sign(num):
        return 1 if num == 0 else -1

    denominator = step_size**order
    accuracy = len(field_data)

    coefficients = get_central_derivatives_coefficients(accuracy=accuracy, order=order)

    derivative = data_0.get_array(vector_name)[-1] * (coefficients.pop())

    for i, coefficient in enumerate(coefficients):
        for j in (0, 1):
            derivative = (
                derivative
                + (sign(j) ** order) * field_data[str(int(2 * i + j))].get_array(vector_name)[-1] * coefficient
            )

    return derivative / denominator


def _build_tensor_from_voigt(voigt, order, index=None):
    """Auxiliary function for reconstructing tensors from voigt notation."""
    if order == 1:
        tensor = np.zeros((3, 3))
        for j in range(3):
            if index is not None:
                tensor[j] = voigt[j][index]
            else:
                tensor[j] = voigt[j]
        return tensor

    if order == 2:
        tensor = np.zeros((3, 3, 3))
        for k in range(3):
            for j in range(6):
                if index is not None:
                    value = voigt[j][index][k]
                else:
                    value = voigt[j][k]
                if j in (0, 1, 2):
                    tensor[k][j][j] = value
                    tensor[k][1][2] = 0.5 * (value - tensor[k][1][1] - tensor[k][2][2])
                    tensor[k][2][1] = tensor[k][1][2]
                elif j == 4:
                    tensor[k][0][2] = 0.5 * (value - tensor[k][0][0] - tensor[k][2][2])
                    tensor[k][2][0] = tensor[k][0][2]
                elif j == 5:
                    tensor[k][0][1] = 0.5 * (value - tensor[k][0][0] - tensor[k][1][1])
                    tensor[k][1][0] = tensor[k][0][1]
        return tensor


@calcfunction
def compute_susceptibility_derivatives(
    structure: orm.StructureData, electric_field: orm.Float, diagonal_scale: orm.Float, **kwargs
):
    """
    Return the third derivative of the total energy in respect to
    one phonon and two electric fields, times volume factor.
    """
    volume = structure.get_cell_volume()  # angstrom^3
    volume_au_units = volume / (CONSTANTS.bohr_to_ang**3)  # bohr^3

    data = {}
    for key, value in kwargs.items():
        if key == 'null_field':
            data.update({key: value})
        else:
            data.update({key: {}})
            for subkey, subvalue in value.items():
                data[key].update({subkey: subvalue})

    # Conversion factors
    dchi_factor = forces_si_to_au * CONSTANTS.bohr_to_ang  # --> angstrom^2
    chi2_factor = 0.5 * (4 * pi) * 100 / (volume_au_units * efield_au_to_si)  # --> pm/Volt

    # Variables
    field_step = electric_field.value
    scale = diagonal_scale.value
    num_atoms = len(structure.sites)
    max_accuracy = len(data['field_index_0'])
    data_0 = data.pop('null_field')

    dchi_data = orm.ArrayData()
    chi2_data = orm.ArrayData()

    # First, I calculate all possible second order accuracy tensors with all possible steps.
    if max_accuracy > 2:
        accuracies = np.arange(2, max_accuracy + 2, 2)[::-1].tolist()
    else:
        accuracies = []

    for accuracy in accuracies:
        # This is the dChi/du tensor, where Chi is the electronic susceptibility.
        # It is a tensor of the shape (num_atoms, k, i, j), k is the index relative to forces,
        # while i,j relative to the electric field (i.e. to Chi)
        dchi_tensor = []
        # This is the Chi2 tensor, where Chi2 is the non linear optical susceptibility.
        # It is a tensor of the shape (k, i, j), i,j,k relative to the electric field direction.
        chi2_tensor = np.zeros((3, 3, 3))

        scale_step = accuracy / 2

        # We first compute the dChi/du tensor using Voigt notation.
        dchi_voigt = [0 for _ in range(6)]
        chi2_voigt = [0 for _ in range(6)]

        # i.e. {'field_index_0':{'0':Traj,'1':Traj, ...}, 'field_index_1':{...}, ..., 'field_index_5':{...} }
        for key, value in data.items():
            step_value = {'0': value[str(accuracy - 2)], '1': value[str(accuracy - 1)]}
            if int(key[-1]) in (0, 1, 2):
                dchi_voigt[int(key[-1])] = central_derivatives_calculator(
                    step_size=scale_step * field_step,
                    order=2,
                    vector_name='forces',
                    data_0=data_0,
                    **step_value
                )
                chi2_voigt[int(key[-1])] = central_derivatives_calculator(
                    step_size=scale_step * field_step,
                    order=2,
                    vector_name='electronic_dipole_cartesian_axes',
                    data_0=data_0,
                    **step_value,
                )
            else:
                dchi_voigt[int(key[-1])] = central_derivatives_calculator(
                    step_size=scale_step * scale * field_step,
                    order=2,
                    vector_name='forces',
                    data_0=data_0,
                    **step_value
                )
                chi2_voigt[int(key[-1])] = central_derivatives_calculator(
                    step_size=scale_step * scale * field_step,
                    order=2,
                    vector_name='electronic_dipole_cartesian_axes',
                    data_0=data_0,
                    **step_value,
                )

        # Now we build the actual tensor, using the symmetry properties of i <--> j .
        # Building dChi[I,k;i,j] from dChi[I,k;l]
        for index in range(num_atoms):
            tensor_ =_build_tensor_from_voigt(voigt=dchi_voigt, order=2, index=index)
            dchi_tensor.append(tensor_)
        dchi_tensor = np.array(dchi_tensor)
        # Now we build the actual tensor, using the symmetry properties of i <--> j .
        # Building Chi2[k;i,j] from Chi2[k;l]
        chi2_tensor = _build_tensor_from_voigt(voigt=chi2_voigt, order=2)

        # Setting arrays
        dchi_data.set_array(f'numerical_accuracy_2_step_{int(scale_step)}', dchi_tensor * dchi_factor)
        chi2_data.set_array(f'numerical_accuracy_2_step_{int(scale_step)}', chi2_tensor * chi2_factor)

    # Second, I calculate all possible accuracy tensors.
    if max_accuracy > 2:
        accuracies.pop()
    else:
        accuracies = [2]

    for accuracy in accuracies:
        dchi_tensor = []
        chi2_tensor = np.zeros((3, 3, 3))

        dchi_voigt = [0 for _ in range(6)]
        chi2_voigt = [0 for _ in range(6)]

        for key, value in data.items():
            if int(key[-1]) in (0, 1, 2):
                dchi_voigt[int(key[-1])] = central_derivatives_calculator(
                    step_size=field_step,
                    order=2,
                    vector_name='forces',
                    data_0=data_0,
                    **value
                )
                chi2_voigt[int(key[-1])] = central_derivatives_calculator(
                    step_size=field_step,
                    order=2,
                    vector_name='electronic_dipole_cartesian_axes',
                    data_0=data_0,
                    **value
                )
            else:
                dchi_voigt[int(key[-1])] = central_derivatives_calculator(
                    step_size=scale * field_step,
                    order=2,
                    vector_name='forces',
                    data_0=data_0,
                    **value
                )
                chi2_voigt[int(key[-1])] = central_derivatives_calculator(
                    step_size=scale * field_step,
                    order=2,
                    vector_name='electronic_dipole_cartesian_axes',
                    data_0=data_0,
                    **value,
                )
            data[key].pop(str(accuracy - 1))
            data[key].pop(str(accuracy - 2))

        for index in range(num_atoms):
            dchi_tensor.append(_build_tensor_from_voigt(voigt=dchi_voigt, order=2, index=index))
        dchi_tensor = np.array(dchi_tensor)

        chi2_tensor = _build_tensor_from_voigt(voigt=chi2_voigt, order=2)

        # Setting arrays
        dchi_data.set_array(f'numerical_accuracy_{accuracy}', dchi_tensor * dchi_factor)
        chi2_data.set_array(f'numerical_accuracy_{accuracy}', chi2_tensor * chi2_factor)

    units_data = orm.Dict(
        dict={
            'dph0_susceptibility': r'$\AA^2$',
            'nlo_susceptibility': 'pm/V',
        }
    )

    return {'nlo_susceptibility': chi2_data, 'dph0_susceptibility': dchi_data, 'units': units_data}


@calcfunction
def compute_nac_parameters(structure: orm.StructureData, electric_field: orm.Float, **kwargs):
    """
    Return epsilon and born charges to second order in finite electric fields (central difference).
    """
    volume_au_units = structure.get_cell_volume() / (CONSTANTS.bohr_to_ang**3)  # in bohr^3

    data = {}
    for key, value in kwargs.items():
        if key == 'null_field':
            data.update({key: value})
        else:
            data.update({key: {}})
            for subkey, subvalue in value.items():
                data[key].update({subkey: subvalue})

    # Conversion factors
    bec_factor = forces_si_to_au / sqrt(2)
    chi_factor = 4 * pi / volume_au_units

    # Variables
    field_step = electric_field.value
    num_atoms = len(structure.sites)
    max_accuracy = len(data['field_index_0'])
    data_0 = data.pop('null_field')

    dielectric_data = orm.ArrayData()
    bec_data = orm.ArrayData()

    # First, I calculate all possible second order accuracy tensors with all possible steps.
    if max_accuracy > 2:
        accuracies = np.arange(2, max_accuracy + 2, 2)[::-1].tolist()
    else:
        accuracies = []

    for accuracy in accuracies:
        bec_tensor = []
        scale_step = accuracy / 2

        # We first compute the tensors using an analogous of Voigt notation.
        chi_voigt = [0 for _ in range(3)]
        bec_voigt = [0 for _ in range(3)]

        # i.e. {'field_index_0':{'0':Traj,'1':Traj, ...}, 'field_index_1':{...}, ..., 'field_index_5':{...} }
        for key, value in data.items():
            step_value = {'0': value[str(accuracy - 2)], '1': value[str(accuracy - 1)]}
            if int(key[-1]) in (0, 1, 2):
                chi_voigt[int(key[-1])] = central_derivatives_calculator(
                    step_size=scale_step * field_step,
                    order=1,
                    vector_name='electronic_dipole_cartesian_axes',
                    data_0=data_0,
                    **step_value,
                )
                bec_voigt[int(key[-1])] = central_derivatives_calculator(
                    step_size=scale_step * field_step, order=1, vector_name='forces', data_0=data_0, **step_value
                )

        # Now we build the actual tensor.
        # Epsilon
        chi_tensor = _build_tensor_from_voigt(voigt=chi_voigt, order=1)
        eps_tensor = chi_factor * chi_tensor + np.eye(3)  # eps = 4.pi.X +1
        # Effective Born charges
        for index in range(num_atoms):
            # ATTENTION: here we need to remember to take the transpose of each single tensor from finite differences.
            bec_ = _build_tensor_from_voigt(voigt=bec_voigt, order=1, index=index)
            bec_tensor.append(bec_.T)
        bec_tensor = np.array(bec_tensor)

        dielectric_data.set_array(f'numerical_accuracy_2_step_{int(scale_step)}', eps_tensor)
        bec_data.set_array(f'numerical_accuracy_2_step_{int(scale_step)}', bec_tensor * bec_factor)

    # Second, I calculate all possible accuracy tensors.
    if max_accuracy > 2:
        accuracies.pop()
    else:
        accuracies = [2]

    for accuracy in accuracies:
        bec_tensor = []

        # We first compute the tensors using an analogous of Voigt notation.
        chi_voigt = [0 for _ in range(3)]
        bec_voigt = [0 for _ in range(3)]

        # i.e. {'field_index_1':{'0':Traj,'1':Traj, ...}, 'field_index_1':{...}, ..., 'field_index_5':{...} }
        for key, value, in data.items():
            if int(key[-1]) in (0, 1, 2):
                chi_voigt[int(key[-1])] = central_derivatives_calculator(
                    step_size=field_step,
                    order=1,
                    vector_name='electronic_dipole_cartesian_axes',
                    data_0=data_0,
                    **value
                )
                bec_voigt[int(key[-1])] = central_derivatives_calculator(
                    step_size=field_step,
                    order=1,
                    vector_name='forces',
                    data_0=data_0,
                    **value
                )
            data[key].pop(str(accuracy - 1))
            data[key].pop(str(accuracy - 2))

        # Now we build the actual tensor.
        # Epsilon
        chi_tensor = _build_tensor_from_voigt(voigt=chi_voigt, order=1)
        eps_tensor = chi_factor * chi_tensor + np.eye(3)  # eps = 4.pi.X +1
        # Effective Born charges
        for index in range(num_atoms):
            bec_tensor.append(_build_tensor_from_voigt(voigt=bec_voigt, order=1, index=index))
        bec_tensor = np.array(bec_tensor)

        dielectric_data.set_array(f'numerical_accuracy_{accuracy}', eps_tensor)
        bec_data.set_array(f'numerical_accuracy_{accuracy}', bec_tensor * bec_factor)

    return {'dielectric': dielectric_data, 'born_charges': bec_data}


def validate_data(data, _):
    """
    Validate the `data` namespace inputs.
    """
    length = len(data)  # must be 4 or 7
    control_null_namespace = 0  # must be 1

    if not length in [4, 7]:
        return f'invalid total number of inputs for namespace `data`: expected 4 or 7, given {length}'

    for label, trajectory in data.items():
        # first, control if `null`
        if label.startswith('null'):
            control_null_namespace += 1
        elif not label[-1] in ['0', '1', '2', '3', '4', '5']:
            return f'`{label[-1]}` is an invalid label ending for field labels`'
        else:
            if not len(trajectory) % 2 == 0:
                return 'field index data must contains even number of key:TrajectoryData pairs'

    if not control_null_namespace == 1:
        return f'invalid number of `null_field` namespaces: expected 1, given {control_null_namespace}'


class NumericalDerivativesWorkChain(WorkChain):
    """
    Workchain that computes first and second order derivatives of forces and polarization in respect to
    polarization, to obtain dielectric tensor, Born effective charges, susceptibility derivatives in
    respect to atomic position and polarization (i.e. non linear optical susceptibility) to be used
    to compute Raman susceptibility tensors.

    Forces and polarization must passed as TrajectoryData as a dictionary in `data`.
    Numerical derivatives can have different number of evaluation points, depending on order and accuracy.
    The price to pay is the standardization of the structure of the dictionary to pass to this namespace.

    To understand, let's review the approach.In central differencs approach we need the evaluation
    of the function at the value we want the derivative (in our case at E=0, E is the electric field),
    and at displaced positions from this value. The evaluation of the function at these points will
    have weights (or coefficients), which depend on order and accuracy. For example:
        * df/dx   = ( 0.5*f(+1*h) -0.5*f(-1*h) )/h +O(h^2)
        * d2f/dx2 = ( 1*f(+1*h) -2*f(0*h) +1*f(-1*h) )/h^2 +O(h^2)

    Referring to the coefficients for each step as `ci`, where `i` is an integer, our convention is
    to put in sequence the Trajectory data with increasing numbers as labels, for example:
        * {
        '0': TrajectoryData for c1,
        `1`: TrajectoryData for c-1,
        '2': TrajectoryData for c2,
        `3`: TrajectoryData for c-2,
        ...}
    This way to creating an analogous of an array with coefficients [c1,c-1,c2,c-2,...].

    These dictionaries are going to be put as sub-dictionary in a general `data` dictionary. Each sub-dict
    has to be put with a key with suffix a number indicating which tensor component is referring to.
    In our case, we use a similar Voigt notation. Namely we have two cases:
        * first order derivatives: keys suffices are 0,1,2;
            0 for [i,x], 1 for [i,y], 2 for [i,z] (with i={x,y,z})
        * second order derivatives: keys suffices are 0,...5;
            0 for [i,x,x], (as in Voigt), 5 for [i,x,y] (with i={x,y,z})

    The prefix can be anything. Best practice is using 'field_' as prefix.
    The Trajectory data for the c0 coefficient (i.e. the one with E=0) must be passed with a different key,
    namely `null_field`. This is to avoid errors and due to the fact that is common to the all derivatives.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input_namespace(
            'data',
            validator=validate_data,
            help='Namespace for passing TrajectoryData containing forces and polarization.',
        )
        spec.input('data.null_field', valid_type=orm.TrajectoryData, required=True)
        spec.input_namespace('data.field_index_0', valid_type=orm.TrajectoryData, required=True)
        spec.input_namespace('data.field_index_1', valid_type=orm.TrajectoryData, required=True)
        spec.input_namespace('data.field_index_2', valid_type=orm.TrajectoryData, required=True)
        spec.input_namespace('data.field_index_3', valid_type=orm.TrajectoryData, required=False)
        spec.input_namespace('data.field_index_4', valid_type=orm.TrajectoryData, required=False)
        spec.input_namespace('data.field_index_5', valid_type=orm.TrajectoryData, required=False)

        spec.input('electric_field', valid_type=orm.Float, validator=validate_positive)
        spec.input('diagonal_scale', valid_type=orm.Float, required=False, validator=validate_positive)
        spec.input('structure', valid_type=orm.StructureData)

        spec.outline(
            cls.run_results,
        )

        spec.output(
            'dielectric',
            valid_type=orm.ArrayData,
            help='Contains high frequency dielectric tensor computed in Cartesian coordinates.',
        )
        spec.output(
            'born_charges',
            valid_type=orm.ArrayData,
            help='Contains Born effective charges tensors computed in Cartesian coordinates.',
        )
        spec.output(
            'dph0_susceptibility',
            valid_type=orm.ArrayData,
            help=('Contains the derivatives of the susceptibility in respect'
                'to the atomic positions in Cartesian coordinates.'),
        )
        spec.output(
            'nlo_susceptibility',
            valid_type=orm.ArrayData,
            help='Contains the non linear optical susceptibility tensor in Cartesian coordinates.',
        )
        spec.output(
            'units',
            valid_type=orm.Dict,
            help='Units of the susceptibility derivatives tensors.'
        )

    def run_results(self):
        """Wrap up results from previous calculations."""

        # Non analytical constants
        out_nac_parameters = compute_nac_parameters(
            structure=self.inputs.structure, electric_field=self.inputs.electric_field, **self.inputs.data
        )
        for key, output in out_nac_parameters.items():
            self.out(key, output)

        # Derivatives of the susceptibility
        if len(self.inputs.data) == 7:
            out_dchis = compute_susceptibility_derivatives(
                structure=self.inputs.structure,
                electric_field=self.inputs.electric_field,
                diagonal_scale=self.inputs.diagonal_scale,
                **self.inputs.data,
            )
            for key, output in out_dchis.items():
                self.out(key, output)
