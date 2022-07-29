# -*- coding: utf-8 -*-
"""Calcfunctions utils for numerical derivatives workchain."""
from math import pi, sqrt

from aiida import orm
from aiida.engine import calcfunction

import numpy as np
from qe_tools import CONSTANTS

from aiida_quantumespresso_vibroscopy.common import UNITS_FACTORS

# Local constants
eVinv_to_ang = UNITS_FACTORS.eVinv_to_ang
efield_au_to_si = UNITS_FACTORS.efield_au_to_si
forces_si_to_au = UNITS_FACTORS.forces_si_to_au

__all__ = (
    'get_central_derivatives_coefficients',
    'central_derivatives_calculator',
    'compute_susceptibility_derivatives',
    'compute_nac_parameters'
)


def get_central_derivatives_coefficients(accuracy: int, order: int):
    """Return an array with the central derivatives coefficients in 0.

    .. note: non standard format. They are provided as:
        :math:`[c_1, c_{-1}, c_2, c_{-2}, \\dots, c_0]` where :math:`c_i` is
        the coefficient for f(i*step_size)
    """
    index = int(accuracy/2)-1

    if order == 1:

        coefficients = [
            [1/2, 0],
            [2/3, -1/12, 0],
            [3/4, -3/20, 1/60,  0],
            [4/5, -1/5,  4/105, -1/280, 0],
        ]

    if order == 2:

        coefficients = [
            [1,  -2],
            [4/3, -1/12, -5/2],
            [3/2, -3/20, 1/90,  -49/18],
            [8/5, -1/5,  8/315, -1/560, -205/72],
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
        stringed numbers as labels, in the order: :math:`[c_1, c_{-1}, c_2, c_{-2}, \\dots , c_0]`
        where :math:`c_i` is the coefficient for f(i*step_size)
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
                elif j==3:
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

    :note: the number of arrays depends on the accuracy.

    :return: dictionary with the following keys:
        * `dph0_susceptibility` as orm.ArrayData containing (num_atoms, 3, 3, 3) arrays (third index refers to forces);
        * `nlo_susceptibility` as orm.ArrayData containing (3, 3, 3) arrays;
        * `units` as orm.Dict containing the units of the tensors.
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
    dchi_factor = forces_si_to_au * CONSTANTS.bohr_to_ang**2  # --> angstrom^2
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
        # This is the dChi/dR tensor, where Chi is the electronic susceptibility.
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

    :note: the number of arrays depends on the accuracy.

    :return: dictionary with keys:
        * `born_charges` as ArrayData containing (num_atoms, 3, 3) arrays (third index refers to forces);
        * `dielectric` as ArrayData containing (3, 3) arrays.
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
            bec_ = _build_tensor_from_voigt(voigt=bec_voigt, order=1, index=index)
            bec_tensor.append(bec_.T)
        bec_tensor = np.array(bec_tensor)

        dielectric_data.set_array(f'numerical_accuracy_{accuracy}', eps_tensor)
        bec_data.set_array(f'numerical_accuracy_{accuracy}', bec_tensor * bec_factor)

    return {'dielectric': dielectric_data, 'born_charges': bec_data}
