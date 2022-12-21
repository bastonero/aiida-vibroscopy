# -*- coding: utf-8 -*-
"""Calcfunctions utils for numerical derivatives workchain."""
from copy import deepcopy
from math import pi, sqrt

from aiida import orm
from aiida.engine import calcfunction
from aiida_phonopy.data import PreProcessData
import numpy as np
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon
from qe_tools import CONSTANTS

from aiida_vibroscopy.calculations.symmetry import (
    get_trajectories_from_symmetries,
    symmetrize_susceptibility_derivatives,
)
from aiida_vibroscopy.common import UNITS_FACTORS

# Local constants
eVinv_to_ang = UNITS_FACTORS.eVinv_to_ang
efield_au_to_si = UNITS_FACTORS.efield_au_to_si
forces_si_to_au = UNITS_FACTORS.forces_si_to_au

__all__ = (
    'get_central_derivatives_coefficients', 'central_derivatives_calculator', 'compute_susceptibility_derivatives',
    'compute_nac_parameters'
)


def get_central_derivatives_coefficients(accuracy: int, order: int):
    """Return an array with the central derivatives coefficients in 0.

    .. note: non standard format. They are provided as:
        :math:`[c_1, c_{-1}, c_2, c_{-2}, \\dots, c_0]` where :math:`c_i` is
        the coefficient for f(i*step_size)
    """
    index = int(accuracy / 2) - 1

    if order == 1:

        coefficients = [
            [1 / 2, 0],
            [2 / 3, -1 / 12, 0],
            [3 / 4, -3 / 20, 1 / 60, 0],
            [4 / 5, -1 / 5, 4 / 105, -1 / 280, 0],
        ]

    if order == 2:

        coefficients = [
            [1, -2],
            [4 / 3, -1 / 12, -5 / 2],
            [3 / 2, -3 / 20, 1 / 90, -49 / 18],
            [8 / 5, -1 / 5, 8 / 315, -1 / 560, -205 / 72],
        ]

    return coefficients[index]


def central_derivatives_calculator(
    step_size: float, order: int, vector_name: str, data_0: orm.TrajectoryData, **field_data: orm.TrajectoryData
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
                derivative + (sign(j)**order) * field_data[str(int(2 * i + j))].get_array(vector_name)[-1] * coefficient
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
                elif j == 3:
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
    preprocess_data: PreProcessData, electric_field: orm.Float, diagonal_scale: orm.Float, accuracy_order: orm.Int,
    **kwargs
):
    """
    Return the third derivative of the total energy in respect to
    one phonon and two electric fields, times volume factor.

    :note: the number of arrays depends on the accuracy.

    :return: dictionary with ArrayData having arraynames:
        * `raman_susceptibility` containing (num_atoms, 3, 3, 3) arrays (second index refers to atomic displacements);
        * `nlo_susceptibility` containing (3, 3, 3) arrays;
        And a key `units` as orm.Dict containing the units of the tensors.
    """
    structure = preprocess_data.get_unitcell()
    volume = structure.get_cell_volume()  # angstrom^3
    volume_au_units = volume / (CONSTANTS.bohr_to_ang**3)  # bohr^3

    # Loading the data
    raw_data = {}
    for key, value in kwargs.items():
        if key == 'null_field':
            raw_data.update({key: value})
        else:
            raw_data.update({key: {}})
            if key.startswith('field_index_'):
                for subkey, subvalue in value.items():
                    raw_data[key].update({subkey: subvalue})

    data_0 = raw_data.pop('null_field')

    # Taking the missing data from symmetry
    if preprocess_data.is_symmetry:
        data = get_trajectories_from_symmetries(
            preprocess_data=preprocess_data, data=raw_data, data_0=data_0, accuracy_order=accuracy_order.value
        )

    # Conversion factors
    dchi_factor = forces_si_to_au * CONSTANTS.bohr_to_ang**2  # --> angstrom^2
    chi2_factor = 0.5 * (4 * pi) * 100 / (volume_au_units * efield_au_to_si)  # --> pm/Volt

    # Variables
    field_step = electric_field.value
    scale = diagonal_scale.value
    max_accuracy = accuracy_order.value

    num_atoms = len(structure.sites)

    chis_data = {}

    # First, I calculate all possible second order accuracy tensors with all possible steps.
    if max_accuracy > 2:
        accuracies = np.arange(2, max_accuracy + 2, 2)[::-1].tolist()
    else:
        accuracies = []

    for accuracy in accuracies:
        scale_step = accuracy / 2

        # We first compute the tensor using Voigt notation.
        dchi_tensor = []
        dchi_voigt = [0 for _ in range(6)]
        chi2_tensor = np.zeros((3, 3, 3))
        chi2_voigt = [0 for _ in range(6)]

        # i.e. {'field_index_0':{'0':Traj,'1':Traj, ...}, 'field_index_1':{...}, ..., 'field_index_5':{...} }
        for key, value in data.items():
            step_value = {'0': value[str(accuracy - 2)], '1': value[str(accuracy - 1)]}

            if int(key[-1]) in (0, 1, 2):
                applied_scale = 1.0
            else:
                applied_scale = scale

            dchi_voigt[int(key[-1])] = central_derivatives_calculator(
                step_size=scale_step * applied_scale * field_step,
                order=2,
                vector_name='forces',
                data_0=data_0,
                **step_value
            )
            chi2_voigt[int(key[-1])] = central_derivatives_calculator(
                step_size=scale_step * applied_scale * field_step,
                order=2,
                vector_name='electronic_dipole_cartesian_axes',
                data_0=data_0,
                **step_value,
            )

        # Now we build the actual tensor, using the symmetry properties of i <--> j .
        # Building dChi[I,k;i,j] from dChi[I,k;l]
        for index in range(num_atoms):
            tensor_ = _build_tensor_from_voigt(voigt=dchi_voigt, order=2, index=index)
            dchi_tensor.append(tensor_)
        dchi_tensor = np.array(dchi_tensor)
        # Now we build the actual tensor, using the symmetry properties of i <--> j .
        # Building Chi2[k;i,j] from Chi2[k;l]
        chi2_tensor = _build_tensor_from_voigt(voigt=chi2_voigt, order=2)

        # Doing the symmetrization in case
        dchi_tensor, chi2_tensor = symmetrize_susceptibility_derivatives(
            raman_susceptibility=dchi_tensor,
            nlo_susceptibility=chi2_tensor,
            ucell=preprocess_data.get_phonopy_instance().unitcell,
            symprec=preprocess_data.symprec,
            is_symmetry=preprocess_data.is_symmetry
        )

        # Setting arrays
        chis_array_data = orm.ArrayData()
        chis_array_data.set_array('raman_susceptibility', dchi_tensor * dchi_factor)
        chis_array_data.set_array('nlo_susceptibility', chi2_tensor * chi2_factor)

        key_order = f'numerical_accuracy_2_step_{int(scale_step)}'
        chis_data.update({key_order: deepcopy(chis_array_data)})

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
                applied_scale = 1.0
            else:
                applied_scale = scale

            dchi_voigt[int(key[-1])] = central_derivatives_calculator(
                step_size=field_step * applied_scale, order=2, vector_name='forces', data_0=data_0, **value
            )
            chi2_voigt[int(key[-1])] = central_derivatives_calculator(
                step_size=field_step * applied_scale,
                order=2,
                vector_name='electronic_dipole_cartesian_axes',
                data_0=data_0,
                **value
            )

            data[key].pop(str(accuracy - 1))
            data[key].pop(str(accuracy - 2))

        for index in range(num_atoms):
            dchi_tensor.append(_build_tensor_from_voigt(voigt=dchi_voigt, order=2, index=index))
        dchi_tensor = np.array(dchi_tensor)

        chi2_tensor = _build_tensor_from_voigt(voigt=chi2_voigt, order=2)

        # Doing the symmetrization in case
        dchi_tensor, chi2_tensor = symmetrize_susceptibility_derivatives(
            raman_susceptibility=dchi_tensor,
            nlo_susceptibility=chi2_tensor,
            ucell=preprocess_data.get_phonopy_instance().unitcell,
            symprec=preprocess_data.symprec,
            is_symmetry=preprocess_data.is_symmetry
        )

        # Setting arrays
        chis_array_data = orm.ArrayData()
        chis_array_data.set_array('raman_susceptibility', dchi_tensor * dchi_factor)
        chis_array_data.set_array('nlo_susceptibility', chi2_tensor * chi2_factor)

        key_order = f'numerical_accuracy_{accuracy}'
        chis_data.update({key_order: deepcopy(chis_array_data)})

    units_data = orm.Dict({
        'raman_susceptibility': r'$\AA^2$',
        'nlo_susceptibility': 'pm/V',
    })

    return {**chis_data, 'units': units_data}


@calcfunction
def compute_nac_parameters(
    preprocess_data: PreProcessData, electric_field: orm.Float, accuracy_order: orm.Int, **kwargs
) -> dict:
    """
    Return epsilon and born charges to second order in finite electric fields (central difference).

    :note: the number of arrays depends on the accuracy.

    :return: dictionary with ArrayData having arraynames:
        * `born_charges` as containing (num_atoms, 3, 3) arrays (third index refers to atomic displacements);
        * `dielectric` as containing (3, 3) arrays.
    """
    structure = preprocess_data.get_unitcell()
    volume_au_units = structure.get_cell_volume() / (CONSTANTS.bohr_to_ang**3)  # in bohr^3

    # Loading the data
    raw_data = {}
    for key, value in kwargs.items():
        if key == 'null_field':
            raw_data.update({key: value})
        else:
            if key.startswith('field_index_'):
                raw_data.update({key: {}})
                for subkey, subvalue in value.items():
                    raw_data[key].update({subkey: subvalue})

    data_0 = raw_data.pop('null_field')

    # Taking the missing data from symmetry
    if preprocess_data.is_symmetry:
        data = get_trajectories_from_symmetries(
            preprocess_data=preprocess_data, data=raw_data, data_0=data_0, accuracy_order=accuracy_order.value
        )

    # Conversion factors
    bec_factor = forces_si_to_au / sqrt(2)
    chi_factor = 4 * pi / volume_au_units

    # Variables
    field_step = electric_field.value
    max_accuracy = accuracy_order.value
    num_atoms = len(structure.sites)

    nac_data = {}

    # First, I calculate all possible second order accuracy tensors with all possible steps.
    if max_accuracy > 2:
        accuracies = np.arange(2, max_accuracy + 2, 2)[::-1].tolist()
    else:
        accuracies = []

    for accuracy in accuracies:
        scale_step = accuracy / 2

        # We first compute the tensors using an analogous of Voigt notation.
        chi_voigt = [0 for _ in range(3)]
        bec_tensor = []
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
                    step_size=scale_step * field_step,
                    order=1,
                    vector_name='forces',
                    data_0=data_0,
                    **step_value,
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

        # Doing the symmetrization in case
        bec_tensor, eps_tensor = symmetrize_borns_and_epsilon(
            borns=bec_tensor,
            epsilon=eps_tensor,
            ucell=preprocess_data.get_phonopy_instance().unitcell,
            symprec=preprocess_data.symprec,
            is_symmetry=preprocess_data.is_symmetry
        )

        # Settings arrays
        array_data = orm.ArrayData()

        array_data.set_array('dielectric', eps_tensor)
        array_data.set_array('born_charges', bec_tensor * bec_factor)

        key_order = f'numerical_accuracy_2_step_{int(scale_step)}'
        nac_data.update({key_order: deepcopy(array_data)})

    # Second, I calculate all possible accuracy tensors.
    if max_accuracy > 2:
        accuracies.pop()
    else:
        accuracies = [2]

    for accuracy in accuracies:
        # We first compute the tensors using an analogous of Voigt notation.
        bec_tensor = []
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
                    step_size=field_step, order=1, vector_name='forces', data_0=data_0, **value
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

        # Doing the symmetrization in case
        bec_tensor, eps_tensor = symmetrize_borns_and_epsilon(
            borns=bec_tensor,
            epsilon=eps_tensor,
            ucell=preprocess_data.get_phonopy_instance().unitcell,
            symprec=preprocess_data.symprec,
            is_symmetry=preprocess_data.is_symmetry
        )

        # Settings arrays
        array_data = orm.ArrayData()

        array_data.set_array('dielectric', eps_tensor)
        array_data.set_array('born_charges', bec_tensor * bec_factor)

        key_order = f'numerical_accuracy_{accuracy}'
        nac_data.update({key_order: deepcopy(array_data)})

    return nac_data


@calcfunction
def join_tensors(nac_parameters: orm.ArrayData, susceptibilities: orm.ArrayData) -> orm.ArrayData:
    """Join the NAC and susceptibilities tensors under a unique ArrayData."""
    tensors = orm.ArrayData()

    keys = ['dielectric', 'born_charges']
    for key in keys:
        tensors.set_array(key, nac_parameters.get_array(key))

    keys = ['raman_susceptibility', 'nlo_susceptibility']
    for key in keys:
        tensors.set_array(key, susceptibilities.get_array(key))

    return tensors
