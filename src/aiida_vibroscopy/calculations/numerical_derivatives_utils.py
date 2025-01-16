# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Calcfunctions utils for numerical derivatives workchain."""
from __future__ import annotations

from copy import deepcopy

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
from aiida_vibroscopy.common import UNITS

# Local constants
eVinv_to_ang = UNITS.eVinv_to_ang
efield_au_to_si = UNITS.efield_au_to_si
evang_to_rybohr = UNITS.evang_to_rybohr

__all__ = (
    'get_central_derivatives_coefficients', 'central_derivatives_calculator', 'compute_susceptibility_derivatives',
    'compute_nac_parameters'
)

# def map_polarization(polarization: np.ndarray, cell: np.ndarray, sign: Literal[-1, 1]) -> np.ndarray:
#     """Map the polarization within a quantum of polarization.

#     It maps P(dE) in [0, Pq] and P(-dE) in [-Pq, 0].

#     :param polarization: (3,) vector in Cartesian coordinates, Ry atomic units
#     :param cell: (3, 3) matrix cell in Cartesian coordinates
#         (rows are lattice vectors, i.e. cell[0] = v1, ...)
#     :param sign: sign (1 or -1) to select the side of the branch
#     :return: (3,) vector in Cartesian coordinates, Ry atomic units
#     """
#     inv_cell = np.linalg.inv(cell)
#     lengths = np.sqrt(np.sum(cell**2, axis=1)) / CONSTANTS.bohr_to_ang  # in Bohr
#     volume = float(abs(np.dot(np.cross(cell[0], cell[1]), cell[2]))) / CONSTANTS.bohr_to_ang**3  # in Bohr^3

#     pol_quantum = np.sqrt(2) * lengths / volume
#     pol_crys = lengths * np.dot(polarization, inv_cell)  # in Bohr

#     pol_branch = pol_quantum * (pol_crys // pol_quantum)
#     pol_crys -= pol_branch

#     if sign < 0:
#         pol_crys -= pol_quantum

#     return np.dot(pol_crys / lengths, cell)


def get_central_derivatives_coefficients(accuracy: int, order: int) -> list[int]:
    r"""Return an array with the central derivatives coefficients.

    Implementation following `Math. Comp. 51 (1988), 699-706`.

    .. note: non standard format. They are provided as:
        :math:`[c_1, c_{-1}, c_2, c_{-2}, \\dots, c_0]` where :math:`c_i` is
        the coefficient for f(i*step_size)
    """
    alpha = [0]
    for n in range(int(accuracy / 2)):
        alpha += [n + 1, -(n + 1)]

    M = order
    N = len(alpha) - 1
    delta = np.zeros((M + 1, N + 1, N + 1))
    delta[0, 0, 0] = 1
    c1 = 1

    for n in range(1, N + 1):
        c2 = 1
        for nu in range(0, n):
            c3 = alpha[n] - alpha[nu]
            c2 = c2 * c3
            if n <= M:
                delta[n, n - 1, nu] = 0
            for m in range(0, min(n, M) + 1):
                delta[m, n, nu] = (alpha[n] * delta[m, n - 1, nu] - m * delta[m - 1, n - 1, nu]) / c3
        for m in range(0, min(n, M) + 1):
            delta[m, n, n] = (c1 / c2) * (m * delta[m - 1, n - 1, n - 1] - alpha[n - 1] * delta[m, n - 1, n - 1])
        c1 = c2

    coefficients = delta[order, accuracy, :(accuracy + 1)].tolist()
    c0 = coefficients.pop(0)
    coefficients.append(c0)

    for n in range(int(accuracy / 2)):
        coefficients.pop(n + 1)

    return coefficients


def central_derivatives_calculator(
    step_size: float, order: int, vector_name: str, data_0: orm.TrajectoryData, **field_data: orm.TrajectoryData
):
    r"""Calculate the central difference derivatives.

    The accuracy of the central finite difference is determined by the number of keys in data.

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


def build_tensor_from_voigt(voigt, order: int, index: int | None = None) -> np.ndarray:
    """Auxiliary function for reconstructing tensors from voigt notation.

    The Voigt notation is as follows (in parenthesis in case of 2nd order derivatives):
        * X(X) -> 0
        * Y(Y) -> 1
        * Z(Z) -> 2
        * YZ   -> 3
        * XZ   -> 4
        * XY   -> 5

    :param voigt: tensors in contracted Voigt form
    :param order: tensor rank; if 2, it uses the Taylor expansion as in
        `Umari & Pasquarello, Diamond and Rel. Mat., (2005)` to reconstruct
        the offdiagonal tensors
    :param index: atomic index; used for Born effective charges and Raman tensors

    :return: tensors in Cartesian coordinates (no contractions)
    """
    if order == 1:  # effective charges, dielectric tensors
        tensor = np.zeros((3, 3))
        for l in range(3):  # l is the `polarization` index
            if index is not None:
                tensor[l] = voigt[l][index]
            else:
                tensor[l] = voigt[l]
        return tensor

    if order == 2:  # chi(2), raman tensors
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
) -> dict:
    """Return the Raman (1/Angstrom) and the non-linear optical susceptibility (pm/V) tensors.

    ..note:
        * If the numerical accuracy order is greater than 2, arrays at lower orders are given as well.
        * Units are 1/Angstrom for Raman tensors, normalized using the UNITCELL volume.
        * Units are pm/V for non-linear optical susceptibility
        * Raman tensors indecis: (atomic,  atomic displacement, electric field, electric field)

    :return: dictionaries of numerical accuracies with :class:`~aiida.orm.ArrayData` having keys:
        * `raman_tensors` containing (num_atoms, 3, 3, 3) arrays;
        * `nlo_susceptibility` containing (3, 3, 3) arrays;
        * `units` as :class:`~aiida.orm.Dict` containing the units of the tensors.
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

    data_0 = raw_data.pop('null_field', None)

    if data_0 is None:
        key = list(raw_data.keys())[0]
        subkey = list(raw_data[key].keys())[0]
        traj = raw_data[key][subkey].clone()
        forces_shape = traj.get_array('forces').shape
        dipole_shape = traj.get_array('electronic_dipole_cartesian_axes').shape
        traj.set_array('forces', np.zeros(forces_shape))
        traj.set_array('electronic_dipole_cartesian_axes', np.zeros(dipole_shape))
        data_0 = traj

    # Taking the missing data from symmetry
    if preprocess_data.is_symmetry:
        data = get_trajectories_from_symmetries(
            preprocess_data=preprocess_data, data=raw_data, data_0=data_0, accuracy_order=accuracy_order.value
        )
    else:
        data = raw_data

    # Conversion factors
    dchi_factor = (evang_to_rybohr * CONSTANTS.bohr_to_ang**2) / volume  # --> 4*pi / angstrom
    chi2_factor = 0.5 * (4 * np.pi) * 100 / (volume_au_units * efield_au_to_si)  # --> pm/Volt

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
            tensor_ = build_tensor_from_voigt(voigt=dchi_voigt, order=2, index=index)
            dchi_tensor.append(tensor_)
        dchi_tensor = np.array(dchi_tensor)
        # Now we build the actual tensor, using the symmetry properties of i <--> j .
        # Building Chi2[k;i,j] from Chi2[k;l]
        chi2_tensor = build_tensor_from_voigt(voigt=chi2_voigt, order=2)

        # Doing the symmetrization in case
        dchi_tensor, chi2_tensor = symmetrize_susceptibility_derivatives(
            raman_tensors=dchi_tensor,
            nlo_susceptibility=chi2_tensor,
            ucell=preprocess_data.get_phonopy_instance().unitcell,
            symprec=preprocess_data.symprec,
            is_symmetry=preprocess_data.is_symmetry
        )

        # Setting arrays
        chis_array_data = orm.ArrayData()
        chis_array_data.set_array('raman_tensors', dchi_tensor * dchi_factor)
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
            dchi_tensor.append(build_tensor_from_voigt(voigt=dchi_voigt, order=2, index=index))
        dchi_tensor = np.array(dchi_tensor)

        chi2_tensor = build_tensor_from_voigt(voigt=chi2_voigt, order=2)

        # Doing the symmetrization in case
        dchi_tensor, chi2_tensor = symmetrize_susceptibility_derivatives(
            raman_tensors=dchi_tensor,
            nlo_susceptibility=chi2_tensor,
            ucell=preprocess_data.get_phonopy_instance().unitcell,
            symprec=preprocess_data.symprec,
            is_symmetry=preprocess_data.is_symmetry
        )

        # Setting arrays
        chis_array_data = orm.ArrayData()
        chis_array_data.set_array('raman_tensors', dchi_tensor * dchi_factor)
        chis_array_data.set_array('nlo_susceptibility', chi2_tensor * chi2_factor)

        key_order = f'numerical_accuracy_{accuracy}'
        chis_data.update({key_order: deepcopy(chis_array_data)})

    units_data = orm.Dict({
        'raman_tensors': r'$1/\AA$',
        'nlo_susceptibility': 'pm/V',
    })

    return {**chis_data, 'units': units_data}


@calcfunction
def compute_nac_parameters(
    preprocess_data: PreProcessData, electric_field: orm.Float, accuracy_order: orm.Int, **kwargs
) -> dict:
    """Return high frequency dielectric and Born charge tensors using central difference schemes.

    ..note:
        * Units are in atomic units, meaning:
            1. Dielectric tensor in vacuum permittivity units.
            2. Born charges in electric charge units.
        * Born charges tensors indecis: (atomic, electric field, atomic displacement)

    :return: dictionary with ArrayData having keys:
        * `born_charges` as containing (num_atoms, 3, 3) arrays
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

    data_0 = raw_data.pop('null_field', None)

    if data_0 is None:
        key = list(raw_data.keys())[0]
        subkey = list(raw_data[key].keys())[0]
        traj = raw_data[key][subkey].clone()
        forces_shape = traj.get_array('forces').shape
        dipole_shape = traj.get_array('electronic_dipole_cartesian_axes').shape
        traj.set_array('forces', np.zeros(forces_shape))
        traj.set_array('electronic_dipole_cartesian_axes', np.zeros(dipole_shape))
        data_0 = traj

    # Taking the missing data from symmetry
    if preprocess_data.is_symmetry:
        data = get_trajectories_from_symmetries(
            preprocess_data=preprocess_data, data=raw_data, data_0=data_0, accuracy_order=accuracy_order.value
        )
    else:
        data = raw_data

    # Conversion factors
    bec_factor = evang_to_rybohr / np.sqrt(2)
    chi_factor = 4 * np.pi / volume_au_units

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
        chi_tensor = build_tensor_from_voigt(voigt=chi_voigt, order=1)
        eps_tensor = chi_factor * chi_tensor + np.eye(3)  # eps = 4.pi.X +1
        # Effective Born charges
        for index in range(num_atoms):
            bec_ = build_tensor_from_voigt(voigt=bec_voigt, order=1, index=index)
            bec_tensor.append(bec_)
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
        chi_tensor = build_tensor_from_voigt(voigt=chi_voigt, order=1)
        eps_tensor = chi_factor * chi_tensor + np.eye(3)  # eps = 4.pi.X +1
        # Effective Born charges
        for index in range(num_atoms):
            bec_tensor.append(build_tensor_from_voigt(voigt=bec_voigt, order=1, index=index))
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

    keys = ['raman_tensors', 'nlo_susceptibility']
    for key in keys:
        tensors.set_array(key, susceptibilities.get_array(key))

    return tensors
