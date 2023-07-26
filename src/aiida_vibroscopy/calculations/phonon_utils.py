# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Calcfunctions utils for extracting outputs related to phonon workflows."""

__all__ = (
    'extract_max_order', 'extract_orders', 'get_forces', 'get_energy', 'get_non_analytical_constants',
    'elaborate_non_analytical_constants'
)

from aiida import orm
from aiida.engine import calcfunction
from aiida.plugins import DataFactory

PreProcessData = DataFactory('phonopy.preprocess')


@calcfunction
def extract_max_order(**kwargs):
    """Extract max order accuracy tensor."""
    output = {}
    for key, value in kwargs.items():
        max_accuracy = 0
        for name in value.get_arraynames():
            if int(name[-1]) > max_accuracy:
                max_accuracy = int(name[-1])

        array_data = orm.ArrayData()
        array_data.set_array(key, value.get_array(f'numerical_accuracy_{max_accuracy}'))
        output.update({key: array_data})

    return output


@calcfunction
def extract_orders(**kwargs):
    """Extract all the orders of all the tensors of a `DielectricWorkChain` output."""
    order_outputs = {}

    all_dielectric = kwargs['dielectric']
    all_born_charges = kwargs['born_charges']

    for order_name in all_dielectric.get_arraynames():
        dielectric_array = all_dielectric.get_array(order_name)
        born_charges_array = all_born_charges.get_array(order_name)

        order_array = orm.ArrayData()
        order_array.set_array('dielectric', dielectric_array)
        order_array.set_array('born_charges', born_charges_array)

        order_outputs[order_name] = {'nac_parameters': order_array}

    try:
        all_dph0 = kwargs['raman_tensors']
        all_nlo = kwargs['nlo_susceptibility']

        for order_name in all_dielectric.get_arraynames():
            dph0_array = all_dph0.get_array(order_name)
            nlo_array = all_nlo.get_array(order_name)

            order_dph0_array = orm.ArrayData()
            order_dph0_array.set_array('raman_tensors', dph0_array)

            order_nlo_array = orm.ArrayData()
            order_nlo_array.set_array('nlo_susceptibility', nlo_array)

            order_outputs[order_name].update({'raman_tensors': order_dph0_array})
            order_outputs[order_name].update({'nlo_susceptibility': order_nlo_array})
    except KeyError:
        pass

    return order_outputs


@calcfunction
def get_forces(trajectory: orm.TrajectoryData) -> orm.ArrayData:
    """Extract the `forces` array from a TrajectoryData."""
    from aiida.orm import ArrayData
    forces = ArrayData()
    forces.set_array('forces', trajectory.get_array('forces')[-1])
    return forces


@calcfunction
def get_energy(parameters: orm.Dict) -> orm.Float:
    """Convert the `energy` attribute of `parameters` into a Float."""
    from aiida.orm import Float
    return Float(parameters.base.attributes.get('energy'))


@calcfunction
def get_non_analytical_constants(dielectric: orm.ArrayData, born_charges: orm.ArrayData) -> orm.ArrayData:
    """Return a joint ArrayData  with dielectric and Born effective charges tensors.

    :param dielectric: ArrayData having an arrayname `dielectric`
    :param born_charges: ArrayData having an arrayname `born_charges`
    """
    nac_parameters = orm.ArrayData()
    nac_parameters.set_array('dielectric', dielectric.get_array('dielectric'))
    nac_parameters.set_array('born_charges', born_charges.get_array('born_charges'))

    return nac_parameters


@calcfunction
def elaborate_non_analytical_constants(
    preprocess_data: PreProcessData,
    dielectric=None,
    born_charges=None,
    nac_parameters=None,
):
    """Return the non analytical constants in the primitive cell.

    It uses the unique atoms referring to the supercell matrix.
    """
    from aiida.orm import ArrayData
    from phonopy.structure.symmetry import symmetrize_borns_and_epsilon

    if nac_parameters is None:
        ref_dielectric = dielectric.get_array('dielectric')
        ref_born_charges = born_charges.get_array('born_charges')
    else:
        ref_dielectric = nac_parameters.get_array('dielectric')
        ref_born_charges = nac_parameters.get_array('born_charges')

    nacs = symmetrize_borns_and_epsilon(
        # nac info
        borns=ref_born_charges,
        epsilon=ref_dielectric,
        # preprocess info
        ucell=preprocess_data.get_phonopy_instance().unitcell,
        primitive_matrix=preprocess_data.primitive_matrix,
        supercell_matrix=preprocess_data.supercell_matrix,
        is_symmetry=preprocess_data.is_symmetry,
        symprec=preprocess_data.symprec,
    )

    nac_parameters = ArrayData()
    nac_parameters.set_array('dielectric', nacs[1])
    nac_parameters.set_array('born_charges', nacs[0])

    return nac_parameters


@calcfunction
def extract_symmetry_info(preprocess_data: PreProcessData):
    """Return symmetry info for analysis."""
    return {
        'symprec': orm.Float(preprocess_data.symprec),
        'distinguish_kinds': orm.Bool(preprocess_data.distinguish_kinds),
        'is_symmetry': orm.Bool(preprocess_data.is_symmetry),
    }
