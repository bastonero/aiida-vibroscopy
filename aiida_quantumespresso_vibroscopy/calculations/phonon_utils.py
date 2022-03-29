# -*- coding: utf-8 -*-
"""Calcfunctions utils for extracting outputs related to phonon workflows."""

from aiida import orm
from aiida.engine import calcfunction
from aiida.plugins import DataFactory

PreProcessData = DataFactory('phonopy.preprocess')

__all__ = (
    'extract_max_order',
    'get_forces',
    'get_energy',
    'get_non_analytical_constants',
    'elaborate_non_analytical_constants'
)

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
        output.update({key:array_data})

    return array_data

@calcfunction
def get_forces(trajectory: orm.TrajectoryData):
    from aiida.orm import ArrayData
    forces = ArrayData()
    forces.set_array('forces', trajectory.get_array('forces'))
    return forces

@calcfunction
def get_energy(parameters: orm.Dict):
    from aiida.orm import Float
    return Float(parameters.get_attribute('energy'))

@calcfunction
def get_non_analytical_constants(dielectric: orm.ArrayData, born_charges: orm.ArrayData):
    nac_parameters = orm.ArrayData()
    nac_parameters.set_array('dielectric', dielectric.get_array('dielectric'))
    nac_parameters.set_array('born_charges', born_charges.get_array('born_charges'))

    return nac_parameters

@calcfunction
def elaborate_non_analytical_constants(
        preprocess_data: PreProcessData,
        dielectric: orm.ArrayData,
        born_charges: orm.ArrayData,
    ):
    from aiida.orm import ArrayData
    from phonopy.structure.symmetry import elaborate_borns_and_epsilon

    ref_dielectric = dielectric.get_array('dielectric')
    ref_born_charges = born_charges.get_array('born_charges')

    nacs= elaborate_borns_and_epsilon(
        # preprocess info
        ucell=preprocess_data.get_phonopy_instance().unitcell,
        primitive_matrix=preprocess_data.primitive_matrix,
        supercell_matrix=preprocess_data.supercell_matrix,
        is_symmetry=preprocess_data.is_symmetry,
        symprec=preprocess_data.symprec,
        # nac info
        borns=ref_born_charges,
        epsilon=ref_dielectric,
    )

    nac_parameters = ArrayData()
    nac_parameters.set_array('dielectric', nacs[1])
    nac_parameters.set_array('born_charges', nacs[0])

    return nac_parameters
