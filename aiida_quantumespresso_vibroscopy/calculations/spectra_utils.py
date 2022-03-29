# -*- coding: utf-8 -*-
"""Calcfunctions utils for spectra workflows."""

from aiida import orm
from aiida.engine import calcfunction
from aiida.plugins import DataFactory

PreProcessData = DataFactory('phonopy.preprocess')

__all__ = (
    'get_supercells_for_hubbard',
    'generate_vibrational_data',
)

@calcfunction
def get_supercells_for_hubbard(preprocess_data: PreProcessData, ref_structure: orm.StructureData):

    displacements = preprocess_data.get_displacements()
    structures = {}

    for i, displacement in enumerate(displacements):
        traslation = [ [0.,0.,0.] for _ in ref_structure.sites]

        traslation[displacement[0]] = displacement[1:4]
        ase = ref_structure.get_ase().copy()
        ase.translate(traslation)
        ase_dict = ase.todict()
        cell = ase_dict['cell']
        positions = ase_dict['positions']
        kinds = [site.kind_name for site in ref_structure.sites ]
        symbols = ase.get_chemical_symbols()

        structure = orm.StructureData(cell=cell)

        for position, symbol, name in zip(positions, symbols, kinds):
            structure.append_atom(position=position, symbols=symbol, name=name)

        structures.update({f'supercell_{i+1}':structure})

    return structures

@calcfunction
def generate_vibrational_data(
    preprocess_data: PreProcessData,
    nac_parameters: orm.ArrayData,
    dph0_susceptibility = None,
    nlo_susceptibility = None,
    **forces_dict):
    """Create a VibrationalFrozenPhononData node from a PreProcessData node, storing forces and dielectric properties
    for spectra calculation.

    `Forces` must be passed as **kwargs**, since we are calling a calcfunction with a variable
    number of supercells forces.

    :param forces_dict: dictionary of supercells forces as ArrayData stored as `forces`, each Data
        labelled in the dictionary in the format `{prefix}_{suffix}`.
        The prefix is common and the suffix corresponds to the suffix number of the supercell with
        displacement label given from the `get_supercells_with_displacements` method.

        For example:
            {'forces_1':ArrayData, 'forces_2':ArrayData} <==> {'supercell_1':StructureData, 'supercell_2':StructureData}
            and forces in each ArrayData stored as 'forces', i.e. ArrayData.get_array('forces') must not raise error

        .. note: if residual forces would be stored, label it with 0 as suffix.
    """
    import numpy as np
    from aiida_quantumespresso_vibroscopy.data.vibro_fp import VibrationalFrozenPhononData

    # Getting the prefix
    for key in forces_dict.keys():
        try:
            int(key.split('_')[-1])
            # dumb way of getting the prefix including cases of multiple `_`, e.g. `force_calculation_001`
            prefix = key[:-(len(key.split('_')[-1])+1)]
        except ValueError:
            raise ValueError(f'{key} is not an acceptable key, must finish with an int number.')

    forces_0 = forces_dict.pop(f'{prefix}_0', None)

    # Setting force sets array
    force_sets = [0 for _ in forces_dict.keys()]

    # Filling arrays in numeric order determined by the key label
    for key, value in forces_dict.items():
        index = int(key.split('_')[-1])
        force_sets[index - 1] = value.get_array('forces')

    # Finilizing force sets array
    sets_of_forces = np.array(force_sets)

    # Setting data on a new PhonopyData
    vibrational_data = VibrationalFrozenPhononData(preprocess_data=preprocess_data)
    vibrational_data.set_forces(sets_of_forces=sets_of_forces)

    if forces_0 is not None:
        vibrational_data.set_residual_forces(forces=forces_0.get_array('forces'))

    vibrational_data.set_dielectric(nac_parameters.get_array('dielectric'))
    vibrational_data.set_born_charges(nac_parameters.get_array('born_charges'))
    if dph0_susceptibility is not None:
        vibrational_data.set_dph0_susceptibility(dph0_susceptibility.get_array('dph0_susceptibility'))
    if nlo_susceptibility is not None:
        vibrational_data.set_nlo_susceptibility(nlo_susceptibility.get_array('nlo_susceptibility'))

    return vibrational_data
