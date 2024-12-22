# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Tests for the :mod:`workflows.phonons.phonon` module."""
from aiida import orm
from aiida.common import AttributeDict
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
import pytest

from aiida_vibroscopy.workflows.phonons.base import PhononWorkChain


@pytest.fixture
def generate_workchain_phonon(generate_workchain, generate_inputs_pw_base):
    """Generate an instance of a `PhononWorkChain`."""

    def _generate_workchain_phonon(structure=None, append_inputs=None, return_inputs=False, the_inputs=None):
        entry_point = 'vibroscopy.phonons.phonon'
        scf_inputs = generate_inputs_pw_base()

        if structure is not None:
            scf_inputs['pw'].pop('structure')
            scf_inputs['pw']['structure'] = structure

        inputs = {
            'scf': scf_inputs,
            'settings': {
                'sleep_submission_time': 0.,
                'max_concurrent_base_workchains': -1,
            },
            'symmetry': {},
        }

        if return_inputs:
            return AttributeDict(inputs)

        if append_inputs is not None:
            inputs.update(append_inputs)

        if the_inputs is not None:
            inputs = the_inputs

        process = generate_workchain(entry_point, inputs)

        return process

    return _generate_workchain_phonon


@pytest.mark.usefixtures('aiida_profile')
def test_valididation_inputs(generate_workchain_phonon):
    """Test `PhononWorkChain` inizialisation with standard inputs."""
    generate_workchain_phonon()


@pytest.mark.parametrize(
    ('parameters', 'message'),
    (({
        'supercell_matrix': [1, 1]
    }, 'need exactly 3 diagonal elements or 3x3 arrays.'), ({
        'supercell_matrix': [[1], [1], [1]]
    }, 'matrix need to have 3x1 or 3x3 shape.'), ({
        'displacement_generator': {
            'invalid': 1
        }
    }, "Unknown flags in 'displacements': {'invalid'}."), ({
        'displacement_generator': {
            'distance': True
        }
    }, 'Displacement options must be of the correct type; got invalid values [True].')),
)
@pytest.mark.usefixtures('aiida_profile')
def test_invalid_inputs(generate_workchain_phonon, parameters, message):
    """Test `PhononWorkChain` validation methods."""
    if 'supercell_matrix' in parameters:
        inputs = generate_workchain_phonon(return_inputs=True)
        inputs.update({'supercell_matrix': orm.List(parameters['supercell_matrix'])})
        parameters = inputs

    if 'displacement_generator' in parameters:
        inputs = generate_workchain_phonon(return_inputs=True)
        inputs.update({'displacement_generator': orm.Dict(parameters['displacement_generator'])})
        parameters = inputs

    with pytest.raises(ValueError) as exception:
        generate_workchain_phonon(append_inputs=parameters)

    assert message in str(exception.value)


@pytest.mark.usefixtures('aiida_profile')
def test_setup(generate_workchain_phonon):
    """Test `PhononWorkChain` setup method."""
    process = generate_workchain_phonon()
    process.setup()

    assert process.ctx.is_magnetic == False
    assert process.ctx.is_insulator == True
    assert process.ctx.plus_hubbard == False
    assert process.ctx.old_plus_hubbard == False
    assert 'preprocess_data' in process.ctx

    data = process.ctx.preprocess_data
    assert data.is_symmetry
    assert data.symprec == 1e-5
    assert data.supercell_matrix == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert not data.distinguish_kinds

    assert 'supercell' in process.ctx


@pytest.mark.usefixtures('aiida_profile')
def test_setup_phonon_options(generate_workchain_phonon):
    """Test `PhononWorkChain.setup` with input options."""
    append_inputs = {
        'supercell_matrix': orm.List([2, 2, 2]),
        'primitive_matrix': orm.List([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        'displacement_generator': orm.Dict({'number_of_snapshots': 5}),
        'symmetry': {
            'symprec': orm.Float(1.0),
            'distinguish_kinds': orm.Bool(False),
            'is_symmetry': orm.Bool(False),
        }
    }
    process = generate_workchain_phonon(append_inputs=append_inputs)
    process.setup()

    data = process.ctx.preprocess_data
    assert data.is_symmetry == append_inputs['symmetry']['is_symmetry'].value
    assert data.symprec == append_inputs['symmetry']['symprec'].value
    assert data.distinguish_kinds == append_inputs['symmetry']['distinguish_kinds'].value
    assert data.supercell_matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert data.primitive_matrix == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert len(data.displacements) == 5


@pytest.mark.usefixtures('aiida_profile')
def test_magnetic_old_hubbard_setup(generate_workchain_phonon):
    """Test `PhononWorkChain.setup` with magnetic and old hubbard."""
    inputs = generate_workchain_phonon(return_inputs=True)
    inputs['scf']['pw']['parameters']['SYSTEM']['lda_plus_u_kind'] = 2
    inputs['scf']['pw']['parameters']['SYSTEM']['nspin'] = 2
    process = generate_workchain_phonon(append_inputs=inputs)

    process.setup()

    assert process.ctx.is_magnetic
    assert not process.ctx.plus_hubbard
    assert process.ctx.old_plus_hubbard

    assert 'preprocess_data' in process.ctx

    data = process.ctx.preprocess_data
    assert data.is_symmetry
    assert data.symprec == 1e-5
    assert data.supercell_matrix == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert not data.distinguish_kinds

    assert 'supercell' in process.ctx


@pytest.mark.usefixtures('aiida_profile')
def test_hubbard_setup(generate_workchain_phonon, generate_structure):
    """Test `PhononWorkChain.setup` with hubbard structure."""
    hubbard_structure = generate_structure(hubbard=True)
    process = generate_workchain_phonon(structure=hubbard_structure)

    process.setup()

    assert not process.ctx.is_magnetic
    assert process.ctx.plus_hubbard
    assert not process.ctx.old_plus_hubbard

    assert 'preprocess_data' in process.ctx

    data = process.ctx.preprocess_data
    assert data.is_symmetry
    assert data.symprec == 1e-5
    assert data.supercell_matrix == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert not data.distinguish_kinds

    assert 'supercell' in process.ctx
    assert isinstance(process.ctx.supercell, HubbardStructureData)


@pytest.mark.usefixtures('aiida_profile')
def test_set_reference_kpoints(generate_workchain_phonon):
    """Test `PhononWorkChain.set_reference_kpoints` method."""
    process = generate_workchain_phonon()
    process.setup()
    process.set_reference_kpoints()

    assert 'kpoints' in process.ctx

    inputs = generate_workchain_phonon(return_inputs=True)
    inputs['scf'].pop('kpoints')
    inputs['scf']['kpoints_distance'] = 0.1
    process = generate_workchain_phonon(the_inputs=inputs)
    process.setup()
    process.set_reference_kpoints()

    assert 'kpoints' in process.ctx


@pytest.mark.usefixtures('aiida_profile')
def test_run_base_supercell(generate_workchain_phonon):
    """Test `PhononWorkChain.run_base_supercell`."""
    process = generate_workchain_phonon()

    process.setup()
    process.set_reference_kpoints()
    process.run_base_supercell()

    assert 'scf_supercell_0' in process.ctx


@pytest.mark.parametrize(('expected_result', 'exit_status'),
                         ((None, 0), (PhononWorkChain.exit_codes.ERROR_FAILED_BASE_SCF, 312)))
@pytest.mark.usefixtures('aiida_profile')
def test_inspect_base_supercell(
    generate_workchain_phonon, generate_base_scf_workchain_node, expected_result, exit_status
):
    """Test `PhononWorkChain.inspect_base_supercell`."""
    process = generate_workchain_phonon()

    process.setup()
    process.set_reference_kpoints()

    process.ctx.scf_supercell_0 = generate_base_scf_workchain_node(exit_status=exit_status)
    result = process.inspect_base_supercell()
    assert result == expected_result
    assert process.ctx.is_insulator


@pytest.mark.usefixtures('aiida_profile')
def test_run_supercells(generate_workchain_phonon):
    """Test `PhononWorkChain.run_supercells` method."""
    process = generate_workchain_phonon()
    process.setup()
    process.run_supercells()

    assert 'supercells' in process.outputs
    assert 'supercells' in process.ctx
    assert 'supercell_1' in process.outputs['supercells']


@pytest.mark.usefixtures('aiida_profile')
def test_should_run_forces(generate_workchain_phonon):
    """Test `PhononWorkChain.should_run_forces` method."""
    process = generate_workchain_phonon()
    process.setup()
    process.run_supercells()
    assert process.should_run_forces()


@pytest.mark.usefixtures('aiida_profile')
def test_run_forces(generate_workchain_phonon, generate_base_scf_workchain_node):
    """Test `PhononWorkChain.run_forces` method."""
    append_inputs = {
        'settings': {
            'sleep_submission_time': 0.,
            'max_concurrent_base_workchains': 1,
        }
    }
    process = generate_workchain_phonon(append_inputs=append_inputs)

    process.setup()
    process.set_reference_kpoints()
    process.run_base_supercell()
    process.run_supercells()

    assert 'scf_supercell_0'

    num_supercells = len(process.ctx.supercells)
    process.ctx.scf_supercell_0 = generate_base_scf_workchain_node()
    process.run_forces()

    assert 'supercells' in process.outputs
    assert 'supercell_1' in process.outputs['supercells']
    assert 'scf_supercell_1' in process.ctx
    assert num_supercells == len(process.ctx.supercells) + 1


@pytest.mark.usefixtures('aiida_profile')
def test_run_forces_with_hubbard(generate_workchain_phonon, generate_base_scf_workchain_node, generate_structure):
    """Test `PhononWorkChain.run_forces` with HubbardStructureData."""
    hubbard_structure = generate_structure(hubbard=True)
    process = generate_workchain_phonon(structure=hubbard_structure)

    process.setup()
    process.set_reference_kpoints()
    process.run_base_supercell()
    process.run_supercells()

    assert 'scf_supercell_0'

    process.ctx.scf_supercell_0 = generate_base_scf_workchain_node()
    process.run_forces()

    assert 'supercells' in process.outputs
    assert 'supercell_1' in process.outputs['supercells']
    assert isinstance(process.outputs['supercells']['supercell_1'], HubbardStructureData)
    assert 'scf_supercell_1' in process.ctx
    assert len(process.ctx.supercells) == 0


@pytest.mark.parametrize(('expected_result', 'exit_status'),
                         ((None, 0), (PhononWorkChain.exit_codes.ERROR_SUB_PROCESS_FAILED, 312)))
@pytest.mark.usefixtures('aiida_profile')
def test_inspect_all_runs(generate_workchain_phonon, generate_base_scf_workchain_node, expected_result, exit_status):
    """Test `PhononWorkChain.inspect_all_runs`."""
    process = generate_workchain_phonon()

    process.setup()
    process.set_reference_kpoints()

    process.ctx.scf_supercell_1 = generate_base_scf_workchain_node(exit_status=0, with_trajectory=True)
    process.ctx.scf_supercell_2 = generate_base_scf_workchain_node(exit_status=exit_status, with_trajectory=True)

    result = process.inspect_all_runs()
    assert result == expected_result


@pytest.mark.usefixtures('aiida_profile')
def test_set_phonopy_data(generate_workchain_phonon, generate_trajectory):
    """Test `PhononWorkChain.set_phonopy_data`."""
    process = generate_workchain_phonon()

    process.setup()
    process.set_reference_kpoints()

    forces_1 = generate_trajectory()
    forces_2 = generate_trajectory()

    process.out(f'supercells_forces.forces_1', forces_1)
    process.out(f'supercells_forces.forces_2', forces_2)

    process.set_phonopy_data()
    assert 'phonopy_data' in process.ctx
    assert 'phonopy_data' in process.outputs


@pytest.mark.usefixtures('aiida_profile')
def test_should_run_phonopy(generate_workchain_phonon, generate_inputs_phonopy):
    """Test `PhononWorkChain.should_run_phonopy` method."""
    process = generate_workchain_phonon()
    assert not process.should_run_phonopy()

    inputs = {'phonopy': generate_inputs_phonopy()}
    process = generate_workchain_phonon(append_inputs=inputs)
    assert process.should_run_phonopy()


@pytest.mark.usefixtures('aiida_profile')
def test_run_phonopy(generate_workchain_phonon, generate_inputs_phonopy, generate_trajectory):
    """Test `PhononWorkChain.run_phonopy` method."""
    inputs = {'phonopy': generate_inputs_phonopy()}
    process = generate_workchain_phonon(append_inputs=inputs)
    process.setup()
    forces_1 = generate_trajectory()
    forces_2 = generate_trajectory()
    process.out(f'supercells_forces.forces_1', forces_1)
    process.out(f'supercells_forces.forces_2', forces_2)
    process.set_phonopy_data()
    process.run_phonopy()

    assert 'phonopy_calculation' in process.ctx


@pytest.mark.parametrize(('expected_result', 'exit_status'),
                         ((None, 0), (PhononWorkChain.exit_codes.ERROR_PHONOPY_CALCULATION_FAILED, 312)))
@pytest.mark.usefixtures('aiida_profile')
def test_inspect_phonopy(generate_workchain_phonon, generate_phonopy_calcjob_node, expected_result, exit_status):
    """Test `PhononWorkChain.inspect_phonopy` method."""
    process = generate_workchain_phonon()
    process.ctx.phonopy_calculation = generate_phonopy_calcjob_node(exit_status=exit_status)

    results = process.inspect_phonopy()
    assert results == expected_result
    if exit_status == 0:
        assert 'output_phonopy' in process.outputs
        assert 'output_parameters' in process.outputs['output_phonopy']
