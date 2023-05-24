# -*- coding: utf-8 -*-
"""Tests for the :mod:`workflows.phonons.harmonic` module."""
from aiida import orm
import pytest

from aiida_vibroscopy.workflows.phonons.harmonic import HarmonicWorkChain


@pytest.fixture
def generate_phonon_workchain_node(generate_vibrational_data_from_forces):
    """Generate an instance of `WorkflowNode`."""

    def _generate_phonon_workchain_node(exit_status=0):
        from aiida.common import LinkType
        from plumpy import ProcessState

        node = orm.WorkflowNode().store()
        node.set_process_state(ProcessState.FINISHED)
        node.set_exit_status(exit_status)

        phonopy_data = generate_vibrational_data_from_forces().store()
        phonopy_data.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='phonopy_data')

        return node

    return _generate_phonon_workchain_node


@pytest.fixture
def generate_dielectric_workchain_node():
    """Generate an instance of `WorkflowNode`."""

    def _generate_dielectric_workchain_node(exit_status=0):
        from aiida.common import LinkType
        import numpy as np
        from plumpy import ProcessState

        node = orm.WorkflowNode().store()
        node.set_process_state(ProcessState.FINISHED)
        node.set_exit_status(exit_status)

        tensors = orm.ArrayData()
        tensors.set_array('dielectric', np.eye(3))
        tensors.set_array('born_charges', np.array([np.eye(3), -np.eye(3)]))
        tensors.set_array('raman_tensors', np.zeros((2, 3, 3, 3)))
        tensors.set_array('nlo_susceptibility', np.zeros((3, 3, 3)))
        tensors.store()
        tensors.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='tensors__numerical_accuracy_2')

        return node

    return _generate_dielectric_workchain_node


@pytest.fixture
def generate_workchain_harmonic(generate_workchain, generate_inputs_pw_base):
    """Generate an instance of a `HarmonicWorkChain`."""

    def _generate_workchain_harmonic(append_inputs=None, phonon_inputs=None, return_inputs=False):
        entry_point = 'vibroscopy.phonons.harmonic'
        scf_inputs = generate_inputs_pw_base()
        structure = scf_inputs['pw'].pop('structure')

        inputs = {
            'structure': structure,
            'phonon': {
                'scf': scf_inputs,
                'settings': {
                    'sleep_submission_time': 0.,
                }
            },
            'dielectric': {
                'property': 'raman',
                'scf': scf_inputs,
                'settings': {
                    'sleep_submission_time': 0.,
                }
            },
            'settings': {},
            'symmetry': {},
        }

        if return_inputs:
            return inputs

        if phonon_inputs is not None:
            inputs['phonon'].update(phonon_inputs)

        if append_inputs is not None:
            inputs.update(append_inputs)

        process = generate_workchain(entry_point, inputs)

        return process

    return _generate_workchain_harmonic


@pytest.mark.usefixtures('aiida_profile')
def test_valididation_inputs(generate_workchain_harmonic):
    """Test `HarmonicWorkChain` inizialisation with secure inputs."""
    generate_workchain_harmonic()


@pytest.mark.usefixtures('aiida_profile')
def test_invalid_inputs(generate_workchain_harmonic, generate_structure):
    """Test `HarmonicWorkChain` validation methods."""
    inputs = {'structure': generate_structure(hubbard=True), 'settings': {'use_primitive_cell': orm.Bool(True)}}
    match = r'`use_primitive_cell` cannot currently be used with `HubbardStructureData` inputs.'
    with pytest.raises(ValueError, match=match):
        generate_workchain_harmonic(append_inputs=inputs)


@pytest.mark.usefixtures('aiida_profile')
def test_validation_no_dielectric(generate_workchain_harmonic):
    """Test `HarmonicWorkChain` initialization method."""
    inputs = generate_workchain_harmonic(return_inputs=True)
    inputs.pop('dielectric')
    generate_workchain_harmonic(append_inputs=inputs)


@pytest.mark.usefixtures('aiida_profile')
def test_setup(generate_workchain_harmonic):
    """Test `HarmonicWorkChain.setup`."""
    symmetry = {'symprec': orm.Float(1.0), 'distinguish_kinds': orm.Bool(True), 'is_symmetry': orm.Bool(False)}
    process = generate_workchain_harmonic(append_inputs={'symmetry': symmetry})
    process.setup()
    assert 'preprocess_data' in process.ctx
    data = process.ctx.preprocess_data
    assert data.symprec == 1.0
    assert data.distinguish_kinds == True
    assert data.is_symmetry == False


@pytest.mark.usefixtures('aiida_profile')
def test_run_phonon(generate_workchain_harmonic):
    """Test `HarmonicWorkChain.run_phonon`."""
    process = generate_workchain_harmonic()
    process.setup()
    process.run_phonon()
    assert 'phonon' in process.ctx


@pytest.mark.usefixtures('aiida_profile')
def test_run_dielectric(generate_workchain_harmonic):
    """Test `HarmonicWorkChain.run_dielectric`."""
    process = generate_workchain_harmonic()
    process.setup()
    process.run_dielectric()
    assert 'dielectric' in process.ctx


@pytest.mark.usefixtures('aiida_profile')
def test_run_dielectric_with_primitive(generate_workchain_harmonic):
    """Test `HarmonicWorkChain.run_dielectric` with primitive cell."""
    process = generate_workchain_harmonic(append_inputs={'settings': {'use_primitive_cell': orm.Bool(True)}})
    process.setup()
    process.run_dielectric()
    assert 'dielectric' in process.ctx


@pytest.mark.parametrize(('expected_result', 'exit_status'),
                         ((None, 0), (HarmonicWorkChain.exit_codes.ERROR_PHONON_WORKCHAIN_FAILED, 312)))
@pytest.mark.usefixtures('aiida_profile')
def test_inspect_processes(generate_workchain_harmonic, generate_phonon_workchain_node, expected_result, exit_status):
    """Test `HarmonicWorkChain.inspect_processes` only with PhononWorkChain."""
    process = generate_workchain_harmonic()
    process.ctx.phonon = generate_phonon_workchain_node(exit_status=exit_status)
    result = process.inspect_processes()
    assert result == expected_result
    if exit_status == 0:
        assert 'output_phonon' in process.outputs


@pytest.mark.parametrize(('expected_result', 'exit_status'),
                         ((None, 0), (HarmonicWorkChain.exit_codes.ERROR_DIELECTRIC_WORKCHAIN_FAILED, 312)))
@pytest.mark.usefixtures('aiida_profile')
def test_inspect_processes_with_dielectric(
    generate_workchain_harmonic, generate_phonon_workchain_node, generate_dielectric_workchain_node, expected_result,
    exit_status
):
    """Test `HarmonicWorkChain.inspect_processes` with `DielectricWorkChain`."""
    process = generate_workchain_harmonic()
    process.ctx.phonon = generate_phonon_workchain_node()
    process.ctx.dielectric = generate_dielectric_workchain_node(exit_status=exit_status)
    result = process.inspect_processes()
    assert result == expected_result
    if exit_status == 0:
        assert 'output_phonon' in process.outputs
        assert 'output_dielectric' in process.outputs


@pytest.mark.usefixtures('aiida_profile')
def test_run_vibrational_data(
    generate_workchain_harmonic,
    generate_phonon_workchain_node,
    generate_dielectric_workchain_node,
):
    """Test `HarmonicWorkChain.run_vibrational_data`."""
    process = generate_workchain_harmonic()
    process.setup()
    process.ctx.phonon = generate_phonon_workchain_node()
    process.ctx.dielectric = generate_dielectric_workchain_node()
    process.run_vibrational_data()
    assert 'numerical_accuracy_2' in process.outputs['vibrational_data']


@pytest.mark.usefixtures('aiida_profile')
def test_should_run_phonopy(generate_workchain_harmonic, generate_inputs_phonopy):
    """Test `HarmonicWorkChain.should_run_phonopy` method."""
    process = generate_workchain_harmonic()
    assert not process.should_run_phonopy()

    inputs = {'phonopy': generate_inputs_phonopy()}
    process = generate_workchain_harmonic(append_inputs=inputs)
    assert process.should_run_phonopy()


@pytest.mark.usefixtures('aiida_profile')
def test_run_phonopy(
    generate_workchain_harmonic, generate_inputs_phonopy, generate_phonon_workchain_node,
    generate_dielectric_workchain_node
):
    """Test `HarmonicWorkChain.run_phonopy` method."""
    inputs = {'phonopy': generate_inputs_phonopy()}
    process = generate_workchain_harmonic(append_inputs=inputs)
    process.setup()
    process.ctx.phonon = generate_phonon_workchain_node()
    process.ctx.dielectric = generate_dielectric_workchain_node()
    process.run_vibrational_data()
    process.run_phonopy()

    assert 'phonopy' in process.ctx


@pytest.mark.parametrize(('expected_result', 'exit_status'),
                         ((None, 0), (HarmonicWorkChain.exit_codes.ERROR_PHONOPY_CALCULATION_FAILED, 312)))
@pytest.mark.usefixtures('aiida_profile')
def test_inspect_phonopy(generate_workchain_harmonic, generate_phonopy_calcjob_node, expected_result, exit_status):
    """Test `HarmonicWorkChain.inspect_phonopy` method."""
    process = generate_workchain_harmonic()
    process.ctx.phonopy = generate_phonopy_calcjob_node(exit_status=exit_status)

    results = process.inspect_phonopy()
    assert results == expected_result
    if exit_status == 0:
        assert 'output_phonopy' in process.outputs
        assert 'output_parameters' in process.outputs['output_phonopy']
