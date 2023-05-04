# -*- coding: utf-8 -*-
"""Tests for the :mod:`workflows.phonons.iraman` module."""
import pytest


@pytest.fixture
def generate_workchain_iraman(generate_workchain, generate_inputs_iraman):
    """Generate an instance of a `DielectricWorkChain`."""

    def _generate_workchain_iraman(inputs=None, **kwargs):
        entry_point = 'vibroscopy.spectra.iraman'

        if inputs is None:
            inputs = generate_inputs_iraman(**kwargs)

        process = generate_workchain(entry_point, inputs)

        return process

    return _generate_workchain_iraman


@pytest.fixture
def generate_intensities_workchain_node():
    """Generate an instance of `WorkflowNode`."""

    def _generate_intensities_workchain_node():
        from aiida.common import LinkType
        from aiida.orm import ArrayData, WorkflowNode

        node = WorkflowNode().store()

        ir_array = ArrayData().store()
        raman_array = ArrayData().store()

        ir_array.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='ir_averaged')
        raman_array.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='raman_averaged')

        return node

    return _generate_intensities_workchain_node


@pytest.mark.usefixtures('aiida_profile')
def test_setup(generate_workchain_iraman):
    """Test `IRamanSpectraWorkChain` setup method."""
    process = generate_workchain_iraman()
    process.setup()

    for key in ('preprocess_data', 'is_magnetic', 'plus_hubbard'):
        assert key in process.ctx

    assert process.ctx.run_parallel == True
    assert process.ctx.plus_hubbard == False


@pytest.mark.usefixtures('aiida_profile')
def test_run_forces(generate_workchain_iraman, generate_base_scf_workchain_node):
    """Test `IRamanSpectraWorkChain.run_forces` method."""
    append_inputs = {'options': {'sleep_submission_time': 0.1}}
    process = generate_workchain_iraman(append_inputs=append_inputs)

    process.setup()
    process.run_base_supercell()
    process.set_reference_kpoints()

    assert 'scf_supercell_0' in process.ctx

    process.ctx.scf_supercell_0 = generate_base_scf_workchain_node()
    process.run_forces()

    assert 'supercells' in process.outputs
    assert 'supercell_1' in process.outputs['supercells']
    assert 'supercell_2' in process.outputs['supercells']
    assert 'scf_supercell_1' in process.ctx


@pytest.mark.usefixtures('aiida_profile')
def test_run_dielectric(generate_workchain_iraman, generate_base_scf_workchain_node):
    """Test `IRamanSpectraWorkChain.run_dielectric` method."""
    append_inputs = {'options': {'sleep_submission_time': 0.1}}
    process = generate_workchain_iraman(append_inputs=append_inputs)

    process.setup()
    process.run_base_supercell()
    process.set_reference_kpoints()

    process.ctx.scf_supercell_0 = generate_base_scf_workchain_node()

    process.run_dielectric()

    assert 'dielectric_workchain' in process.ctx


@pytest.mark.parametrize(
    ('raman'),
    ((True), (False)),
)
@pytest.mark.usefixtures('aiida_profile')
def test_run_raw_results(generate_workchain_iraman, generate_dielectric_workchain_node, raman, generate_trajectory):
    """Test `IRamanSpectraWorkChain.run_raw_results` method."""
    process = generate_workchain_iraman()

    process.setup()
    process.ctx.dielectric_workchain = generate_dielectric_workchain_node(raman=raman)

    forces_1 = generate_trajectory()
    forces_2 = generate_trajectory()

    process.out(f'supercells_forces.forces_1', forces_1)
    process.out(f'supercells_forces.forces_2', forces_2)

    process.set_phonopy_data()
    process.run_raw_results()

    assert 'vibrational_data' in process.outputs
    assert 'vibrational_data' in process.ctx

    assert 'numerical_accuracy_2_step_1' in process.outputs['vibrational_data']
    assert 'numerical_accuracy_4' in process.outputs['vibrational_data']


@pytest.mark.usefixtures('aiida_profile')
def test_run_intensities_averaged(generate_workchain_iraman, generate_vibrational_data_from_forces):
    """Test `IRamanSpectraWorkChain.run_intensities_averaged` method."""
    from aiida.orm import Dict
    options = Dict({'quadrature_order': 3})
    process = generate_workchain_iraman(append_inputs={'intensities_average': {'options': options}})

    process.ctx.vibrational_data = {
        'numerical_order_2_step_1': generate_vibrational_data_from_forces(),
        'numerical_order_4': generate_vibrational_data_from_forces(),
    }

    process.run_intensities_averaged()

    assert 'intensities_average' in process.ctx
    assert 'numerical_order_2_step_1' in process.ctx.intensities_average
    assert 'numerical_order_4' in process.ctx.intensities_average

    for key in process.ctx.intensities_average:
        assert key in ['numerical_order_2_step_1', 'numerical_order_4']


@pytest.mark.usefixtures('aiida_profile')
def test_show_results(generate_workchain_iraman, generate_intensities_workchain_node):
    """Test `IRamanSpectraWorkChain.show_results` method."""
    from aiida.common import AttributeDict
    process = generate_workchain_iraman()

    process.ctx.intensities_average = AttributeDict({
        'numerical_order_2_step_1': generate_intensities_workchain_node(),
        'numerical_order_4': generate_intensities_workchain_node(),
    })

    # Sanity check
    assert 'ir_averaged' in process.ctx.intensities_average['numerical_order_4'].outputs
    assert 'raman_averaged' in process.ctx.intensities_average['numerical_order_4'].outputs

    process.show_results()

    assert 'output_intensities_average' in process.outputs
    assert 'numerical_order_4' in process.outputs['output_intensities_average']
    assert 'numerical_order_2_step_1' in process.outputs['output_intensities_average']

    for key in ['ir_averaged', 'raman_averaged']:
        assert key in process.outputs['output_intensities_average']['numerical_order_4']
        assert key in process.outputs['output_intensities_average']['numerical_order_2_step_1']
