# -*- coding: utf-8 -*-
"""Tests for the :mod:`workflows.phonons.iraman` module."""
import pytest


@pytest.fixture
def generate_workchain_iraman(
    generate_workchain, generate_structure, generate_inputs_pw_base, generate_inputs_dielectric
):
    """Generate an instance of a `IRamanSpectraWorkChain`."""

    def _generate_workchain_iraman(append_inputs=None, return_inputs=False):
        entry_point = 'vibroscopy.spectra.iraman'

        structure = generate_structure()

        phonon_inputs = generate_inputs_pw_base()
        phonon_inputs['pw'].pop('structure')

        dielectric_inputs = generate_inputs_dielectric(electric_field_scale=1.0)
        dielectric_inputs['scf']['pw'].pop('structure')
        dielectric_inputs.pop('clean_workdir')

        inputs = {
            'structure': structure,
            'phonon_workchain': {
                'scf': phonon_inputs,
            },
            'dielectric_workchain': dielectric_inputs,
        }

        if return_inputs:
            return inputs

        if append_inputs is not None:
            inputs.update(append_inputs)

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


#@pytest.mark.usefixtures('aiida_profile_clean')
def test_setup(generate_workchain_iraman):
    """Test `IRamanSpectraWorkChain` setup method."""
    process = generate_workchain_iraman()
    process.setup()

    for key in ('preprocess_data', 'is_magnetic', 'plus_hubbard'):
        assert key in process.ctx

    assert process.ctx.run_parallel == True
    assert process.ctx.plus_hubbard == False


#@pytest.mark.usefixtures('aiida_profile_clean')
def test_run_forces(generate_workchain_iraman, generate_base_scf_workchain_node):
    """Test `IRamanSpectraWorkChain.run_forces` method."""
    append_inputs = {'options': {'sleep_submission_time': 0.1}}
    process = generate_workchain_iraman(append_inputs=append_inputs)

    process.setup()
    process.run_base_supercell()

    assert 'scf_supercell_0' in process.ctx

    process.ctx.scf_supercell_0 = generate_base_scf_workchain_node()
    process.run_forces()

    assert 'supercells' in process.outputs
    assert 'supercell_1' in process.outputs['supercells']
    assert 'supercell_2' in process.outputs['supercells']
    assert 'scf_supercell_1' in process.ctx


#@pytest.mark.usefixtures('aiida_profile_clean')
def test_run_dielectric(generate_workchain_iraman, generate_base_scf_workchain_node):
    """Test `IRamanSpectraWorkChain.run_dielectric` method."""
    append_inputs = {'options': {'sleep_submission_time': 0.1}}
    process = generate_workchain_iraman(append_inputs=append_inputs)

    process.setup()
    process.run_base_supercell()
    process.ctx.scf_supercell_0 = generate_base_scf_workchain_node()

    process.run_dielectric()

    assert 'dielectric_workchain' in process.ctx


@pytest.mark.parametrize(
    ('raman'),
    ((True), (False)),
)
#@pytest.mark.usefixtures('aiida_profile_clean')
def test_run_raw_results(generate_workchain_iraman, generate_dielectric_workchain_node, raman):
    """Test `IRamanSpectraWorkChain.run_raw_results` method."""
    from aiida import orm
    import numpy

    process = generate_workchain_iraman()

    process.setup()
    process.ctx.dielectric_workchain = generate_dielectric_workchain_node(raman=raman)

    forces_1 = orm.ArrayData()
    forces_1.set_array('forces', numpy.full((2, 3), 1))
    forces_1.store()
    forces_2 = orm.ArrayData()
    forces_2.set_array('forces', numpy.full((2, 3), -1))
    forces_2.store()

    process.out(f'supercells_forces.scf_supercell_1', forces_1)
    process.out(f'supercells_forces.scf_supercell_2', forces_2)

    process.run_raw_results()

    assert 'vibrational_data' in process.outputs
    assert 'vibrational_data' in process.ctx

    assert 'numerical_accuracy_2_step_1' in process.outputs['vibrational_data']
    assert 'numerical_accuracy_4' in process.outputs['vibrational_data']


#@pytest.mark.usefixtures('aiida_profile_clean')
def test_run_intensities_averaged(generate_workchain_iraman, generate_vibrational_data):
    """Test `IRamanSpectraWorkChain.run_intensities_averaged` method."""
    process = generate_workchain_iraman()

    process.ctx.vibrational_data = {
        'numerical_order_2_step_1': generate_vibrational_data(),
        'numerical_order_4': generate_vibrational_data(),
    }

    process.run_intensities_averaged()

    assert 'intensities_average' in process.ctx
    assert 'numerical_order_2_step_1' in process.ctx.intensities_average
    assert 'numerical_order_4' in process.ctx.intensities_average

    for key in process.ctx.intensities_average:
        assert key in ['numerical_order_2_step_1', 'numerical_order_4']


#@pytest.mark.usefixtures('aiida_profile_clean')
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
