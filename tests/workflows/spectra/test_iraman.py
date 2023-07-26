# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Tests for the :mod:`workflows.phonons.iraman` module."""
import pytest

from aiida_vibroscopy.workflows.spectra.iraman import IRamanSpectraWorkChain


@pytest.fixture
def generate_workchain_iraman(generate_workchain, generate_inputs_pw_base):
    """Generate an instance of a `IRamanSpectraWorkChain`."""

    def _generate_workchain_iraman(append_inputs=None):
        entry_point = 'vibroscopy.spectra.iraman'

        scf_inputs = generate_inputs_pw_base()
        structure = scf_inputs['pw'].pop('structure')

        inputs = {
            'structure': structure,
            'phonon': {
                'scf': scf_inputs
            },
            'dielectric': {
                'property': 'raman',
                'scf': scf_inputs,
                'settings': {
                    'sleep_submission_time': 0.
                }
            },
            'settings': {},
            'symmetry': {},
        }

        if append_inputs is not None:
            inputs.update(append_inputs)

        process = generate_workchain(entry_point, inputs)

        return process

    return _generate_workchain_iraman


@pytest.fixture
def generate_intensities_workchain_node():
    """Generate an instance of `WorkflowNode`."""

    def _generate_intensities_workchain_node(exit_status=0):
        from aiida.common import LinkType
        from aiida.orm import ArrayData, WorkflowNode
        from plumpy import ProcessState

        node = WorkflowNode().store()
        node.set_process_state(ProcessState.FINISHED)
        node.set_exit_status(exit_status)

        ir_array = ArrayData().store()
        raman_array = ArrayData().store()

        ir_array.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='ir_averaged')
        raman_array.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='raman_averaged')

        return node

    return _generate_intensities_workchain_node


@pytest.mark.usefixtures('aiida_profile')
def test_initialization(generate_workchain_iraman):
    """Test `IRamanSpectraWorkChain` initialization."""
    generate_workchain_iraman()


@pytest.mark.usefixtures('aiida_profile')
def test_run_spectra(generate_workchain_iraman):
    """Test `IRamanSpectraWorkChain.run_spectra`."""
    process = generate_workchain_iraman()
    process.run_spectra()
    assert 'harmonic' in process.ctx


@pytest.mark.usefixtures('aiida_profile')
def test_inspect_process(generate_workchain_iraman):
    """Test `IRamanSpectraWorkChain.inspect_process`."""
    from aiida.orm import WorkflowNode
    from plumpy import ProcessState

    node = WorkflowNode().store()
    node.set_process_state(ProcessState.FINISHED)
    node.set_exit_status(400)

    process = generate_workchain_iraman()
    process.ctx.harmonic = node
    result = process.inspect_process()
    assert result == IRamanSpectraWorkChain.exit_codes.ERROR_HARMONIC_WORKCHAIN_FAILED


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
def test_inspect_averaging(generate_workchain_iraman, generate_intensities_workchain_node):
    """Test `IRamanSpectraWorkChain.inspect_averaging` method."""
    from aiida.common import AttributeDict
    process = generate_workchain_iraman()

    process.ctx.intensities_average = AttributeDict({
        'numerical_order_2_step_1': generate_intensities_workchain_node(),
        'numerical_order_4': generate_intensities_workchain_node(),
    })

    # Sanity check
    assert 'ir_averaged' in process.ctx.intensities_average['numerical_order_4'].outputs
    assert 'raman_averaged' in process.ctx.intensities_average['numerical_order_4'].outputs

    process.inspect_averaging()

    assert 'output_intensities_average' in process.outputs
    assert 'numerical_order_4' in process.outputs['output_intensities_average']
    assert 'numerical_order_2_step_1' in process.outputs['output_intensities_average']

    for key in ['ir_averaged', 'raman_averaged']:
        assert key in process.outputs['output_intensities_average']['numerical_order_4']
        assert key in process.outputs['output_intensities_average']['numerical_order_2_step_1']


@pytest.mark.usefixtures('aiida_profile')
def test_inspect_averaging_error(generate_workchain_iraman, generate_intensities_workchain_node):
    """Test `IRamanSpectraWorkChain.inspect_averaging` exit code."""
    process = generate_workchain_iraman()

    process.ctx.intensities_average = {
        'numerical_order_2_step_1': generate_intensities_workchain_node(exit_status=300),
        'numerical_order_4': generate_intensities_workchain_node(),
    }

    result = process.inspect_averaging()
    assert result == IRamanSpectraWorkChain.exit_codes.ERROR_AVERAGING_WORKCHAIN_FAILED
