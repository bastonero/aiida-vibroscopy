# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name
"""Tests for the :mod:`aiida_common_workflows.workflows.common_force_sets` module."""
import pytest

from aiida.engine import WorkChain
from aiida.plugins import WorkflowFactory

from aiida_common_workflows.plugins import get_workflow_entry_point_names
from aiida_common_workflows.workflows.phonons import common_force_sets
from aiida_common_workflows.workflows.relax.workchain import CommonRelaxWorkChain

@pytest.fixture
def generate_inputs_pw_base(generate_inputs_pw, generate_structure):
    """Generate default inputs for a `PwBaseWorkChain`."""

    def _generate_inputs_pw_base():
        """Generate default inputs for a `PwBaseWorkChain`."""
        from aiida.orm import Dict, Float

        structure = generate_structure()
        inputs_scf = generate_inputs_pw(structure=structure)

        kpoints = inputs_scf.pop('kpoints')
        
        inputs = {
            'pw': inputs_scf,
            'kpoints': kpoints,
        }

        return inputs

    return _generate_inputs_pw_base

@pytest.fixture
def generate_workchain_force_sets(generate_workchain, generate_structure, generate_inputs_pw_base):
    """Generate an instance of a `ForceSetsWorkChain`."""

    def _generate_workchain_force_sets(append_inputs=None, return_inputs=False):
        from aiida.orm import List
        entry_point = 'quantumespresso.vibroscopy.phonons.force_sets'
        scf_inputs = generate_inputs_pw_base()
        scf_inputs['pw'].pop('structure')
        
        inputs = {
            'structure': generate_structure(), 
            'supercell_matrix': List(list=[1,1,1]),
            'scf': scf_inputs,
            }

        if return_inputs:
            return inputs
        
        if append_inputs is not None:
            inputs.update(append_inputs)

        process = generate_workchain(entry_point, inputs)

        return process

    return _generate_workchain_force_sets


def test_run_forces(generate_workchain_force_sets):
    """Test `ForceSetsWorkChain.run_forces`."""
    process = generate_workchain_force_sets()
    process.setup()
    process.run_forces()

    for key in ['cells', 'primitive_matrix', 'displacement_dataset']:
        assert key in process.outputs

    for key in ['primitive', 'supercell', 'supercell_1']:
        assert key in process.outputs['cells']

    # Double check for the `setup` method (already tested in `aiida-phonopy`).
    assert 'primitive' not in process.ctx.supercells
    assert 'supercell' not in process.ctx.supercells
    assert 'supercell_1' in process.ctx.supercells    

    # Check for 
    assert 'force_calc_1' in process.ctx


def test_outline(generate_workchain_force_sets):
    """Test `CommonForceSetsWorkChain` outline."""
    from plumpy.process_states import ProcessState
    from aiida.common import LinkType
    from aiida.orm import WorkflowNode, ArrayData, Float
    import numpy as np
    
    process = generate_workchain_force_sets()
    
    node = WorkflowNode().store()
    node.label = 'force_calc_1'
    forces = ArrayData()
    forces.set_array('forces', np.array([[[0.,0.,0.],[0.,0.,0.]]])) # TrajectoryData like
    forces.store()
    forces.add_incoming(node, link_type=LinkType.RETURN, link_label='forces')
    energy = Float(0.).store()
    energy.add_incoming(node, link_type=LinkType.RETURN, link_label='total_energy')
    
    node.set_process_state(ProcessState.FINISHED)
    node.set_exit_status(0)
    
    process.ctx.force_calc_1 = node
    
    process.inspect_forces()

    assert 'force_calc_1' in process.outputs['supercells_forces']
    assert 'forces' in process.ctx
    assert 'forces_1' in process.ctx.forces
    
    process.run_results()
    
    assert 'force_sets' in process.outputs
    
    
def test_run_outline_with_subtracting_residual_forces(generate_workchain_force_sets):
    """Test `CommonForceSetsWorkChain.run_forces`."""
    from aiida.orm import Bool
    process = generate_workchain_force_sets(append_inputs={'subtract_residual_forces':Bool(True)})
    process.setup()
    process.run_forces()

    for key in ['cells', 'primitive_matrix', 'displacement_dataset']:
        assert key in process.outputs

    for key in ['primitive', 'supercell', 'supercell_1']:
        assert key in process.outputs['cells']

    # Double check for the `setup` method (already tested in `aiida-phonopy`).
    assert 'primitive' not in process.ctx.supercells
    assert 'supercell' in process.ctx.supercells
    assert 'supercell_1' in process.ctx.supercells    

    # Check for 
    assert 'force_calc_0' in process.ctx
    assert 'force_calc_1' in process.ctx