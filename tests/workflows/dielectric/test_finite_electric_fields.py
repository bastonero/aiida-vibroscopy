# -*- coding: utf-8 -*-
# pylint: disable=no-member,redefined-outer-name
"""Tests for the `FiniteElectricFieldsWorkChain` class."""
import pytest
from aiida.orm import Dict, WorkChainNode


@pytest.fixture
def generate_workchain_finite_electric_fields(generate_workchain, generate_inputs_finite_electric_fields):
    """Generate an instance of a `FiniteElectricFieldsWorkChain`."""

    def _generate_workchain_finite_electric_fields(inputs=None):
        entry_point = 'quantumespresso.fd.finite_electric_fields'

        if inputs is None:
            inputs = generate_inputs_finite_electric_fields()

        process = generate_workchain(entry_point, inputs)

        return process

    return _generate_workchain_finite_electric_fields
    

@pytest.fixture
def generate_elfield_scf_workchain_node():
    """Generate an instance of `WorkflowNode`."""
    
    def _generate_scf_workchain_node(polarization=None,forces=None, volume=None):
        from aiida.common import LinkType
        from aiida.orm import WorkflowNode, TrajectoryData
        import numpy as np
        
        node = WorkflowNode().store()

        if volume is None:
            parameters = Dict(dict={'volume': 1.0}).store()
        else:
            parameters = Dict(dict={'volume': volume}).store()
            
        parameters.add_incoming(node, link_type=LinkType.RETURN, link_label='output_parameters')
        
        trajectory = TrajectoryData()
        
        if polarization is None:
            trajectory.set_array('electronic_dipole_cartesian_axes',np.array([[0.,0.,0.]]) )
        else:
            trajectory.set_array('electronic_dipole_cartesian_axes',np.array(polarization) )
            
        if polarization is None:
            trajectory.set_array('forces',np.array([[[0.,0.,0.],[0.,0.,0.]]]) )
        else:
            trajectory.set_array('forces',np.array(forces) )
        stepids = np.array([1])
        times = stepids * 0.0
        cells = np.array([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
        positions = np.array([[[0., 0., 0.],[0.,0.,0.]]])
        trajectory.set_trajectory(stepids=stepids, cells=cells, symbols=['Mg','O'], positions=positions, times=times)
        trajectory.store()
        trajectory.add_incoming(node, link_type=LinkType.RETURN, link_label='output_trajectory')

        return node
    
    return _generate_scf_workchain_node


@pytest.mark.usefixtures('aiida_profile')
def test_invalid_inputs(generate_workchain_finite_electric_fields, generate_inputs_finite_electric_fields):
    """Test `FiniteElectricFieldsWorkChain` validation methods."""
    with pytest.raises(ValueError):
        direction = 3 # out of range direction input
        inputs = generate_inputs_finite_electric_fields(selected_elfield=direction)
        process = generate_workchain_finite_electric_fields(inputs=inputs)
    with pytest.raises(ValueError):
        nberrycyc = -1 # negative nberrycyc input
        inputs = generate_inputs_finite_electric_fields(nberrycyc=nberrycyc)
        process = generate_workchain_finite_electric_fields(inputs=inputs)
    with pytest.raises(ValueError):
        elfield = -0.01 # negative nberrycyc input
        inputs = generate_inputs_finite_electric_fields(elfield=elfield)
        process = generate_workchain_finite_electric_fields(inputs=inputs)


@pytest.mark.usefixtures('aiida_profile')
def test_setup(generate_workchain_finite_electric_fields, generate_inputs_finite_electric_fields):
    """Test `FiniteElectricFieldsWorkChain.setup`."""
    inputs = generate_inputs_finite_electric_fields()
    dE = inputs['elfield']
    process = generate_workchain_finite_electric_fields(inputs=inputs)
    process.setup()
    
    assert process.ctx.elfield_card == [[dE,0,0],[0,dE,0],[0,0,dE]]
    assert process.ctx.only_one_elfield == False
    assert process.ctx.has_parent_scf == False
    assert process.ctx.should_run_init_scf == False


@pytest.mark.usefixtures('aiida_profile')
def test_find_direction(generate_workchain_finite_electric_fields, generate_inputs_finite_electric_fields):
    """Test `FiniteElectricFieldsWorkChain.find_direction`."""
    inputs = generate_inputs_finite_electric_fields()
    dE = inputs['elfield']
    process = generate_workchain_finite_electric_fields(inputs=inputs)
    process.setup()
    index = 0
    for card in process.ctx.elfield_card:
        direction, _ = process.find_direction(card)
        assert index == direction
        index += 1


@pytest.mark.usefixtures('aiida_profile')
def test_run_elfield_scf(generate_workchain_finite_electric_fields, generate_inputs_finite_electric_fields):
    """Test `FiniteElectricFieldsWorkChain.run_elfield_scf`."""
    inputs = generate_inputs_finite_electric_fields()
    process = generate_workchain_finite_electric_fields(inputs=inputs)
    process.setup()
    process.run_elfield_scf()
    # asserting that PwBaseWorkChains have been launched
    assert 'null_electric_field' in process.ctx
    for direction in range(3):
        key = f'electric_field_{direction}'
        assert key in process.ctx


@pytest.mark.usefixtures('aiida_profile')
def test_run_one_elfield_scf(generate_workchain_finite_electric_fields, generate_inputs_finite_electric_fields):
    """Test `FiniteElectricFieldsWorkChain` one electric field."""
    for direction in range(3):
        inputs = generate_inputs_finite_electric_fields(selected_elfield=direction)
        process = generate_workchain_finite_electric_fields(inputs=inputs)
        # asserting the setup options
        process.setup()
        assert len(process.ctx.elfield_card)==1
        assert process.ctx.only_one_elfield
        
        # asserting that PwBaseWorkChains have been launched
        process.run_elfield_scf()
        assert 'null_electric_field' in process.ctx
        key = f'electric_field_{direction}'
        assert key in process.ctx


@pytest.mark.usefixtures('aiida_profile')
def test_run_results(
    generate_workchain_finite_electric_fields, generate_inputs_finite_electric_fields,
    generate_structure, generate_elfield_scf_workchain_node):
    """Test `FiniteElectricFieldsWorkChain.run_results`."""
    import numpy as np
    inputs = generate_inputs_finite_electric_fields()
    process = generate_workchain_finite_electric_fields(inputs=inputs)
    process.setup()

    # Mock the run workchain context variables as if a `PwBaseWorkChain` has been run in
    process.ctx.null_electric_field = generate_elfield_scf_workchain_node()
    process.ctx.electric_field_0 = generate_elfield_scf_workchain_node()
    process.ctx.electric_field_1 = generate_elfield_scf_workchain_node()
    process.ctx.electric_field_2 = generate_elfield_scf_workchain_node()
    process.run_results()
    assert 'numerical_derivatives' in process.ctx
