# -*- coding: utf-8 -*-
"""Tests for the `DielectricWorkChain` class."""
import pytest
from aiida.orm import Dict

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

@pytest.fixture
def generate_workchain_dielectric(generate_workchain, generate_inputs_dielectric):
    """Generate an instance of a `DielectricWorkChain`."""

    def _generate_workchain_dielectric(inputs=None, **kwargs):
        entry_point = 'quantumespresso.vibroscopy.dielectric'

        if inputs is None:
            inputs = generate_inputs_dielectric(**kwargs)

        process = generate_workchain(entry_point, inputs)

        return process

    return _generate_workchain_dielectric


@pytest.fixture
def generate_base_scf_workchain_node(fixture_localhost):
    """Generate an instance of `WorkflowNode`."""

    def _generate_base_scf_workchain_node():
        from aiida.common import LinkType
        from aiida.orm import WorkflowNode, RemoteData

        node = WorkflowNode().store()

        parameters = Dict(dict={'number_of_bands': 5}).store()
        parameters.add_incoming(node, link_type=LinkType.RETURN, link_label='output_parameters')

        remote_folder = RemoteData(computer=fixture_localhost, remote_path='/tmp').store()
        remote_folder.add_incoming(node, link_type=LinkType.RETURN, link_label='remote_folder')
        remote_folder.store()

        return node

    return _generate_base_scf_workchain_node

@pytest.mark.usefixtures('aiida_profile')
def test_valididation_inputs(generate_workchain_dielectric, generate_inputs_dielectric):
    """Test `DielectricWorkChain` validation methods."""
    inputs = generate_inputs_dielectric(electric_field=0.1)
    generate_workchain_dielectric(inputs=inputs)

@pytest.mark.parametrize(
    ('parameters', 'message'),
    (
        ({'electric_field':-1.0},
         'specified value is negative.'),
        ({'electric_field_scale':-1.0},
         'specified value is negative.'),
        ({'electric_field':1.0, 'electric_field_scale':1.0, },
         'cannot specify both `electric_field*` inputs, only one is accepted'),
        ({},
         'one between `electric_field` and `electric_field_scale` must be specified'),
        ({'electric_field':+1.0, 'property':'not valid'},
         'Got invalid or not implemented property value not valid.'),
        ({'diagonal_scale':-0.9, 'electric_field':1.0,},
         'specified value is negative.'),
        ({'accuracy':3, 'electric_field':1.0,},
         'specified accuracy is negative or not even.'),
    ),
)
@pytest.mark.usefixtures('aiida_profile')
def test_invalid_inputs(generate_workchain_dielectric, generate_inputs_dielectric, parameters, message):
    """Test `DielectricWorkChain` validation methods."""
    with pytest.raises(ValueError) as exception:
        inputs = generate_inputs_dielectric(**parameters)
        generate_workchain_dielectric(inputs=inputs)

    assert message in str(exception.value)


@pytest.mark.usefixtures('aiida_profile')
def test_valid_property_inputs(generate_workchain_dielectric, generate_inputs_dielectric):
    """Test `DielectricWorkChain` validation methods."""
    properties = (
        'ir','born-charges','dielectric','nac','raman','bec',
        'susceptibility-derivative','non-linear-susceptibility',
        'IR','Born-Charges','Dielectric','NAC','BEC',
        'Raman','susceptibility-derivative','non-linear-susceptibility'
    )
    for property in properties:
        inputs = generate_inputs_dielectric(**{'property':property, 'electric_field':1.0})
        generate_workchain_dielectric(inputs=inputs)


@pytest.mark.parametrize(
    ('parameters', 'values'),
    (
        ({'electric_field':1.0}, [False, True, 6, True]),
        ({'property':'ir', 'electric_field':1.0}, [False, True, 3, True]),
        ({'property':'ir', 'electric_field_scale':1.0}, [True, False, 3, True]),
        ({'electric_field':1.0, 'clean_workdir':False}, [False, True, 6, False]),
    ),
)
@pytest.mark.usefixtures('aiida_profile')
def test_setup(generate_workchain_dielectric, generate_inputs_dielectric, parameters, values):
    """Test `DielectricWorkChain.setup`."""
    inputs = generate_inputs_dielectric(**parameters)
    process = generate_workchain_dielectric(inputs=inputs)
    process.setup()

    assert process.ctx.should_run_base_scf == True
    assert process.ctx.should_estimate_electric_field == values[0]
    assert ('electric_field' in process.ctx) == values[1]
    assert process.ctx.numbers == values[2]
    assert process.ctx.is_magnetic == False
    assert process.ctx.clean_workdir == values[3]


@pytest.mark.usefixtures('aiida_profile')
def test_run_base_scf(generate_workchain_dielectric, generate_inputs_dielectric):
    """Test `DielectricWorkChain.run_base_scf`."""
    inputs = generate_inputs_dielectric(electric_field=1.0)
    process = generate_workchain_dielectric(inputs=inputs)
    process.setup()
    process.run_base_scf()
    # asserting that PwBaseWorkChain have been launched
    assert 'base_scf' in process.ctx


@pytest.mark.usefixtures('aiida_profile')
def test_run_nscf(generate_workchain_dielectric, generate_inputs_dielectric, generate_base_scf_workchain_node):
    """Test `DielectricWorkChain.run_nscf`."""
    inputs = generate_inputs_dielectric(electric_field_scale=1.0)
    process = generate_workchain_dielectric(inputs=inputs)

    process.setup()
    process.ctx.base_scf = generate_base_scf_workchain_node()

    process.run_nscf()
    assert 'nscf' in process.ctx


@pytest.mark.parametrize(
    ('parameters', 'values'),
    (
        ({'electric_field':1.0e-3}, [2,1]),
        ({'electric_field':1.0e-5}, [4,2]),
        ({'electric_field':1.0, 'accuracy':4}, [4,2]),
    ),
)
@pytest.mark.usefixtures('aiida_profile')
def test_run_null_field_scf(generate_workchain_dielectric, generate_inputs_dielectric, generate_base_scf_workchain_node, parameters, values):
    """Test `DielectricWorkChain.run_null_field_scf`."""
    inputs = generate_inputs_dielectric(**parameters)
    process = generate_workchain_dielectric(inputs=inputs)

    process.setup()
    process.ctx.base_scf = generate_base_scf_workchain_node()

    process.run_null_field_scf()

    assert 'null_field' in process.ctx
    assert process.ctx.accuracy.value == values[0]
    assert process.ctx.steps == values[1]
    assert process.ctx.iteration == 0

@pytest.mark.usefixtures('aiida_profile')
def test_run_electric_field_scfs(generate_workchain_dielectric, generate_inputs_dielectric, generate_base_scf_workchain_node):
    """Test `DielectricWorkChain.run_null_field_scf`."""
    inputs = generate_inputs_dielectric(electric_field=1.0, accuracy=4)
    process = generate_workchain_dielectric(inputs=inputs)

    process.setup()
    process.ctx.base_scf = generate_base_scf_workchain_node()
    process.run_null_field_scf()
    process.ctx.null_field = generate_base_scf_workchain_node()

    # First iteration faked simulation
    process.run_electric_field_scfs()
    for i in range(6):
        key = f'field_index_{i}'
        assert key in process.ctx
        assert type(process.ctx[key])==list
        assert len(process.ctx[key]) == 2
        process.ctx[key] = [
            generate_base_scf_workchain_node(),
            generate_base_scf_workchain_node()
        ]

    assert process.ctx.iteration == 1

    # Second iteration faked simulation
    process.run_electric_field_scfs()
    for i in range(1):
        key = f'field_index_{i}'
        assert key in process.ctx
        assert len(process.ctx[key]) == 4

    assert process.ctx.iteration == 2


@pytest.mark.usefixtures('aiida_profile')
def test_run_numerical_derivatives(generate_workchain_dielectric, generate_inputs_dielectric, generate_elfield_scf_workchain_node):
    """Test `DielectricWorkChain.run_numerical_derivatives`."""
    from aiida.orm import Int
    inputs = generate_inputs_dielectric(electric_field=1.0, accuracy=2)
    process = generate_workchain_dielectric(inputs=inputs)
    process.setup()
    process.ctx.accuracy = Int(2)

    process.ctx.null_field = generate_elfield_scf_workchain_node()
    for i in range(6):
        process.ctx[f'field_index_{i}'] = [
            generate_elfield_scf_workchain_node(),
            generate_elfield_scf_workchain_node(),
        ]

    process.run_numerical_derivatives()
    assert 'numerical_derivatives' in process.ctx
