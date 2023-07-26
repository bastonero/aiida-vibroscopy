# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Tests for the `DielectricWorkChain` class."""
from aiida.orm import Dict
import pytest

from aiida_vibroscopy.workflows.dielectric.base import DielectricWorkChain


@pytest.fixture
def generate_elfield_scf_workchain_node():
    """Generate an instance of `WorkflowNode`."""

    def _generate_scf_workchain_node(exit_status=0, polarization=None, forces=None, volume=None):
        from aiida.common import LinkType
        from aiida.orm import KpointsData, TrajectoryData, WorkflowNode
        import numpy as np
        from plumpy.process_states import ProcessState

        node = WorkflowNode()

        kpoints = KpointsData()
        kpoints.set_kpoints_mesh([2, 2, 2])
        kpoints.store()
        node.base.links.add_incoming(kpoints, link_type=LinkType.INPUT_WORK, link_label='kpoints')

        node.store()
        node.set_exit_status(exit_status)
        node.set_process_state(ProcessState.FINISHED)

        if volume is None:
            parameters = Dict({'volume': 1.0}).store()
        else:
            parameters = Dict({'volume': volume}).store()

        parameters.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='output_parameters')

        trajectory = TrajectoryData()

        if polarization is None:
            trajectory.set_array('electronic_dipole_cartesian_axes', np.array([[0., 0., 0.]]))
        else:
            trajectory.set_array('electronic_dipole_cartesian_axes', np.array(polarization))

        if forces is None:
            trajectory.set_array('forces', np.array([[[0., 0., 0.], [0., 0., 0.]]]))
        else:
            trajectory.set_array('forces', np.array(forces))
        stepids = np.array([1])
        times = stepids * 0.0
        cells = np.array([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
        positions = np.array([[[0., 0., 0.], [0., 0., 0.]]])
        trajectory.set_trajectory(stepids=stepids, cells=cells, symbols=['Mg', 'O'], positions=positions, times=times)
        trajectory.store()
        trajectory.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='output_trajectory')

        return node

    return _generate_scf_workchain_node


@pytest.fixture
def generate_workchain_dielectric(generate_workchain, generate_inputs_dielectric):
    """Generate an instance of a `DielectricWorkChain`."""

    def _generate_workchain_dielectric(inputs=None, **kwargs):
        entry_point = 'vibroscopy.dielectric'

        if inputs is None:
            inputs = generate_inputs_dielectric(**kwargs)

        process = generate_workchain(entry_point, inputs)

        return process

    return _generate_workchain_dielectric


@pytest.mark.usefixtures('aiida_profile')
def test_valididation_inputs(generate_workchain_dielectric, generate_inputs_dielectric):
    """Test `DielectricWorkChain` validation methods."""
    inputs = generate_inputs_dielectric()
    generate_workchain_dielectric(inputs=inputs)


@pytest.mark.parametrize(
    ('parameters', 'message'),
    (
        ({
            'electric_field_step': -1.0,
            'accuracy': 2,
        }, 'specified value is negative.'),
        ({
            'electric_field_step': +1.0,
            'accuracy': 2,
            'property': 'not valid'
        }, 'Got invalid or not implemented property value not valid.'),
        ({
            'diagonal_scale': -0.9,
        }, 'specified value is negative.'),
        ({
            'accuracy': 3,
        }, 'specified accuracy is negative or not even.'),
        ({
            'electric_field_step': 0.001,
            'property': 'raman',
        }, (
            'cannot evaluate numerical accuracy when `electric_field_step` '
            'is specified but `accuracy` is not in `central_difference`'
        )),
    ),
)
@pytest.mark.usefixtures('aiida_profile')
def test_invalid_inputs(generate_workchain_dielectric, generate_inputs_dielectric, parameters, message):
    """Test `DielectricWorkChain` validation methods."""
    with pytest.raises(ValueError, match=message):
        inputs = generate_inputs_dielectric(**parameters)
        generate_workchain_dielectric(inputs=inputs)


@pytest.mark.parametrize('key', ('-nk', '-npools'))
def test_invalid_scf_parallelization(generate_workchain_dielectric, generate_inputs_dielectric, key):
    """Test `DielectricWorkChain` validation method for parallelization."""
    from aiida.orm import Dict

    match = 'pool parallelization for electric field is not implemented'
    inputs = generate_inputs_dielectric()
    inputs['scf']['pw']['settings'] = Dict({'cmdline': [key, '2']})

    with pytest.raises(ValueError, match=match):
        generate_workchain_dielectric(inputs=inputs)


@pytest.mark.parametrize(
    'property_input', (
        'ir', 'born-charges', 'dielectric', 'nac', 'raman', 'bec', 'susceptibility-derivative',
        'non-linear-susceptibility', 'IR', 'Born-Charges', 'Dielectric', 'NAC', 'BEC', 'Raman',
        'susceptibility-derivative', 'non-linear-susceptibility'
    )
)
@pytest.mark.usefixtures('aiida_profile')
def test_valid_property_inputs(generate_workchain_dielectric, generate_inputs_dielectric, property_input):
    """Test `DielectricWorkChain` validation methods."""
    inputs = generate_inputs_dielectric(**{'property': property_input})
    generate_workchain_dielectric(inputs=inputs)


@pytest.mark.parametrize(
    ('parameters', 'values'),
    (
        ({
            'electric_field_step': 1.0,
            'accuracy': 2
        }, [False, True, 6, True]),
        ({
            'property': 'ir',
        }, [True, False, 3, True]),
        ({
            'clean_workdir': False
        }, [True, False, 6, False]),
    ),
)
@pytest.mark.usefixtures('aiida_profile')
def test_setup(generate_workchain_dielectric, generate_inputs_dielectric, parameters, values):
    """Test `DielectricWorkChain.setup`."""
    inputs = generate_inputs_dielectric(**parameters)
    process = generate_workchain_dielectric(inputs=inputs)
    process.setup()

    # assert process.ctx.should_run_base_scf == True
    assert process.ctx.should_estimate_electric_field == values[0]
    assert ('electric_field_step' in process.ctx) == values[1]
    assert 'numbers' in process.ctx
    assert 'signs' in process.ctx
    assert process.ctx.is_magnetic == False
    assert process.ctx.clean_workdir == values[3]

    # The followings are True only for MgO or equivalent symmetric systems
    numbers = [2, 3] if values[2] == 6 else [2]
    assert process.ctx.numbers == numbers


@pytest.mark.usefixtures('aiida_profile')
def test_set_reference_kpoints(generate_workchain_dielectric, generate_inputs_dielectric):
    """Test the `DielectricWorkChain.set_reference_kpoints` method."""
    from aiida.orm import Float

    inputs = generate_inputs_dielectric()
    process = generate_workchain_dielectric(inputs=inputs)

    process.setup()
    assert not process.ctx.is_parallel_distance

    process.set_reference_kpoints()

    assert 'kpoints_parallel_distance' not in process.inputs
    assert 'kpoints' in process.ctx
    assert 'kpoints_dict' not in process.ctx

    inputs = generate_inputs_dielectric()
    inputs.scf.pop('kpoints')
    inputs.scf.kpoints_distance = Float(0.15)
    process = generate_workchain_dielectric(inputs=inputs)

    process.setup()
    assert not process.ctx.is_parallel_distance

    process.set_reference_kpoints()

    assert 'kpoints' in process.ctx
    assert 'kpoints_dict' not in process.ctx

    inputs = generate_inputs_dielectric()
    inputs.scf.pop('kpoints')
    inputs.scf.kpoints_distance = Float(0.15)
    inputs.kpoints_parallel_distance = Float(0.15)
    process = generate_workchain_dielectric(inputs=inputs)

    process.setup()
    assert process.ctx.is_parallel_distance

    process.set_reference_kpoints()

    assert 'kpoints' in process.ctx
    assert 'kpoints_dict' in process.ctx
    assert 'kpoints_list' in process.ctx
    assert 'meshes' in process.ctx
    assert len(list(process.ctx['kpoints_dict'].values())) == 2


@pytest.mark.usefixtures('aiida_profile')
def test_run_base_scf(generate_workchain_dielectric, generate_inputs_dielectric):
    """Test `DielectricWorkChain.run_base_scf`."""
    inputs = generate_inputs_dielectric()
    process = generate_workchain_dielectric(inputs=inputs)
    process.setup()
    process.set_reference_kpoints()
    process.run_base_scf()
    # asserting that PwBaseWorkChain have been launched
    assert 'base_scf' in process.ctx


@pytest.mark.parametrize(('expected_result', 'exit_status'),
                         ((None, 0), (DielectricWorkChain.exit_codes.ERROR_FAILED_BASE_SCF, 312)))
@pytest.mark.usefixtures('aiida_profile')
def test_inspect_base_scf(
    generate_workchain_dielectric, generate_base_scf_workchain_node, expected_result, exit_status
):
    """Test `DielectricWorkChain.inspect_base_scf`."""
    process = generate_workchain_dielectric()
    process.ctx.base_scf = generate_base_scf_workchain_node(exit_status=exit_status)
    result = process.inspect_base_scf()
    assert result == expected_result


@pytest.mark.usefixtures('aiida_profile')
def test_run_nscf(generate_workchain_dielectric, generate_inputs_dielectric, generate_base_scf_workchain_node):
    """Test `DielectricWorkChain.run_nscf`."""
    inputs = generate_inputs_dielectric()
    process = generate_workchain_dielectric(inputs=inputs)

    process.setup()
    process.set_reference_kpoints()

    process.ctx.base_scf = generate_base_scf_workchain_node(with_trajectory=True)

    process.run_nscf()
    assert 'nscf' in process.ctx


@pytest.mark.parametrize(('expected_result', 'exit_status'),
                         ((None, 0), (DielectricWorkChain.exit_codes.ERROR_FAILED_NSCF, 312)))
@pytest.mark.usefixtures('aiida_profile')
def test_inspect_nscf(generate_workchain_dielectric, generate_base_scf_workchain_node, expected_result, exit_status):
    """Test `DielectricWorkChain.inspect_nscf`."""
    process = generate_workchain_dielectric()
    process.ctx.nscf = generate_base_scf_workchain_node(exit_status=exit_status)
    result = process.inspect_nscf()
    assert result == expected_result


@pytest.mark.parametrize(
    ('parameters', 'values'),
    (
        ({
            'accuracy': 4,
            'critical_electric_field': 0.002
        }, [4, 0.0005]),
        ({
            'accuracy': 2,
            'electric_field_step': 1.0e-5
        }, [2, 1.0e-5]),
        ({
            'critical_electric_field': 0.002
        }, [4, 1.0e-3 / 2]),
        ({
            'critical_electric_field': 0.0005
        }, [4, 2.5e-4]),
        ({
            'critical_electric_field': 5.0e-5
        }, [2, 5.0e-5]),
    ),
)
@pytest.mark.usefixtures('aiida_profile')
def test_set_step_and_accuracy(generate_workchain_dielectric, generate_inputs_dielectric, parameters, values):
    """Test `DielectricWorkChain.set_step_and_accuracy` method."""
    from aiida.orm import Float

    inputs = generate_inputs_dielectric(**parameters)
    process = generate_workchain_dielectric(inputs=inputs)

    if 'electric_field_step' not in parameters:
        process.ctx.critical_electric_field = Float(parameters['critical_electric_field'])

    process.setup()
    process.set_step_and_accuracy()

    assert 'accuracy' in process.ctx
    assert 'electric_field_step' in process.ctx
    assert 'iteration' in process.ctx
    assert 'max_iteration' in process.ctx

    assert process.ctx.accuracy.value == values[0]
    assert process.ctx.electric_field_step.value - values[1] < 1.0e-8
    assert process.ctx.max_iteration == values[0] / 2
    assert process.ctx.iteration == 0


@pytest.mark.parametrize(
    ('parameters', 'values'),
    (
        ({
            'electric_field_step': 2.0e-3,
            'accuracy': 6,
        }, [6, 2, 1]),
        ({
            'electric_field_step': 1.0e-5,
            'accuracy': 2,
        }, [2, 1, 1]),
        ({
            'electric_field_step': 1.0,
            'accuracy': 4
        }, [4, 2, 1]),
        ({
            'electric_field_step': 1.0,
            'accuracy': 4,
            'parallel_distance': 0.1,
        }, [4, 2, 2]),
    ),
)
@pytest.mark.usefixtures('aiida_profile')
def test_run_null_field_scfs(
    generate_workchain_dielectric, generate_inputs_dielectric, generate_base_scf_workchain_node, parameters, values
):
    """Test `DielectricWorkChain.run_null_field_scfs`."""
    from aiida.orm import Float

    distance = Float(parameters.pop('parallel_distance')) if 'parallel_distance' in parameters else None
    inputs = generate_inputs_dielectric(**parameters)

    if distance is not None:
        inputs.scf.pop('kpoints')
        inputs.scf.kpoints_distance = distance * 4
        inputs.kpoints_parallel_distance = distance

    process = generate_workchain_dielectric(inputs=inputs)

    process.setup()
    process.set_reference_kpoints()
    process.set_step_and_accuracy()

    process.ctx.base_scf = generate_base_scf_workchain_node(with_trajectory=True)

    process.run_null_field_scfs()

    assert 'null_fields' in process.ctx
    assert process.ctx.accuracy.value == values[0]

    if distance is None:
        assert len(process.ctx.null_fields) == values[2]
    else:
        assert process.ctx.is_parallel_distance
        assert len(process.ctx.null_fields) == len(process.ctx.kpoints_list)


@pytest.mark.parametrize(('expected_result', 'exit_status'),
                         ((None, 0),
                          (DielectricWorkChain.exit_codes.ERROR_FAILED_ELFIELD_SCF.format(direction='`null`'), 312)))
@pytest.mark.usefixtures('aiida_profile')
def test_inspect_null_field_scfs(
    generate_workchain_dielectric, generate_base_scf_workchain_node, expected_result, exit_status
):
    """Test `DielectricWorkChain.inspect_null_field_scfs`."""
    process = generate_workchain_dielectric()
    process.ctx.null_fields = [generate_base_scf_workchain_node(exit_status=exit_status)]
    result = process.inspect_null_field_scfs()
    assert result == expected_result


@pytest.mark.usefixtures('aiida_profile')
def test_run_electric_field_scfs(
    generate_workchain_dielectric, generate_inputs_dielectric, generate_base_scf_workchain_node
):
    """Test `DielectricWorkChain.run_null_field_scfs`."""
    inputs = generate_inputs_dielectric(electric_field_step=1.0, accuracy=4)
    process = generate_workchain_dielectric(inputs=inputs)

    process.setup()
    process.set_reference_kpoints()
    process.set_step_and_accuracy()

    process.ctx.base_scf = generate_base_scf_workchain_node(with_trajectory=True)

    process.run_null_field_scfs()
    process.ctx.null_fields = [generate_base_scf_workchain_node(with_trajectory=True)]

    # First iteration faked simulation
    process.run_electric_field_scfs()
    for i in [2, 3]:  # This is for MgO or equivalent
        key = f'field_index_{i}'
        assert key in process.ctx
        assert isinstance(process.ctx[key], list)
        assert len(process.ctx[key]) == 1
        process.ctx[key] = [
            generate_base_scf_workchain_node(with_trajectory=True),
        ]

    assert process.ctx.iteration == 1

    # Second iteration faked simulation
    process.run_electric_field_scfs()
    for i in [2, 3]:  # This is for MgO or equivalent
        key = f'field_index_{i}'
        assert key in process.ctx
        assert len(process.ctx[key]) == 2

    assert process.ctx.iteration == 2


@pytest.mark.usefixtures('aiida_profile')
def test_run_no_sym(generate_workchain_dielectric, generate_inputs_dielectric, generate_base_scf_workchain_node):
    """Test `DielectricWorkChain.run_null_field_scfs`."""
    from aiida.orm import Bool
    inputs = generate_inputs_dielectric(electric_field_step=1.0, accuracy=4)
    inputs['symmetry']['is_symmetry'] = Bool(False)
    process = generate_workchain_dielectric(inputs=inputs)

    process.setup()
    process.set_reference_kpoints()
    process.set_step_and_accuracy()

    process.ctx.base_scf = generate_base_scf_workchain_node(with_trajectory=True)

    process.run_null_field_scfs()
    process.ctx.null_fields = [generate_base_scf_workchain_node(with_trajectory=True)]

    assert len(process.ctx.numbers) == 6

    for bool_signs in process.ctx.signs:
        assert bool_signs == [True, True]

    # First iteration faked simulation
    process.run_electric_field_scfs()
    for i in range(6):
        key = f'field_index_{i}'
        assert key in process.ctx
        assert isinstance(process.ctx[key], list)
        assert len(process.ctx[key]) == 2
        process.ctx[key] = [
            generate_base_scf_workchain_node(with_trajectory=True),
            generate_base_scf_workchain_node(with_trajectory=True),
        ]

    assert process.ctx.iteration == 1

    # Second iteration faked simulation
    process.run_electric_field_scfs()
    for i in range(1):
        key = f'field_index_{i}'
        assert len(process.ctx[key]) == 4

    assert process.ctx.iteration == 2


@pytest.mark.usefixtures('aiida_profile')
def test_run_fields_with_directional_kpoints(
    generate_workchain_dielectric, generate_inputs_dielectric, generate_base_scf_workchain_node
):
    """Test `DielectricWorkChain.run_electric_field_scfs` with directional density."""
    from aiida.orm import Float
    inputs = generate_inputs_dielectric(electric_field_step=1.0, accuracy=4)
    inputs['kpoints_parallel_distance'] = Float(0.1)
    inputs['scf']['kpoints_distance'] = Float(0.4)
    inputs['scf'].pop('kpoints')

    process = generate_workchain_dielectric(inputs=inputs)

    process.setup()
    process.set_reference_kpoints()
    process.set_step_and_accuracy()

    num_kpoints = len(process.ctx.kpoints_list)
    assert 'kpoints_list' in process.ctx
    assert num_kpoints == 2

    process.ctx.base_scf = generate_base_scf_workchain_node(with_trajectory=True)

    process.run_null_field_scfs()
    process.ctx.null_fields = [generate_base_scf_workchain_node(with_trajectory=True) for _ in range(num_kpoints)]

    process.run_electric_field_scfs()


@pytest.mark.parametrize(('expected_result', 'exit_status'),
                         ((None, 0),
                          (DielectricWorkChain.exit_codes.ERROR_FAILED_ELFIELD_SCF.format(direction='1'), 312)))
@pytest.mark.usefixtures('aiida_profile')
def test_test_inspect_electric_field_scfs(
    generate_workchain_dielectric, generate_elfield_scf_workchain_node, expected_result, exit_status
):
    """Test `DielectricWorkChain.test_inspect_electric_field_scfs`."""
    process = generate_workchain_dielectric()
    process.ctx.field_index_1 = [generate_elfield_scf_workchain_node(exit_status=exit_status)]
    result = process.inspect_electric_field_scfs()
    assert 'data' in process.ctx
    assert 'meshes_dict' in process.ctx
    assert result == expected_result
    if exit_status == 0:
        assert 'field_index_1' in process.outputs['fields_data']


@pytest.mark.usefixtures('aiida_profile')
def test_run_numerical_derivatives(
    generate_workchain_dielectric, generate_inputs_dielectric, generate_elfield_scf_workchain_node,
    generate_base_scf_workchain_node
):
    """Test `DielectricWorkChain.run_numerical_derivatives`."""
    # from aiida.orm import Int
    inputs = generate_inputs_dielectric(electric_field_step=1.0, accuracy=2)
    process = generate_workchain_dielectric(inputs=inputs)
    process.setup()
    process.set_step_and_accuracy()
    process.set_reference_kpoints()

    process.ctx.base_scf = generate_base_scf_workchain_node(with_trajectory=True)
    process.inspect_base_scf()

    process.ctx['null_fields'] = [generate_elfield_scf_workchain_node()]
    for i in range(6):
        process.ctx[f'field_index_{i}'] = [
            generate_elfield_scf_workchain_node(),
            generate_elfield_scf_workchain_node(),
        ]

    process.inspect_electric_field_scfs()
    process.remove_reference_forces()
    assert 'new_data' in process.ctx

    for i in range(6):
        key = f'field_index_{i}'
        assert key in process.ctx['new_data']
        assert len(process.ctx['new_data'][key]) == 2

    process.run_numerical_derivatives()
    assert 'numerical_derivatives' in process.ctx
