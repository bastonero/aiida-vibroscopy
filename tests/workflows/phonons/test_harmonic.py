# -*- coding: utf-8 -*-
"""Tests for the :mod:`workflows.phonons.harmonic` module."""
from aiida import orm
import numpy
import pytest


@pytest.fixture
def generate_workchain_harmonic(generate_workchain, generate_inputs_pw_base):
    """Generate an instance of a `HarmonicWorkChain`."""

    def _generate_workchain_harmonic(append_inputs=None, phonon_inputs=None, return_inputs=False):
        entry_point = 'vibroscopy.phonons.harmonic'
        scf_inputs = generate_inputs_pw_base()
        scf_inputs['pw'].pop('structure')

        inputs = {
            'phonon_workchain': {
                'scf': scf_inputs,
            },
            'options': {
                'sleep_submission_time': 0
            },
        }

        if return_inputs:
            return inputs

        if phonon_inputs is not None:
            inputs['phonon_workchain'].update(phonon_inputs)

        if append_inputs is not None:
            inputs.update(append_inputs)

        process = generate_workchain(entry_point, inputs)

        return process

    return _generate_workchain_harmonic


def generate_invalid_nac_parameters():
    """Generate invalid nac parameters input for `HarmonicWorkChain`."""
    nac = orm.ArrayData()
    nac.set_array('invalid', numpy.eye(3))

    return nac


def generate_valid_nac_parameters():
    """Generate valid nac parameters input for `HarmonicWorkChain`."""
    nac = orm.ArrayData()
    nac.set_array('dielectric', numpy.eye(3))
    bec = numpy.array([numpy.eye(3), -1 * numpy.eye(3)])
    nac.set_array('born_charges', bec)

    return nac


@pytest.mark.usefixtures('aiida_profile')
def test_valididation_inputs(generate_workchain_harmonic, generate_preprocess_data):
    """Test `HarmonicWorkChain` inizialisation with secure inputs."""
    preprocess_data = generate_preprocess_data()
    append_inputs = {'preprocess_data': preprocess_data}
    generate_workchain_harmonic(append_inputs=append_inputs)


@pytest.mark.parametrize(
    ('parameters', 'message'),
    (({}, 'at least one between `preprocess_data` and `structure` must be provided in input'), ({
        'structure': True,
        'preprocess_data': True
    }, 'too many inputs have been provided'), ({
        'nac_parameters': True
    }, 'data does not contain `dieletric` and/or `born_charges` arraynames.'), ({
        'nac_parameters': False,
        'dielectric_workchain': True
    }, 'too many inputs for non-analytical constants'), ({
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
def test_invalid_inputs(
    generate_workchain_harmonic, generate_inputs_dielectric, generate_preprocess_data, generate_structure, parameters,
    message
):
    """Test `DielectricWorkChain` validation methods."""

    if 'dielectric_workchain' in parameters:
        dielectric_inputs = generate_inputs_dielectric(electric_field_scale=1.0)
        dielectric_inputs['scf']['pw'].pop('structure')
        dielectric_inputs.pop('clean_workdir')
        parameters.update({'dielectric_workchain': dielectric_inputs})

    if 'structure' in parameters:
        parameters.update({'structure': generate_structure()})

    if 'preprocess_data' in parameters:
        parameters.update({'preprocess_data': generate_preprocess_data()})

    if 'nac_parameters' in parameters:
        if parameters['nac_parameters']:
            parameters.update({'nac_parameters': generate_invalid_nac_parameters()})
        else:
            parameters.update({'nac_parameters': generate_valid_nac_parameters()})
            parameters.update({'structure': generate_structure()})

    if 'supercell_matrix' in parameters:
        inputs = generate_workchain_harmonic(return_inputs=True)
        inputs['phonon_workchain'].update({'supercell_matrix': orm.List(list=parameters['supercell_matrix'])})
        parameters = inputs

    if 'displacement_generator' in parameters:
        inputs = generate_workchain_harmonic(return_inputs=True)
        inputs['phonon_workchain'].update({
            'displacement_generator': orm.Dict(dict=parameters['displacement_generator'])
        })
        parameters = inputs

    with pytest.raises(ValueError) as exception:
        generate_workchain_harmonic(append_inputs=parameters)

    assert message in str(exception.value)


@pytest.mark.parametrize(
    ('parameters'),
    (
        (True),
        (False),
    ),
)
@pytest.mark.usefixtures('aiida_profile')
def test_setup(generate_workchain_harmonic, generate_preprocess_data, generate_structure, parameters):
    """Test `HarmonicWorkChain` setep method."""
    from aiida.plugins import DataFactory
    PreProcessData = DataFactory('phonopy.preprocess')

    if parameters:
        preprocess_data = generate_preprocess_data()
        append_inputs = {'preprocess_data': preprocess_data}
    else:
        append_inputs = {'structure': generate_structure()}

    process = generate_workchain_harmonic(append_inputs=append_inputs)

    process.setup()

    assert 'preprocess_data' in process.ctx
    assert isinstance(process.ctx.preprocess_data, PreProcessData)

    assert process.ctx.run_parallel == False
    assert process.ctx.is_magnetic == False
    assert process.ctx.plus_hubbard == False


@pytest.mark.usefixtures('aiida_profile')
def test_setup_with_dielectric(generate_workchain_harmonic, generate_structure, generate_inputs_dielectric):
    """Test `HarmonicWorkChain` setep method."""
    append_inputs = {'structure': generate_structure()}

    dielectric_inputs = generate_inputs_dielectric(electric_field_scale=1.0)
    dielectric_inputs['scf']['pw'].pop('structure')
    dielectric_inputs.pop('clean_workdir')
    append_inputs.update({'dielectric_workchain': dielectric_inputs})

    process = generate_workchain_harmonic(append_inputs=append_inputs)
    process.setup()

    assert process.ctx.run_parallel == True


def test_run_forces(generate_workchain_harmonic, generate_structure, generate_base_scf_workchain_node):
    """Test `HarmonicWorkChain.run_forces` method."""
    append_inputs = {'structure': generate_structure(), 'options': {'sleep_submission_time': 0.1}}
    process = generate_workchain_harmonic(append_inputs=append_inputs)

    process.setup()
    process.run_base_supercell()

    assert 'scf_supercell_0'

    process.ctx.scf_supercell_0 = generate_base_scf_workchain_node()
    process.run_forces()

    assert 'supercells' in process.outputs
    assert 'supercell_1' in process.outputs['supercells']
    assert 'scf_supercell_1' in process.ctx


def test_run_dielectric(
    generate_workchain_harmonic, generate_structure, generate_base_scf_workchain_node, generate_inputs_dielectric
):
    """Test `HarmonicWorkChain.run_dielectric` method."""
    dielectric_inputs = generate_inputs_dielectric(electric_field_scale=1.0)
    dielectric_inputs['scf']['pw'].pop('structure')
    dielectric_inputs.pop('clean_workdir')
    append_inputs = {
        'structure': generate_structure(),
        'dielectric_workchain': dielectric_inputs,
        'options': {
            'sleep_submission_time': 0
        },
    }
    process = generate_workchain_harmonic(append_inputs=append_inputs)

    process.setup()
    process.run_base_supercell()
    process.ctx.scf_supercell_0 = generate_base_scf_workchain_node()

    process.run_dielectric()

    assert 'dielectric_workchain' in process.ctx


@pytest.mark.parametrize(
    ('is_magnetic'),
    ((True), (False)),
)
def test_run_results(
    generate_workchain_harmonic, generate_structure, generate_dielectric_workchain_node, generate_inputs_dielectric,
    is_magnetic
):
    """Test `HarmonicWorkChain.run_results` method."""
    from aiida import orm
    import numpy

    dielectric_inputs = generate_inputs_dielectric(electric_field_scale=1.0)
    dielectric_inputs['scf']['pw'].pop('structure')
    dielectric_inputs.pop('clean_workdir')
    append_inputs = {
        'structure': generate_structure(),
        'dielectric_workchain': dielectric_inputs,
        'options': {
            'sleep_submission_time': 0.1
        },
    }
    process = generate_workchain_harmonic(append_inputs=append_inputs)

    process.setup()
    process.ctx.dielectric_workchain = generate_dielectric_workchain_node(raman=False)
    process.ctx.is_magnetic = is_magnetic

    forces_1 = orm.ArrayData()
    forces_1.set_array('forces', numpy.full((2, 3), 1))
    forces_2 = orm.ArrayData()
    forces_2.set_array('forces', numpy.full((2, 3), -1))

    process.out(f'supercells_forces.scf_supercell_1', forces_1)
    process.out(f'supercells_forces.scf_supercell_2', forces_2)

    process.run_results()

    assert 'phonopy_data' in process.outputs
