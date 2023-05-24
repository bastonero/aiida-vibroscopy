# -*- coding: utf-8 -*-
"""Tests for the ``PhononWorkChain.get_builder_from_protocol`` method."""
from aiida.engine import ProcessBuilder
from aiida_quantumespresso.common.types import ElectronicType, SpinType
import pytest

from aiida_vibroscopy.common.properties import PhononProperty
from aiida_vibroscopy.workflows.phonons.base import PhononWorkChain


def test_get_available_protocols():
    """Test ``PhononWorkChain.get_available_protocols``."""
    protocols = PhononWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ['fast', 'moderate', 'precise']
    assert all('description' in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``PhononWorkChain.get_default_protocol``."""
    assert PhononWorkChain.get_default_protocol() == 'moderate'


def test_default(fixture_code, generate_structure, data_regression, serialize_builder):
    """Test ``PhononWorkChain.get_builder_from_protocol`` for the default protocol."""
    code = fixture_code('quantumespresso.pw')
    structure = generate_structure(structure_id='silicon')
    builder = PhononWorkChain.get_builder_from_protocol(code, structure)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_electronic_type(fixture_code, generate_structure):
    """Test ``PhononWorkChain.get_builder_from_protocol`` with ``electronic_type`` keyword."""
    code = fixture_code('quantumespresso.pw')
    structure = generate_structure(structure_id='silicon')

    with pytest.raises(NotImplementedError):
        for electronic_type in [ElectronicType.AUTOMATIC]:
            PhononWorkChain.get_builder_from_protocol(code, structure, electronic_type=electronic_type)

    builder = PhononWorkChain.get_builder_from_protocol(code, structure, electronic_type=ElectronicType.INSULATOR)

    for namespace in [builder.scf]:
        parameters = namespace['pw']['parameters'].get_dict()
        assert parameters['SYSTEM']['occupations'] == 'fixed'
        assert 'degauss' not in parameters['SYSTEM']
        assert 'smearing' not in parameters['SYSTEM']


def test_spin_type(fixture_code, generate_structure):
    """Test ``PhononWorkChain.get_builder_from_protocol`` with ``spin_type`` keyword."""
    code = fixture_code('quantumespresso.pw')
    structure = generate_structure('silicon')

    with pytest.raises(NotImplementedError):
        for spin_type in [SpinType.NON_COLLINEAR, SpinType.SPIN_ORBIT]:
            PhononWorkChain.get_builder_from_protocol(code, structure, spin_type=spin_type)

    builder = PhononWorkChain.get_builder_from_protocol(code, structure, spin_type=SpinType.COLLINEAR)

    for namespace in [builder.scf]:
        parameters = namespace['pw']['parameters'].get_dict()
        assert parameters['SYSTEM']['nspin'] == 2
        assert parameters['SYSTEM']['starting_magnetization'] == {'Si': 0.1}


def test_overrides(fixture_code, generate_structure):
    """Test ``PhononWorkChain.get_builder_from_protocol`` with overrides."""
    code = fixture_code('quantumespresso.pw')
    structure = generate_structure('silicon')

    overrides = {
        'primitive_matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'displacement_generator': {
            'distance': 0.005
        },
    }
    builder = PhononWorkChain.get_builder_from_protocol(code, structure, overrides=overrides)

    assert builder.primitive_matrix.get_list() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert builder.displacement_generator.get_dict() == {'distance': 0.005}


def test_phonon_properties(fixture_code, generate_structure):
    """Test ``PhononWorkChain.get_builder_from_protocol`` with phonon properties."""
    pw_code = fixture_code('quantumespresso.pw')
    phonopy_code = fixture_code('phonopy.phonopy')
    structure = generate_structure('silicon')
    phonon_property = PhononProperty.BANDS

    builder = PhononWorkChain.get_builder_from_protocol(
        pw_code, structure, phonopy_code=phonopy_code, phonon_property=phonon_property
    )

    assert builder.phonopy.parameters.get_dict() == {'band': 'auto'}


def test_phonon_properties_raise(fixture_code, generate_structure):
    """Test ``PhononWorkChain.get_builder_from_protocol`` raising error while not specifying phonopy code."""
    pw_code = fixture_code('quantumespresso.pw')
    structure = generate_structure('silicon')
    phonon_property = PhononProperty.BANDS

    match = '`PhononProperty` is specified, but `phonopy_code` is None'
    with pytest.raises(ValueError, match=match):
        PhononWorkChain.get_builder_from_protocol(pw_code, structure, phonon_property=phonon_property)


def test_overrides_phonopy(fixture_code, generate_structure):
    """Test ``PhononWorkChain.get_builder_from_protocol`` with overriding phonopy inputs."""
    pw_code = fixture_code('quantumespresso.pw')
    phonopy_code = fixture_code('phonopy.phonopy')
    structure = generate_structure('silicon')
    phonon_property = PhononProperty.BANDS

    overrides = {'phonopy': {'parameters': {'band': 'fancy'}}}
    builder = PhononWorkChain.get_builder_from_protocol(
        pw_code, structure, phonopy_code=phonopy_code, overrides=overrides, phonon_property=phonon_property
    )

    assert builder.phonopy.parameters.get_dict() == {'band': 'fancy'}


def test_options(fixture_code, generate_structure):
    """Test specifying ``options`` for the ``get_builder_from_protocol()`` method."""
    code = fixture_code('quantumespresso.pw')
    structure = generate_structure()

    queue_name = 'super-fast'
    withmpi = False  # The protocol default is ``True``

    options = {'queue_name': queue_name, 'withmpi': withmpi}
    builder = PhononWorkChain.get_builder_from_protocol(code, structure, options=options)

    assert builder.scf.pw.metadata['options']['queue_name'] == queue_name
