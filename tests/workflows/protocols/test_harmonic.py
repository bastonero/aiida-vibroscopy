# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Tests for the ``HarmonicWorkChain.get_builder_from_protocol`` method."""
from aiida.engine import ProcessBuilder
from aiida_quantumespresso.common.types import ElectronicType, SpinType
import pytest

from aiida_vibroscopy.common.properties import PhononProperty
from aiida_vibroscopy.workflows.phonons.harmonic import HarmonicWorkChain


def test_get_available_protocols():
    """Test ``HarmonicWorkChain.get_available_protocols``."""
    protocols = HarmonicWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == sorted(['fast', 'balanced', 'stringent'])
    assert all('description' in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``HarmonicWorkChain.get_default_protocol``."""
    assert HarmonicWorkChain.get_default_protocol() == 'balanced'


def test_default(fixture_code, generate_structure, data_regression, serialize_builder):
    """Test ``HarmonicWorkChain.get_builder_from_protocol`` for the default protocol."""
    code = fixture_code('quantumespresso.pw')
    structure = generate_structure('silicon')
    builder = HarmonicWorkChain.get_builder_from_protocol(code, structure)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_electronic_type(fixture_code, generate_structure):
    """Test ``HarmonicWorkChain.get_builder_from_protocol`` with ``electronic_type`` keyword."""
    code = fixture_code('quantumespresso.pw')
    structure = generate_structure('silicon')

    with pytest.raises(NotImplementedError):
        for electronic_type in [ElectronicType.AUTOMATIC]:
            HarmonicWorkChain.get_builder_from_protocol(code, structure, electronic_type=electronic_type)

    builder = HarmonicWorkChain.get_builder_from_protocol(code, structure, electronic_type=ElectronicType.INSULATOR)

    for namespace in [builder.phonon.scf]:
        parameters = namespace['pw']['parameters'].get_dict()
        assert parameters['SYSTEM']['occupations'] == 'fixed'
        assert 'degauss' not in parameters['SYSTEM']
        assert 'smearing' not in parameters['SYSTEM']


def test_spin_type(fixture_code, generate_structure):
    """Test ``HarmonicWorkChain.get_builder_from_protocol`` with ``spin_type`` keyword."""
    code = fixture_code('quantumespresso.pw')
    structure = generate_structure(structure_id='silicon')

    builder = HarmonicWorkChain.get_builder_from_protocol(code, structure, spin_type=SpinType.COLLINEAR)

    for namespace in [builder.phonon.scf]:
        parameters = namespace['pw']['parameters'].get_dict()
        assert parameters['SYSTEM']['nspin'] == 2
        assert parameters['SYSTEM']['starting_magnetization'] == {'Si': 0.1}


def test_options(fixture_code, generate_structure):
    """Test specifying ``options`` for the ``get_builder_from_protocol()`` method."""
    code = fixture_code('quantumespresso.pw')
    structure = generate_structure()

    queue_name = 'super-fast'
    withmpi = False  # The protocol default is ``True``

    options = {'queue_name': queue_name, 'withmpi': withmpi}
    builder = HarmonicWorkChain.get_builder_from_protocol(code, structure, options=options)

    for subspace in (builder.phonon.scf.pw.metadata,):
        assert subspace['options']['queue_name'] == queue_name, subspace


def test_phonon_properties(fixture_code, generate_structure):
    """Test ``HarmonicWorkChain.get_builder_from_protocol`` with phonon properties."""
    pw_code = fixture_code('quantumespresso.pw')
    phonopy_code = fixture_code('phonopy.phonopy')
    structure = generate_structure('silicon')
    phonon_property = PhononProperty.BANDS

    builder = HarmonicWorkChain.get_builder_from_protocol(
        pw_code, structure, phonopy_code=phonopy_code, phonon_property=phonon_property
    )

    assert builder.phonopy.parameters.get_dict() == {'band': 'auto'}


def test_phonon_properties_raise(fixture_code, generate_structure):
    """Test ``HarmonicWorkChain.get_builder_from_protocol`` raising error while not specifying phonopy code."""
    pw_code = fixture_code('quantumespresso.pw')
    structure = generate_structure('silicon')
    phonon_property = PhononProperty.BANDS

    match = '`PhononProperty` is specified, but `phonopy_code` is None'
    with pytest.raises(ValueError, match=match):
        HarmonicWorkChain.get_builder_from_protocol(pw_code, structure, phonon_property=phonon_property)


def test_overrides_phonopy(fixture_code, generate_structure):
    """Test ``HarmonicWorkChain.get_builder_from_protocol`` with overriding phonopy inputs."""
    pw_code = fixture_code('quantumespresso.pw')
    phonopy_code = fixture_code('phonopy.phonopy')
    structure = generate_structure('silicon')
    phonon_property = PhononProperty.BANDS

    overrides = {'phonopy': {'parameters': {'band': 'fancy'}}}
    builder = HarmonicWorkChain.get_builder_from_protocol(
        pw_code, structure, phonopy_code=phonopy_code, overrides=overrides, phonon_property=phonon_property
    )

    assert builder.phonopy.parameters.get_dict() == {'band': 'fancy'}
