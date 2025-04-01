# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Tests for the ``DielectricWorkChain.get_builder_from_protocol`` method."""
from aiida.engine import ProcessBuilder
from aiida_quantumespresso.common.types import ElectronicType, SpinType
import pytest

from aiida_vibroscopy.workflows.dielectric.base import DielectricWorkChain


def test_get_available_protocols():
    """Test ``DielectricWorkChain.get_available_protocols``."""
    protocols = DielectricWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == sorted(['fast', 'balanced', 'stringent'])
    assert all('description' in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``DielectricWorkChain.get_default_protocol``."""
    assert DielectricWorkChain.get_default_protocol() == 'balanced'


def test_default(fixture_code, generate_structure, data_regression, serialize_builder):
    """Test ``DielectricWorkChain.get_builder_from_protocol`` for the default protocol."""
    code = fixture_code('quantumespresso.pw')
    structure = generate_structure(structure_id='silicon')
    builder = DielectricWorkChain.get_builder_from_protocol(code, structure)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_electronic_type(fixture_code, generate_structure):
    """Test ``DielectricWorkChain.get_builder_from_protocol`` with ``electronic_type`` keyword."""
    code = fixture_code('quantumespresso.pw')
    structure = generate_structure(structure_id='silicon')

    with pytest.raises(NotImplementedError):
        for electronic_type in [ElectronicType.AUTOMATIC]:
            DielectricWorkChain.get_builder_from_protocol(code, structure, electronic_type=electronic_type)

    builder = DielectricWorkChain.get_builder_from_protocol(code, structure, electronic_type=ElectronicType.INSULATOR)

    for namespace in [builder.scf]:
        parameters = namespace['pw']['parameters'].get_dict()
        assert parameters['SYSTEM']['occupations'] == 'fixed'
        assert 'degauss' not in parameters['SYSTEM']
        assert 'smearing' not in parameters['SYSTEM']


def test_spin_type(fixture_code, generate_structure):
    """Test ``DielectricWorkChain.get_builder_from_protocol`` with ``spin_type`` keyword."""
    code = fixture_code('quantumespresso.pw')
    structure = generate_structure('silicon')

    with pytest.raises(NotImplementedError):
        for spin_type in [SpinType.NON_COLLINEAR, SpinType.SPIN_ORBIT]:
            DielectricWorkChain.get_builder_from_protocol(code, structure, spin_type=spin_type)

    builder = DielectricWorkChain.get_builder_from_protocol(code, structure, spin_type=SpinType.COLLINEAR)

    for namespace in [builder.scf]:
        parameters = namespace['pw']['parameters'].get_dict()
        assert parameters['SYSTEM']['nspin'] == 2
        assert parameters['SYSTEM']['starting_magnetization'] == {'Si': 0.1}


def test_overrides_dielectric(fixture_code, generate_structure):
    """Test ``DielectricWorkChain.get_builder_from_protocol`` with overrides."""
    code = fixture_code('quantumespresso.pw')
    structure = generate_structure('silicon')

    step = 0.001
    accuracy = 4

    overrides = {'central_difference': {'electric_field_step': step, 'accuracy': 4}}
    builder = DielectricWorkChain.get_builder_from_protocol(code, structure, overrides=overrides)

    assert builder.central_difference.electric_field_step == step
    assert builder.central_difference.accuracy == accuracy


def test_options(fixture_code, generate_structure):
    """Test specifying ``options`` for the ``get_builder_from_protocol()`` method."""
    code = fixture_code('quantumespresso.pw')
    structure = generate_structure()

    queue_name = 'super-fast'
    withmpi = False  # The protocol default is ``True``

    options = {'queue_name': queue_name, 'withmpi': withmpi}
    builder = DielectricWorkChain.get_builder_from_protocol(code, structure, options=options)

    assert builder.scf.pw.metadata['options']['queue_name'] == queue_name
