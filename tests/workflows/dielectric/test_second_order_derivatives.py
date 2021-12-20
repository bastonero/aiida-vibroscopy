# -*- coding: utf-8 -*-
# pylint: disable=no-member,redefined-outer-name
"""Tests for the `SecondOrderDerivativesWorkChain` class."""
import pytest
from aiida.orm import Dict


@pytest.fixture
def generate_workchain_second_derivatives(generate_workchain, generate_inputs_second_derivatives):
    """Generate an instance of a `SecondOrderDerivativesWorkChain`."""

    def _generate_workchain_second_derivatives(inputs=None):
        entry_point = 'quantumespresso.fd.second_order_derivatives'

        if inputs is None:
            inputs = generate_inputs_second_derivatives()

        process = generate_workchain(entry_point, inputs)

        return process

    return _generate_workchain_second_derivatives


@pytest.mark.usefixtures('aiida_profile')
def test_setup_and_results(generate_workchain_second_derivatives, generate_inputs_second_derivatives):
    """Test `SecondOrderDerivativesWorkChain`."""
    inputs = generate_inputs_second_derivatives()
    process = generate_workchain_second_derivatives(inputs=inputs)
    process.setup()
    process.run_results()


@pytest.mark.usefixtures('aiida_profile')
def test_one_direction(generate_workchain_second_derivatives, generate_inputs_second_derivatives):
    """Test `SecondOrderDerivativesWorkChain`."""
    inputs = generate_inputs_second_derivatives(trial=0)
    process = generate_workchain_second_derivatives(inputs=inputs)
    process.setup()
    process.run_results()


@pytest.mark.usefixtures('aiida_profile')
def test_invalid_inputs(generate_workchain_second_derivatives, generate_inputs_second_derivatives):
    """Test `SecondOrderDerivativesWorkChain` validation method."""
    for trial in (1,2):
        with pytest.raises(ValueError):
            inputs = generate_inputs_second_derivatives(trial=trial)
            process = generate_workchain_second_derivatives(inputs=inputs)
