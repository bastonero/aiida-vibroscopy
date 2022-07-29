# -*- coding: utf-8 -*-
"""Tests for the :mod:`workflows.phonons.iraman` module."""
import pytest


@pytest.fixture
def generate_workchain_average(generate_workchain, generate_vibrational_data):
    """Generate an instance of a `IntensitiesAverageWorkChain`."""

    def _generate_workchain_average(append_inputs=None, return_inputs=False):
        from aiida import orm
        entry_point =  'quantumespresso.vibroscopy.spectra.intensities_average'
        vibrational_data = generate_vibrational_data()
        options = orm.Dict(dict={'quadrature_order':3})

        inputs = {
            'vibrational_data': vibrational_data,
            'options': options,
        }

        if return_inputs:
            return inputs

        if append_inputs is not None:
            inputs.update(append_inputs)

        process = generate_workchain(entry_point, inputs)

        return process

    return _generate_workchain_average


def test_run_results(generate_workchain_average):
    """Test `IntensitiesAverageWorkChain.run_results` method."""
    process = generate_workchain_average()
    process.run_results()

    assert 'ir_averaged' in process.outputs
    assert 'raman_averaged' in process.outputs
