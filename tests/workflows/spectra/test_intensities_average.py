# -*- coding: utf-8 -*-
"""Tests for the :mod:`workflows.phonons.iraman` module."""
import pytest


@pytest.fixture(name='generate_workchain_average')
def generate_workchain_average_fixture(generate_workchain, generate_vibrational_data_from_forces):
    """Generate an instance of a `IntensitiesAverageWorkChain`."""

    def _generate_workchain_average(append_inputs=None, return_inputs=False):
        from aiida import orm
        entry_point = 'vibroscopy.spectra.intensities_average'
        vibrational_data = generate_vibrational_data_from_forces()
        parameters = orm.Dict({'quadrature_order': 3})

        inputs = {
            'vibrational_data': vibrational_data,
            'parameters': parameters,
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
