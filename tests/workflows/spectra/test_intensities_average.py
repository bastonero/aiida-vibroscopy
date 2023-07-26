# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Tests for the :mod:`workflows.phonons.iraman` module."""
import pytest


@pytest.fixture
def generate_workchain_average(generate_workchain, generate_vibrational_data_from_forces):
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


@pytest.mark.usefixtures('aiida_profile')
def test_run_results(generate_workchain_average):
    """Test `IntensitiesAverageWorkChain.run_results` method."""
    process = generate_workchain_average()
    process.run_results()

    assert 'ir_averaged' in process.outputs
    assert 'raman_averaged' in process.outputs
