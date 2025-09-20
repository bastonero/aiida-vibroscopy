# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Tests for the :mod:`data` module."""
import pytest


@pytest.mark.usefixtures('aiida_profile')
def test_methods(generate_vibrational_data_from_forces, ndarrays_regression):
    """Test `VibrationalMixin` methods."""
    import numpy as np

    vibrational_data = generate_vibrational_data_from_forces()

    results = {}
    vibrational_data.run_raman_susceptibility_tensors()[0]
    # results['raman_susceptibility_tensors'] = vibrational_data.run_raman_susceptibility_tensors()[0]
    vibrational_data.run_polarization_vectors()[0]
    # results['polarization_vectors'] = vibrational_data.run_polarization_vectors()[0]
    results['single_crystal_raman_intensities'] = vibrational_data.run_single_crystal_raman_intensities([1, 0, 0],
                                                                                                        [-1, 0, 0])[0]
    results['powder_raman_intensities'] = vibrational_data.run_powder_raman_intensities()[0]
    results['single_crystal_ir_intensities'] = vibrational_data.run_single_crystal_ir_intensities([1, 0, 0])[0]
    results['powder_ir_intensities'] = vibrational_data.run_powder_ir_intensities()[0]
    freq_range = np.arange(10, 1000, 10)
    results['complex_dielectric_function'] = vibrational_data.run_complex_dielectric_function(freq_range=freq_range)
    results['normal_reflectivity_spectrum'] = vibrational_data.run_normal_reflectivity_spectrum([
        0, 0, 1
    ], **dict(freq_range=freq_range))

    ndarrays_regression.check(results, default_tolerance=dict(atol=1e-2, rtol=1e-2))


@pytest.mark.usefixtures('aiida_profile')
def test_powder_methods(generate_vibrational_data_from_forces):
    """Test `VibrationalMixin` powder spectra methods."""
    vibrational_data = generate_vibrational_data_from_forces()

    vibrational_data.run_powder_raman_intensities(quadrature_order=3)
    vibrational_data.run_powder_ir_intensities(quadrature_order=3)
