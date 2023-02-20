# -*- coding: utf-8 -*-
"""Tests for the :mod:`data` module."""
import pytest


@pytest.mark.usefixtures('aiida_profile')
def test_methods(generate_vibrational_data_from_forces):
    """Test `VibrationalMixin` methods."""
    vibrational_data = generate_vibrational_data_from_forces()

    vibrational_data.run_raman_susceptibility_tensors()
    vibrational_data.run_polarization_vectors()
    vibrational_data.run_polarized_raman_intensities([1, 0, 0], [-1, 0, 0])
    vibrational_data.run_powder_raman_intensities()
    vibrational_data.run_polarized_ir_intensities([1, 0, 0])
    vibrational_data.run_powder_ir_intensities()


@pytest.mark.usefixtures('aiida_profile')
def test_powder_methods(generate_vibrational_data_from_forces):
    """Test `VibrationalMixin` powder spectra methods."""
    vibrational_data = generate_vibrational_data_from_forces()

    vibrational_data.run_powder_raman_intensities(quadrature_order=3)
    vibrational_data.run_powder_ir_intensities(quadrature_order=3)
