# -*- coding: utf-8 -*-
"""Test the :mod:`utils.broadenings`."""
import numpy as np
import pytest


@pytest.fixture
def generate_lorentz_inputs():
    """Generate an input for Lorentz funciton."""

    def _generate_lorentz_inputs(multi=False):
        """Generate the inputs."""
        x_range = np.arange(0, 100, 0.1)
        peak = 50.0
        intensity = 1.0
        sigma = 10.0

        if multi:
            peak = [20.0, 30.0, 40.0]
            sigma = peak
            intensity = peak

        return [x_range, peak, intensity, sigma]

    return _generate_lorentz_inputs


@pytest.fixture
def generate_voigt_inputs():
    """Generate an input for Voigt profile."""

    def _generate_voigt_inputs(multi=False, sigma_gaussian=1.0):
        """Generate the inputs."""
        x_range = np.arange(0, 100, 0.1)
        peak = 50.0
        intensity = 1.0
        sigma = 10.0

        if multi:
            peak = [20.0, 30.0, 40.0]
            sigma = peak
            intensity = peak

        return [x_range, peak, intensity, sigma, sigma_gaussian]

    return _generate_voigt_inputs


def test_lorentz_inputs(generate_lorentz_inputs):
    """Test the `lorentz` function."""
    from aiida_vibroscopy.utils.broadenings import lorentz

    lorentz(*generate_lorentz_inputs())


def test_multilorentz_inputs(generate_lorentz_inputs):
    """Test the `multilorentz` function."""
    from aiida_vibroscopy.utils.broadenings import multilorentz

    multilorentz(*generate_lorentz_inputs(multi=True))

    # Testing sigmas as single float, i.e. same for each peak
    inputs = generate_lorentz_inputs(multi=True)
    inputs[-1] = 8.0
    multilorentz(*inputs)


def test_multilorentz_error(generate_lorentz_inputs):
    """Test `multilorentz` raising error."""
    from aiida_vibroscopy.utils.broadenings import multilorentz

    inputs = generate_lorentz_inputs(multi=True)
    inputs[-1] = [8.0]

    match = r"length of `gammas` and `peaks` doesn't match"
    with pytest.raises(ValueError, match=match):
        multilorentz(*inputs)

    inputs = generate_lorentz_inputs(multi=True)
    inputs[-2] = [8.0]

    match = r"length of `intensities` and `peaks` doesn't match"
    with pytest.raises(ValueError, match=match):
        multilorentz(*inputs)


def test_voigt_against_lorentz(generate_voigt_inputs, generate_lorentz_inputs):
    """Test the `voigt_profile` function against Lorentz.

    When `sigma_gaussian = 0`, Voigt and Lorentz coincide.
    """
    from aiida_vibroscopy.utils.broadenings import lorentz, voigt_profile

    voigt_res = voigt_profile(*generate_voigt_inputs(sigma_gaussian=0))
    lorentz_res = lorentz(*generate_lorentz_inputs())

    assert np.abs(voigt_res - lorentz_res).max() < 1e-10


def test_voigt_against_gaussian(generate_voigt_inputs, generate_lorentz_inputs):
    """Test the `voigt_profile` function against Gaussian.

    When `sigma_lorentz = 0`, Voigt and Gaussian coincide.
    """
    from aiida_vibroscopy.utils.broadenings import gaussian, voigt_profile

    x_range = np.arange(0, 100, 0.1)
    peak, intensity, sigma = 50, 1, 2
    voigt_res = voigt_profile(x_range, peak, intensity, gamma_lorentz=0, sigma_gaussian=sigma)
    gaussian_res = gaussian(x_range, peak, intensity, sigma)

    assert np.abs(voigt_res - gaussian_res).max() < 1e-10


def test_voigt_inputs(generate_voigt_inputs):
    """Test the `voigt_profile` function."""
    from aiida_vibroscopy.utils.broadenings import voigt_profile

    voigt_profile(*generate_voigt_inputs())


def test_multilorentz_inputs(generate_voigt_inputs):
    """Test the `multilorentz` function."""
    from aiida_vibroscopy.utils.broadenings import multilvoigt

    multilvoigt(*generate_voigt_inputs(multi=True))

    # Testing sigmas as single float, i.e. same for each peak
    inputs = generate_voigt_inputs(multi=True)
    inputs[-1] = 8.0
    multilvoigt(*inputs)


def test_multilvoigt_error(generate_voigt_inputs):
    """Test `multilvoigt` raising error."""
    from aiida_vibroscopy.utils.broadenings import multilvoigt

    inputs = generate_voigt_inputs(multi=True)
    inputs[-2] = [8.0]

    match = r"length of `gammas_lorentz` and `peaks` doesn't match"
    with pytest.raises(ValueError, match=match):
        multilvoigt(*inputs)

    inputs = generate_voigt_inputs(multi=True)
    inputs[-3] = [8.0]

    match = r"length of `intensities` and `peaks` doesn't match"
    with pytest.raises(ValueError, match=match):
        multilvoigt(*inputs)
