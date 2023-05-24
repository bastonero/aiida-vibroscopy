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


def test_lorentz(generate_lorentz_inputs):
    """Test the `lorentz` function."""
    from aiida_vibroscopy.utils.broadenings import lorentz

    lorentz(*generate_lorentz_inputs())


def test_multilorentz(generate_lorentz_inputs):
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

    match = r"length of `sigmas` and `peaks` doesn't match"
    with pytest.raises(ValueError, match=match):
        multilorentz(*inputs)

    inputs = generate_lorentz_inputs(multi=True)
    inputs[-2] = [8.0]

    match = r"length of `intensities` and `peaks` doesn't match"
    with pytest.raises(ValueError, match=match):
        multilorentz(*inputs)
