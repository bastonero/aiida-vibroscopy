# -*- coding: utf-8 -*-
"""Broadening functions for plottin spectra."""
from __future__ import annotations

import numpy as np


def lorentz(x_range: np.ndarray, peak: float, intensity: float, sigma: float):
    """Compute a Lorentzian function.

    .. note:: assuming all quantities in cm-1, but intensity at arb. units

    :param x_range: frequency range
    :param peak: peak position
    :param intensity: intensity of the peak
    :param sigma: broadening, full width at half maximum (FWHM)
    :return: :class:`numpy.ndarray`
    """
    gamma = sigma / 2.0
    return intensity * gamma / ((x_range - peak)**2 + gamma**2) / np.pi


def multilorentz(x_range: np.ndarray, peaks: list[float], intensities: list[float], sigmas: float | list[float]):
    """Compute Lorentzian function for multiple peaks, and sum it.

    .. note:: assuming all quantities in cm-1, but intensities at arb. units

    :param x_range: frequency range
    :param peaks: list of peak positions
    :param intensities: list of intensities for each position
    :param sigmas: list of broadenings, i.e. full width at half maximum (FWHM)
    :return: :class:`numpy.ndarray`
    """
    tot = np.zeros(np.array(x_range).shape)

    if isinstance(sigmas, float):
        sigmas = [sigmas for _ in peaks]
    elif isinstance(sigmas, list):
        if len(sigmas) != len(peaks):
            raise ValueError("length of `sigmas` and `peaks` doesn't match")
    else:
        sigmas = float(sigmas)

    if len(intensities) != len(peaks):
        raise ValueError("length of `intensities` and `peaks` doesn't match")

    for peak, intensity, sigma in zip(peaks, intensities, sigmas):
        tot += lorentz(x_range, peak, intensity, sigma)
    return tot
