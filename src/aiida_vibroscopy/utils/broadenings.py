# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Broadening functions for plottin spectra."""
from __future__ import annotations

from copy import deepcopy

import numpy as np


def lorentz(x_range: np.ndarray, peak: float, intensity: float, gamma: float):
    """Compute a Lorentzian function.

    .. note:: assuming all quantities in cm-1, but intensity at arb. units

    :param x_range: frequency range
    :param peak: peak position
    :param intensity: intensity of the peak
    :param gamma: broadening, full width at half maximum (FWHM)
    :return: :class:`numpy.ndarray`
    """
    hwhm = gamma / 2.0
    return intensity * hwhm / ((x_range - peak)**2 + hwhm**2) / np.pi


def multilorentz(x_range: np.ndarray, peaks: list[float], intensities: list[float], gammas: float | list[float]):
    """Compute Lorentzian function for multiple peaks, and sum it.

    .. note:: assuming all quantities in cm-1, but intensities at arb. units

    :param x_range: frequency range
    :param peaks: list of peak positions
    :param intensities: list of intensities for each position
    :param gammas: list of broadenings, i.e. full width at half maximum (FWHM)
    :return: :class:`numpy.ndarray`
    """
    tot = np.zeros(np.array(x_range).shape)

    if isinstance(gammas, float):
        sigmas = [gammas for _ in peaks]
    elif isinstance(gammas, (list, np.ndarray)):
        if len(gammas) != len(peaks):
            raise ValueError("length of `gammas` and `peaks` doesn't match")
        sigmas = deepcopy(gammas)
    else:
        sigmas = float(gammas)

    if len(intensities) != len(peaks):
        raise ValueError("length of `intensities` and `peaks` doesn't match")

    for peak, intensity, sigma in zip(peaks, intensities, sigmas):
        tot += lorentz(x_range, peak, intensity, sigma)
    return tot


def voigt_profile(x_range: np.ndarray, peak: float, intensity: float, gamma_lorentz: float, sigma_gaussian: float):
    """Compute a Voigt profile convolution.

    .. note:: assuming all quantities in cm-1, but intensities at arb. units;
        implementation as from `Ida et al. J. Appl. Cryst. (2000). 33, 1311`

    .. important:: FWHM_Gaussian = 2 * sqrt(2 * ln(2)) * sigma_gaussian;
        FWHM_Gaussian ~ 2.355 * sigma_gaussian

    :param x_range: frequency range
    :param peak: peak position
    :param intensity: intensity of the peak
    :param gamma_lorentz: Lorentzian broadening, full width at half maximum (FWHM)
    :param sigma_gaussian: Guassian broadening, corresponding to the deviation standard (not FWHM)
    :return: :class:`numpy.ndarray`
    """
    # f_L = FWHM parameter for Lorenztian
    # f_G = FWHM parameter for Gaussian
    f_L = deepcopy(gamma_lorentz)
    f_G = 2.0 * np.sqrt(2 * np.log(2.0)) * deepcopy(sigma_gaussian)
    rho = f_L / (f_G + f_L)
    x = x_range - peak
    #
    list_a = [0.66000, 0.15021, -1.24984, 4.74052, -9.48291, 8.48252, -2.95553]
    list_b = [-0.42179, -1.25693, 10.30003, -23.45651, 29.14158, -16.50453, 3.19974]
    list_c = [1.19913, 1.43021, -15.36331, 47.06071, -73.61822, 57.92559, -17.80614]
    list_d = [1.10186, -0.47745, -0.68688, 2.76622, -4.55466, 4.05475, -1.26571]
    list_f = [-0.30165, -1.38927, 9.31550, -24.10743, 34.96491, -21.18862, 3.70290]
    list_g = [0.25437, -0.14107, 3.23653, -11.09215, 22.10544, -24.12407, 9.76947]
    list_h = [1.01579, 1.50429, -9.21815, 23.59717, -39.71134, 32.83023, -10.02142]
    #
    w_G = 1.
    w_L = 1.
    w_I = 0.
    w_P = 0.
    eta_L = rho
    eta_I = 0.
    eta_P = 0.
    #
    for index, _ in enumerate(list_a):
        i = index  #fortran convention
        w_G = w_G - rho * list_a[index] * (rho**i)
        w_L = w_L - (1. - rho) * list_b[index] * (rho**i)
        w_I = w_I + list_c[index] * (rho**i)
        w_P = w_P + list_d[index] * (rho**i)
        eta_L = eta_L + rho * (1. - rho) * list_f[index] * (rho**i)
        eta_I = eta_I + rho * (1. - rho) * list_g[index] * (rho**i)
        eta_P = eta_P + rho * (1. - rho) * list_h[index] * (rho**i)
    #
    #
    big_W_G = w_G * (f_G + f_L)
    big_W_L = w_L * (f_G + f_L)
    big_W_I = w_I * (f_G + f_L)
    big_W_P = w_P * (f_G + f_L)
    #
    gamma_G = big_W_G / (2.0 * (np.log(2.0)**0.5))
    gamma_L = big_W_L / 2.
    gamma_I = big_W_I / (2. * (2**(2. / 3.) - 1.)**0.5)
    #this parameter has been checked numerically, the original paper has a typo
    # in the formula for gamma_P
    gamma_P = big_W_P / (2. * (np.log(np.sqrt(2.0) + 1.0)))
    #
    #this because the paper uses a rescaled sigma.
    res_Gauss = np.sqrt(1.0 / np.pi) * (1.0 / gamma_G) * np.exp(-(x / gamma_G)**2)
    #
    res_Lorentzian = (1. / np.pi) * gamma_L / (x**2 + gamma_L**2)
    #
    res_f_I = 1. / (2. * gamma_I) * ((1. + (x / gamma_I)**2)**(-3. / 2.))
    #
    res_f_P = 1. / (2. * gamma_P) * (2. / (np.exp(x / gamma_P) + np.exp(-x / gamma_P)))**2
    #
    res = (1. - eta_L - eta_I - eta_P) * res_Gauss + eta_L * res_Lorentzian + eta_I * res_f_I + eta_P * res_f_P
    return intensity * res


def multilvoigt(
    x_range: np.ndarray,
    peaks: list[float],
    intensities: list[float],
    gammas_lorentz: float | list[float],
    sigma_gaussian: float,
):
    """Compute Lorentzian function for multiple peaks, and sum it.

    .. note:: assuming all quantities in cm-1, but intensities at arb. units

    .. important:: FWHM_Gaussian = 2 * sqrt(2 * ln(2)) * sigma_gaussian;
        FWHM_Gaussian ~ 2.355 * sigma_gaussian

    :param x_range: frequency range
    :param peaks: list of peak positions
    :param intensities: list of intensities for each position
    :param gammas_lorentz: list (or single value) of Lorentzian broadenings, i.e. full width at half maximum (FWHM)
    :param sigma_gaussian: Guassian broadening, corresponding to sigma
    :return: :class:`numpy.ndarray`
    """
    tot = np.zeros(np.array(x_range).shape)

    if isinstance(gammas_lorentz, float):
        sigmas = [gammas_lorentz for _ in peaks]
    elif isinstance(gammas_lorentz, (list, np.ndarray)):
        if len(gammas_lorentz) != len(peaks):
            raise ValueError("length of `gammas_lorentz` and `peaks` doesn't match")
        sigmas = deepcopy(gammas_lorentz)
    else:
        sigmas = float(gammas_lorentz)

    if len(intensities) != len(peaks):
        raise ValueError("length of `intensities` and `peaks` doesn't match")

    for peak, intensity, sigma in zip(peaks, intensities, sigmas):
        tot += voigt_profile(x_range, peak, intensity, sigma, sigma_gaussian)
    return tot


def gaussian(x_range: np.ndarray, peak: float, intensity: float, sigma: float):
    """Compute a Gaussian function.

    .. note:: assuming all quantities in cm-1, but intensity at arb. units

    .. important:: FWHM_Gaussian = 2 * sqrt(2 * ln(2)) * sigma_gaussian;
        FWHM_Gaussian ~ 2.355 * sigma_gaussian

    :param x_range: frequency range
    :param peak: peak position
    :param intensity: intensity of the peak
    :param sigma: standard deviation of the Gaussian
    :return: :class:`numpy.ndarray`
    """
    prefactor = intensity / (np.sqrt(2 * np.pi) * sigma)
    return prefactor * np.exp(-(x_range - peak)**2 / (2 * sigma**2))
