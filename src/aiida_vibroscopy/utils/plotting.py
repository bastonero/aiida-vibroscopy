# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Minimal plotting module for plotting spectra."""
from __future__ import annotations

import numpy as np

from aiida_vibroscopy.utils.broadenings import multilorentz


def get_spectra_plot(
    frequencies: list[float],
    intensities: list[float],
    broadening: float = 10.0,
    x_range: list[float] | str = 'auto',
    broadening_function=multilorentz,
    normalize: bool = True,
):
    """Plot a spectra using Matplotlib.

    :param frequencies: frequency modes (peaks) in cm^-1
    :param intensities: intensities of the modes
    :param broadening: broadening of the function (usually FWHM)
    :param x_range: range for plotting in cm^-1
    :param broadening_function: multi broadening function
    :param normalize: whether normalize the spectra to have maximum peak to 1

    :return: :mod:`matplotlib.pyplot`
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    # Some options to make the plot nice
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['text.usetex'] = False

    plt.rcParams['xtick.major.size'] = 7.0
    plt.rcParams['xtick.minor.size'] = 4.
    plt.rcParams['ytick.major.size'] = 7.0
    plt.rcParams['ytick.minor.size'] = 4.

    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['legend.fontsize'] = 15

    frequencies = np.array(frequencies)
    intensities = np.array(intensities)

    if x_range == 'auto':
        xi = max(0, frequencies.min() - 200)
        xf = frequencies.max() + 200
        x_range = np.arange(xi, xf, 1.)

    # Canvas
    _, ax = plt.subplots()

    y_range = broadening_function(x_range, frequencies, intensities, broadening)

    if normalize:
        y_range /= y_range.max()

    ax.plot(
        x_range,
        y_range,
        linewidth=2.0,
        linestyle='-',
    )

    # ----- Ticks and Labels
    ax.set_yticklabels('')
    ax.set_yticks([])
    ax.tick_params(axis='both', which='both', direction='in')
    ax.set_ylabel('Intensity (arb. units)')  # Add a y-label to the axes.
    ax.set_xlabel('Wavenumber (cm$^{-1}$)')

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    return plt
