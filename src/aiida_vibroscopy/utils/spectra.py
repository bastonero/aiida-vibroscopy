# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Utility functions for plotting spectra."""
from __future__ import annotations

import numpy as np


def boson_factor(frequency: float, temperature: float = 300) -> float:
    """Return boson factor.

    .. note:: boson factor for Raman as (n+1).

    :param frequency: frequency in cm^-1
    :param temperature: temperature in Kelvin
    :return: boson occupation factor (adimensional)
    """
    from phonopy.units import CmToEv, Kb

    return 1.0 / (1.0 - np.exp(-CmToEv * frequency / (temperature * Kb)))


def nanometer_to_cm(value: float):
    """Convert laser frequency from nm to cm^-1."""
    return 10**7 / value


def raman_prefactor(frequency: float, frequency_laser: float, temperature: float, absolute: bool = True) -> float:
    r"""Return the Raman prefactor.

    :param frequency: frequency in cm^-1
    :param frequency_laser: frequency in nm (nanometer)
    :param temperature: temperature in Kelvin
    :param absolute: whether to use the conversion factor for theoretical Raman cross-section,
        which gives the intensities in (sterad^-1 cm^-2), default to True
    :return: the scalar :math:`(\\omega_L-\\omega)^4 [n(\\omega,T)+1] / \\omega`
    """
    from aiida_vibroscopy.common.constants import DEFAULT

    pre = DEFAULT.raman_xsection if absolute else 1
    laser = nanometer_to_cm(frequency_laser)

    return pre * boson_factor(frequency, temperature) * (laser - frequency)**4 / frequency
