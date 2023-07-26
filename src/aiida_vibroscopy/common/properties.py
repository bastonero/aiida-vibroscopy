# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Module with common properties."""
import enum


class PhononProperty(enum.Enum):
    """Enumeration to indicate the phonon properties to extract for a system."""

    NONE = None
    BANDS = {'band': 'auto'}
    DOS = {'dos': True, 'mesh': 1000, 'write_mesh': False}
    THERMODYNAMIC = {'tprop': True, 'mesh': 1000, 'write_mesh': False}
