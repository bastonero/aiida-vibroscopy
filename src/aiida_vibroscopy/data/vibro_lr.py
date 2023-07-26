# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Mixin for vibrational data."""
from aiida_phonopy.data.force_constants import ForceConstantsData

from .vibro_mixin import VibrationalMixin

__all__ = ('VibrationalData',)


class VibrationalData(ForceConstantsData, VibrationalMixin):  # pylint: disable=too-many-ancestors
    """Vibrational data for IR and Raman spectra."""
