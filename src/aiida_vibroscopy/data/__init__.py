# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""AiiDA data type for vibrational properties."""
from .vibro_fp import VibrationalFrozenPhononData
from .vibro_lr import VibrationalData

__all__ = ('VibrationalFrozenPhononData', 'VibrationalData')
