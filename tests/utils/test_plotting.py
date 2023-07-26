# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Test the :mod:`utils.plotting`."""


def test_get_spectra_plot():
    """Test the inputs for `get_spectra_plot`."""
    from aiida_vibroscopy.utils.plotting import get_spectra_plot

    get_spectra_plot([50, 100, 200], [1, 1, 1])
