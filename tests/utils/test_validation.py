# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Test the :mod:`utils.validation`."""
import numpy as np
import pytest


def test_validate_tot_magnetization():
    """Test `validate_tot_magnetization`."""
    from aiida_vibroscopy.utils.validation import validate_tot_magnetization

    magnetization = 1.1
    assert not validate_tot_magnetization(magnetization)

    magnetization = 1.3
    assert validate_tot_magnetization(magnetization)


@pytest.mark.parametrize(('value', 'message'), (
    (5.0, 'value is not of the right type; only `list`, `aiida.orm.List` and `numpy.ndarray`'),
    ([0, 0], 'need exactly 3 diagonal elements or 3x3 arrays.'),
    ([[0, 0, 0], [0], [0, 0, 0]], 'matrix need to have 3x1 or 3x3 shape.'),
    (np.ones((3)), None),
    (np.ones((3, 3)), None),
))
def test_validate_matrix(
    value,
    message,
):
    """Test `validate_tot_magnetization`."""
    from aiida_vibroscopy.utils.validation import validate_matrix
    assert validate_matrix(value, None) == message


def test_validate_positive():
    """Test `validate_positive`."""
    from aiida.orm import Float

    from aiida_vibroscopy.utils.validation import validate_positive

    assert validate_positive(Float(1.0), None) is None
    assert validate_positive(Float(-1.0), None) == 'specified value is negative.'
