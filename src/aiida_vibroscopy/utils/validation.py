# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Validation function utilities."""
from __future__ import annotations

__all__ = ('validate_tot_magnetization', 'validate_matrix', 'validate_positive')


def validate_tot_magnetization(tot_magnetization: float, thr: float = 0.2) -> bool:
    """Round the total magnetization input and return true if outside the threshold.

    This is needed because 'tot_magnetization' must be an integer in the aiida-quantumespresso input parameters.
    """
    int_tot_magnetization = round(tot_magnetization, 0)
    return abs(tot_magnetization - int_tot_magnetization) > thr


def validate_matrix(value, _):
    """Validate the `supercell_matrix` and `primitive_matrix` inputs."""
    from aiida.orm import List
    import numpy as np

    if not isinstance(value, (list, List, np.ndarray)):
        return 'value is not of the right type; only `list`, `aiida.orm.List` and `numpy.ndarray`'

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if not len(value) == 3:
        return 'need exactly 3 diagonal elements or 3x3 arrays.'

    for row in value:
        if isinstance(row, list):
            if not len(row) in [0, 3]:
                return 'matrix need to have 3x1 or 3x3 shape.'
            for element in row:
                if not isinstance(element, (int, float)):
                    return (
                        f'type `{type(element)}` of {element} is not an accepted '
                        'type in matrix; only `int` and `float` are valid.'
                    )


def validate_positive(value, _):
    """Validate that `value` is positive."""
    if not value.value > 0:
        return 'specified value is negative.'


# def validate_nac(value, _):
#     """Validate that `value` is a valid non-analytical ArrayData input."""
#     try:
#         value.get_array('dielectric')
#         value.get_array('born_charges')
#     except KeyError:
#         return 'data does not contain `dieletric` and/or `born_charges` arraynames.'
