# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Functions helping in the generation of the multiple direction electric field cards."""
from __future__ import annotations

__all__ = ('get_vector_from_number', 'get_tuple_from_vector')


def get_vector_from_number(number: int, value: float) -> tuple[float, float, float]:
    """Get the electric field vector from the number for finite differences.

    The Voigt notation is used:

    * 0, 1, 2 for first order derivatives: l --> {l}j ; e.g. 0 does 00, 01, 02
    * 0, 1, 2, 3, 4, 5 for second order derivatives: l <--> ij --> {ij}k ;
        precisely (k = 0, 1, 2):
        - 0 --> {00}k
        - 1 --> {11}k
        - 2 --> {22}k
        - 3 --> {12}k
        - 4 --> {02}k
        - 5 --> {01}k

    :param number: the number according to Voigt notation to get the associated electric field vector
    :param value: value of the electric field

    :return: (3,) shape list
    """
    if not number in (0, 1, 2, 3, 4, 5):
        raise ValueError('Only numbers from 0 to 5 are accepted')

    if number == 0:
        return [value, 0, 0]
    if number == 1:
        return [0, value, 0]
    if number == 2:
        return [0, 0, value]
    if number == 3:
        return [0, value, value]
    if number == 4:
        return [value, 0, value]
    if number == 5:
        return [value, value, 0]


def get_tuple_from_vector(vector: tuple[float, float, float]) -> tuple[int, int]:
    """Return a tuple referring to the Voigt number and the sing of the direction.

    :return: tuple(Voigt number, sign), the number is between 0 and 5, the sign 1 or -1.
    """
    import numpy as np

    sign = 1 if (np.array(vector) >= 0).all() else -1
    direction = np.abs(vector)
    direction = np.array(direction, dtype='int').tolist()

    if direction == [1, 0, 0]:
        return 0, sign
    if direction == [0, 1, 0]:
        return 1, sign
    if direction == [0, 0, 1]:
        return 2, sign
    if direction == [0, 1, 1]:
        return 3, sign
    if direction == [1, 0, 1]:
        return 4, sign
    if direction == [1, 1, 0]:
        return 5, sign
