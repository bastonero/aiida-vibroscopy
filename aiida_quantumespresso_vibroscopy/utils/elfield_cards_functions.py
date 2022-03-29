# -*- coding: utf-8 -*-
"""Functions helping in the generation of the multiple direction electric field cards."""

__all__ = ('get_vector_from_number',)


def get_vector_from_number(number, value):
    """Get the electric field vector from the number for finite differences.

    The Voigt notation is used:
        * 0,1,2 for first order derivatives: l --> {l}j ; e.g. 0 does 00, 01, 02
        * 0,1,2,3,4,5 for second order derivatives: l <--> ij --> {ij}k ;
            precisely 0 > {00}k; 1 > {11}k; 2 > {22}k; 3 > {12}k; 4 > {02}k; 5 --> {01}k | k=0,1,2.

    :param number: the number according to Voigt notation to get the associated electric field vector
    :param value: value of the electric field

    :returns: (3,) shape list
    """
    if not number in (0, 1, 2, 3, 4, 5):
        raise ValueError('Only numbers from 0 to 5 are accepted')

    if number == 0:
        vector = [value, 0, 0]
    if number == 1:
        vector = [0, value, 0]
    if number == 2:
        vector = [0, 0, value]
    if number == 3:
        vector = [0, value, value]
    if number == 4:
        vector = [value, 0, value]
    if number == 5:
        vector = [value, value, 0]

    return vector
