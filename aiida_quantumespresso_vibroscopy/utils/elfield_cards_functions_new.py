# -*- coding: utf-8 -*-
"""Functions helping in the generation of the multiple direction electric field cards."""

from math import sqrt

import numpy as np

root2 = sqrt(2)

def generate_electric_fields(dE: float, order: int, ttype: str, scale=root2):
    """
    Generate an array of electric field directions which serve as discrete points
    for the central derivatives scheme.
    The directions depend whether it is for NAC, IR or Raman tensors.

    :param dE: value of the electric field (in Ry a.u.)
    :param order: order of the central difference numerical differentiation
    :param scale: scaling factor for direction non parallel to the cartesiaan axis
    :return: a list of electric fields directions
    
    :NOTE: null direction is excluded.
    """
    cards = central_differences_points(dE, scale)
    if order in (4,6):
        cards += central_differences_points(2*dE, scale)
    if order == 6:
        cards += central_differences_points(3*dE, scale)
        
    return cards


def central_differences_points(dE, scale):
    """Numerical central derivatives points."""
    parallel = dE*np.array([
        [[+1,0,0],[0,+1,0],[0,0,+1],],
        [[-1,0,0],[0,-1,0],[0,0,-1],]
        ])
    
    diagonal = (dE/scale)*np.array([
        [[+1,+1,0],[+1,0,+1],[0,+1,+1],],
        [[-1,-1,0],[-1,0,-1],[0,-1,-1],]
        ])
    
    return [[parallel.tolist()]+[diagonal.tolist()]]


def find_directions(vector):
    """Return the index of first non zero value in the vector."""
    direction = ''
    for i in range(len(vector)):
        if abs(vector[i])>0.0:
            direction+=str(i)
            if vector[i]>0.0:
                direction+='p'
            else:
                direction+='m'
                
    return direction