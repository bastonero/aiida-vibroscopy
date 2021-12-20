# -*- coding: utf-8 -*-
"""Functions helping in the generation of the multiple direction electric field cards."""


def generate_cards_first_order(dE):
    """
    Generate an array of arrays, each containing the direction of the electric field
    in which to perturb for first order calculation of epsilon and born charges.
    Generates 3 efield_cards.
    """
    cards = []
    
    for i in range(3):
        vector = []
        for j in range(3):
            if j==i:
                vector.append(dE)
            else:
                vector.append(0.0)
        cards.append(vector)
        
    return cards


def find_direction(vector):
    """Return the index of first non zero value in the vector"""
    found = False
    index = 0
    try:
        while(not found):
            if(vector[index]!=0.):
                found=True
            else:
                index+=1
    except IndexError:
        return index, found
    else:
        return index, found


def generate_cards_second_order(dE):
    """
    Generate an array of arrays, each containing the direction of the electric field
    in which to perturb for second order derivatives calculation of the susceptibility.
    Generates 18 efield_cards.
    """
    from copy import deepcopy
    
    def sign(i,j):
        """Help function, used for imply symmetry."""
        if i<j: 
            return +1
        else:
            return -1

    cards=[]
    vector = [0.,0.,0.]
    for i in range(3):
        for j in range(3):
            if j==i:
                vector[j] = dE
                cards.append(deepcopy(vector))
                vector[j] = -dE
                cards.append(deepcopy(vector))
                vector = [0.,0.,0.]
            else:
                vector[i] = dE*sign(i,j)
                vector[j] = dE*sign(i,j)
                cards.append(deepcopy(vector))
                vector[i] = dE
                vector[j] = -dE
                cards.append(deepcopy(vector))
                vector = [0.,0.,0.]

    return cards


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
            