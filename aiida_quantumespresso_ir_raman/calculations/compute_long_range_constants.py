# -*- coding: utf-8 -*-
"""Calculation functions to return the calculated long-range constants."""
from aiida.engine import calcfunction
from aiida.orm import ArrayData, Float
import numpy as np


#@calcfunction                     
def compute_high_frequency_dielectric_tensor(parameters, outputs_null, outputs_finites, E):
    """Return the high frequency dielectric tensor using finite differences at first order ."""
    from math import pi as PI
    
    def delta(a,b):
        """Delta Kronecker"""
        if a==b:
            return 1.
        else:
            return 0.
    
    volume = parameters.get_attribute('volume')
    dE = E.value
    
    dielectric_tensor = []
    
    polarization_0 = outputs_null.get_array('electronic_dipole_cartesian_axes')[-1]
    polarizations_E = [out_finite.get_array('electronic_dipole_cartesian_axes')[-1] for out_finite in outputs_finites]
     
    for i in range(len(polarizations_E)): # takes single cartesian direction
        diff_ij = []
        for j in range(len(polarizations_E[i])):
            diff_ij.append( 4*PI*(polarizations_E[i][j]-polarization_0[j])/(volume*dE) + delta(i,j) )    
        dielectric_tensor.append(diff_ij)
        
    epsilon = ArrayData()
    epsilon.set_array('epsilon', np.array(dielectric_tensor))
    
    return epsilon

#@calcfunction                     
def compute_high_frequency_dielectric_array(parameters, outputs_null, outputs_finites, E, direction):
    """Return the high frequency dielectric tensor using finite differences at first order."""
    from math import pi as PI
    
    def delta(a,b):
        """Delta Kronecker"""
        if a==b:
            return 1.
        else:
            return 0.
        
    volume = parameters.get_attribute('volume')
    dE = E.value
    
    dielectric_array = []
    
    polarization_0 = outputs_null.get_array('electronic_dipole_cartesian_axes')[-1]
    polarization_E = outputs_finites[0].get_array('electronic_dipole_cartesian_axes')[-1] 
    
    for j in range(len(polarization_E)):
        dielectric_array.append( 4*PI*(polarization_E[j]-polarization_0[j])/(volume*dE) + delta(direction,j) )    
    
    epsilon = ArrayData()
    epsilon.set_array('epsilon', np.array(dielectric_array))
    
    return epsilon

#@calcfunction                     
def compute_effective_charge_tensors(outputs_null, outputs_finites, E):
    """
    Return the Born effective charge tensors using finite differences at first order.
    
    ::NOTE ON CHARGE:: the results are divided by the elementary electronic charge in atomic units (sqrt(2)).
    """
    from math import sqrt    
    
    dE = E.value
    
    effective_charge_tensors = []
    
    forces_atoms_0 = outputs_null.get_array('forces')[-1]
    forces_atoms_Es = [out_finite.get_array('forces')[-1] for out_finite in outputs_finites]
    
    for I in range(len(forces_atoms_0)): # running on atom index (tot=#atoms)
        single_atom_tensor = []
        # building single atomic effective charges
        for B in range(len(forces_atoms_Es)): # running on direction of electric field index (tot=3)
            single_atom_tensor.append( (forces_atoms_Es[B][I]-forces_atoms_0[I])/(sqrt(2)*dE) ) # auto running on force directions (tot=3)
        effective_charge_tensors.append(single_atom_tensor)
    
    born_charges = ArrayData()
    born_charges.set_array('born_charges', np.array(effective_charge_tensors))
    
    return born_charges

#@calcfunction                     
def compute_effective_charge_arrays(outputs_null, outputs_finites, E):
    """
    Return the Born effective charge tensors using finite differences at first order.
    
    ::NOTE ON CHARGE:: the results are divided by the elementary electronic charge in atomic units (sqrt(2)).
    """
    from math import sqrt    
    
    dE = E.value
    
    effective_charge_arrays = []
    
    forces_atoms_0 = outputs_null.get_array('forces')[-1]
    forces_atoms_E = outputs_finites[0].get_array('forces')[-1] 
    
    for I in range(len(forces_atoms_0)): # running on atom index (tot=#atoms)
        # building single atomic effective charge array
        effective_charge_arrays.append( (forces_atoms_E[I]-forces_atoms_0[I])/(sqrt(2)*dE) ) # auto running on force directions (tot=3)

    born_charges = ArrayData()
    born_charges.set_array('born_charges', np.array(effective_charge_arrays))
    
    return born_charges
