# -*- coding: utf-8 -*-
"""Calculations and workflows that call and wrap the output results."""
from aiida_quantumespresso_ir_raman.calculations.compute_long_range_constants import *
from aiida.engine import calcfunction, workfunction

#@calcfunction                     
def wrap_tensors(epsilon, born_charges):
    """Gives as a results a unique ArrayData containing the computed tensors."""
    from aiida.orm import ArrayData
    
    results = ArrayData()
    
    results.set_array('epsilon', epsilon.get_array('epsilon'))
    results.set_array('born_charges', born_charges.get_array('born_charges'))
    
    return results
    
#@workfunction
def run_tensors(parameters, outputs_null, outputs_finites, E):
    """Work function to compute the output_tensors."""
    epsilon = compute_high_frequency_dielectric_tensor(parameters, outputs_null, outputs_finites, E)
    born_charges = compute_effective_charge_tensors(outputs_null, outputs_finites, E)
    wrapped_results = wrap_tensors(epsilon, born_charges)
    
    return wrapped_results

#@workfunction
def run_arrays(parameters, outputs_null, outputs_finites, E, direction):
    """Work function to compute the output_arrays."""
    epsilon = compute_high_frequency_dielectric_array(parameters, outputs_null, outputs_finites, E, direction)
    born_charges = compute_effective_charge_arrays(outputs_null, outputs_finites, E)
    wrapped_results = wrap_tensors(epsilon, born_charges)
    
    return wrapped_results
