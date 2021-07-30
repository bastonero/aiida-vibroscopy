# -*- coding: utf-8 -*-
"""Turn-key solution to automatically compute the self-consistent Hubbard parameters for a given structure."""
from aiida import orm
from aiida.engine import WorkChain, if_, calcfunction
from aiida.plugins import WorkflowFactory
from math import pi as PI
from math import sqrt  
import numpy as np

def delta(a,b):
    """Delta Kronecker"""
    if a==b:
        return 1.
    else:
        return 0.

@calcfunction                     
def compute_tensors(vol, elfield, **data):
    """
    Return the high frequency dielectric tensor and Born effective charges
    using finite differences at first order .

    ::NOTE ON CHARGE:: the results are divided by the elementary electronic charge in atomic units (sqrt(2)).
    """
    volume = vol.value
    dE = elfield.value
    dielectric_tensor = []
    data_fields = []
    
    for label, trajectory in data.items():
        if label.startswith('null'):
            data_null = trajectory
        else:
            data_fields.append(trajectory)
    
    polarization_0 = data_null.get_array('electronic_dipole_cartesian_axes')[-1]
    polarizations_E = [data_field.get_array('electronic_dipole_cartesian_axes')[-1] for data_field in data_fields]
     
    for i in range(len(polarizations_E)): # takes single cartesian direction
        diff_ij = []
        for j in range(len(polarizations_E[i])):
            diff_ij.append( 4*PI*(polarizations_E[i][j]-polarization_0[j])/(volume*dE) + delta(i,j) )    
        dielectric_tensor.append(diff_ij)
        
    tensors = orm.ArrayData()
    tensors.set_array('epsilon', np.array(dielectric_tensor))
    
    effective_charge_tensors = []
    
    forces_atoms_0 = data_null.get_array('forces')[-1]
    forces_atoms_Es = [data_field.get_array('forces')[-1] for data_field in data_fields]
    
    for I in range(len(forces_atoms_0)): # running on atom index
        single_atom_tensor = []
        # building single atomic effective charges
        for B in range(len(forces_atoms_Es)): # running on direction of electric field index (tot=3)
            single_atom_tensor.append( (forces_atoms_Es[B][I]-forces_atoms_0[I])/(sqrt(2)*dE) ) # auto running on force directions (tot=3)
        effective_charge_tensors.append(single_atom_tensor)
    
    tensors.set_array('born_charges', np.array(effective_charge_tensors))
    
    return tensors


@calcfunction
def compute_arrays(vol, elfield, **data):
    """
    Return the high frequency dielectric tensor and Born effective charges
    using finite differences at first order .

    ::NOTE ON CHARGE:: the results are divided by the elementary electronic charge in atomic units (sqrt(2)).
    """
    volume = vol.value
    dE = elfield.value
    dielectric_array = []
    data_fields = []
    
    for label, trajectory in data.items():
        if label.startswith('null'):
            data_null = trajectory
        else:
            data_fields.append(trajectory)
            direction = int(label[-1])
    
    polarization_0 = data_null.get_array('electronic_dipole_cartesian_axes')[-1]
    polarization_E = data_fields[0].get_array('electronic_dipole_cartesian_axes')[-1] 
    
    for j in range(len(polarization_E)):
        dielectric_array.append( 4*PI*(polarization_E[j]-polarization_0[j])/(volume*dE) + delta(direction,j) )    
    
    arrays = orm.ArrayData()
    arrays.set_array('epsilon', np.array(dielectric_array))
    
    effective_charge_arrays = []
    
    forces_atoms_0 = data_null.get_array('forces')[-1]
    forces_atoms_E = data_fields[0].get_array('forces')[-1] 
    
    for I in range(len(forces_atoms_0)): # running on atom index (tot=#atoms)
        # building single atomic effective charge array
        effective_charge_arrays.append( (forces_atoms_E[I]-forces_atoms_0[I])/(sqrt(2)*dE) ) # auto running on force directions (tot=3)

    arrays.set_array('born_charges', np.array(effective_charge_arrays))
    
    return arrays


def validate_data(data, _):
    """
    Validate the `data` namespace inputs.
    """
    length = len(data) # must be 2 or 4
    control_null_namespace = 0 # must be 1
    
    for label, trajectory in data.items():
        # first, control if `null`
        if label.startswith('null'):
            control_null_namespace+=1
        elif not label[-1] in ['0','1','2']:
            return f'`{label[-1]}` is an invalid label ending for label `{label}`'
    
    if not control_null_namespace==1:
        return f'invalid number of `null` namespaces: expected 1, given {control_null_namespace}'
    
    if not length in [2,4]:
        return f'invalid total number of inputs: expected 2 or 4, given {length}'


class SecondOrderDerivativesWorkChain(WorkChain):
    """
    Workchain that computes the second order derivatives via finite differences,
    providing force and polarization vectors as TrajectoryData or ArrayData
    from previous workflow(s). 
    
    ::NOTE:: input namespaces are standardized - look into the validation 
             fucntions for more info.
    """
    
    @classmethod
    def define(cls, spec):
        super().define(spec)
        # yapf: disable
        spec.input_namespace('data', valid_type=orm.TrajectoryData, validator=validate_data)
        spec.input('elfield', valid_type=orm.Float)
        spec.input('volume', valid_type=orm.Float)
        spec.outline(
            cls.setup,
            cls.run_results,
        )
        spec.output('output_tensors', valid_type=orm.ArrayData, required=False,
            help='Contains high frequency dielectric tensor and Born effective charge tensors, computed at first order in electric field.')
        spec.output('output_arrays', valid_type=orm.ArrayData, required=False,
            help='Contains partial (i.e. `arrays`) high frequency dielectric tensor and Born effective charge tensors, computed at first order in the selected direction of the electric field.')

    def setup(self):
        """Set up the context."""
        if len(self.inputs['data'])==2:
            self.ctx.only_one_elfield = True
        else:
            self.ctx.only_one_elfield = False

    def run_results(self):
        """Wrap up results from previous calculations."""
        if not self.ctx.only_one_elfield:
            out = compute_tensors(self.inputs['volume'], self.inputs['elfield'], **self.inputs['data'])
            self.out('output_tensors', out)
        else:
            out = compute_arrays(self.inputs['volume'], self.inputs['elfield'], **self.inputs['data'])
            self.out('output_arrays', out) 
