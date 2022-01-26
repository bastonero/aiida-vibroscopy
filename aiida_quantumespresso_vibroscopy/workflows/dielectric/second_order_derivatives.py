# -*- coding: utf-8 -*-
"""Code agnostic workflow which wraps up calculations. """
from aiida import orm
from aiida.engine import WorkChain, if_, calcfunction
from aiida.plugins import WorkflowFactory
from math import pi as PI
from math import sqrt  
import numpy as np
from qe_tools import CONSTANTS

def delta(a,b):
    """Delta Kronecker"""
    if a==b:
        return 1.
    else:
        return 0.

    
def symmetrization(tensor):
    """Symmetrizes a 3x3x3 tensor."""
    sym_tensor = np.zeros((3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                sym_tensor[i][j][k] = (1./6.)*(tensor[i][j][k]+tensor[i][k][j]+tensor[j][i][k]+
                                               tensor[j][k][i]+tensor[k][i][j]+tensor[k][j][i] )
    return sym_tensor

                          
@calcfunction                     
def compute_phonon_derivative_susceptibility(vol, elfield, **data):
    """
    Return the third derivative of the total energy in respect to
    one phonon and two electric fields, times volume factor.
    
    """
    volume = vol.value # in ang**3
    dE = elfield.value
    dE2 = dE*dE

    forces_Es_diag = {}
    forces_Es_off  = {}
    
    for label, trajectory in data.items():
        if label.startswith('null'):
            forces_0 = trajectory.get_array('forces')[-1]
        elif label[-3]=='_':
            forces_Es_diag.update({label[-2:]:trajectory.get_array('forces')[-1]})
        else:
            forces_Es_off.update({label[-4:]:trajectory.get_array('forces')[-1]})           
    
    num_atoms = len(forces_0)
    tensors = np.zeros((num_atoms,3,3,3))
    tensors_efficient = np.zeros((num_atoms,3,3,3))
    
    # Building Chi[I,k;i,j]
    for I in range(num_atoms): # running on atom index
        for k in range(3): # building single atomic tensors, one 3x3 for each spatial direction k
            for i in range(3):
                for j in range(3):
                    if i==j:
                        fp = forces_Es_diag[str(i)+'p'][I][k]
                        fm = forces_Es_diag[str(i)+'m'][I][k]
                        f0 = forces_0[I][k]
                        tensors[I][k][i][j] = (fp -2.0*f0 +fm)/dE2
                        tensors_efficient[I][k][i][j] = (fp -2.0*f0 +fm)/dE2
                    else:
                        if i<j:
                            a=i
                            b=j
                        else:
                            a=j
                            b=i
                        fpp = forces_Es_off[str(a)+'p'+str(b)+'p'][I][k]
                        fmm = forces_Es_off[str(a)+'m'+str(b)+'m'][I][k]
                        fpm = forces_Es_off[str(a)+'p'+str(b)+'m'][I][k]
                        fmp = forces_Es_off[str(a)+'m'+str(b)+'p'][I][k]
                        fpi = forces_Es_diag[str(i)+'p'][I][k]
                        fmi = forces_Es_diag[str(i)+'m'][I][k]
                        fpj = forces_Es_diag[str(j)+'p'][I][k]
                        fmj = forces_Es_diag[str(j)+'m'][I][k]
                        f0 = forces_0[I][k]
                        tensors[I][k][i][j] = 0.25*(fpp -fpm -fmp +fmm)/dE2
                        tensors_efficient[I][k][i][j] = 0.5*(fpp + fmm -fpi -fmi +2.0*f0 -fpj -fmj)/dE2
    
    #for I in range(num_atoms):
    #!!!
    # CAN DO IF LOOP OVER K, i.e. symmetrize just the second derivative over el. fields
    #!!!
    #    tensors[I] = symmetrization(tensors[I])
    #    tensors_efficient[I] = symmetrization(tensors_efficient[I])
    
    tensors_data = orm.ArrayData()
    tensors_data.set_array('central', tensors/volume)
    tensors_data.set_array('efficiency', tensors_efficient/volume)
    
    return tensors_data

@calcfunction                     
def compute_nonlinear_optical_susceptibility(vol, elfield, **data):
    """
    Return the third derivative of the total energy in respect to three electric fields.
    """
    ang_to_bohr = 1./CONSTANTS.bohr_to_ang
    volume = vol.value*(ang_to_bohr**3) # in bohr**3
    dE = elfield.value
    dE2 = dE*dE

    polarization_Es_diag = {}
    polarization_Es_off  = {}
    
    for label, trajectory in data.items():
        if label.startswith('null'):
             polarization_0 = trajectory.get_array('electronic_dipole_cartesian_axes')[-1]
        elif label[-3]=='_':
             polarization_Es_diag.update({label[-2:]:trajectory.get_array('electronic_dipole_cartesian_axes')[-1]})
        else:
             polarization_Es_off.update({label[-4:]:trajectory.get_array('electronic_dipole_cartesian_axes')[-1]})           
    
    tensors = np.zeros((3,3,3))
    tensors_efficient = np.zeros((3,3,3))
    
    # Building Chi[k,i,j]
    for k in range(3): # building tensors, one 3x3 for each spatial direction k
        for i in range(3):
            for j in range(3):
                if i==j:
                    fp = polarization_Es_diag[str(i)+'p'][k]
                    fm = polarization_Es_diag[str(i)+'m'][k]
                    f0 = polarization_0[k]
                    tensors[k][i][j] = (fp -2.0*f0 +fm)/dE2
                    tensors_efficient[k][i][j] = (fp -2.0*f0 +fm)/dE2
                else:
                    if i<j:
                        a=i
                        b=j
                    else:
                        a=j
                        b=i
                    fpp = polarization_Es_off[str(a)+'p'+str(b)+'p'][k]
                    fmm = polarization_Es_off[str(a)+'m'+str(b)+'m'][k]
                    fpm = polarization_Es_off[str(a)+'p'+str(b)+'m'][k]
                    fmp = polarization_Es_off[str(a)+'m'+str(b)+'p'][k]
                    fpi = polarization_Es_diag[str(i)+'p'][k]
                    fmi = polarization_Es_diag[str(i)+'m'][k]
                    fpj = polarization_Es_diag[str(j)+'p'][k]
                    fmj = polarization_Es_diag[str(j)+'m'][k]
                    f0 = polarization_0[k]
                    tensors[k][i][j] = 0.25*(fpp -fpm -fmp +fmm)/dE2
                    tensors_efficient[k][i][j] = 0.5*(fpp -fpi -fmi +2.0*f0 -fpj -fmj +fmm)/dE2
                        
    tensors = symmetrization(tensors)
    tensors_efficient = symmetrization(tensors_efficient)
                        
    tensors_data = orm.ArrayData()
    tensors_data.set_array('central', 0.5*tensors/volume)
    tensors_data.set_array('efficiency', 0.5*tensors_efficient/volume)
    
    return tensors_data

@calcfunction                     
def compute_nac_constants(vol, elfield, **data):
    """
    Return epsilon and born charges to second order in finite electric fields (central difference).
    """
    volume = vol.value/(CONSTANTS.bohr_to_ang**3)
    dE = elfield.value
    au_units = CONSTANTS.bohr_to_ang/CONSTANTS.ry_to_ev
    
    # --- Epsilon
    polarization_Es_diag = {}
    polarization_Es_off  = {}
    
    for label, trajectory in data.items():
        if label.startswith('null'):
             polarization_0 = trajectory.get_array('electronic_dipole_cartesian_axes')[-1]
        elif label[-3]=='_':
             polarization_Es_diag.update({label[-2:]:trajectory.get_array('electronic_dipole_cartesian_axes')[-1]})
        else:
             polarization_Es_off.update({label[-4:]:trajectory.get_array('electronic_dipole_cartesian_axes')[-1]})           
    
    epsilon = np.zeros((3,3))
    
    # Building eps[i,j]
    for i in range(3):
        for j in range(3):
            fp = polarization_Es_diag[str(i)+'p'][j]
            fm = polarization_Es_diag[str(i)+'m'][j]
            epsilon[i][j] = 4.0*PI*(fp -fm)/(2.0*dE*volume) +delta(i,j)
                        
    tensors_data = orm.ArrayData()
    tensors_data.set_array('epsilon', epsilon)
    
    # --- Born charges
    forces_Es_diag = {}
    forces_Es_off  = {}
    
    for label, trajectory in data.items():
        if label.startswith('null'):
            forces_0 = au_units*trajectory.get_array('forces')[-1]
        elif label[-3]=='_':
            forces_Es_diag.update({label[-2:]:au_units*trajectory.get_array('forces')[-1]})
        else:
            forces_Es_off.update({label[-4:]:au_units*trajectory.get_array('forces')[-1]})
    
    num_atoms = len(forces_0)
    born_charges = np.zeros((num_atoms,3,3))
    
    # Building born_charges[I;i,j]
    for I in range(num_atoms):
        for i in range(3):
            for j in range(3):
                fp = forces_Es_diag[str(i)+'p'][I][j]
                fm = forces_Es_diag[str(i)+'m'][I][j]
                born_charges[I][i][j] = (fp -fm)/(2.0*dE*sqrt(2))
            
    tensors_data.set_array('born_charges', born_charges)
    
    return tensors_data


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
        elif not label[-1] in ['m','p']:
            return f'`{label[-1]}` is an invalid label ending for label `{label}`'
    
    if not control_null_namespace==1:
        return f'invalid number of `null` namespaces: expected 1, given {control_null_namespace}'
    
    #if not length in [2,4]:
    #    return f'invalid total number of inputs: expected 2 or 4, given {length}'


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
            cls.run_results,
        )
        spec.output('nac_constants', valid_type=orm.ArrayData,
            help=('Contains high frequency dielectric tensor and Born effective charge tensors, '
                  'computed at second order in electric field.') )
        spec.output('phonon_derivative_susceptibility', valid_type=orm.ArrayData)
        spec.output('nonlinear_optical_susceptibility', valid_type=orm.ArrayData)


    def run_results(self):
        """Wrap up results from previous calculations."""
        out_long_range = compute_nac_constants(self.inputs['volume'], self.inputs['elfield'], **self.inputs['data'])
        out_first = compute_phonon_derivative_susceptibility(self.inputs['volume'], self.inputs['elfield'], **self.inputs['data'])
        out_second = compute_nonlinear_optical_susceptibility(self.inputs['volume'], self.inputs['elfield'], **self.inputs['data'])
        self.out('nac_constants', out_long_range) 
        self.out('phonon_derivative_susceptibility', out_first) 
        self.out('nonlinear_optical_susceptibility', out_second) 
