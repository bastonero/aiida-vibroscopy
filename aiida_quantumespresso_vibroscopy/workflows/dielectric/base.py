# -*- coding: utf-8 -*-
"""Base workflow for dielectric properties calculation from finite fields."""
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import WorkChain, ToContext, if_, append_, calcfunction
from aiida.plugins import CalculationFactory, WorkflowFactory

from aiida_quantumespresso_vibroscopy.utils.validation import set_tot_magnetization
from aiida_quantumespresso_vibroscopy.utils.elfield_cards_functions import generate_cards_second_order, find_directions

import numpy as np

from math import sqrt

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PwCalculation = CalculationFactory('quantumespresso.pw')
SecondOrderDerivativesWorkChain = WorkflowFactory('quantumespresso.vibroscopy.dielectric.second_order_derivatives')

# SHOULD COME FROM STRUCTURE <<BETTER>>
@calcfunction
def get_volume(parameters):
    """Take the volume from outputs.output_parameters and link it for provenance."""
    volume = parameters.attributes['volume']
    return orm.Float(volume)

def validate_positive(value, _):
    """Validate the value of the electric field."""
    if value.value < 0:
        return f'{value} specified is negative.' 
    
def validate_parent_scf(parent_scf, _):
    """Validate the `parent_scf` input. Make sure that it is created by a `PwCalculation`."""
    creator = parent_scf.creator

    if not creator:
        return f'could not determine the creator of {parent_scf}.'

    if creator.process_class is not PwCalculation:
        return f'creator of `parent_scf` {creator} is not a `PwCalculation`.'


class DielectircWorkChain(WorkChain):
    """Workchain that for a given input structure can compute the dielectric tensor at
    high frequency, the Born effective charges, the derivatives of the susceptibility (dielectric) tensor
    using finite fields in the electric enthalpy.
    """
    
    _DEFAULT_NBERRYCYC = 3
    _AVAILABLE_PROPERTIES = ('ir','raman','nac','nos')
    
    @classmethod
    def define(cls, spec):
        super().define(spec)
        # yapf: disable
        spec.input('electric_field', valid_type=orm.Float, required=False,
                   help='Electric field value in Ry atomic units. Only positive value. If not specified, '
                   'an nscf is run in order to get the best possible value under the critical field (recommended).',
                  validator = validate_positive)
        spec.input('property', valid_type=str, required=True, non_db=True,
                   help=("String for the property to calculate. Valid inputs are:\n"
                   +"\n ".join(f"{flag_name}" for flag_name in cls._AVAILABLE_PROPERTIES)),
                   validator=cls._validate_properties,
                   ),
        spec.input('parent_scf', valid_type=orm.RemoteData, validator=validate_parent_scf, required=False,
                   help='Scf parent folder from where restarting the scfs with electric fields.') # CAREFUL: THE NBND MUST BE SPECIFIED
        spec.expose_inputs(PwBaseWorkChain, namespace='scf', 
                           namespace_options={
                               'require': True,
                               'help': ('Inputs for the `PwBaseWorkChain` that, will be used to run different .')},
                           exclude=('pw.parent_folder',) )
        spec.input_namespace('central_difference', help='The inputs for the central difference scheme.')
        spec.input('central_difference.diagonal_scale', valid_type=float, default=sqrt(2), required=False, non_db=True,
                   help='Scaling factor for electric fields non parallel to cartesiaan axis.')
        spec.input('central_difference.order', valid_type=int, required=False, non_db=True,
                   help='Central difference scheme order to employ. '
                   'If not specified, an automatic choice is made upon the intensity of the electric field.')
        
        spec.outline(
            cls.setup,
            if_(cls.should_run_base_scf)(
                cls.run_base_scf,
                cls.inspect_base_scf,
            ),
            if_(cls.should_estimate_electric_field)(
                cls.run_nscf,
                cls.inspect_nscf,
                cls.estimate_electric_field,
            )
            cls.run_null_field_scf,
            cls.run_electric_field_scfs,
            cls.inspect_electric_field_scfs,
            cls.run_numerical_derivatives,
            cls.results,
        )
        
        spec.expose_outputs(SecondOrderDerivativesWorkChain)
        
        spec.exit_code(400, 'ERROR_FAILED_BASE_SCF',
            message='The initial scf work chain failed.')
        spec.exit_code(401, 'ERROR_FAILED_NSCF',
            message='The nscf work chain failed.') 
        spec.exit_code(402, 'ERROR_FAILED_ELFIELD_SCF',
            message='The electric field scf work chain failed for direction {direction}.') 
        spec.exit_code(403, 'ERROR_EFIELD_CARD_FATAL_FAIL ',
            message='One of the electric field card is abnormally all zeros or the direction finding failed.') 
        spec.exit_code(404, 'ERROR_NUMERICAL_DERIVATIVES ',
            message='The numerical derivatives calculation failed.') 
        spec.exit_code(405, 'ERROR_NON_INTEGER_TOT_MAGNETIZATION',
            message=('The scf PwBaseWorkChain sub process in iteration '
                    'returned a non integer total magnetization (threshold exceeded).'))
    
    @classmethod
    def _validate_displacements(cls, value, _):
        """Validate the ``property`` input namespace."""
        if value.lower() not in cls._AVAILABLE_PROPERTIES:
            invalid_value = value.lower()
        if invalid_value:
            return f"Got invalid or not implemented property value {invalid_value}."

    def setup(self):
        """Set up the context and the outline."""       
        
        if 'parent_scf' in self.inputs:
            self.ctx.should_run_base_scf = True
        else:
            self.ctx.should_run_base_scf = False
            
        if 'electric_field' in self.inputs:
            self.ctx.should_estimate_electric_field = False
        else:
            self.ctx.should_estimate_electric_field = True
            
        # Determine whether the system is to be treated as magnetic
        if 'init_scf' in self.inputs:
            parameters = self.inputs.init_scf.pw.parameters.get_dict()
            nspin      = parameters.get('SYSTEM', {}).get('nspin', 1)
            if  nspin != 1:
                self.report('system is treated to be magnetic because `nspin != 1` in `scf.pw.parameters` input.')
                self.ctx.is_magnetic = True
                if nspin == 2:                   
                    if parameters.get('SYSTEM', {}).get('starting_magnetization') == None and parameters.get('SYSTEM', {}).get('tot_magnetization') == None:
                        raise NameError('Missing `*_magnetization` input in `scf.pw.parameters` while `nspin == 2`.')
                else: 
                    raise NotImplementedError(f'nspin=`{nspin}` is not implemented in the code.') # are we sure???
            else:
                # self.report('system is treated to be non-magnetic because `nspin == 1` in `scf.pw.parameters` input.')
                self.ctx.is_magnetic = False

    def should_run_base_scf(self): 
        """Return whether a ground-state scf calculation needs to be run."""
        return self.ctx.should_run_base_scf
    
    def should_estimate_electric_field(self): 
        """Return whether a nscf calculation needs to be run to estimate the electric field."""
        return self.ctx.should_estimate_electric_field
    
    def is_magnetic(self):
        """Return whether the current structure is magnetic."""
        return self.ctx.is_magnetic
    
    def get_inputs(self, elfield_array):
        """Return the inputs for the electric enthalpy scf."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        parameters = inputs.pw.parameters.get_dict()
        parameters.setdefault('CONTROL', {})
        parameters.setdefault('SYSTEM', {})
        parameters.setdefault('ELECTRONS', {})
        # --- Compulsory keys for electric enthalpy     
        parameters['SYSTEM']['occupations'] = 'fixed'
        parameters['SYSTEM'].pop('degauss', None)
        parameters['SYSTEM'].pop('smearing', None)
        parameters['CONTROL']['lelfield'] = True
        parameters['ELECTRONS']['efield_cart'] = elfield_array
        # --- Field dependent settings
        if elfield_array == [0,0,0]:
            parameters['CONTROL']['nberrycyc'] = 1
            if 'parent_scf' in self.inputs:
                inputs.pw.parent_folder = self.inputs.parent_scf # NEED TO COPY JUST THE CHARGE DENSITY !!!
            else:
                inputs.pw.parent_folder = self.ctx.base_scf.outputs.remote_folder # NEED TO COPY JUST THE CHARGE DENSITY !!!
        else:
            nberrycyc = parameters['CONTROL'].pop('nberrycyc', self._DEFAULT_NBERRYCYC) 
            parameters['CONTROL']['nberrycyc'] = nberrycyc
            inputs.pw.parent_folder = self.ctx.null_electric_field.outputs.remote_folder # NEED TO COPY JUST THE CHARGE DENSITY !!!
        # --- Restarting from file        
        parameters['ELECTRONS']['startingpot'] = 'file'
        # --- Magnetic ground state
        if self.is_magnetic():
            parameters['SYSTEM'].pop('starting_magnetization', None)
            parameters['SYSTEM']['nbnd'] = self.ctx.initial_scf.outputs.output_parameters.get_dict()['number_of_bands']
            if set_tot_magnetization( inputs.pw.parameters, self.ctx.initial_scf.outputs.output_parameters.get_dict()['total_magnetization'] ):
                return self.exit_codes.ERROR_NON_INTEGER_TOT_MAGNETIZATION
        # --- Return
        inputs.pw.parameters = orm.Dict(dict=parameters)
        return inputs

    def run_base_scf(self):
        """Run initial scf for ground-state ."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        parameters = inputs.pw.parameters.get_dict
        for key in ('nberrycyc, lelfield', 'efield_cart'):
            parameters.pop(key, None)
        inputs.pw.parameters = orm.Dict(dict=parameters)
        
        inputs.metadata.call_link_label = 'base_scf'
        if inputs.clean_workdir.value:
            inputs.clean_workdir = orm.Bool(False) # the folder is needed for next calculations
        
        node = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launched base scf PwBaseWorkChain<{node.pk}>')
        return ToContext(initial_scf=node)

    def inspect_base_scf(self):
        """Verify that the scf PwBaseWorkChain finished successfully."""
        workchain = self.ctx.base_scf

        if not workchain.is_finished_ok:
            self.report(f'base scf failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_FAILED_BASE_SCF   
    
    def run_nscf(self):
        """Run nscf."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        parameters = inputs.pw.parameters.get_dict
        parameters['CONTROL']['calculation'] = 'nscf'
        for key in ('nberrycyc, lelfield', 'efield_cart'):
            parameters.pop(key, None)
        if 'parent_folder' not in self.inputs:
            parameters['SYSTEM']['nbnd'] = self.ctx.base_scf.outputs.output_parameters.get_dict()['number_of_bands']+10
        inputs.pw.parameters = orm.Dict(dict=parameters)
        
        kpoints = inputs.kpoints.clone()
        mesh = ( 2*np.array(kpoints.get_attribute('mesh')) ).tolist()
        kpoints.kpoints.set_kpoints_mesh(mesh)
        inputs.kpoints = kpoints
        
        if 'parent_folder' in self.inputs: # JUST LINK THE FOLDER OR COPY OVER ONLY THE DENSITY
            inputs.pw.parent_folder = self.inputs.parent_folder
        else:
            inputs.pw.parent_folder = self.ctx.base_scf
            
        inputs.metadata.call_link_label = 'nscf'
        
        node = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launched base scf PwBaseWorkChain<{node.pk}>')
        return ToContext(nscf=node)

    def inspect_nscf(self):
        """Verify that the nscf PwBaseWorkChain finished successfully."""
        workchain = self.ctx.nscf

        if not workchain.is_finished_ok:
            self.report(f'nscf failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_FAILED_NSCF   
    
    def estimate_electric_field(self):
        """Estimate the electric field to be lower than the critical one. E ~ Egap/(e*a*Nk)"""
        # call a `calcfunction` which takes in input structure, kpoints, output_bands (?)
        # calculate_electric_field --> orm.Float
    
    def run_null_field_scf(self):
        """Run electric enthalpy scf with zero electric field."""      
        inputs = self.get_inputs(elfield_array=[0.,0.,0.]) 
        
        key = 'null_electric_field'
        inputs.metadata.call_link_label = key
        
        node = self.submit(PwBaseWorkChain, **inputs)
        self.to_context(**{key: node})
        
        self.report(f'launched PwBaseWorkChain<{node.pk}> with null electric field') 
    
    def run_elfield_scf(self):
        """Running scf with different electric fields for central difference."""
        for card in self.ctx.elfield_card: 
            direction = find_directions(card)            
            inputs = self.get_inputs(elfield_array=card)  
            
            # Here I label:
            # * 0,1,2 for first order derivatives: l --> {l}j ; e.g. 0 does 00, 01, 02
            # * 0,1,2,3,4,5 for second order derivatives: l <--> ij --> {ij}k ; 
            #   precisely 0 > {00}k; 1 > {11}k; 2 > {22}k; 3 > {01}k; 4 > {02}k; 5 --> {12}k | k=0,1,2
            key =  f'electric_field_{direction}' 
            inputs.metadata.call_link_label = key
            
            node = self.submit(PwBaseWorkChain, **inputs)
            self.to_context(**{key: node})
            self.report(f'launched PwBaseWorkChain<{node.pk}> with electric field in direction {direction}')  
        
    def inspect_elfield_scf(self):
        """Inspect all previous pw workchains before computing final results."""       
        # 1. Inspecting scf with null electric field
        workchain = self.ctx.null_electric_field

        if not workchain.is_finished_ok:
            self.report(f'null electric field scf failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_FAILED_ELFIELD_SCF.format(direction='null')    
        
        # 2. Inspecting scf with different electric fields 
        for key, workchain in self.ctx.items():
            if key.startswith('electric_field_'):
                if not workchain.is_finished_ok:
                    self.report(f'electric field scf failed with exit status {workchain.exit_status}')
                    return self.exit_codes.ERROR_FAILED_ELFIELD_SCF.format(direction=key[-1])

    def run_results(self):
        """Compute outputs from previous calculations."""
        data = {label: wc.outputs.output_trajectory for label, wc in self.ctx.items() if (label.startswith('null') or label[-2] in ['0','1','2']) }
        elfield = self.inputs['elfield']
        volume = get_volume(self.ctx.null_electric_field.outputs.output_parameters)
        key = 'numerical_derivatives'
        
        inputs = {'data':data,
                  'elfield':elfield,
                  'volume':volume,
                  'metadata':{'call_link_label':key}
                  }
        
        node = self.submit(SecondOrderDerivativesWorkChain, **inputs)
        self.to_context(**{key: node})
        self.report(f'launched SecondOrderDerivativesWorkChain<{node.pk}> for computing numerical derivatives.')   

    def results(self):
        """Show outputss."""
        # Inspecting numerical derivative work chain
        workchain = self.ctx.numerical_derivatives
        if not workchain.is_finished_ok:
            self.report(f'computation of numerical derivatives failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_NUMERICAL_DERIVATIVES  
        
        if self.should_run_init_scf():
            inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='init_scf'))
            if inputs.clean_workdir.value:
                self.report(f'clean_workdir was True for initial scf, cleaning as final step. (not implemented yet)')
                # to be done...
        
        self.out_many(self.exposed_outputs(self.ctx.numerical_derivatives, SecondOrderDerivativesWorkChain))
