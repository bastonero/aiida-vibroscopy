# -*- coding: utf-8 -*-
"""Turn-key solution to automatically compute the self-consistent Hubbard parameters for a given structure."""
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import WorkChain, ToContext, while_, if_, append_
from aiida.orm.nodes.data.array.bands import find_bandgap
from aiida.plugins import CalculationFactory, WorkflowFactory

from aiida_quantumespresso_finite_differences.calculations.compute_long_range_constants import *
from aiida_quantumespresso_finite_differences.calculations.wrappers import *

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PwCalculation = CalculationFactory('quantumespresso.pw')

def validate_elfield(elfield, _):
    """Validate the value of the electric field."""
    if elfield.value <= 0:
        return 'Electric field value specified is negative.' 

def validate_nberrycyc(nberrycyc, _):
    """Validate the value of nberrycyc."""
    if not nberrycyc.value > 0:
        return 'nberrycyc value must be at least 1.' 

def validate_direction(selected_elfield, _):
    """Validate the validity of the direction requested."""
    direction = selected_elfield.value
    if not direction in [0,1,2]:
        return f'Direction {direction} is not a valid input. Choose among 0 (x), 1 (y), 2 (z).' 
    
def validate_parent_scf(parent_scf, _):
    """
    Validate the `parent_scf` input.
    Make sure that it is created by a `PwCalculation`.
    """
    creator = parent_scf.creator

    if not creator:
        return f'could not determine the creator of {parent_scf}'

    if creator.process_class is not PwCalculation:
        return f'creator of `parent_scf` {creator} is not a `PwCalculation`'

    try:
        parameters = creator.inputs.parameters.get_dict()
    except AttributeError:
        return f'could not retrieve the input parameters node from the parent calculation {creator}'


class FiniteElectricFieldsWorkChain(WorkChain):
    """
    Workchain that for a given input structure will compute the dielectric tensor at
    high frequency and the Born effective charges using finite differences.
    """
    
    @classmethod
    def define(cls, spec):
        super().define(spec)
        # yapf: disable
        spec.input('elfield', valid_type=orm.Float, default=lambda: orm.Float(0.001), 
                   help='Electric field value to be used in the computation of the quantities. Only positive value.',
                  validator = validate_elfield)
        spec.input('nberrycyc', valid_type=orm.Int, default=lambda: orm.Int(3), 
                   help='Number of iterations for init_scferging the wavefunctions in the electric field Hamiltonian, '\
                   'for each external iteration on the charge density (the same as the one on the pw.x doc).',
                  validator = validate_nberrycyc)
        spec.input('selected_elfield', valid_type=orm.Int, required=False, 
                   help='Single direction of electric field calculation. Intended for test '\
                   'and convergence test purposes. Valid values are: 0 (x), 1 (y), 2 (z).',
                  validator = validate_direction)
        spec.input('parent_scf', valid_type=orm.RemoteData, validator=validate_parent_scf, required=False)
        spec.expose_inputs(PwBaseWorkChain, namespace='init_scf', 
                           namespace_options={'required': False, 'populate_defaults': False,
                                              'help': 'Inputs for the `PwBaseWorkChain` that, '\
                                              'when defined, should run an scf calculation to find, '\
                                              'a well converged ground-state to be used as a base '\
                                              'for the scfs with finite electric field.'} )
        spec.expose_inputs(PwBaseWorkChain, namespace='elfield_scf', 
                           namespace_options={'help': 'Inputs for the `PwBaseWorkChain` that, '\
                                              'will be used to run different .'},
                           exclude=('pw.parent_folder',) )
        spec.outline(
            cls.setup,
            cls.validate_inputs,
            if_(cls.should_run_init_scf)(
                cls.run_init_scf,
                cls.inspect_init_scf,
            ),
            cls.run_elfield_scf,
            cls.inspect_elfield_scf,
            cls.run_results,
        )
        spec.output('output_tensors', valid_type=orm.ArrayData, required=False,
            help='Contains high frequency dielectric tensor and Born effective charge tensors, computed at first order in electric field.')
        spec.output('output_arrays', valid_type=orm.ArrayData, required=False,
            help='Contains partial (i.e. `arrays`) high frequency dielectric tensor and Born effective charge tensors, computed at first order in the selected direction of the electric field.')
        spec.exit_code(401, 'ERROR_FAILED_INIT_SCF',
            message='The initial scf work chain failed.') 
        spec.exit_code(402, 'ERROR_FAILED_ELFIELD_SCF',
            message='The electric field scf work chain failed for direction {direction}.') 
        spec.exit_code(403, 'ERROR_EFIELD_CARD_FATAL_FAIL ',
            message='One of the electric field card is abnormally all zeros or the direction finding failed.') 
        
    def setup(self):
        """Set up the context."""
        self.ctx.effective_charges = []
        self.ctx.high_freq_dielectric = []
       
        # constructing the elfield_cards for the different scf calculations
        self.ctx.elfield_card = []
        for i in range(3):
            vector = []
            for j in range(3):
                if j==i:
                    vector.append(self.inputs.elfield.value)
                else:
                    vector.append(0.0)
            self.ctx.elfield_card.append(vector)
        
        if 'selected_elfield' in self.inputs:
            #setting the elfield card array to one direction only
            self.ctx.only_one_elfield = True
            self.ctx.elfield_card = [self.ctx.elfield_card[self.inputs.selected_elfield.value]]
        else:
            self.ctx.only_one_elfield = False
        
        if 'init_scf' in self.inputs:
            self.ctx.should_run_init_scf = True
        else:
            self.ctx.should_run_init_scf = False
            
        if ('init_scf' in self.inputs) or ('parent_scf' in self.inputs):      
            self.ctx.has_parent_scf = True
        else:
            self.ctx.has_parent_scf = False
            
        
            
    def validate_inputs(self):
        """Validate inputs."""
        if ('init_scf' in self.inputs) and ('parent_scf' in self.inputs):
            self.ctx.should_run_init_scf = False # priority to less computational cost
            self.report('both ´init_scf´ and ´parent_scf´ are specified. Disregarding ´init_scf´ and continuing...')
   
    def should_run_init_scf(self): 
        """Return whether a ground-state scf calculation needs to be run, which is true if `init_scf` is specified in inputs."""
        return self.ctx.should_run_init_scf

    def only_one_elfield(self):
        """Return whether a single direction has to be run."""
        return self.ctx.only_one_elfield
    
    def get_inputs(self, elfield_array):
        """Return the inputs for one of the subprocesses whose inputs are exposed in the given namespace.

        :elfield_array: elfield_card array.
        :return:  dictionary with inputs.
        """
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='elfield_scf'))
        parameters = inputs.pw.parameters.get_dict()
#       inputs.pw.parameters.setdefault('CONTROL', {})
#       inputs.pw.parameters.setdefault('SYSTEM', {})
#       inputs.pw.parameters.setdefault('ELECTRONS', {})
        # --- Compulsory keys for electric enthalpy        
        parameters['CONTROL']['lelfield'] = True
        parameters['ELECTRONS']['efield_cart'] = elfield_array
        # --- Field dependent settings
        if elfield_array == [0,0,0]:
            parameters['CONTROL']['nberrycyc'] = 1
        else:
            parameters['CONTROL']['nberrycyc'] = self.inputs.nberrycyc.value 
        # --- Parent scf        
        if self.ctx.has_parent_scf:
            parameters['ELECTRONS']['startingpot'] = 'file'
            if 'parent_scf' in self.inputs:
                inputs.pw.parent_folder = self.inputs.parent_scf
            else:
                inputs.pw.parent_folder = self.ctx.initial_scf.outputs.remote_folder
        # --- Return
        inputs.pw.parameters = orm.Dict(dict=parameters)
        return inputs
    
    def find_direction(self, vector):
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

    def run_init_scf(self):
        """Run initial scf for ground-state ."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='init_scf'))
        inputs.metadata.call_link_label = 'initial_scf'

        node = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launched initial scf PwBaseWorkChain<{node.pk}>')
        return ToContext(initial_scf=node)

    def inspect_init_scf(self):
        """Verify that the scf PwBaseWorkChain finished successfully."""
        workchain = self.ctx.initial_scf

        if not workchain.is_finished_ok:
            self.report(f'initial scf failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_FAILED_INIT_SCF    
        
    def run_elfield_scf(self):
        """Run pw scf for computing tensors."""
        
        inputs = self.get_inputs(elfield_array=[0.,0.,0.]) # 1. Running scf with null electric field
        key = 'null_electric_field'
        inputs.metadata.call_link_label = key
        
        node = self.submit(PwBaseWorkChain, **inputs)
        self.to_context(**{key: node})
        self.report(f'launched PwBaseWorkChain<{node.pk}> with null electric field')    
             
        for card in self.ctx.elfield_card: # 2. Running scf with different electric fields  
            direction, found = self.find_direction(card)
            #if not found:
            #    self.report(f'unexpected empty electric card')
            #    return self.exit_codes.ERROR_EFIELD_CARD_FATAL_FAIL           
            inputs = self.get_inputs(elfield_array=card)  
            #inputs.pw.parameters = orm.Dict(dict=inputs.pw.parameters)
            
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
        # 1. Output trajectory of null electric field
        outputs_null_ef = self.ctx.null_electric_field.outputs.output_trajectory
        outputs_param = self.ctx.null_electric_field.outputs.output_parameters
                           
        # 2. Output trajectory of finite electric fields
        outputs_finite_efs = orm.List(list=[ wc.outputs.output_trajectory for key, wc in self.ctx.items() if key.startswith('electric_field_') ] )
        
        # 3. Compute, run and output of the results 
        if not self.ctx.only_one_elfield:
            epsilon = compute_high_frequency_dielectric_tensor(outputs_param, outputs_null_ef, outputs_finite_efs, self.inputs.elfield)
            born_charges = compute_effective_charge_tensors(outputs_null_ef, outputs_finite_efs, self.inputs.elfield)
            computed_tensors = wrap_tensors(epsilon, born_charges)
            #computed_tensors = run_tensors(outputs_null_ef, outputs_finite_efs, self.inputs.elfield)
            self.out('output_tensors', computed_tensors)
        else:
            computed_arrays = run_arrays(outputs_null_ef, outputs_finite_efs, 
                                        self.inputs.elfield, self.inputs.selected_elfield)
            self.out('output_arrays', computed_arrays) 
