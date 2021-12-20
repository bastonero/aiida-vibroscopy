# -*- coding: utf-8 -*-
"""
Turn-key solution to automatically compute the second order derivatives of the 
total energy in respect to electric field and atom displacements.
"""
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import WorkChain, ToContext, if_, append_, calcfunction
from aiida.plugins import CalculationFactory, WorkflowFactory

from aiida_quantumespresso_ir_raman.utils.validation import set_tot_magnetization
from aiida_quantumespresso_ir_raman.utils.elfield_cards_functions import generate_cards_first_order, find_direction

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PwCalculation = CalculationFactory('quantumespresso.pw')
FirstOrderDerivativesWorkChain = WorkflowFactory('quantumespresso.fd.first_order_derivatives')

@calcfunction
def get_volume(parameters):
    '''Take the volume from outputs.output_parameters and link it for provenance.'''
    volume = parameters.attributes['volume']
    return orm.Float(volume)

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
            cls.results,
        )
        spec.expose_outputs(FirstOrderDerivativesWorkChain)
        spec.exit_code(401, 'ERROR_FAILED_INIT_SCF',
            message='The initial scf work chain failed.') 
        spec.exit_code(402, 'ERROR_FAILED_ELFIELD_SCF',
            message='The electric field scf work chain failed for direction {direction}.') 
        spec.exit_code(403, 'ERROR_EFIELD_CARD_FATAL_FAIL ',
            message='One of the electric field card is abnormally all zeros or the direction finding failed.') 
        spec.exit_code(404, 'ERROR_NUMERICAL_DERIVATIVES ',
            message='The numerical derivatives calculation failed.') 
        spec.exit_code(405, 'ERROR_NON_INTEGER_TOT_MAGNETIZATION',
            message=('The scf PwBaseWorkChain sub process in iteration '
                    'returned a non integer total magnetization (threshold exceeded).'))
        
    def setup(self):
        """Set up the context."""       
        # constructing the elfield_cards for the different scf calculations
        self.ctx.elfield_card = generate_cards_first_order(self.inputs.elfield.value)
        
        if 'selected_elfield' in self.inputs:
            # setting the elfield card array to one direction only
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
            
        # Determine whether the system is to be treated as magnetic
        if 'init_scf' in self.inputs:
            parameters = self.inputs.init_scf.pw.parameters.get_dict()
            nspin      = parameters.get('SYSTEM', {}).get('nspin', 1)
            if  nspin != 1:
                self.report('system is treated to be magnetic because `nspin != 1` in `init_scf.pw.parameters` input.')
                self.ctx.is_magnetic = True
                if nspin == 2:                   
                    if parameters.get('SYSTEM', {}).get('starting_magnetization') == None and parameters.get('SYSTEM', {}).get('tot_magnetization') == None:
                        raise NameError('Missing `starting_magnetization` input in `init_scf.pw.parameters` while `nspin == 2`.')
                else: 
                    raise NotImplementedError(f'nspin=`{nspin}` is not implemented in the code.') # are we sure???
            else:
                # self.report('system is treated to be non-magnetic because `nspin == 1` in `scf.pw.parameters` input.')
                self.ctx.is_magnetic = False
            
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
    
    def is_magnetic(self):
        """Return whether the current structure is magnetic."""
        return self.ctx.is_magnetic
    
    def get_inputs(self, elfield_array):
        """Return the inputs for one of the subprocesses whose inputs are exposed in the given namespace.

        :elfield_array: elfield_card array.
        :return:  dictionary with inputs.
        """
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='elfield_scf'))
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
        else:
            parameters['CONTROL']['nberrycyc'] = self.inputs.nberrycyc.value 
        # --- Parent scf        
        if self.ctx.has_parent_scf:
            parameters['ELECTRONS']['startingpot'] = 'file'
            if 'parent_scf' in self.inputs:
                inputs.pw.parent_folder = self.inputs.parent_scf
            else:
                inputs.pw.parent_folder = self.ctx.initial_scf.outputs.remote_folder
        # --- Magnetic ground state
        if 'init_scf' in self.inputs and self.is_magnetic():
            parameters['SYSTEM'].pop('starting_magnetization', None)
            parameters['SYSTEM']['nbnd'] = self.ctx.initial_scf.outputs.output_parameters.get_dict()['number_of_bands']
            if set_tot_magnetization( inputs.pw.parameters,  self.ctx.initial_scf.outputs.output_parameters.get_dict()['total_magnetization'] ):
                return self.exit_codes.ERROR_NON_INTEGER_TOT_MAGNETIZATION
        # --- Return
        inputs.pw.parameters = orm.Dict(dict=parameters)
        return inputs

    def run_init_scf(self):
        """Run initial scf for ground-state ."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='init_scf'))
        inputs.metadata.call_link_label = 'initial_scf'
        if inputs.clean_workdir.value:
            inputs.clean_workdir = orm.Bool(False) # the folder is needed for next calculations

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
        # 1. Running scf with null electric field
        inputs = self.get_inputs(elfield_array=[0.,0.,0.]) 
        key = 'null_electric_field'
        inputs.metadata.call_link_label = key
        
        node = self.submit(PwBaseWorkChain, **inputs)
        self.to_context(**{key: node})
        self.report(f'launched PwBaseWorkChain<{node.pk}> with null electric field')    
             
        # 2. Running scf with different electric fields  
        for card in self.ctx.elfield_card: 
            direction, found = find_direction(card)            
            inputs = self.get_inputs(elfield_array=card)  
            
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
        data = {label: wc.outputs.output_trajectory for label, wc in self.ctx.items() if (label.startswith('null') or label[-1] in ['0','1','2']) }
        elfield = self.inputs['elfield']
        volume = get_volume(self.ctx.null_electric_field.outputs.output_parameters)
        key = 'numerical_derivatives'
        
        inputs = {'data':data,
                  'elfield':elfield,
                  'volume':volume,
                  'metadata':{'call_link_label':key}
                  }
        
        node = self.submit(FirstOrderDerivativesWorkChain, **inputs)
        self.to_context(**{key: node})
        self.report(f'launched FirstOrderDerivativesWorkChain<{node.pk}> for computing numerical derivatives.')   

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
        
        self.out_many(self.exposed_outputs(self.ctx.numerical_derivatives, FirstOrderDerivativesWorkChain))
