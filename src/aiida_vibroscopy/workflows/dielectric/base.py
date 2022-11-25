# -*- coding: utf-8 -*-
"""Base workflow for dielectric properties calculation from finite fields."""
from math import sqrt
import time

from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import WorkChain, append_, calcfunction, if_, while_
from aiida.orm.nodes.data.array.bands import find_bandgap
from aiida.plugins import CalculationFactory, WorkflowFactory
from aiida_phonopy.data import PreProcessData
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
import numpy as np

from aiida_vibroscopy.calculations.symmetry import get_irreducible_numbers_and_signs
from aiida_vibroscopy.common import UNITS_FACTORS
from aiida_vibroscopy.utils.elfield_cards_functions import get_vector_from_number
from aiida_vibroscopy.utils.validation import validate_positive, validate_tot_magnetization

from .numerical_derivatives import NumericalDerivativesWorkChain

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PwCalculation = CalculationFactory('quantumespresso.pw')


@calcfunction
def compute_electric_field(
    parameters: orm.Dict, bands: orm.BandsData, structure: orm.StructureData, scale_factor: orm.Float
):
    'Return the estimated electric field as Egap/(e*a*Nk) in Ry a.u. .'
    _, band_gap = find_bandgap(bands)
    scaling = scale_factor.value

    kmesh = np.array(parameters.base.attributes.get('monkhorst_pack_grid'))
    cell = np.array(structure.cell)

    denominator = np.fabs(np.dot(cell.T, kmesh)).max() * UNITS_FACTORS.efield_au_to_si

    return orm.Float(scaling * band_gap / denominator)


@calcfunction
def get_electric_field_step(electric_field: orm.Float, accuracy: orm.Int):
    """Return the central difference displacement step."""
    return orm.Float(2 * electric_field.value / accuracy.value)


@calcfunction
def get_accuracy_from_field_intensity(norm: orm.Float):
    """Return the central difference accuracy.

    :param norm: intensity of electric field in Ry a.u.
    """
    return orm.Int(4) if norm.value > 5.e-4 else orm.Int(2)


def validate_accuracy(value, _):
    """Validate the value of the numerical accuracy. Only positive integer even numbers, 0 excluded."""
    if value.value <= 0 or value.value % 2 != 0:
        return 'specified accuracy is negative or not even.'


def validate_parent_scf(parent_scf, _):
    """Validate the `parent_scf` input. Make sure that it is created by a `PwCalculation`."""
    creator = parent_scf.creator

    if not creator:
        return f'could not determine the creator of {parent_scf}.'

    if creator.process_class is not PwCalculation:
        return f'creator of `parent_scf` {creator} is not a `PwCalculation`.'


def validate_inputs(inputs, _):
    """Validate the entire inputs namespace."""
    if 'electric_field' in inputs and 'electric_field_scale' in inputs:
        return 'cannot specify both `electric_field*` inputs, only one is accepted'

    if not ('electric_field' in inputs or 'electric_field_scale' in inputs):
        return 'one between `electric_field` and `electric_field_scale` must be specified'


class DielectricWorkChain(WorkChain, ProtocolMixin):
    """Workchain that for a given input structure can compute the dielectric tensor at
    high frequency, the Born effective charges, the derivatives of the susceptibility (dielectric) tensor
    using finite fields in the electric enthalpy.
    """

    _DEFAULT_NBERRYCYC = 1
    _AVAILABLE_PROPERTIES = (
        'ir', 'born-charges', 'dielectric', 'nac', 'bec', 'raman', 'susceptibility-derivative',
        'non-linear-susceptibility'
    )

    @classmethod
    def define(cls, spec):
        super().define(spec)
        # yapf: disable
        spec.input(
            'electric_field',
            valid_type=orm.Float,
            required=False,
            validator=validate_positive,
            help=(
                'Electric field value in Ry atomic units, referred as '
                'the maximum step size used in the numerical differenciation. '
                'Only positive value. If not specified, an nscf is run in '
                'order to get the best possible value under the critical field (recommended).'
            )
        )
        spec.input(
            'electric_field_scale',
            valid_type=orm.Float,
            required=False,
            validator=validate_positive,
            help=(
                'Scaling factor for evaluating the electric field from its critical'
                'value (i.e. it will multiply on the numerator the critical electric value).'
            )
        )
        spec.input(
            'property',
            valid_type=str,
            required=True,
            non_db=True,
            validator=cls._validate_properties,
            help=(
                'Valid inputs are: \n \n * '.join(f'{flag_name}' for flag_name in cls._AVAILABLE_PROPERTIES)
            )
        )
        spec.input(
            'parent_scf',
            valid_type=orm.RemoteData,
            validator=validate_parent_scf,
            required=False,
            help='Scf parent folder from where restarting the scfs with electric fields.'
        )
        spec.expose_inputs(PwBaseWorkChain, namespace='scf',
            namespace_options={
                'required': True,
                'help': ('Inputs for the `PwBaseWorkChain` that will be used to run the electric enthalpy scfs.')
            },
            exclude=('clean_workdir', 'pw.parent_folder',)
        )
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(True),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.'
        )
        spec.input_namespace('central_difference', help='The inputs for the central difference scheme.')
        spec.input('central_difference.diagonal_scale',
            valid_type=orm.Float, default=lambda: orm.Float(1/sqrt(2)), validator=validate_positive,
            help='Scaling factor for electric fields non parallel to cartesiaan axis (i.e. E --> scale*E).'
        )
        spec.input('central_difference.accuracy', valid_type=orm.Int, required=False, validator=validate_accuracy,
            help=('Central difference scheme accuracy to employ (i.e. number of points for derivative evaluation). '
                  'This must be an EVEN positive integer number. If not specified, an automatic '
                  'choice is made upon the intensity of the critical electric field.')
        )
        spec.input_namespace(
            'options',
            help='Options for how to run the workflow.',
        )
        spec.input(
            'options.sleep_submission_time', valid_type=(int, float), non_db=True, default=3.0,
            help='Time in seconds to wait before submitting subsequent displaced structure scf calculations.',
        )
        spec.input(
            'options.symprec', valid_type=orm.Float, default=lambda:orm.Float(1e-5),
            help='Symmetry tolerance for space group analysis on the input structure.',
        )
        spec.input(
            'options.distinguish_kinds', valid_type=orm.Bool, default=lambda:orm.Bool(False),
            help='Whether or not to distinguish atom with same species but different names with symmetries.',
        )
        spec.input(
            'options.is_symmetry', valid_type=orm.Bool, default=lambda:orm.Bool(True),
            help='Whether using or not the space group symmetries.',
        )
        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            cls.run_base_scf,
            cls.inspect_base_scf,
            if_(cls.should_estimate_electric_field)(
                cls.run_nscf,
                cls.inspect_nscf,
                cls.estimate_electric_field,
            ),
            cls.run_null_field_scf,
            cls.inspect_null_field_scf,
            while_(cls.should_run_electric_field_scfs)(
                cls.run_electric_field_scfs,
                cls.inspect_electric_field_scfs,
            ),
            cls.run_numerical_derivatives,
            cls.results,
        )

        spec.expose_outputs(NumericalDerivativesWorkChain)
        spec.output('estimated_electric_field', valid_type = orm.Float, required=False)
        spec.output('electric_field_step', valid_type = orm.Float, required=False)
        spec.output('accuracy_order', valid_type = orm.Int, required=False)
        spec.output_namespace(
            'fields_data',
            help='Namespace for passing TrajectoryData containing forces and polarization.',
        )
        spec.output('fields_data.null_field', valid_type=orm.TrajectoryData, required=True)
        spec.output_namespace('fields_data.field_index_0', valid_type=orm.TrajectoryData, required=False)
        spec.output_namespace('fields_data.field_index_1', valid_type=orm.TrajectoryData, required=False)
        spec.output_namespace('fields_data.field_index_2', valid_type=orm.TrajectoryData, required=False)
        spec.output_namespace('fields_data.field_index_3', valid_type=orm.TrajectoryData, required=False)
        spec.output_namespace('fields_data.field_index_4', valid_type=orm.TrajectoryData, required=False)
        spec.output_namespace('fields_data.field_index_5', valid_type=orm.TrajectoryData, required=False)

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
    def _validate_properties(cls, value, _):
        """Validate the ``property`` input namespace."""
        if value.lower() not in cls._AVAILABLE_PROPERTIES:
            invalid_value = value.lower()
        else:
            invalid_value=None

        if invalid_value is not None:
            return f'Got invalid or not implemented property value {invalid_value}.'

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from ..protocols import dielectric as dielectric_protocols
        return files(dielectric_protocols) / 'base.yaml'

    @classmethod
    def get_builder_from_protocol(cls, code, structure, protocol=None, overrides=None, options=None, **kwargs):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        from aiida.orm import to_aiida_type

        inputs = cls.get_protocol_inputs(protocol, overrides)

        args = (code, structure, protocol)
        scf = PwBaseWorkChain.get_builder_from_protocol(
            *args, overrides=inputs.get('scf', None), options=options, **kwargs
        )

        scf.pop('clean_workdir', None)
        scf['pw'].pop('parent_folder', None)

        builder = cls.get_builder()

        if 'options' in inputs:
            builder_options = {}

            if 'sleep_submission_time' in inputs['options']:
                builder_options.update({'sleep_submission_time': inputs['options']['sleep_submission_time']})

            non_default_options = ['symprec', 'distinguish_kinds', 'is_symmetry']
            for name in non_default_options:
                if name in inputs['options']:
                    builder_options.update({name:to_aiida_type(inputs['options'][name])})

            builder.options = builder_options

        name = 'electric_field' if 'electric_field' in inputs else 'electric_field_scale'
        builder[name] = to_aiida_type(inputs[name])

        non_default_difference = ['diagonal_scale', 'accuracy']
        if 'central_difference' in inputs:
            central_difference = {}
            for name in non_default_difference:
                if name in inputs['central_difference']:
                    central_difference.update({name:to_aiida_type(inputs['central_difference'][name])})
            builder['central_difference'] = central_difference

        builder.scf = scf
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder.property = inputs['property']

        return builder

    def setup(self):
        """Set up the context and the outline."""
        self.ctx.clean_workdir = self.inputs.clean_workdir.value

        preprocess_data = PreProcessData(
            structure=self.inputs.scf.pw.structure,
            symprec=self.inputs.options.symprec.value,
            is_symmetry=self.inputs.options.is_symmetry.value,
            distinguish_kinds=self.inputs.options.distinguish_kinds.value
        )

        self.ctx.should_estimate_electric_field = True

        if 'electric_field' in self.inputs:
            self.ctx.should_estimate_electric_field = False
            self.ctx.electric_field = self.inputs.electric_field

        if self.inputs.property in ('ir','nac','born-charges','bec','dielectric'):
            self.ctx.numbers, self.ctx.signs = get_irreducible_numbers_and_signs(preprocess_data, 3)
        elif self.inputs.property in ('raman','susceptibility-derivative','non-linear-susceptibility'):
            self.ctx.numbers, self.ctx.signs = get_irreducible_numbers_and_signs(preprocess_data, 6)

        else: # it is impossible to get here due to input validation
            raise NotImplementedError(f'calculation of {self.inputs.property} not available')

        # Determine whether the system is to be treated as magnetic
        parameters = self.inputs.scf.pw.parameters.get_dict()
        nspin = parameters.get('SYSTEM', {}).get('nspin', 1)
        if  nspin != 1:
            self.report('system is treated to be magnetic because `nspin != 1` in `scf.pw.parameters` input.')
            self.ctx.is_magnetic = True
            if nspin == 2:
                starting_magnetization = parameters.get('SYSTEM', {}).get('starting_magnetization')
                tot_magnetization = parameters.get('SYSTEM', {}).get('tot_magnetization')
                if  starting_magnetization is None and  tot_magnetization is None:
                    raise NameError('Missing `*_magnetization` input in `scf.pw.parameters` while `nspin == 2`.')
            else:
                raise NotImplementedError(f'nspin=`{nspin}` is not implemented in the code.') # are we sure???
        else:
            # self.report('system is treated to be non-magnetic because `nspin == 1` in `scf.pw.parameters` input.')
            self.ctx.is_magnetic = False

    def should_estimate_electric_field(self):
        """Return whether a nscf calculation needs to be run to estimate the electric field."""
        return self.ctx.should_estimate_electric_field

    def should_run_electric_field_scfs(self):
        """Return whether to run the next electric field scfs."""
        return not self.ctx.steps == self.ctx.iteration

    def is_magnetic(self):
        """Return whether the current structure is magnetic."""
        return self.ctx.is_magnetic

    def get_inputs(self, electric_field_vector):
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
        parameters['CONTROL']['tprnfor'] = True # sanity check
        parameters['ELECTRONS']['efield_cart'] = electric_field_vector
        # --- Field dependent settings
        if electric_field_vector == [0,0,0]:
            parameters['CONTROL']['nberrycyc'] = 1
        else:
            nberrycyc = parameters['CONTROL'].pop('nberrycyc', self._DEFAULT_NBERRYCYC)
            parameters['CONTROL']['nberrycyc'] = nberrycyc
            parameters['ELECTRONS']['startingwfc'] = 'file' # MAYBE OPTIONAL FROM INPUTS?
        # --- Restarting from file
        parameters['ELECTRONS']['startingpot'] = 'file'
        # --- Magnetic ground state
        if self.is_magnetic():
            base_out = self.ctx.base_scf.outputs
            parameters['SYSTEM'].pop('starting_magnetization', None)
            parameters['SYSTEM']['occupations'] = 'fixed'
            parameters['SYSTEM'].pop('smearing', None)
            parameters['SYSTEM'].pop('degauss', None)
            parameters['SYSTEM']['nbnd'] = base_out.output_parameters.base.attributes.get('number_of_bands')
            tot_magnetization = base_out.output_parameters.base.attributes.get('total_magnetization')
            parameters['SYSTEM']['tot_magnetization'] = tot_magnetization
            if validate_tot_magnetization(tot_magnetization):
                return self.exit_codes.ERROR_NON_INTEGER_TOT_MAGNETIZATION
        # --- Return
        inputs.pw.parameters = orm.Dict(parameters)
        if self.ctx.clean_workdir:
            inputs.clean_workdir = orm.Bool(False)

        return inputs

    def run_base_scf(self):
        """Run initial scf for ground-state ."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        parameters = inputs.pw.parameters.get_dict()

        for key in ('nberrycyc', 'lelfield', 'efield_cart'):
            parameters['CONTROL'].pop(key, None)
            parameters['ELECTRONS'].pop(key, None)

        if 'parent_scf' in self.inputs:
            inputs.pw.parent_folder = self.inputs.parent_scf
            parameters['ELECTRONS']['startingpot'] = 'file'

        inputs.pw.parameters = orm.Dict(parameters)

        key = 'base_scf'
        inputs.metadata.call_link_label = key

        inputs.clean_workdir = orm.Bool(False) # the folder is needed for next calculations

        node = self.submit(PwBaseWorkChain, **inputs)
        self.to_context(**{key: node})
        self.report(f'launched base scf PwBaseWorkChain<{node.pk}>')

    def inspect_base_scf(self):
        """Verify that the scf PwBaseWorkChain finished successfully."""
        workchain = self.ctx.base_scf

        if not workchain.is_finished_ok:
            self.report(f'base scf failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_FAILED_BASE_SCF

    def run_nscf(self):
        """Run nscf."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        outputs = self.ctx.base_scf.outputs

        parameters = inputs.pw.parameters.get_dict()
        parameters['CONTROL']['calculation'] = 'nscf'

        nbnd = outputs.output_parameters.base.attributes.get('number_of_bands')+10
        parameters['SYSTEM']['nbnd'] = nbnd

        for key in ('nberrycyc', 'lelfield', 'efield_cart'):
            parameters['CONTROL'].pop(key, None)
            parameters['ELECTRONS'].pop(key, None)

        inputs.pw.parameters = orm.Dict(parameters)
        inputs.pw.parent_folder = outputs.remote_folder

        key = 'nscf'
        inputs.metadata.call_link_label = key

        node = self.submit(PwBaseWorkChain, **inputs)
        self.to_context(**{key: node})
        self.report(f'launched base scf PwBaseWorkChain<{node.pk}>')

    def inspect_nscf(self):
        """Verify that the nscf PwBaseWorkChain finished successfully."""
        workchain = self.ctx.nscf

        if not workchain.is_finished_ok:
            self.report(f'nscf failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_FAILED_NSCF

    def estimate_electric_field(self):
        """Estimate the electric field to be lower than the critical one. E ~ Egap/(e*a*Nk)"""
        nscf_outputs = self.ctx.nscf.outputs
        value_node = compute_electric_field(
            parameters=nscf_outputs.output_parameters,
            bands=nscf_outputs.output_band,
            structure=self.inputs.scf.pw.structure,
            scale_factor=self.inputs.electric_field_scale,
        )
        self.ctx.electric_field = value_node
        self.out('estimated_electric_field', self.ctx.electric_field)

    def run_null_field_scf(self):
        """Run electric enthalpy scf with zero electric field."""
        # First we quickly put in the ctx the value of the numerical accuracy
        if 'accuracy' in self.inputs.central_difference:
            self.ctx.accuracy = self.inputs.central_difference.accuracy
        else:
            self.ctx.accuracy = get_accuracy_from_field_intensity(self.ctx.electric_field)

        self.ctx.steps = int(self.ctx.accuracy.value/2)
        self.ctx.iteration = 0

        inputs = self.get_inputs(electric_field_vector=[0.,0.,0.])
        if 'parent_scf' in self.inputs:
            inputs.pw.parent_folder = self.inputs.parent_scf # NEED TO COPY JUST THE CHARGE DENSITY !!!
        else:
            inputs.pw.parent_folder = self.ctx.base_scf.outputs.remote_folder # NEED TO COPY JUST THE CHARGE DENSITY !!!

        key = 'null_field'
        inputs.metadata.call_link_label = key

        node = self.submit(PwBaseWorkChain, **inputs)
        self.to_context(**{key: node})
        self.report(f'launched PwBaseWorkChain<{node.pk}> with null electric field')

    def inspect_null_field_scf(self):
        """Verify that the scf PwBaseWorkChain with null electric field finished successfully."""
        workchain = self.ctx.null_field

        if not workchain.is_finished_ok:
            self.report(f'electric field scf failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_FAILED_ELFIELD_SCF.format(direction='`null`')

    def run_electric_field_scfs(self):
        """Running scf with different electric fields for central difference."""
        # Here we can already truncate `numbers`` from the very beginning using symmetry analysis
        signs = [1.0,-1.0]
        for number, bool_signs in zip(self.ctx.numbers, self.ctx.signs):
            norm = self.ctx.electric_field.value*(self.ctx.iteration +1)/self.ctx.steps
            if number in (3,4,5):
                norm = norm*self.inputs.central_difference.diagonal_scale

            # Here we can put a zip with True or False, but it depends on numbers, so even before check
            for sign, bool_sign in zip(signs, bool_signs):
                if bool_sign:
                    electric_field_vector = get_vector_from_number(number=number, value=sign*norm)
                    inputs = self.get_inputs(electric_field_vector=electric_field_vector)

                    # Here I label:
                    # * 0,1,2 for first order derivatives: l --> {l}j ; e.g. 0 does 00, 01, 02
                    # * 0,1,2,3,4,5 for second order derivatives: l <--> ij --> {ij}k ;
                    #   precisely 0 > {00}k; 1 > {11}k; 2 > {22}k; 3 > {12}k; 4 > {02}k; 5 --> {01}k | k=0,1,2
                    key =  f'field_index_{number}' # adding the iteration as well?
                    inputs.metadata.call_link_label = key

                    if self.ctx.iteration == 0:
                        inputs.pw.parent_folder = self.ctx.null_field.outputs.remote_folder
                    else:
                        index_folder = -2 if bool_signs[1] else -1
                        inputs.pw.parent_folder = self.ctx[key][index_folder].outputs.remote_folder

                    # We fill in the ctx arrays in order to have at the end something like:
                    # field_index_0 = [+1,-1,+2,-2] (e.g. for accuracy = 4, no symmetry)
                    # field_index_0 = [+1, +2] (e.g. for accuracy = 4, with e.g. inversion symmetry)
                    # where the numbers refers to the multiplication factor to the base electric field E/accuracy,
                    # since in central finite differences we always have a number of evaluations equal to the accuracy,
                    # half for positive and half for negative evaluation of the function.
                    # Symmetries can reduce this.
                    node = self.submit(PwBaseWorkChain, **inputs)
                    self.to_context(**{key: append_(node)})
                    self.report(f'launched PwBaseWorkChain<{node.pk}> with electric field index {number}')
                    time.sleep(self.inputs.options.sleep_submission_time)

        self.ctx.iteration = self.ctx.iteration +1

    def inspect_electric_field_scfs(self):
        """Inspect all previous pw workchains with electric fields."""
        self.ctx.data = {'null_field': self.ctx.null_field.outputs.output_trajectory}

        for label, workchains in self.ctx.items():
            if label.startswith('field_index_'):
                field_data = {str(i):wc.outputs.output_trajectory for i, wc in enumerate(workchains)}
                self.ctx.data.update({label:field_data})
        self.out_many({'fields_data':self.ctx.data})

        for key, workchains in self.ctx.items():
            if key.startswith('field_index_'):
                for workchain in workchains:
                    if not workchain.is_finished_ok:
                        self.report(f'electric field scf failed with exit status {workchain.exit_status}')
                        return self.exit_codes.ERROR_FAILED_ELFIELD_SCF.format(direction=key[-1])

    def run_numerical_derivatives(self):
        """Compute numerical derivatives from previous calculations."""
        electric_field_step = get_electric_field_step(self.ctx.electric_field, self.ctx.accuracy)
        self.out('electric_field_step', electric_field_step)
        key = 'numerical_derivatives'

        inputs = {
            'data':self.ctx.data,
            'electric_field_step':electric_field_step,
            'structure':self.inputs.scf.pw.structure,
            'accuracy_order':self.ctx.accuracy,
            'diagonal_scale':self.inputs.central_difference.diagonal_scale,
            'options':{
                'symprec':self.inputs.options.symprec,
                'is_symmetry':self.inputs.options.is_symmetry,
                'distinguish_kinds':self.inputs.options.distinguish_kinds,
            },
            'metadata':{'call_link_label':key}
        }

        node = self.submit(NumericalDerivativesWorkChain, **inputs)
        self.to_context(**{key: node})
        self.report(f'launched NumericalDerivativesWorkChain<{node.pk}> for computing numerical derivatives.')

    def results(self):
        """Expose outputss."""
        # Inspecting numerical derivative work chain
        workchain = self.ctx.numerical_derivatives
        if not workchain.is_finished_ok:
            self.report(f'computation of numerical derivatives failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_NUMERICAL_DERIVATIVES

        self.out_many(self.exposed_outputs(self.ctx.numerical_derivatives, NumericalDerivativesWorkChain))

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report(f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}")
