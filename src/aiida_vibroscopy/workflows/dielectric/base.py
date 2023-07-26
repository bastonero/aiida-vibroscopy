# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Base workflow for dielectric properties calculation from finite fields."""
from __future__ import annotations

import time

from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import WorkChain, append_, calcfunction, if_, while_
from aiida.orm.nodes.data.array.bands import find_bandgap
from aiida.plugins import CalculationFactory, WorkflowFactory
from aiida_phonopy.data import PreProcessData
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
import numpy as np

from aiida_vibroscopy.calculations.symmetry import get_irreducible_numbers_and_signs
from aiida_vibroscopy.common import UNITS
from aiida_vibroscopy.utils.elfield_cards_functions import get_vector_from_number
from aiida_vibroscopy.utils.validation import validate_positive, validate_tot_magnetization

from .numerical_derivatives import NumericalDerivativesWorkChain

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PwCalculation = CalculationFactory('quantumespresso.pw')


@calcfunction
def compute_critical_electric_field(
    parameters: orm.Dict,
    bands: orm.BandsData,
    structure: orm.StructureData,
) -> orm.Float:
    """Return the estimated electric field as Egap/(e*a*Nk) in Ry a.u. ."""
    _, band_gap = find_bandgap(bands, number_electrons=parameters['number_of_electrons'])

    kmesh = np.array(parameters.base.attributes.get('monkhorst_pack_grid'))
    cell = np.array(structure.cell)

    denominator = np.fabs(np.dot(cell.T, kmesh)).max() * UNITS.efield_au_to_si

    return orm.Float(band_gap / denominator)


@calcfunction
def add_zero_polarization(trajectory: orm.TrajectoryData) -> orm.TrajectoryData:
    """Add a null `electronic_dipole_cartesian_axes` to a converged SCF TrajectoryData.

    :return: a clone of the input `TrajectoryData` with a null electric dipole
    """
    new_trajectory = trajectory.clone()
    new_trajectory.set_array('electronic_dipole_cartesian_axes', np.array([[0., 0., 0.]]))

    return new_trajectory


@calcfunction
def get_electric_field_step(critical_electric_field: orm.Float, accuracy: orm.Int) -> orm.Float:
    """Return the central difference displacement step."""
    norm = critical_electric_field.value
    norm = 0.001 if norm > 1.e-3 else norm
    return orm.Float(2 * norm / accuracy.value)


@calcfunction
def get_accuracy_from_critical_field(norm: orm.Float) -> orm.Int:
    """Return the central difference accuracy.

    :param norm: intensity of critical electric field in Ry a.u.
    :return: even Int in aiida type.
    """
    return orm.Int(4) if norm.value > 1.e-4 else orm.Int(2)


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
    if 'electric_field_step' in inputs['central_difference'] and 'accuracy' not in inputs['central_difference']:
        return (
            'cannot evaluate numerical accuracy when `electric_field_step` '
            'is specified but `accuracy` is not in `central_difference`'
        )

    if 'kpoints_parallel_distance' in inputs and 'kpoints_distance' not in inputs['scf']:
        return '`kpoints_parallel_distance` works only when specifying `scf.kpoints_distance`'

    if 'settings' in inputs['scf']['pw']:
        settings = inputs['scf']['pw']['settings'].get_dict()
        cmdline = settings.get('cmdline', [])

        for key in ['-nk', '-npools']:
            if key in cmdline:
                index = cmdline.index(key)
                if int(cmdline[index + 1]) != 1:
                    return 'pool parallelization for electric field is not implemented'


class DielectricWorkChain(WorkChain, ProtocolMixin):  # pylint: disable=too-many-public-methods
    """Workchain computing different second and third order tensors.

    It computes the high frequency dielectric tensor, the Born effective charges,
    the non-linear optical susceptibility and Raman tensors
    using homogeneous small electric fields via the electric enthalpy functional.
    """

    _DEFAULT_NBERRYCYC = 1
    _AVAILABLE_PROPERTIES = (
        'ir', 'born-charges', 'dielectric', 'nac', 'bec', 'raman', 'susceptibility-derivative',
        'non-linear-susceptibility'
    )

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        # yapf: disable
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
            exclude=('clean_workdir', 'pw.parent_folder')
        )
        spec.input('kpoints_parallel_distance', valid_type=orm.Float, required=False,
            help=(
                'Distance of the k-points in reciprocal space along the '
                'parallel direction of each applied electric field.'
            )
        )
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(True),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.'
        )
        spec.input_namespace(
            'central_difference',
            help='The inputs for the central difference scheme.'
        )
        spec.input(
            'central_difference.electric_field_step', valid_type=orm.Float, required=False,
            help=(
                'Electric field step in Ry atomic units used in the numerical differenciation. '
                'Only positive values. If not specified, an NSCF is run to evaluate the critical '
                'electric field; an electric field step is then extracted to secure a stable SCF.'
            ),
            validator=validate_positive,
        )
        spec.input(
            'central_difference.diagonal_scale', valid_type=orm.Float, default=lambda: orm.Float(1/np.sqrt(2)),
            help='Scaling factor for electric fields non parallel to cartesiaan axis (i.e. E --> scale*E).',
            validator=validate_positive,
        )
        spec.input(
            'central_difference.accuracy', valid_type=orm.Int, required=False,
            help=('Central difference scheme accuracy to employ (i.e. number of points for derivative evaluation). '
                  'This must be an EVEN positive integer number. If not specified, an automatic '
                  'choice is made upon the intensity of the critical electric field.'),
            validator=validate_accuracy,
        )
        spec.input_namespace(
            'settings',
            help='Options for how to run the workflow.',
        )
        spec.input(
            'settings.sleep_submission_time', valid_type=(int, float), non_db=True, default=3.0,
            help='Time in seconds to wait before submitting subsequent displaced structure scf calculations.',
        )
        spec.input_namespace(
            'symmetry',
            help='Namespace for symmetry related inputs.',
        )
        spec.input(
            'symmetry.symprec', valid_type=orm.Float, default=lambda:orm.Float(1e-5),
            help='Symmetry tolerance for space group analysis on the input structure.',
        )
        spec.input(
            'symmetry.distinguish_kinds', valid_type=orm.Bool, default=lambda:orm.Bool(False),
            help='Whether or not to distinguish atom with same species but different names with symmetries.',
        )
        spec.input(
            'symmetry.is_symmetry', valid_type=orm.Bool, default=lambda:orm.Bool(True),
            help='Whether using or not the space group symmetries.',
        )
        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            cls.set_reference_kpoints,
            cls.run_base_scf,
            cls.inspect_base_scf,
            if_(cls.should_estimate_electric_field)(
                cls.run_nscf,
                cls.inspect_nscf,
                cls.estimate_critical_electric_field,
            ),
            cls.set_step_and_accuracy,
            cls.run_null_field_scfs,
            cls.inspect_null_field_scfs,
            while_(cls.should_run_electric_field_scfs)(
                cls.run_electric_field_scfs,
                cls.inspect_electric_field_scfs,
            ),
            cls.remove_reference_forces,
            cls.run_numerical_derivatives,
            cls.results,
        )

        spec.expose_outputs(NumericalDerivativesWorkChain)
        spec.output('critical_electric_field', valid_type = orm.Float, required=False)
        spec.output('electric_field_step', valid_type = orm.Float, required=False)
        spec.output('accuracy_order', valid_type = orm.Int, required=False)
        spec.output_namespace(
            'fields_data',
            help='Namespace for passing TrajectoryData containing forces and polarization.',
        )
        # spec.output('fields_data.null_field', valid_type=orm.TrajectoryData, required=True)
        for i in range(6):
            spec.output_namespace(f'fields_data.field_index_{i}', valid_type=orm.TrajectoryData, required=False)

        spec.exit_code(400, 'ERROR_FAILED_BASE_SCF',
            message='The initial scf work chain failed.')
        spec.exit_code(401, 'ERROR_FAILED_NSCF',
            message='The nscf work chain failed.')
        spec.exit_code(402, 'ERROR_FAILED_ELFIELD_SCF',
            message='The electric field scf work chain failed for direction {direction}.')
        spec.exit_code(403, 'ERROR_NUMERICAL_DERIVATIVES',
            message='The numerical derivatives calculation failed.')
        spec.exit_code(404, 'ERROR_NON_INTEGER_TOT_MAGNETIZATION',
            message=('The scf PwBaseWorkChain sub process in iteration '
                    'returned a non integer total magnetization (threshold exceeded).'))
        # yapf: enable

    @classmethod
    def _validate_properties(cls, value, _):
        """Validate the ``property`` input namespace."""
        if value.lower() not in cls._AVAILABLE_PROPERTIES:
            invalid_value = value.lower()
        else:
            invalid_value = None

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

        name = 'kpoints_parallel_distance'
        if name in inputs:
            builder[name] = orm.Float(inputs[name])

        non_default_difference = ['diagonal_scale', 'accuracy', 'electric_field_step']
        if 'central_difference' in inputs:
            central_difference = {}
            for name in non_default_difference:
                if name in inputs['central_difference']:
                    value = to_aiida_type(inputs['central_difference'][name])
                    central_difference.update({name: value})
            builder['central_difference'] = central_difference

        builder.scf = scf
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder.property = inputs['property']
        builder['symmetry']['symprec'] = orm.Float(inputs['symmetry']['symprec'])
        builder['symmetry']['distinguish_kinds'] = orm.Bool(inputs['symmetry']['distinguish_kinds'])
        builder['symmetry']['is_symmetry'] = orm.Bool(inputs['symmetry']['is_symmetry'])
        builder['settings']['sleep_submission_time'] = inputs['settings']['sleep_submission_time']

        return builder

    def setup(self):
        """Set up the context and the outline."""
        self.ctx.clean_workdir = self.inputs.clean_workdir.value

        preprocess_data = PreProcessData(
            structure=self.inputs.scf.pw.structure,
            symprec=self.inputs.symmetry.symprec.value,
            is_symmetry=self.inputs.symmetry.is_symmetry.value,
            distinguish_kinds=self.inputs.symmetry.distinguish_kinds.value
        )

        self.ctx.should_estimate_electric_field = True
        self.ctx.is_parallel_distance = 'kpoints_parallel_distance' in self.inputs

        if 'electric_field_step' in self.inputs.central_difference:
            self.ctx.should_estimate_electric_field = False
            self.ctx.electric_field_step = self.inputs.central_difference.electric_field_step

        if self.inputs.property in ('ir', 'nac', 'born-charges', 'bec', 'dielectric'):
            self.ctx.numbers, self.ctx.signs = get_irreducible_numbers_and_signs(preprocess_data, 3)
        elif self.inputs.property in ('raman', 'susceptibility-derivative', 'non-linear-susceptibility'):
            self.ctx.numbers, self.ctx.signs = get_irreducible_numbers_and_signs(preprocess_data, 6)

        else:  # it is impossible to get here due to input validation
            raise NotImplementedError(f'calculation of {self.inputs.property} not available')

        # Determine whether the system is to be treated as magnetic
        parameters = self.inputs.scf.pw.parameters.get_dict()
        nspin = parameters.get('SYSTEM', {}).get('nspin', 1)
        if nspin != 1:
            self.report('system is treated to be magnetic because `nspin != 1` in `scf.pw.parameters` input.')
            self.ctx.is_magnetic = True
            if nspin == 2:
                starting_magnetization = parameters.get('SYSTEM', {}).get('starting_magnetization', None)
                tot_magnetization = parameters.get('SYSTEM', {}).get('tot_magnetization', None)
                if starting_magnetization is None and tot_magnetization is None:
                    raise NameError('Missing `*_magnetization` input in `scf.pw.parameters` while `nspin == 2`.')
            else:
                raise NotImplementedError(f'nspin=`{nspin}` is not implemented in the code.')  # are we sure???
        else:
            # self.report('system is treated to be non-magnetic because `nspin == 1` in `scf.pw.parameters` input.')
            self.ctx.is_magnetic = False

    def set_reference_kpoints(self):
        """Set the Context variables for the kpoints for the sub WorkChains.

        It allows for calling only once the `create_kpoints_from_distance` calcfunction.
        """
        from aiida_vibroscopy.calculations.create_directional_kpoints import create_directional_kpoints

        try:
            kpoints = self.inputs.scf.kpoints
        except AttributeError:
            if self.ctx.is_parallel_distance:
                distance = self.inputs.kpoints_parallel_distance
            else:
                distance = self.inputs.scf.kpoints_distance
            inputs = {
                'structure': self.inputs.scf.pw.structure,
                'distance': distance,
                'force_parity': self.inputs.scf.get('kpoints_force_parity', orm.Bool(False)),
                'metadata': {
                    'call_link_label': 'create_kpoints_from_distance'
                }
            }
            kpoints = create_kpoints_from_distance(**inputs)  # pylint: disable=unexpected-keyword-arg

        self.ctx.kpoints = kpoints  # Needed for first SCF, and finite electric fields if needed

        if self.ctx.is_parallel_distance:
            self.ctx.kpoints_dict = {}
            for number in self.ctx.numbers:
                inputs = {
                    'structure': self.inputs.scf.pw.structure,
                    'direction': orm.List(get_vector_from_number(number, 1.)),
                    'parallel_distance': self.inputs.kpoints_parallel_distance,
                    'orthogonal_distance': self.inputs.scf.kpoints_distance,
                    'force_parity': self.inputs.scf.get('kpoints_force_parity', orm.Bool(False)),
                    'metadata': {
                        'call_link_label': 'create_directional_kpoints',
                    },
                }

                self.ctx.kpoints_dict[number] = create_directional_kpoints(**inputs)  # pylint: disable=unexpected-keyword-arg

            self.ctx.meshes = []
            self.ctx.kpoints_list = []

            for kpoints in self.ctx.kpoints_dict.values():
                mesh = kpoints.get_kpoints_mesh()[0]  # not the offset
                if not mesh in self.ctx.meshes:
                    self.ctx.meshes.append(mesh)
                    self.ctx.kpoints_list.append(kpoints)

    def should_estimate_electric_field(self):
        """Return whether a nscf calculation needs to be run to estimate the electric field."""
        return self.ctx.should_estimate_electric_field

    def should_run_electric_field_scfs(self):
        """Return whether to run the next electric field scfs."""
        return not self.ctx.max_iteration == self.ctx.iteration

    def get_inputs(self, electric_field_vector):
        """Return the inputs for the electric enthalpy scf."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        parameters = inputs.pw.parameters.get_dict()
        parameters.setdefault('CONTROL', {})
        parameters.setdefault('SYSTEM', {})
        parameters.setdefault('ELECTRONS', {})

        # --- Compulsory keys for electric enthalpy
        parameters['SYSTEM'].pop('degauss', None)
        parameters['SYSTEM'].pop('smearing', None)

        parameters['SYSTEM']['occupations'] = 'fixed'
        parameters['CONTROL'].update({
            'restart_mode': 'from_scratch',
            'lelfield': True,
            'tprnfor': True,  # must be True to compute forces
            'tstress': False,  # do not waste time computing stress tensor - not needed
        })
        parameters['ELECTRONS'].update({
            'efield_cart': electric_field_vector,
            'startingpot': 'file',
        })
        parameters['CONTROL'].setdefault('disk_io', 'medium')  # this allows smoother restart

        base_out = self.ctx.base_scf.outputs

        # --- Field dependent settings
        if electric_field_vector == [0, 0, 0]:
            # thr_init = max(1e-10, base_out.output_parameters.get_dict()['convergence_info']['scf_conv']['scf_error'])
            parameters['CONTROL']['nberrycyc'] = 1
            parameters['ELECTRONS'].update({
                # 'diago_thr_init': thr_init,
                'efield_phase': 'write',  # write the polarization phase
            })
        else:
            parameters['CONTROL'].setdefault('nberrycyc', self._DEFAULT_NBERRYCYC)
            parameters['ELECTRONS'].update({
                'startingwfc': 'file',
                'efield_phase': 'read',
            })

        # --- Magnetic ground state
        if self.ctx.is_magnetic:
            parameters['SYSTEM'].pop('starting_magnetization', None)

            tot_magnetization = base_out.output_parameters.base.attributes.get('total_magnetization')
            if validate_tot_magnetization(tot_magnetization):
                return self.exit_codes.ERROR_NON_INTEGER_TOT_MAGNETIZATION

            parameters['SYSTEM'].update({
                # In some rare cases, this makes the code to crash.
                # 'nbnd': base_out.output_parameters.base.attributes.get('number_of_bands'),
                'tot_magnetization': abs(round(tot_magnetization)),
            })

        # --- Fill the inputs
        inputs.pw.parameters = orm.Dict(parameters)
        if self.ctx.clean_workdir:
            inputs.clean_workdir = orm.Bool(False)

        # --- Set non-redundant kpoints
        for name in ('kpoints_distance', 'kpoints_force_parity', 'kpoints'):
            inputs.pop(name, None)
        inputs.kpoints = self.ctx.kpoints

        return inputs

    def run_base_scf(self):
        """Run initial scf for ground-state ."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        parameters = inputs.pw.parameters.get_dict()

        for name in ('kpoints_distance', 'kpoints_force_parity', 'kpoints'):
            inputs.pop(name, None)
        inputs.kpoints = self.ctx.kpoints

        for key in ('nberrycyc', 'lelfield', 'efield_cart'):
            parameters['CONTROL'].pop(key, None)
            parameters['ELECTRONS'].pop(key, None)

        if 'parent_scf' in self.inputs:
            inputs.pw.parent_folder = self.inputs.parent_scf
            parameters['CONTROL']['restart_mode'] = 'from_scratch'
            parameters['ELECTRONS']['startingpot'] = 'file'

        inputs.pw.parameters = orm.Dict(parameters)

        key = 'base_scf'
        inputs.metadata.call_link_label = key

        inputs.clean_workdir = orm.Bool(False)  # the folder is needed for next calculations

        node = self.submit(PwBaseWorkChain, **inputs)
        self.to_context(**{key: node})
        self.report(f'launching base scf PwBaseWorkChain<{node.pk}>')

    def inspect_base_scf(self):
        """Verify that the scf PwBaseWorkChain finished successfully."""
        workchain = self.ctx.base_scf

        if not workchain.is_finished_ok:
            #     self.ctx.data = {'null_field': add_zero_polarization(self.ctx.base_scf.outputs.output_trajectory)}
            # else:
            self.report(f'base scf failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_FAILED_BASE_SCF

    def run_nscf(self):
        """Run a NSCF PwBaseWorkChain to evalute the band gap."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))

        for name in ('kpoints_distance', 'kpoints_force_parity', 'kpoints'):
            inputs.pop(name, None)
        inputs.kpoints = self.ctx.kpoints

        outputs = self.ctx.base_scf.outputs

        parameters = inputs.pw.parameters.get_dict()
        parameters['CONTROL'].update({
            'calculation': 'nscf',
            'restart_mode': 'from_scratch',
        })

        nbnd = outputs.output_parameters.base.attributes.get('number_of_bands') + 10
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
        self.report(f'launching base scf PwBaseWorkChain<{node.pk}>')

    def inspect_nscf(self):
        """Verify that the nscf PwBaseWorkChain finished successfully."""
        workchain = self.ctx.nscf

        if not workchain.is_finished_ok:
            self.report(f'nscf failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_FAILED_NSCF

    def set_step_and_accuracy(self):
        """Set the electric field step and the accuracy in Context."""
        if 'accuracy' in self.inputs.central_difference:
            self.ctx.accuracy = self.inputs.central_difference.accuracy
        else:
            self.ctx.accuracy = get_accuracy_from_critical_field(self.ctx.critical_electric_field)

        if self.ctx.should_estimate_electric_field:
            self.ctx.electric_field_step = get_electric_field_step(self.ctx.critical_electric_field, self.ctx.accuracy)
            self.out('electric_field_step', self.ctx.electric_field_step)

        self.ctx.max_iteration = int(self.ctx.accuracy.value / 2)
        self.ctx.iteration = 0

    def run_null_field_scfs(self):
        """Run electric enthalpy scf with zero electric field."""
        inputs = self.get_inputs(electric_field_vector=[0., 0., 0.])
        if 'parent_scf' in self.inputs:
            inputs.pw.parent_folder = self.inputs.parent_scf
        else:
            inputs.pw.parent_folder = self.ctx.base_scf.outputs.remote_folder

        key = 'null_fields'
        if not self.ctx.is_parallel_distance:
            inputs.metadata.call_link_label = key

            node = self.submit(PwBaseWorkChain, **inputs)
            self.to_context(**{key: append_(node)})
            self.report(f'launching PwBaseWorkChain<{node.pk}> with null electric field')
        else:
            for index, kpoints in enumerate(self.ctx.kpoints_list):
                inputs.kpoints = kpoints
                inputs.metadata.call_link_label = f'{key[:-1]}_{index}'

                node = self.submit(PwBaseWorkChain, **inputs)
                self.to_context(**{key: append_(node)})
                self.report(f'launching PwBaseWorkChain<{node.pk}> with null electric field {index}')

    def inspect_null_field_scfs(self):
        """Verify that the scf PwBaseWorkChain with null electric field finished successfully."""
        workchains = self.ctx.null_fields

        for workchain in workchains:
            if not workchain.is_finished_ok:
                self.report(f'electric field scf failed with exit status {workchain.exit_status}')
                return self.exit_codes.ERROR_FAILED_ELFIELD_SCF.format(direction='`null`')

    def estimate_critical_electric_field(self):
        """Estimate the critical electric field E ~ Egap/(e*a*Nk)."""
        nscf_outputs = self.ctx.nscf.outputs
        value_node = compute_critical_electric_field(
            parameters=nscf_outputs.output_parameters,
            bands=nscf_outputs.output_band,
            structure=self.inputs.scf.pw.structure,
        )
        self.ctx.critical_electric_field = value_node
        self.out('critical_electric_field', self.ctx.critical_electric_field)

    def run_electric_field_scfs(self):
        """Run scf with different electric fields for central difference."""
        # Here we can already truncate `numbers`` from the very beginning using symmetry analysis
        signs = [1.0, -1.0]
        for number, bool_signs in zip(self.ctx.numbers, self.ctx.signs):
            norm = self.ctx.electric_field_step.value * (self.ctx.iteration + 1)
            if number in (3, 4, 5):
                norm = norm * self.inputs.central_difference.diagonal_scale

            # Here we can put a zip with True or False, but it depends on numbers, so even before check
            for sign, bool_sign in zip(signs, bool_signs):
                if bool_sign:
                    electric_field_vector = get_vector_from_number(number=number, value=sign * norm)
                    inputs = self.get_inputs(electric_field_vector=electric_field_vector)

                    if self.ctx.is_parallel_distance:
                        inputs.kpoints = self.ctx.kpoints_dict[number]

                    # Here I label:
                    # * 0,1,2 for first order derivatives: l --> {l}j ; e.g. 0 does 00, 01, 02
                    # * 0,1,2,3,4,5 for second order derivatives: l <--> ij --> {ij}k ;
                    #   precisely 0 > {00}k; 1 > {11}k; 2 > {22}k; 3 > {12}k; 4 > {02}k; 5 --> {01}k | k=0,1,2
                    key = f'field_index_{number}'  # adding the iteration as well?
                    inputs.metadata.call_link_label = key

                    if not self.ctx.iteration == 0:
                        index_folder = -2 if bool_signs[1] else -1
                        inputs.pw.parent_folder = self.ctx[key][index_folder].outputs.remote_folder
                    else:
                        index_null = 0
                        if self.ctx.is_parallel_distance:
                            mesh = inputs.kpoints.get_kpoints_mesh()[0]
                            index_null = self.ctx.meshes.index(mesh)
                        inputs.pw.parent_folder = self.ctx.null_fields[index_null].outputs.remote_folder

                    # We fill in the ctx arrays in order to have at the end something like:
                    # field_index_0 = [+1,-1,+2,-2] (e.g. for accuracy = 4, no symmetry)
                    # field_index_0 = [+1, +2] (e.g. for accuracy = 4, with e.g. inversion symmetry)
                    # where the numbers refers to the multiplication factor to the base electric field E/accuracy,
                    # since in central finite differences we always have a number of evaluations equal to the accuracy,
                    # half for positive and half for negative evaluation of the function.
                    # Symmetries can reduce this.
                    node = self.submit(PwBaseWorkChain, **inputs)
                    self.to_context(**{key: append_(node)})
                    message = f'with electric field index {number} and sign {sign} iteration #{self.ctx.iteration}'
                    self.report(f'launching PwBaseWorkChain<{node.pk}> ' + message)
                    time.sleep(self.inputs.settings.sleep_submission_time)

        self.ctx.iteration = self.ctx.iteration + 1

    def inspect_electric_field_scfs(self):
        """Inspect all previous pw workchains with electric fields."""
        # output_data = {'fields_data.null_field': self.ctx.null_field.outputs.output_trajectory}
        # self.ctx.data = {'null_field': self.ctx.null_field.outputs.output_trajectory}
        output_data = {}
        self.ctx.data = {}
        self.ctx.meshes_dict = {}

        for label, workchains in self.ctx.items():
            if label.startswith('field_index_'):
                self.ctx.meshes_dict[label] = workchains[0].inputs.kpoints.get_kpoints_mesh()[0]
                field_data = {
                    str(i): wc.outputs.output_trajectory for i, wc in enumerate(workchains) if wc.is_finished_ok
                }  # pylint: disable=locally-disabled, line-too-long
                output_data.update({f'fields_data.{label}': field_data})
                self.ctx.data.update({label: field_data})
        self.out_many(output_data)

        for key, workchains in self.ctx.items():
            if key.startswith('field_index_'):
                for workchain in workchains:
                    if not workchain.is_finished_ok:
                        self.report(f'electric field scf failed with exit status {workchain.exit_status}')
                        return self.exit_codes.ERROR_FAILED_ELFIELD_SCF.format(direction=key[-1])

    def remove_reference_forces(self):
        """Subtract the reference forces to each electric field trajectory."""
        from aiida_vibroscopy.calculations.spectra_utils import subtract_residual_forces
        if 'kpoints_parallel_distance' in self.inputs:
            ref_meshes = orm.List(self.ctx.meshes)
        else:
            ref_meshes = orm.List([self.ctx.kpoints.get_kpoints_mesh()[0]])
        meshes_dict = orm.Dict(self.ctx.meshes_dict)
        ref_trajectories = {}
        for i, wc in enumerate(self.ctx.null_fields):
            ref_trajectories[str(i)] = wc.outputs.output_trajectory
        old_trajectories = self.ctx.data
        kwargs = {'ref_trajectories': ref_trajectories, 'old_trajectories': old_trajectories}
        new_data = subtract_residual_forces(ref_meshes, meshes_dict, **kwargs)
        self.ctx.new_data = new_data

    def run_numerical_derivatives(self):
        """Compute numerical derivatives from previous calculations."""
        key = 'numerical_derivatives'

        inputs = {
            'data': self.ctx.new_data,
            'structure': self.inputs.scf.pw.structure,
            'central_difference': {
                'electric_field_step': self.ctx.electric_field_step,
                'accuracy': self.ctx.accuracy,
                'diagonal_scale': self.inputs.central_difference.diagonal_scale,
            },
            'symmetry': {
                'symprec': self.inputs.symmetry.symprec,
                'is_symmetry': self.inputs.symmetry.is_symmetry,
                'distinguish_kinds': self.inputs.symmetry.distinguish_kinds,
            },
            'metadata': {
                'call_link_label': 'numerical_derivatives'
            }
        }

        node = self.submit(NumericalDerivativesWorkChain, **inputs)
        self.to_context(**{key: node})
        self.report(f'launching NumericalDerivativesWorkChain<{node.pk}> for computing numerical derivatives.')

    def results(self):
        """Expose outputs."""
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
