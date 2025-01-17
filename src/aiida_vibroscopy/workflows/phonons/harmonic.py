# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Automatic harmonic frozen phonons calculations using Phonopy and Quantum ESPRESSO."""

from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.common.lang import type_check
from aiida.engine import WorkChain, if_
from aiida.plugins import CalculationFactory, DataFactory, WorkflowFactory
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_vibroscopy.calculations.spectra_utils import (
    elaborate_tensors,
    generate_vibrational_data_from_force_constants,
    generate_vibrational_data_from_phonopy,
)
from aiida_vibroscopy.common.properties import PhononProperty
from aiida_vibroscopy.data import VibrationalData, VibrationalFrozenPhononData

from ..dielectric.base import DielectricWorkChain
from .base import PhononWorkChain

PreProcessData = DataFactory('phonopy.preprocess')
PhonopyData = DataFactory('phonopy.phonopy')
PhonopyCalculation = CalculationFactory('phonopy.phonopy')
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')


def validate_inputs(inputs, _):
    """Validate `HarmonicWorkChain` inputs."""
    from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
    if inputs['settings']['use_primitive_cell'].value and isinstance(inputs['structure'], HubbardStructureData):
        return '`use_primitive_cell` cannot currently be used with `HubbardStructureData` inputs.'


class HarmonicWorkChain(WorkChain, ProtocolMixin):
    """Workchain for frozen phonons calculations.

    Non-analytical constants (NAC) and higher order mixed  derivatives are computed
    via finite differences through finite electric fields.
    See :class:`~aiida_vibroscopy.workflows.DielectricWorkChain`
    for more details on how they are carried out.
    """

    # yapf: disable
    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)

        # yapf: disable
        spec.input('structure', valid_type=orm.StructureData)
        spec.expose_inputs(
            DielectricWorkChain, namespace='dielectric',
            namespace_options={
                'required': False,
                'help': (
                    'Inputs for the `DielectricWorkChain` that will be'
                    'used to calculate the mixed derivatives with electric field.'
                )
            },
            exclude=('scf.pw.structure', 'symmetry')
        )
        spec.expose_inputs(
            PhononWorkChain, namespace='phonon',
            namespace_options={
                'required': True, 'populate_defaults': True,
                'help': (
                    'Inputs for the `PhononWorkChain` that will be'
                    'used to calculate the force constants.'
                )
            },
            exclude=('scf.pw.structure', 'symmetry')
        )
        spec.expose_inputs(
            PhonopyCalculation, namespace='phonopy',
            namespace_options={
                'required': False, 'populate_defaults': False,
                'help': (
                    'Inputs for the `PhonopyCalculation` that will'
                    'be used to calculate the inter-atomic force constants, or for post-processing.'
                )
            },
            exclude=['phonopy_data', 'force_constants'],
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
        spec.input_namespace(
            'settings',
            help='Options for how to run the workflow.',
        )
        spec.input(
            'settings.use_primitive_cell', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help=(
                'Whether to use the primitive cell for the `DielectricWorkChain`. '
                'WARNING: it is not implemented for HubbardStructureData.'
            ),
        )
        spec.input(
            'settings.run_parallel', valid_type=bool, non_db=True, default=lambda: True,
            help='Whether to run the `DielectricWorkChain` and the `PhononWorkChain` in parallel.',
        )
        spec.input(
            'clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.'
        )
        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            if_(cls.should_run_parallel)(
                cls.run_parallel
            ).else_(
                cls.run_phonon,
                if_(cls.should_run_dielectric)(
                    cls.run_dielectric,
                    )
                ),
            cls.inspect_processes,
            cls.run_vibrational_data,
            if_(cls.should_run_phonopy)(
                cls.run_phonopy,
                cls.inspect_phonopy,
            ),
        )

        spec.expose_outputs(
            PhononWorkChain, namespace='output_phonon',
            namespace_options={'required': True, 'help':'Outputs of the `PhononWorkChain`.'},
        )
        spec.expose_outputs(
            DielectricWorkChain, namespace='output_dielectric',
            namespace_options={'required': False, 'help':'Outputs of the `DielectricWorkChain`.'},
        )
        spec.expose_outputs(
            PhonopyCalculation, namespace='output_phonopy',
            namespace_options={'required': False, 'help':'Outputs of the post-processing via `phonopy`.'},
        )
        spec.output_namespace(
            'vibrational_data', dynamic=True,
            valid_type=(VibrationalData, VibrationalFrozenPhononData),
            help=(
                'The phonopy data with supercells displacements, forces and (optionally)'
                'nac parameters to use in the post-processing calculation.'
            )
        )

        spec.exit_code(400, 'ERROR_PHONON_WORKCHAIN_FAILED', message='The phonon workchain failed.')
        spec.exit_code(401, 'ERROR_DIELECTRIC_WORKCHAIN_FAILED', message='The dielectric workchain failed.')
        spec.exit_code(402, 'ERROR_PHONOPY_CALCULATION_FAILED', message='The phonopy calculation failed.')
        # yapf: enable

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from ..protocols import phonons
        return files(phonons) / 'harmonic.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls,
        pw_code,
        structure,
        protocol=None,
        phonopy_code=None,
        overrides=None,
        options=None,
        phonon_property=PhononProperty.NONE,
        **kwargs
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param phonopy_code: the ``Code`` instance configured for the ``phonopy.phonopy`` plugin.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

        if isinstance(phonopy_code, str):
            phonopy_code = orm.load_code(phonopy_code)

        type_check(phonon_property, PhononProperty)

        if phonopy_code is not None:
            type_check(phonopy_code, orm.AbstractCode)
        elif phonon_property.value is not None:
            raise ValueError('`PhononProperty` is specified, but `phonopy_code` is None')

        inputs = cls.get_protocol_inputs(protocol, overrides)

        args = (pw_code, structure, protocol)
        phonon = PhononWorkChain.get_builder_from_protocol(
            *args, overrides=inputs.get('phonon', None), options=options, **kwargs
        )
        dielectric = DielectricWorkChain.get_builder_from_protocol(
            *args, overrides=inputs.get('dielectric', None), options=options, **kwargs
        )

        phonon['scf']['pw'].pop('structure', None)
        phonon.pop('symmetry', None)
        dielectric['scf']['pw'].pop('structure', None)
        dielectric.pop('symmetry', None)

        builder = cls.get_builder()
        builder.phonon = phonon
        builder.dielectric = dielectric
        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder.symmetry['symprec'] = orm.Float(inputs['symmetry']['symprec'])
        builder.symmetry['distinguish_kinds'] = orm.Bool(inputs['symmetry']['distinguish_kinds'])
        builder.symmetry['is_symmetry'] = orm.Bool(inputs['symmetry']['is_symmetry'])
        builder.settings['use_primitive_cell'] = orm.Bool(inputs['settings']['use_primitive_cell'])
        builder.settings['run_parallel'] = inputs['settings']['run_parallel']

        if phonopy_code is not None:
            builder.phonopy.code = phonopy_code
            parameters = phonon_property.value
            if overrides:
                parameter_overrides = overrides.get('phonopy', {}).get('parameters', {})
                parameters = recursive_merge(parameters, parameter_overrides)
            builder.phonopy.parameters = orm.Dict(parameters)

        return builder

    def setup(self):
        """Set up general context variables."""
        preprocess_inputs = {'structure': self.inputs.structure}

        for input_ in ['supercell_matrix', 'primitive_matrix', 'displacements_generator']:
            if input_ in self.inputs.phonon:
                preprocess_inputs.update({input_: self.inputs.phonon[input_]})

        for input_ in ['symprec', 'is_symmetry', 'distinguish_kinds']:
            if input_ in self.inputs.symmetry:
                preprocess_inputs.update({input_: self.inputs.symmetry[input_]})

        preprocess = PreProcessData.generate_preprocess_data(**preprocess_inputs)

        self.ctx.preprocess_data = preprocess

    def should_run_phonopy(self):
        """Wheter to run a `PhonopyCalculation`."""
        return 'phonopy' in self.inputs

    def should_run_parallel(self):
        """Wheter to run parallelly `DielectricWorkChain` and `PhononWorkChain`."""
        return self.inputs.settings.run_parallel

    def should_run_dielectric(self):
        """Wheter to run the `DielectricWorkChain`."""
        return 'dielectric' in self.inputs

    def run_parallel(self):
        """Run in parallel forces calculations and dielectric workchain."""
        self.run_phonon()

        if 'dielectric' in self.inputs:
            self.run_dielectric()

    def run_phonon(self):
        """Run a PhononWorkChain."""
        key = 'phonon'
        inputs = AttributeDict(self.exposed_inputs(PhononWorkChain, namespace=key))
        inputs.scf.pw.structure = self.inputs.structure
        inputs.symmetry = self.inputs.symmetry
        inputs.metadata.call_link_label = key

        future = self.submit(PhononWorkChain, **inputs)
        self.report(f'submitting `PhononWorkChain` <PK={future.pk}>')
        self.to_context(**{key: future})

    def run_dielectric(self):
        """Run a DielectricWorkChain."""
        key = 'dielectric'
        inputs = AttributeDict(self.exposed_inputs(DielectricWorkChain, namespace=key))
        inputs.scf.pw.structure = self.inputs.structure
        inputs.symmetry = self.inputs.symmetry

        if self.inputs.settings.use_primitive_cell:
            inputs.scf.pw.structure = self.ctx.preprocess_data.calcfunctions.get_primitive_cell()

        inputs.metadata.call_link_label = key

        future = self.submit(DielectricWorkChain, **inputs)
        self.report(f'submitting `DielectricWorkChain` <PK={future.pk}>')
        self.to_context(**{key: future})

    def inspect_processes(self):
        """Inspect all sub-processes."""
        workchain = self.ctx.phonon

        if not workchain.is_finished_ok:
            self.report(f'the child `PhononWorkChain` with <PK={workchain.pk}> failed')
            return self.exit_codes.ERROR_PHONON_WORKCHAIN_FAILED

        self.out_many(self.exposed_outputs(self.ctx.phonon, PhononWorkChain, namespace='output_phonon'))

        if 'dielectric' in self.ctx:
            workchain = self.ctx.dielectric
            if not workchain.is_finished_ok:
                self.report(f'the child `DielectricWorkChain` with <PK={workchain.pk}> failed')
                return self.exit_codes.ERROR_DIELECTRIC_WORKCHAIN_FAILED

            self.out_many(self.exposed_outputs(self.ctx.dielectric, DielectricWorkChain, namespace='output_dielectric'))

    def run_vibrational_data(self):
        """Run results generating outputs for post-processing and visualization."""
        self.ctx.vibrational_data = {}

        if 'dielectric' in self.ctx:

            tensors_dict = self.ctx.dielectric.outputs.tensors

            for key, tensors in tensors_dict.items():

                if not self.inputs.settings.use_primitive_cell.value:
                    tensors = elaborate_tensors(self.ctx.preprocess_data, tensors)

                if 'output_force_constants' in self.ctx.phonon.outputs:
                    vibrational_data = generate_vibrational_data_from_force_constants(
                        preprocess_data=self.ctx.preprocess_data,
                        force_constants=self.ctx.force_constants,
                        tensors=tensors
                    )
                else:
                    vibrational_data = generate_vibrational_data_from_phonopy(
                        phonopy_data=self.ctx.phonon.outputs.phonopy_data, tensors=tensors
                    )

                self.ctx.vibrational_data[key] = vibrational_data

                output_key = f'vibrational_data.{key}'
                self.out(output_key, vibrational_data)

    def run_phonopy(self):
        """Run a PhononWorkChain."""
        key = 'phonopy'
        inputs = AttributeDict(self.exposed_inputs(PhonopyCalculation, namespace=key))
        inputs.force_constants = list(self.outputs['vibrational_data'].values())[-1]
        inputs.metadata.call_link_label = key
        future = self.submit(PhonopyCalculation, **inputs)
        self.report(f'submitting `PhonopyCalculation` <PK={future.pk}>')
        self.to_context(**{key: future})

    def inspect_phonopy(self):
        """Inspect the `PhonopyCalculation`."""
        workchain = self.ctx.phonopy

        if not workchain.is_finished_ok:
            self.report(f'the child `PhonopyCalculation` with <PK={workchain.pk}> failed')
            return self.exit_codes.ERROR_PHONOPY_CALCULATION_FAILED

        self.out_many(self.exposed_outputs(self.ctx.phonopy, PhonopyCalculation, namespace='output_phonopy'))

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
