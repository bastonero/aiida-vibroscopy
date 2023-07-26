# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Automatic IR and Raman spectra calculations using Phonopy and Quantum ESPRESSO."""
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import WorkChain, if_
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from ..dielectric.base import DielectricWorkChain
from ..phonons.base import PhononWorkChain
from ..phonons.harmonic import HarmonicWorkChain
from .intensities_average import IntensitiesAverageWorkChain


class IRamanSpectraWorkChain(WorkChain, ProtocolMixin):
    """Workchain for automatically compute IR and Raman spectra using finite displacements and fields.

    For other details of the sub-workchains used, see also:
        * :class:`~aiida_vibroscopy.workflows.dielectric.base.DielectricWorkChain` for finite fields
        * :class:`~aiida_vibroscopy.workflows.phonons.base.PhononWorkChain` for finite displacements
    """

    # yapf: disable
    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)

        # yapf: disable
        spec.expose_inputs(
            HarmonicWorkChain,
            namespace_options={'required': True, 'populate_defaults': False,},
            exclude=('dielectric', 'phonon.supercell_matrix', 'phonopy'),
        )
        spec.expose_inputs(
            DielectricWorkChain, namespace='dielectric',
            namespace_options={
                'required': True, 'populate_defaults': False,
                'help': (
                    'Inputs for the `DielectricWorkChain` that will be'
                    'used to calculate the mixed derivatives with electric field.'
                )
            },
            exclude=('scf.pw.structure', 'symmetry')
        )
        spec.expose_inputs(
            IntensitiesAverageWorkChain, namespace='intensities_average',
            namespace_options={'required': False, 'populate_defaults': False,
                'help': (
                    'Inputs for the `IntensitiesAverageWorkChain` that will'
                    'be used to run the average calculation over intensities.'
                )
            },
            exclude=('vibrational_data',)
        )

        spec.outline(
            cls.run_spectra,
            cls.inspect_process,
            if_(cls.should_run_average)(
                cls.run_intensities_averaged,
                cls.inspect_averaging,
            )
        )

        spec.expose_outputs(HarmonicWorkChain, exclude=('output_phonopy'), namespace_options={'required': False})
        spec.expose_outputs(IntensitiesAverageWorkChain, namespace='fake', namespace_options={'required': False})
        spec.output_namespace(
            'output_intensities_average',
            dynamic=True,
            required=False,
            help='Intensities average over space and q-points.'
        )
        spec.exit_code(
            400, 'ERROR_HARMONIC_WORKCHAIN_FAILED',
            message='The averaging procedure for intensities had an unexpected error.')
        spec.exit_code(
            401, 'ERROR_AVERAGING_WORKCHAIN_FAILED',
            message='The averaging procedure for intensities had an unexpected error.')
        # yapf: enable

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from ..protocols import spectra as stectra_protocols
        return files(stectra_protocols) / 'iraman.yaml'

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
        inputs = cls.get_protocol_inputs(protocol, overrides)

        args = (code, structure, protocol)
        phonon = PhononWorkChain.get_builder_from_protocol(
            *args, overrides=inputs.get('phonon', None), options=options, **kwargs
        )
        dielectric = DielectricWorkChain.get_builder_from_protocol(
            *args, overrides=inputs.get('dielectric', None), options=options, **kwargs
        )

        phonon['scf']['pw'].pop('structure', None)
        phonon.pop('symmetry', None)
        phonon.pop('supercell_matrix', None)
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

        return builder

    def run_spectra(self):
        """Run an `HarmonicWorkChain` at Gamma."""
        inputs = AttributeDict(self.exposed_inputs(HarmonicWorkChain))
        dielectric = AttributeDict(self.exposed_inputs(DielectricWorkChain, namespace='dielectric'))
        inputs.dielectric = dielectric
        inputs.phonon.supercell_matrix = orm.List([1, 1, 1])

        key = 'harmonic'
        inputs.metadata.call_link_label = key
        future = self.submit(HarmonicWorkChain, **inputs)
        self.report(f'submitting `HarmonicWorkChain` <PK={future.pk}>')
        self.to_context(**{key: future})

    def inspect_process(self):
        """Inspect that the `HarmonicWorkChain` finished successfully."""
        workchain = self.ctx.harmonic

        if workchain.is_failed:
            self.report(f'`HarmonicWorkChain` failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_HARMONIC_WORKCHAIN_FAILED

        self.out_many(self.exposed_outputs(workchain, HarmonicWorkChain))

    def should_run_average(self):
        """Return whether to run the average spectra."""
        return 'intensities_average' in self.inputs

    def run_intensities_averaged(self):
        """Run an `IntensitiesAverageWorkChain` with the calculated vibrational data."""
        self.ctx.intensities_average = AttributeDict({})
        for key, vibrational_data in self.ctx.vibrational_data.items():
            inputs = AttributeDict(self.exposed_inputs(IntensitiesAverageWorkChain, namespace='intensities_average'))
            inputs.vibrational_data = vibrational_data
            inputs.metadata.call_link_label = key

            future = self.submit(IntensitiesAverageWorkChain, **inputs)
            self.report(f'submitting `IntensitiesAverageWorkChain` <PK={future.pk}>.')
            self.to_context(**{f'intensities_average.{key}': future})

    def inspect_averaging(self):
        """Inspect and expose the outputs."""
        for key, workchain in self.ctx.intensities_average.items():

            if workchain.is_failed:
                self.report(f'`IntensitiesAverageWorkChain` failed with exit status {workchain.exit_status}')
                return self.exit_codes.ERROR_AVERAGING_WORKCHAIN_FAILED

            out_key = f'output_intensities_average.{key}'
            out = AttributeDict(
                {**self.exposed_outputs(workchain, IntensitiesAverageWorkChain, namespace='fake', agglomerate=False)}
            )
            out_dict = {out_key: {okey[5:]: ovalue for okey, ovalue in out.items()}}

            self.out_many(out_dict)
