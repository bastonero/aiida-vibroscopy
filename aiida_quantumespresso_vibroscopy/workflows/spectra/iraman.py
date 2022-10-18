# -*- coding: utf-8 -*-
"""Automatic IR and Raman spectra calculations using Phonopy and Quantum ESPRESSO."""

from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import if_

from ..base import BaseWorkChain
from aiida_quantumespresso_vibroscopy.data import VibrationalFrozenPhononData
from aiida_quantumespresso_vibroscopy.calculations.phonon_utils import extract_symmetry_info
from aiida_quantumespresso_vibroscopy.calculations.spectra_utils import elaborate_tensors, generate_vibrational_data
from .intensities_average import IntensitiesAverageWorkChain
from ..dielectric.base import DielectricWorkChain

PreProcessData = DataFactory('phonopy.preprocess')
PhonopyData = DataFactory('phonopy.phonopy')

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')


class IRamanSpectraWorkChain(BaseWorkChain):
    """
    Workchain for automatically compute IR and Raman spectra using finite displacements and fields.
    """

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)

        spec.expose_inputs(IntensitiesAverageWorkChain, namespace='intensities_average',
            namespace_options={
                'required': True, 'populate_defaults': True,
                'help': ('Inputs for the `IntensitiesAverageWorkChain` that will'
                    'be used to run the average calculation over intensities.')
            },
            exclude=('vibrational_data',)
        )

        spec.outline(
            cls.setup,
            cls.run_base_supercell,
            cls.inspect_base_supercell,
            if_(cls.should_run_parallel)(
                cls.run_parallel,
            ).else_(
                cls.run_forces,
                cls.run_dielectric,
            ),
            cls.inspect_all_runs,
            cls.run_raw_results,
            cls.run_intensities_averaged,
            cls.show_results,
        )

        spec.output_namespace('intensities_average', dynamic=True,
            help=('Intensities average over space and q-points.')
        )
        spec.expose_outputs(IntensitiesAverageWorkChain)
        spec.output_namespace('vibrational_data', valid_type=VibrationalFrozenPhononData, dynamic=True,
            help=('The phonopy data with supercells displacements, forces and (optionally)'
                'nac parameters to use in the post-processing calculation.')
        )

        spec.exit_code(403, 'ERROR_AVERAGING_FAILED',
            message='The averaging procedure for intensities had an unexpected error.')

    def setup(self):
        """Setup the workflow generating the PreProcessData."""
        preprocess_inputs = {'structure':self.inputs.structure}
        for pp_input in [
                'symprec', 'is_symmetry',
                'displacement_generator', 'distinguish_kinds']:
            if pp_input in self.inputs:
                preprocess_inputs.update({pp_input:self.inputs.phonon_workchain[pp_input]})
        preprocess_inputs.update({'supercell_matrix':orm.List(list=[1,1,1])})
        preprocess = PreProcessData.generate_preprocess_data(**preprocess_inputs)

        self.ctx.preprocess_data = preprocess
        self.ctx.run_parallel = self.inputs.options.run_parallel

        self.set_ctx_variables()

    def run_dielectric(self):
        """Run a DielectricWorkChain."""
        inputs = AttributeDict(self.exposed_inputs(DielectricWorkChain, namespace='dielectric_workchain'))
        base_key = f'{self._RUN_PREFIX}_0'

        inputs.scf.pw.structure = self.ctx.supercell
        inputs.parent_scf = self.ctx[base_key].outputs.remote_folder
        inputs.clean_workdir = self.inputs.clean_workdir
        inputs.options = extract_symmetry_info(self.ctx.preprocess_data)

        key = 'dielectric_workchain'
        inputs.metadata.call_link_label = key

        future = self.submit(DielectricWorkChain, **inputs)
        self.report(f'submitting `DielectricWorkChain` <PK={future.pk}>')
        self.to_context(**{key: future})

    def run_raw_results(self):
        """Run results generating outputs for post-processing and visualization."""
        self.ctx.vibrational_data = {}
        workchain = self.ctx.dielectric_workchain
        diel_out = self.exposed_outputs(workchain, DielectricWorkChain)
        tensors_dict = diel_out['tensors'] # remember it is a dictionary with the numerical accuracies

        for key, tensors in tensors_dict.items():

            tensors = elaborate_tensors(self.ctx.preprocess_data, tensors)
            kwargs = {'tensors':tensors, **self.outputs['supercells_forces']}

            vibrational_data = generate_vibrational_data(
                preprocess_data=self.ctx.preprocess_data,
                **kwargs,
            )

            self.ctx.vibrational_data[key] = vibrational_data

            output_key = f'vibrational_data.{key}'
            self.out(output_key, vibrational_data)

    def run_intensities_averaged(self):
        """Run an `IntensitiesAverageWorkChain` with the calculated vibrational data."""
        self.ctx.intensities_average = {}
        for key, vibrational_data in self.ctx.vibrational_data.items():
            inputs = AttributeDict(self.exposed_inputs(IntensitiesAverageWorkChain, namespace='intensities_average'))
            inputs.vibrational_data = vibrational_data
            inputs.metadata.call_link_label = key

            future = self.submit(IntensitiesAverageWorkChain, **inputs)
            self.report(f'submitting `IntensitiesAverageWorkChain` <PK={future.pk}>.')
            self.to_context(**{f'intensities_average.{key}': future})

    def show_results(self):
        """Expose the outputs."""
        outputs = {'intensities_average':{}}
        for key, workchain in self.ctx.intensities_average.items():
            if workchain.is_failed:
                self.report('the averaging procedure failed')
                return self.exit_codes.ERROR_AVERAGING_FAILED
            else:
                outs = { key: {**self.exposed_outputs(workchain, IntensitiesAverageWorkChain)} }
                outputs['intensities_average'].update(outs)

        self.out_many(outputs)
