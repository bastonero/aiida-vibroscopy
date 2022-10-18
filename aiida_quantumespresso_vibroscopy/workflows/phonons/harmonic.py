# -*- coding: utf-8 -*-
"""Automatic harmonic frozen phonons calculations using Phonopy and Quantum ESPRESSO."""

from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import if_

from ..base import BaseWorkChain
from ..dielectric.base import DielectricWorkChain
from aiida_quantumespresso_vibroscopy.utils.validation import validate_matrix, validate_nac
from aiida_quantumespresso_vibroscopy.calculations.phonon_utils import extract_symmetry_info
from aiida_quantumespresso_vibroscopy.calculations.spectra_utils import elaborate_tensors


PreProcessData = DataFactory('phonopy.preprocess')
PhonopyData = DataFactory('phonopy.phonopy')
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')


class HarmonicWorkChain(BaseWorkChain):
    """
    Workchain for automatically compute all the pre-process data necessary
    for frozen phonons calculations. Non-analytical constants are computed
    via finite differences as well through finite electric fields.
    """

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)

        spec.input(
            'preprocess_data',
            valid_type=(PhonopyData, PreProcessData),
            required=False,
            help='The preprocess data for frozen phonon calcualtion.'
        )
        spec.input(
            'phonon_workchain.supercell_matrix',
            valid_type=orm.List,
            validator=validate_matrix,
            required=False,
            help='Supercell matrix that defines the supercell from the unitcell.',
        )
        spec.input(
            'nac_parameters',
            valid_type=orm.ArrayData,
            required=False,
            validator=validate_nac,
            help='Non-analytical parameters.',
        )

        spec.outline(
            cls.setup,
            cls.run_base_supercell,
            cls.inspect_base_supercell,
            if_(cls.should_run_parallel)(
                cls.run_parallel,
            ).else_(
                cls.run_forces,
                if_(cls.should_run_dielectric)(
                    cls.run_dielectric,
                )
            ),
            cls.inspect_all_runs,
            cls.run_results,
        )

        spec.output('phonopy_data', valid_type=PhonopyData,
            help='The phonopy data with supercells displacements, forces and (optionally) nac parameters to use in the post-processing calculation.'
        )


    def setup(self):
        """Setup the workflow generating the PreProcessData."""
        if 'preprocess_data' in self.inputs:
            preprocess = self.inputs.preprocess_data
            if 'displacement_generator' in self.inputs.phonon_workchain:
                preprocess = preprocess.calcfunctions.get_preprocess_with_new_displacements(self.inputs.phonon_workchain.displacement_generator)
        else:
            preprocess_inputs = {'structure':self.inputs.structure}
            for input in [
                    'supercell_matrix', 'primitive_matrix',
                    'symprec', 'is_symmetry', 'displacement_generator',
                    'distinguish_kinds'
                ]:
                if input in self.inputs.phonon_workchain:
                    preprocess_inputs.update({input:self.inputs.phonon_workchain[input]})
            preprocess = PreProcessData.generate_preprocess_data(**preprocess_inputs)

        self.ctx.preprocess_data = preprocess

        if 'dielectric_workchain' in self.inputs:
            self.ctx.run_dielectric = True
            self.ctx.run_parallel = self.inputs.options.run_parallel

            parameters = self.inputs.dielectric_workchain.scf.pw.parameters.get_dict()
            nspin = parameters.get('SYSTEM', {}).get('nspin', 1)
            if  nspin != 1:
                if len(preprocess.get_unitcell().sites) != len(preprocess.get_primitive_cell().sites):
                    raise NotImplementedError('a primitive cell smaller than the unitcell with spin polarized is not supported.')
        else:
            self.ctx.run_parallel = False
            self.ctx.run_dielectric = False

        self.set_ctx_variables()

    def should_run_dielectric(self):
        return self.ctx.run_dielectric

    def run_parallel(self):
        """It runs in parallel forces calculations and dielectric workchain."""
        self.run_forces()
        self.run_dielectric()

    def run_dielectric(self):
        """Run a DielectricWorkChain."""
        inputs = AttributeDict(self.exposed_inputs(DielectricWorkChain, namespace='dielectric_workchain'))

        preprocess = self.ctx.preprocess_data

        if self.ctx.is_magnetic or (len(preprocess.get_unitcell().sites) != len(preprocess.get_primitive_cell().sites)):
            inputs.scf.pw.structure = self.ctx.supercell
            # base_key = f'{self._RUN_PREFIX}_0'
            # inputs.parent_scf = self.ctx[base_key].outputs.remote_folder
        else:
            inputs.scf.pw.structure = self.ctx.preprocess_data.calcfunctions.get_primitive_cell()

        inputs.clean_workdir = self.inputs.clean_workdir
        inputs.options = extract_symmetry_info(self.ctx.preprocess_data)

        key = 'dielectric_workchain'
        inputs.metadata.call_link_label = key

        future = self.submit(DielectricWorkChain, **inputs)
        self.report(f'submitting `DielectricWorkChain` <PK={future.pk}>')
        self.to_context(**{key: future})

    def run_results(self):
        """Run results generating outputs for post-processing and visualization."""
        nac_parameters = None

        if 'nac_parameters' in self.inputs:
            nac_parameters = self.inputs.nac_parameters
        if 'dielectric_workchain' in self.inputs:
            tensors_dict = self.ctx.dielectric_workchain.outputs.tensors

            tensor_key = 'numerical_accuracy_2'
            max_accuracy = 2
            for key in tensors_dict.keys():
                if int(key[-1]) > max_accuracy:
                    max_accuracy = int(key[-1])
                    tensor_key = key

            tensors = tensors_dict[tensor_key]

            if not self.ctx.is_magnetic:
                nac_parameters = tensors
            else:
                nac_parameters = elaborate_tensors(self.ctx.preprocess_data, tensors)

        full_phonopy_data = self.ctx.preprocess_data.calcfunctions.generate_full_phonopy_data(
            nac_parameters=nac_parameters,
            **self.outputs['supercells_forces']
        )

        self.out('phonopy_data', full_phonopy_data)
