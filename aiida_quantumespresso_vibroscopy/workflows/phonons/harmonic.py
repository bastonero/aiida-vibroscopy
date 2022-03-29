# -*- coding: utf-8 -*-
"""Automatic harmonic frozen phonons calculations using Phonopy and Quantum ESPRESSO."""

from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import WorkChain, if_

from ..dielectric.base import DielectricWorkChain
from aiida_quantumespresso_vibroscopy.utils.validation import *
from aiida_quantumespresso_vibroscopy.calculations.phonon_utils import *
from aiida_quantumespresso_vibroscopy.calculations.spectra_utils import *


PreProcessData = DataFactory('phonopy.preprocess')
PhonopyData = DataFactory('phonopy.phonopy')
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')


def validate_inputs(inputs, _):
    """Validate the entire inputs namespace."""
    ensamble_inputs = ['structure', 'supercell_matrix', 'primitive_matrix', 'symprec', 'is_symmetry']

    given_inputs = []

    for input in ensamble_inputs:
        if input in inputs:
            given_inputs.append(input)

    if 'preprocess_data' in inputs and given_inputs:
        return 'too many inputs have been provided'

    if given_inputs and not 'structure' in given_inputs:
        return 'a structure data is required'

    if not given_inputs and not 'preprocess_data' in inputs:
        return 'at least one between `preprocess_data` and `structure` must be provided in input'

    if 'nac_parameters' in inputs and 'dielectric_workchain' in inputs:
        return 'too many inputs for non-analytical constants'


class HarmonicWorkChain(WorkChain):
    """
    Workchain for automatically compute all the pre-process data necessary
    for frozen phonons calculations. Non-analytical constants are computed
    via finite differences as well through finite electric fields.
    """

    _ENABLED_DISPLACEMENT_GENERATOR_FLAGS = {
        'distance': [float],
        'is_plusminus': ['auto', float],
        'is_diagonal': [bool],
        'is_trigonal': [bool],
        'number_of_snapshots': [int, None],
        'random_seed': [int, None],
        'temperature': [float, None],
        'cutoff_frequency': [float, None],
    }

    _RUN_PREFIX = 'scf_supercell'

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
            'structure',
            valid_type=orm.StructureData,
            required=False,
            help='The structure at equilibrium volume.'
        )
        spec.input(
            'supercell_matrix',
            valid_type=orm.List,
            required=False,
            validator=validate_matrix,
            help=(
                'The matrix used to generate the supercell from the input '
                'structure in the List format. Allowed shapes are 3x1 and 3x3 lists.'
            ),
        )
        spec.input(
            'primitive_matrix',
            valid_type=orm.List,
            required=False,
            validator=validate_matrix,
            help=(
                'The matrix used to generate the primitive cell from the input '
                'structure in the List format. Allowed shapes are 3x1 and 3x3 lists.'
            ),
        )
        spec.input(
            'symmetry_tolerance',
            valid_type=orm.Float,
            validator=validate_positive,
            required=False,
            help='Symmetry tolerance for space group analysis on the input structure.',
        )
        spec.input(
            'is_symmetry',
            valid_type=orm.Bool,
            required=False,
            help='Whether using or not the space group symmetries.',
        )
        spec.input(
            'displacement_generator',
            valid_type=orm.Dict,
            required=False,
            validator=cls._validate_displacements,
            help=(
                'Info for displacements generation. The following flags are allowed:\n '
                + '\n '.join(f'{flag_name}' for flag_name in cls._ENABLED_DISPLACEMENT_GENERATOR_FLAGS)
            ),
        )
        spec.input(
            'nac_parameters',
            valid_type=orm.ArrayData,
            required=False,
            validator=validate_nac,
            help='Non-analytical parameters.',
        )
        spec.expose_inputs(DielectricWorkChain, namespace='dielectric_workchain',
            namespace_options={
                'required': False, 'populate_defaults': False,
                'help': ('Inputs for the `DielectricWorkChain` that will be used to calculate the non-analytical constants.')
            },
            exclude=('clean_workdir','scf.pw.structure')
        )
        spec.expose_inputs(PwBaseWorkChain, namespace='scf',
            namespace_options={
                'required': True,
                'help': ('Inputs for the `PwBaseWorkChain` that will be used to run the supercell with displacement scfs.')
            },
            exclude=('clean_workdir', 'pw.parent_folder', 'pw.structure')
        )
        spec.input_namespace(
            'options',
            help='Options for how to run the workflow.',
        )
        spec.input(
            'options.run_parallel',
            valid_type=bool,
            non_db=True,
            default=True,
            help='Whether running dielectric workchain and forces calculations in parallel.',
        )
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.'
        )
        spec.inputs.validator = validate_inputs

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
            cls.run_results,
            cls.show_results,
        )

        spec.output_namespace('supercells', valid_type=orm.StructureData, dynamic=True, required=False,
            help='The supercells with displacements.'
        )
        spec.output_namespace('supercells_forces', valid_type=orm.ArrayData, required=True,
            help='The forces acting on the atoms of each supercell.'
        )
        spec.output_namespace('supercells_energies', valid_type=orm.Float, dynamic=True, required=False,
            help='The total energy of each supercell.'
        )
        spec.output('output_phonopy_data', valid_type=PhonopyData,
            help='The phonopy data with supercells displacements, forces and (optionally) nac parameters to use in the post-processing calculation.'
        )
        spec.expose_outputs(DielectricWorkChain, namespace='output_dielectric',
            namespace_options={
                'required': False,
                'help': ('Outputs of the `DielectricWorkChain`.')
            },
        )

        spec.exit_code(400, 'ERROR_FAILED_BASE_SCF',
            message='The initial supercell scf work chain failed.')
        spec.exit_code(401, 'ERROR_NON_INTEGER_TOT_MAGNETIZATION',
            message=('The scf PwBaseWorkChain sub process in iteration '
                    'returned a non integer total magnetization (threshold exceeded).'))
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED', # can't we say exactly which are not finished ok?
            message='At least one sub processe did not finish successfully.')

    @classmethod
    def _validate_displacements(cls, value, _):
        """Validate the ``displacements`` input namespace."""
        if value:
            value_dict = value.get_dict()
            enabled_dict = cls._ENABLED_DISPLACEMENT_GENERATOR_FLAGS
            unknown_flags = set(value_dict.keys()) - set(enabled_dict.keys())
            if unknown_flags:
                return (
                    f"Unknown flags in 'displacements': {unknown_flags}."
                    # f"allowed flags are {cls._ENABLED_DISPLACEMENT_GENERATOR_FLAGS.keys()}."
                )
            invalid_values = [
                value_dict[key]
                for key in value_dict.keys()
                if not (type(value_dict[key]) in enabled_dict[key] or value_dict[key] in enabled_dict[key])
            ]
            if invalid_values:
                return f'Displacement options must be of the correct type; got invalid values {invalid_values}.'

    def setup(self):
        """Setup the workflow generating the PreProcessData."""
        if 'preprocess_data' in self.inputs:
            preprocess = self.inputs.preprocess_data
            if 'displacement_generator' in self.inputs:
                preprocess = preprocess.calcfunctions.get_preprocess_with_new_displacements(self.inputs.displacement_generator)
        else:
            preprocess_inputs = {}
            for input in ['structure', 'supercell_matrix', 'primitive_matrix', 'symprec', 'is_symmetry', 'displacement_generator']:
                if input in self.inputs:
                    preprocess_inputs.update({input:self.inputs[input]})
            preprocess = PreProcessData.generate_preprocess_data(**preprocess_inputs)

        self.ctx.preprocess_data = preprocess

        if 'dielectric_workchain' in self.inputs:
            self.ctx.run_parallel = self.inputs.options.run_parallel

            parameters = self.inputs.dielectric_workchain.scf.pw.parameters.get_dict()
            nspin = parameters.get('SYSTEM', {}).get('nspin', 1)
            if  nspin != 1:
                if len(preprocess.get_unitcell().sites) != len(preprocess.get_primitive_cell().sites):
                    raise NotImplementedError('a primitive cell smaller than the unitcell with spin polarized is not supported.')
        else:
            self.ctx.run_parallel = False

        parameters = self.inputs.scf.pw.parameters.get_dict()
        nspin = parameters.get('SYSTEM', {}).get('nspin', 1)
        self.ctx.is_magnetic = True if  nspin != 1 else False

        if parameters.get('SYSTEM', {}).get('lda_plus_u_kind', None) == 2:
            self.ctx.plus_hubbard = True
        else:
            self.ctx.plus_hubbard = False

    def should_run_parallel(self):
        return self.ctx.run_parallel

    def run_base_supercell(self):
        """Run a `pristine` supercell calculation from where to restart supercell with displacements."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        self.ctx.supercell = self.ctx.preprocess_data.calcfunctions.get_supercell()
        inputs.pw.structure = self.ctx.supercell

        key = 'scf_supercell_0'
        inputs.metadata.call_link_label = key
        inputs.clean_workdir = orm.Bool(False) # the folder is needed for next calculations

        node = self.submit(PwBaseWorkChain, **inputs)
        self.to_context(**{key: node})
        self.report(f'launched base supercell scf PwBaseWorkChain<{node.pk}>')

    def inspect_base_supercell(self):
        """Verify that the scf PwBaseWorkChain finished successfully."""
        workchain = self.ctx.scf_supercell_0

        if not workchain.is_finished_ok:
            self.report(f'base supercell scf failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_FAILED_BASE_SCF

    def run_parallel(self):
        """It runs in parallel forces calculations and dielectric workchain."""
        self.run_forces()
        self.run_dielectric()

    def run_forces(self):
        """Run an scf for each supercell with displacements."""
        # Works only @ Gamma
        if self.ctx.plus_hubbard:
            supercells = get_supercells_for_hubbard(
                preprocess_data=self.ctx.preprocess_data,
                ref_structure=self.inputs.structure
            )
        else:
            supercells = self.ctx.preprocess_data.calcfunctions.get_supercells_with_displacements()

        self.out('supercells', supercells)

        for key, supercell in supercells.items():
            num = key.split('_')[-1]
            label = f'{self._RUN_PREFIX}_{num}'

            inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
            inputs.pw.structure = supercell
            inputs.pw.parent_folder = self.ctx.scf_supercell_0.outputs.remote_folder

            parameters = inputs.pw.parameters.get_dict()
            parameters.setdefault('CONTROL', {})
            parameters.setdefault('SYSTEM', {})
            parameters.setdefault('ELECTRONS', {})
            if self.ctx.is_magnetic:
                parameters['SYSTEM'].pop('starting_magnetization', None)
                parameters['SYSTEM']['nbnd'] = self.ctx.scf_supercell_0.outputs.output_parameters.get_dict()['number_of_bands']
                if set_tot_magnetization( inputs.pw.parameters, self.ctx.scf_supercell_0.outputs.output_parameters.get_dict()['total_magnetization'] ):
                    return self.exit_codes.ERROR_NON_INTEGER_TOT_MAGNETIZATION
            parameters['ELECTRONS']['startingpot'] = 'file'
            inputs.pw.parameters = orm.Dict(dict=parameters)

            inputs.clean_workdir = self.inputs.clean_workdir
            inputs.metadata.label = label
            inputs.metadata.call_link_label = label

            future = self.submit(PwBaseWorkChain, **inputs)
            self.report(f'submitting `PwBaseWorkChain` <PK={future.pk}> with supercell n.o {key}')
            self.to_context(**{label: future})

    def run_dielectric(self):
        """Run a DielectricWorkChain."""
        inputs = AttributeDict(self.exposed_inputs(DielectricWorkChain, namespace='dielectric_workchain'))
        preprocess = self.ctx.preprocess_data

        if self.ctx.is_magnetic or (len(preprocess.get_unitcell().sites) != len(preprocess.get_primitive_cell().sites)):
            inputs.scf.pw.structure = self.ctx.supercell
            inputs.parent_folder = self.ctx.scf_supercell_0.outputs.remote_folder
        else:
            inputs.scf.pw.structure = self.ctx.preprocess_data.calcfunctions.get_primitive_cell()

        inputs.clean_workdir = self.inputs.clean_workdir

        key = 'dielectric_workchain'
        inputs.metadata.call_link_label = key

        future = self.submit(DielectricWorkChain, **inputs)
        self.report(f'submitting `DielectricWorkChain` <PK={future.pk}>')
        self.to_context(**{key: future})

    def inspect_all_runs(self):
        """Inspect all previous workchains."""
        # First we check the forces
        failed_runs = []

        for label, workchain in self.ctx.items():
            if label.startswith(self._RUN_PREFIX):
                if workchain.is_finished_ok:
                    forces = get_forces(workchain.outputs.output_trajectory)
                    energy = get_energy(workchain.outputs.output_parameters)
                    self.out(f'supercells_forces.{label}', forces)
                    self.out(f'supercells_energies.{label}', energy)
                else:
                    self.report(f'PwBaseWorkChain with <PK={workchain.pk}> failed with exit status {workchain.exit_status}')
                    failed_runs.append(workchain.pk)

        if 'dielectric_workchain' in self.ctx:
            workchain = self.ctx.dielectric_workchain
            if not workchain.is_finished_ok:
                self.report(f'DielectricWorkChain failed with exit status {workchain.exit_status}')
                failed_runs.append(workchain.pk)
            else:
                self.out_many(self.exposed_outputs(self.ctx.dielectric_workchain, DielectricWorkChain))

        if failed_runs:
            self.report('one or more workchains did not finish succesfully')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(cls=self.inputs.sub_process_class)  # pylint: disable=no-member


    def run_results(self):
        """Run results generating outputs for post-processing and visualization."""
        nac_parameters = None

        if 'nac_parameters' in self.inputs:
            nac_parameters = self.inputs.nac_parameters
        if 'dielectric_workchain' in self.inputs:
            diel_out = self.ctx.dielectric_workchain.outputs
            nac = extract_max_order({'dielectric':diel_out.dielectric, 'born_charges':diel_out.born_charges})
            if not self.ctx.is_magnetic:
                nac_parameters = get_non_analytical_constants(**nac)
            else:
                # Here we have to `elaborate` the nac parameters, since for magnetic insulators the nac are computed
                # on the unitcell and not the primitive cell. For sanity check, we elaborate them.
                nac_parameters = elaborate_non_analytical_constants(
                    ref_structure=self.inputs.structure,
                    preprocess_data=self.ctx.preprocess_data,
                    **nac
                )

        phonopy_data = PhonopyData(preprocess_data=self.ctx.preprocess_data)
        self.ctx.full_phonopy_data = phonopy_data.calcfunctions.generate_full_phonopy_data(
            nac_parameters=nac_parameters,
            **self.outputs['supercells_forces']
        )

    def show_results(self):
        """Expose the outputs."""
        self.out('output_phonopy_data', self.ctx.full_phonopy_data)

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
