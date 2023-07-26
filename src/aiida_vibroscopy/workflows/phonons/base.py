# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Class for phonons with finite displacements."""
from __future__ import annotations

import time

from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.common.lang import type_check
from aiida.engine import WorkChain, calcfunction, if_
from aiida.plugins import CalculationFactory, DataFactory, WorkflowFactory
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_vibroscopy.calculations.spectra_utils import get_supercells_for_hubbard
from aiida_vibroscopy.common.properties import PhononProperty
from aiida_vibroscopy.utils.validation import validate_matrix, validate_tot_magnetization

__all__ = ('PhononWorkChain',)

PreProcessData = DataFactory('phonopy.preprocess')
PhonopyData = DataFactory('phonopy.phonopy')
HubbardStructureData = DataFactory('quantumespresso.hubbard_structure')

PhonopyCalculation = CalculationFactory('phonopy.phonopy')

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')


@calcfunction
def get_supercell_hubbard_structure(hubbard_structure, supercell, thr: orm.Float = None):
    """Return the HubbardStructureData of the supercell of the primitive structure.

    :param hubbard_structure: (Hubbard)StructureData containing the reference Hubbard parameters
    :param supercell: StructureData referring to a supercell of the input structure. It should be
        commensurate to the pristine one
    :param thr: threshold for symmetry analysis
    """
    from aiida_quantumespresso.utils.hubbard import HubbardUtils

    thr = 1.0e-5 if thr is None else thr.value

    hubbard_utils = HubbardUtils(hubbard_structure=hubbard_structure)
    return hubbard_utils.get_hubbard_for_supercell(supercell=supercell, thr=thr)


class PhononWorkChain(WorkChain, ProtocolMixin):
    """Class for computing force constants of phonons, without non-analytical corrections."""

    _ENABLED_DISPLACEMENT_GENERATOR_FLAGS = {
        'distance': [float],
        'is_plusminus': ['auto', float],
        'is_diagonal': [bool],
        'is_trigonal': [bool],
        'number_of_snapshots': [int, None],
        'random_seed': [int, None],
        'cutoff_frequency': [float, None],
    }

    _RUN_PREFIX = 'scf_supercell'

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)

        # yapf:disable
        spec.input(
            'supercell_matrix', valid_type=orm.List, required=False,
            help='Supercell matrix that defines the supercell from the unitcell.',
            validator=validate_matrix,
        )
        spec.input(
            'primitive_matrix', valid_type=orm.List, required=False,
            help='Primitive matrix that defines the primitive cell from the unitcell.',
            validator=validate_matrix,
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
        spec.input(
            'displacement_generator', valid_type=orm.Dict, required=False,
            help=(
                'Info for displacements generation. The following flags are allowed:\n ' +
                '\n '.join(f'{flag_name}' for flag_name in cls._ENABLED_DISPLACEMENT_GENERATOR_FLAGS)
            ),
            validator=cls._validate_displacements,
        )
        spec.expose_inputs(
            PwBaseWorkChain, namespace='scf',
            namespace_options={
                'required': True,
                'help': ('Inputs for the `PwBaseWorkChain` that will be used to run the electric enthalpy scfs.')
            },
            exclude=('clean_workdir', 'pw.parent_folder')
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
            'settings',
            help='Options for how to run the workflow.',
        )
        spec.input(
            'settings.sleep_submission_time', valid_type=(int, float), non_db=True, default=3.0,
            help='Time in seconds to wait before submitting subsequent displaced structure scf calculations.',
        )
        spec.input(
            'clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.'
        )

        spec.outline(
            cls.setup,
            cls.set_reference_kpoints,
            cls.run_base_supercell,
            cls.inspect_base_supercell,
            cls.run_forces,
            cls.inspect_all_runs,
            cls.set_phonopy_data,
            if_(cls.should_run_phonopy)(
              cls.run_phonopy,
              cls.inspect_phonopy,
            ),
        )

        spec.output_namespace(
            'supercells', valid_type=orm.StructureData, dynamic=True, required=False,
            help='The supercells with displacements.'
        )
        spec.output_namespace(
            'supercells_forces', valid_type=(orm.ArrayData, orm.TrajectoryData), required=True,
            help='The forces acting on the atoms of each supercell.'
        )
        spec.output(
            'phonopy_data', valid_type=PhonopyData, required=True,
            help=(
                'The phonopy data with supercells displacements, forces'
                ' to use in the post-processing calculation.'
            ),
        )
        spec.expose_outputs(PhonopyCalculation, namespace='output_phonopy', namespace_options={'required': False})

        spec.exit_code(
            400, 'ERROR_FAILED_BASE_SCF',
            message='The initial supercell scf work chain failed.'
        )
        spec.exit_code(
            401, 'ERROR_NON_INTEGER_TOT_MAGNETIZATION',
            message='The initial PwBaseWorkChain sub process returned a non integer total magnetization.'
        )
        spec.exit_code(
            402, 'ERROR_SUB_PROCESS_FAILED',
            message='At least one sub processe did not finish successfully.'
        )
        spec.exit_code(
            403, 'ERROR_PHONOPY_CALCULATION_FAILED',
            message='The phonopy calculation did not finish correctly.'
        )
        # yapf: enable

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

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from ..protocols import phonons
        return files(phonons) / 'phonon.yaml'

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
        scf = PwBaseWorkChain.get_builder_from_protocol(
            *args, overrides=inputs.get('scf', None), options=options, **kwargs
        )
        scf.pop('clean_workdir', None)

        builder = cls.get_builder()
        builder.scf = scf
        builder['supercell_matrix'] = orm.List(inputs['supercell_matrix'])
        builder['symmetry']['symprec'] = orm.Float(inputs['symmetry']['symprec'])
        builder['symmetry']['distinguish_kinds'] = orm.Bool(inputs['symmetry']['distinguish_kinds'])
        builder['symmetry']['is_symmetry'] = orm.Bool(inputs['symmetry']['is_symmetry'])
        builder.settings = inputs['settings']
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        if 'displacement_generator' in inputs:
            builder['displacement_generator'] = orm.Dict(inputs['displacement_generator'])
        if 'primitive_matrix' in inputs:
            builder['primitive_matrix'] = orm.List(inputs['primitive_matrix'])

        if phonopy_code is None:
            builder.pop('phonopy')
        else:
            builder.phonopy.code = phonopy_code
            parameters = phonon_property.value
            if overrides:
                parameter_overrides = overrides.get('phonopy', {}).get('parameters', {})
                parameters = recursive_merge(parameters, parameter_overrides)
            builder.phonopy.parameters = orm.Dict(parameters)

        return builder

    def set_ctx_variables(self):
        """Set `is_magnetic` and hubbard-related context variables."""
        parameters = self.inputs.scf.pw.parameters.get_dict()
        nspin = parameters.get('SYSTEM', {}).get('nspin', 1)
        self.ctx.is_magnetic = (nspin != 1)
        self.ctx.is_insulator = True
        self.ctx.plus_hubbard = False
        self.ctx.old_plus_hubbard = False

        if parameters.get('SYSTEM', {}).get('lda_plus_u_kind', None) == 2:
            self.ctx.old_plus_hubbard = True
        if isinstance(self.inputs.scf.pw.structure, HubbardStructureData):
            self.ctx.plus_hubbard = True

    def setup(self):
        """Set up the workflow generating the PreProcessData."""
        preprocess_inputs = {'structure': self.inputs.scf.pw.structure}

        for input_ in [
            'supercell_matrix',
            'primitive_matrix',
            'displacement_generator',
        ]:
            if input_ in self.inputs:
                preprocess_inputs.update({input_: self.inputs[input_]})
        for input_ in ['symprec', 'is_symmetry', 'distinguish_kinds']:
            if input_ in self.inputs['symmetry']:
                preprocess_inputs.update({input_: self.inputs['symmetry'][input_]})

        preprocess = PreProcessData.generate_preprocess_data(**preprocess_inputs)

        self.ctx.preprocess_data = preprocess

        self.set_ctx_variables()

        self.ctx.supercell = self.inputs.scf.pw.structure

        if not self.ctx.old_plus_hubbard:
            self.ctx.supercell = self.ctx.preprocess_data.calcfunctions.get_supercell()

        if self.ctx.plus_hubbard:
            self.ctx.supercell = get_supercell_hubbard_structure(self.inputs.scf.pw.structure, self.ctx.supercell)

    def set_reference_kpoints(self):
        """Set the reference kpoints for the all PwBaseWorkChains."""
        try:
            kpoints = self.inputs['scf']['kpoints']
        except (AttributeError, KeyError):
            inputs = {
                'structure': self.ctx.supercell,
                'distance': self.inputs['scf']['kpoints_distance'],
                'force_parity': self.inputs['scf'].get('kpoints_force_parity', orm.Bool(False)),
                'metadata': {
                    'call_link_label': 'create_kpoints_from_distance'
                }
            }
            kpoints = create_kpoints_from_distance(**inputs)  # pylint: disable=unexpected-keyword-arg

        self.ctx.kpoints = kpoints

    def run_base_supercell(self):
        """Run a `pristine` supercell calculation from where to restart supercell with displacements."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        inputs.pw.structure = self.ctx.supercell

        for name in ('kpoints_distance', 'kpoints_force_parity', 'kpoints'):
            inputs.pop(name, None)

        inputs.kpoints = self.ctx.kpoints

        key = f'{self._RUN_PREFIX}_0'
        inputs.metadata.label = key
        inputs.metadata.call_link_label = key
        inputs.clean_workdir = orm.Bool(False)  # the folder is needed for next calculations

        node = self.submit(PwBaseWorkChain, **inputs)
        self.to_context(**{key: node})
        self.report(f'launching base supercell scf PwBaseWorkChain<{node.pk}>')

    def inspect_base_supercell(self):
        """Verify that the scf PwBaseWorkChain finished successfully."""
        base_key = f'{self._RUN_PREFIX}_0'
        workchain = self.ctx[base_key]

        if not workchain.is_finished_ok:
            self.report(f'base supercell scf failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_FAILED_BASE_SCF

        parameters = workchain.outputs.output_parameters.dict
        if parameters.occupations == 'smearing':
            bands = workchain.outputs.output_band
            fermi_energy = parameters.fermi_energy
            self.ctx.is_insulator, _ = orm.find_bandgap(bands, fermi_energy=fermi_energy)

    def run_forces(self):
        """Run an scf for each supercell with displacements."""
        if self.ctx.plus_hubbard or self.ctx.old_plus_hubbard:
            supercells = get_supercells_for_hubbard(
                preprocess_data=self.ctx.preprocess_data, ref_structure=self.inputs.scf.pw.structure
            )
        else:
            supercells = self.ctx.preprocess_data.calcfunctions.get_supercells_with_displacements()

        self.out('supercells', supercells)

        base_key = f'{self._RUN_PREFIX}_0'
        base_out = self.ctx[base_key].outputs

        for key, supercell in supercells.items():
            num = key.split('_')[-1]
            label = f'{self._RUN_PREFIX}_{num}'

            inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
            inputs.pw.parent_folder = base_out.remote_folder

            for name in ('kpoints_distance', 'kpoints_force_parity', 'kpoints'):
                inputs.pop(name, None)

            inputs.kpoints = self.ctx.kpoints

            inputs.pw.structure = supercell

            parameters = inputs.pw.parameters.get_dict()
            parameters.setdefault('CONTROL', {})
            parameters.setdefault('SYSTEM', {})
            parameters.setdefault('ELECTRONS', {})

            if self.ctx.is_magnetic and self.ctx.is_insulator:
                parameters['SYSTEM']['occupations'] = 'fixed'

                for name in ('smearing', 'degauss', 'starting_magnetization'):
                    parameters['SYSTEM'].pop(name, None)

                parameters['SYSTEM']['nbnd'] = base_out.output_parameters.base.attributes.get('number_of_bands')
                tot_magnetization = base_out.output_parameters.base.attributes.get('total_magnetization')
                parameters['SYSTEM']['tot_magnetization'] = abs(round(tot_magnetization))

                if validate_tot_magnetization(tot_magnetization):
                    return self.exit_codes.ERROR_NON_INTEGER_TOT_MAGNETIZATION

            parameters['CONTROL']['restart_mode'] = 'from_scratch'  # important
            parameters['ELECTRONS']['startingpot'] = 'file'
            inputs.pw.parameters = orm.Dict(parameters)

            inputs.clean_workdir = self.inputs.clean_workdir
            inputs.metadata.label = label
            inputs.metadata.call_link_label = label

            future = self.submit(PwBaseWorkChain, **inputs)
            self.report(f'submitting `PwBaseWorkChain` <PK={future.pk}> with supercell n.o {num}')
            self.to_context(**{label: future})
            time.sleep(self.inputs.settings.sleep_submission_time)

    def inspect_all_runs(self):
        """Inspect all previous workchains."""
        failed_runs = []

        for label, workchain in self.ctx.items():
            if label.startswith(self._RUN_PREFIX):
                if workchain.is_finished_ok:
                    forces = workchain.outputs.output_trajectory
                    self.out(f"supercells_forces.forces_{label.split('_')[-1]}", forces)
                else:
                    self.report(
                        f'PwBaseWorkChain with <PK={workchain.pk}> failed'
                        'with exit status {workchain.exit_status}'
                    )
                    failed_runs.append(workchain.pk)

        if failed_runs:
            self.report('one or more workchains did not finish succesfully')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED

    def set_phonopy_data(self):
        """Set the `PhonopyData` in context for force constants calculation."""
        kwargs = {'forces_index': orm.Int(-1), **self.outputs['supercells_forces']}

        self.ctx.phonopy_data = self.ctx.preprocess_data.calcfunctions.generate_phonopy_data(**kwargs)
        self.out('phonopy_data', self.ctx.phonopy_data)

    def should_run_phonopy(self):
        """Return whether to run a PhonopyCalculation."""
        return 'phonopy' in self.inputs

    def run_phonopy(self):
        """Run a `PhonopyCalculation` to get the force constants."""
        inputs = AttributeDict(self.exposed_inputs(PhonopyCalculation, namespace='phonopy'))
        inputs.phonopy_data = self.ctx.phonopy_data

        key = 'phonopy_calculation'
        inputs.metadata.call_link_label = key

        future = self.submit(PhonopyCalculation, **inputs)
        self.report(f'submitting `PhonopyCalculation` <PK={future.pk}>')
        self.to_context(**{key: future})

    def inspect_phonopy(self):
        """Inspect that the `PhonopyCalculation` finished successfully."""
        calc = self.ctx.phonopy_calculation

        if calc.is_finished_ok:
            try:
                self.ctx.force_constants = calc.outputs.output_force_constants
            except AttributeError:
                self.report('WARNING: no force constants in output')
        else:
            self.report(f'PhonopyCalculation failed with exit status {calc.exit_status}')
            return self.exit_codes.ERROR_PHONOPY_CALCULATION_FAILED

        self.out_many(self.exposed_outputs(calc, PhonopyCalculation, namespace='output_phonopy'))

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
