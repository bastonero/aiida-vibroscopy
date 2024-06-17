# -*- coding: utf-8 -*-
"""Workchain to compute the phonon dispersion from the raw initial unrelaxed structure."""
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import ToContext, WorkChain, if_
from aiida.plugins import CalculationFactory, WorkflowFactory
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
BandsBaseWorkChain = WorkflowFactory('quantumespresso.bands.base')
PhBaseWorkChain = WorkflowFactory('quantumespresso.ph.base')
Q2rBaseWorkChain = WorkflowFactory('quantumespresso.q2r.base')
MatdynBaseWorkChain = WorkflowFactory('quantumespresso.matdyn.base')

PhCalculation = CalculationFactory('quantumespresso.ph')

QERamanBaseWorkChain = WorkflowFactory('vibroscopy.qe_raman.base')


class ResonantQERamanWorkChain(ProtocolMixin, WorkChain):
    """Workchain to compute the resonant Raman spectra for an input structure.

    .. important:: The workflow exploits the QERaman package. Please also cite:
    > N. T. Hung et al., QERaman: A open-source program for calculating resonance Raman spectroscopy
      based on Quantum ESPRESSO, Computer Physics Communications 295, 108967 (2024).
      DOI: https://doi.org/10.1016/j.cpc.2023.108967

    To compile the relevant QERaman binaries, please refer to: https://github.com/nguyen-group/QERaman
    """

    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('structure', valid_type=orm.StructureData, required=False)
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input(
            'parent_folder',
            valid_type=orm.RemoteData,
            required=False,
            help='`RemoteData` folder of a parent `pw.x` calculation.'
        )

        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace='relax',
            exclude=('clean_workdir', 'structure'),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs to relax the structure prior to all the other calculations.'
            }
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace='scf_coarse',
            exclude=('clean_workdir', 'structure'),
            namespace_options={
                'required': True,
                'populate_defaults': False,
                'help': 'Inputs for the coarse SCF using symmetry (used to restart the fine SCF).'
            }
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace='scf_fine',
            exclude=('clean_workdir', 'structure'),
            namespace_options={
                'required': True,
                'populate_defaults': False,
                'help': 'Inputs for the fine SCF not using symmetry'
            }
        )
        spec.expose_inputs(
            BandsBaseWorkChain,
            exclude=('clean_workdir', 'bands.parent_folder', 'bands.parameters'),
            namespace_options={
                'required': True,
                'populate_defaults': False,
                'help': 'Inputs for the `bands.x` calculation to compute the momemtum operator for `raman.x`.'
            }
        )
        spec.expose_inputs(
            PhBaseWorkChain,
            namespace='ph_main',
            exclude=('clean_workdir', 'ph.parent_folder'),
            namespace_options={
                'required': True,
                'populate_defaults': False,
                'help': 'Inputs of the ph calculation, used by `ph_mat` to compute el-ph couplings.'
            }
        )
        spec.expose_inputs(
            PhBaseWorkChain,
            namespace='ph_mat',
            exclude=('clean_workdir', 'ph.parent_folder'),
            namespace_options={
                'required': True,
                'populate_defaults': False,
                'help': 'Inputs of the `ph_mat.x` calculation to compute el-ph couplings for `raman.x`.'
            }
        )
        spec.expose_inputs(
            QERamanBaseWorkChain,
            exclude=('clean_workdir', 'ph.parent_folder'),
            namespace_options={
                'required': True,
                'populate_defaults': False,
                'help': 'Inputs of the `raman.x` calculation to compute the resonant Raman spectra and tensors.'
            }
        )

        spec.outline(
            cls.setup,
            if_(cls.should_run_relax)(
                cls.run_relax,
                cls.inspect_relax,
            ),
            cls.run_scf_coarse,
            cls.inspect_scf_coarse,
            cls.run_scf_fine,
            cls.inspect_scf_fine,
            cls.run_bands_and_ph_main,
            cls.inspect_bands_and_ph_main,
            cls.run_ph_mat,
            cls.inspect_ph_mat,
            cls.run_raman,
            cls.inspect_raman,
            cls.results,
        )

        spec.output(
            'output_structure',
            valid_type=orm.StructureData,
            required=False,
            help='The structure for which the dynamical matrix is computed.'
        )
        spec.output('pw_output_parameters', valid_type=orm.Dict)
        spec.output('ph_output_parameters', valid_type=orm.Dict)
        spec.output('ph_retrieved', valid_type=orm.FolderData)

        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_RELAX', message='The PwRelaxWorkChain sub process failed')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from ...protocols.spectra import qe_raman
        return files(qe_raman) / 'resonant.yaml'

    @classmethod
    def get_builder_from_protocol(cls, pw_code, ph_code, structure, protocol=None, overrides=None, **kwargs):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param ph_code: the ``Code`` instance configured for the ``quantumespresso.ph`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)

        args = (pw_code, structure, protocol)
        relax = PwRelaxWorkChain.get_builder_from_protocol(*args, overrides=inputs.get('relax', None), **kwargs)
        relax.pop('structure', None)
        relax.pop('clean_workdir', None)
        relax.pop('base_final_scf', None)

        args = (ph_code, None, protocol)
        ph_main = PhBaseWorkChain.get_builder_from_protocol(*args, overrides=inputs.get('ph_main', None), **kwargs)
        ph_main.pop('clean_workdir', None)

        builder = cls.get_builder()
        builder.structure = structure
        builder.relax = relax  #pw_base = pw_base
        builder.ph_main = ph_main
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

    def setup(self):
        """Initialise basic context variables and get input structure."""
        self.ctx.current_structure = self.inputs.structure
        if 'parent_folder' in self.inputs:
            self.ctx.current_folder = self.inputs.parent_folder
        else:
            self.report('no parent given')

    def should_run_relax(self):
        """Check if the work chain should run the  ``PwRelaxWorkChain`` - for either relax or scf."""
        return not 'parent_folder' in self.inputs

    def run_relax(self):
        """Run the PwRelaxWorkChain to run a relax PwCalculation."""
        inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace='relax'))
        inputs.metadata.call_link_label = 'relax'
        inputs.structure = self.ctx.current_structure

        running = self.submit(PwRelaxWorkChain, **inputs)

        self.report(f'launching PwRelaxWorkChain<{running.pk}>')

        return ToContext(workchain_relax=running)

    def inspect_relax(self):
        """Verify that the PwRelaxWorkChain finished successfully."""
        workchain = self.ctx.workchain_relax

        if not workchain.is_finished_ok:
            self.report(f'PwRelaxWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX

        self.ctx.current_folder = workchain.outputs.remote_folder
        if 'output_structure' in workchain.outputs:
            self.ctx.current_structure = workchain.outputs.output_structure
            self.out('output_structure', workchain.outputs.output_structure)

    def run_ph(self):
        """Run the PhBaseWorkChain."""
        inputs = AttributeDict(self.inputs.ph_main)
        inputs.ph.parent_folder = self.ctx.current_folder

        workchain_node = self.submit(PhBaseWorkChain, **inputs)

        self.report(f'launching PhBaseWorkChain<{workchain_node.pk}>')

        return ToContext(workchain_ph=workchain_node)

    def results(self):
        """Attach the desired output nodes directly as outputs of the workchain."""
        self.report('workchain succesfully completed')

        self.out('pw_output_parameters', self.ctx.workchain_relax.outputs.output_parameters)
        self.out('ph_output_parameters', self.ctx.workchain_ph.outputs.output_parameters)
        self.out('ph_retrieved', self.ctx.workchain_ph.outputs.retrieved)

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
