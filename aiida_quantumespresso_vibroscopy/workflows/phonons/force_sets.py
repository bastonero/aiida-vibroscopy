# -*- coding: utf-8 -*-
"""Force sets workflow implementation for Quantum ESPRESSO."""

from aiida.common.extendeddicts import AttributeDict
from aiida.plugins import WorkflowFactory

BaseForceSetsWorkChain = WorkflowFactory('phonopy.force_sets')
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')


class ForceSetsWorkChain(BaseForceSetsWorkChain):
    """
    Workflow to compute automatically the force set of a given structure
    using the frozen phonons approach.
    """

    _RUN_PREFIX = 'force_calc'
    
    _FORCE_LABEL = "forces"
    _FORCE_INDEX = -1

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(PwBaseWorkChain, namespace='scf', exclude=('pw.structure',))

        spec.exit_code(400, 'ERROR_SUB_PROCESS_FAILED', # can't we say exactly which are not finished ok?
            message='At least one of the `PwBaseWorkChain` sub processes did not finish successfully.')

    @staticmethod
    def collect_forces_and_energies(ctx):
        """Collect forces and energies from calculation outputs."""
        forces_dict = {}

        for key, workchain in ctx.items(): # key: e.g. "supercell_001"
            if key.startswith('force_calc'):
                num = key.split("_")[-1] # e.g. "001"
                
                output = workchain.outputs

                forces_dict[f'forces_{num}'] = output.output_trajectory

        return forces_dict

    def run_forces(self):
        """Run supercell force calculations."""
        for key, supercell in self.ctx.supercells.items():
            
            num = key.split("_")[-1]
            if num == key:
                num = 0
            label = self._RUN_PREFIX + "_%s" % num
            
            inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
            inputs.pw.structure = supercell
            inputs.metadata.label = label 
            
            future = self.submit(PwBaseWorkChain, **inputs)
            self.report(f"submitting `PwBaseWorkChain` <PK={future.pk}> with {key} as structure")
            self.to_context(**{label: future})

    def inspect_forces(self):
        """Inspect all children workflows to make sure they finished successfully."""
        failed_runs = []
        
        for label, workchain in self.ctx.items():
            if label.startswith(self._RUN_PREFIX):
                if workchain.is_finished_ok:
                    forces = workchain.outputs.forces
                    self.out(f'supercells_forces.{label}', forces)
                else:
                    failed_runs.append(workchain.pk)
                    
        if failed_runs:
            self.report("workchain(s) with <PK={}> did not finish correctly".format(failed_runs))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(cls=self.inputs.sub_process_class)  # pylint: disable=no-member
        
        self.ctx.forces = self.collect_forces_and_energies()
        


