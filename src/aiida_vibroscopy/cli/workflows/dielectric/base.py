# -*- coding: utf-8 -*-
"""Command line scripts to launch a `DielectricWorkChain` for testing and demonstration purposes."""
# pylint: disable=import-error
from aiida.cmdline.utils import decorators
import click
import yaml

from .. import cmd_launch
from ...utils import defaults, launch, options


@cmd_launch.command('dielectric')
@options.PW_CODE()
@options.STRUCTURE(default=defaults.get_structure)
@options.PROTOCOL(type=click.Choice(['fast', 'moderate', 'precise']), default='moderate', show_default=True)
@options.PSEUDO_FAMILY()
@options.KPOINTS_MESH(show_default=False)
@options.OVERRIDES()
@options.DAEMON()
@decorators.with_dbenv()
def launch_workflow(pw_code, structure, protocol, pseudo_family, kpoints_mesh, overrides, daemon):
    """Run an `DielectricWorkChain`.

    It computes dielectric, Born charges, Raman and non-linear optical susceptibility
    tensors for a given structure.
    """
    from aiida.plugins import WorkflowFactory  # pyliny: disable=import-error

    entry_point_name = 'vibroscopy.dielectric'

    if overrides:
        overrides = yaml.safe_load(overrides)

    if pseudo_family:
        if overrides:
            overrides.setdefault('scf', {})['pseudo_family'] = pseudo_family.label
        else:
            overrides = {
                'scf': {
                    'pseudo_family': pseudo_family.label
                },
            }

    builder = WorkflowFactory(entry_point_name).get_builder_from_protocol(
        code=pw_code,
        structure=structure,
        protocol=protocol,
        overrides=overrides,
    )

    if kpoints_mesh:
        builder.pop('kpoints_parallel_distance')
        builder.scf.pop('kpoints_distance')
        builder.scf.kpoints = kpoints_mesh

    launch.launch_process(builder, daemon)
