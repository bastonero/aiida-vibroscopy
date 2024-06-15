# -*- coding: utf-8 -*-
"""Command line scripts to launch a `PhononWorkChain` for testing and demonstration purposes."""
from aiida.cmdline.utils import decorators
import click
import yaml

from .. import cmd_launch
from ...utils import defaults, launch, options


@cmd_launch.command('phonon')
@options.PW_CODE()
@options.STRUCTURE(default=defaults.get_structure)
@options.PROTOCOL(type=click.Choice(['fast', 'moderate', 'precise']), default='moderate', show_default=True)
@options.PSEUDO_FAMILY()
@options.KPOINTS_MESH(show_default=False)
@options.PHONOPY_CODE(required=False)
@options.OVERRIDES()
@options.DAEMON()
@decorators.with_dbenv()
def launch_workflow(pw_code, structure, protocol, pseudo_family, kpoints_mesh, phonopy_code, overrides, daemon):
    """Run an `PhononWorkChain`.

    It computes the force constants in the harmonic approximation.

    .. note:: this workflow does NOT computer non-analytical constants (dielectric and
        Born effective charges tensors). Only the finite displacements of atoms.
    """
    from aiida.plugins import WorkflowFactory

    entry_point_name = 'vibroscopy.phonons.phonon'

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
        pw_code=pw_code,
        structure=structure,
        protocol=protocol,
        phonopy_code=phonopy_code,
        overrides=overrides,
    )

    if kpoints_mesh:
        builder.scf.pop('kpoints_distance')
        builder.scf.kpoints = kpoints_mesh

    launch.launch_process(builder, daemon)
