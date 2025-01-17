# -*- coding: utf-8 -*-
"""Command line scripts to launch a `IRamanSpectraWorkChain` for testing and demonstration purposes."""
from aiida.cmdline.utils import decorators
import click
import yaml

from .. import cmd_launch
from ...utils import defaults, launch, options


@cmd_launch.command('iraman-spectra')
@options.PW_CODE()
@options.STRUCTURE(default=defaults.get_structure)
@options.PROTOCOL(type=click.Choice(['fast', 'moderate', 'precise']), default='moderate', show_default=True)
@options.PSEUDO_FAMILY()
@options.KPOINTS_MESH(show_default=False)
@options.OVERRIDES()
@options.DAEMON()
@decorators.with_dbenv()
def launch_workflow(pw_code, structure, protocol, pseudo_family, kpoints_mesh, overrides, daemon):
    """Run an `IRamanSpectraWorkChain`.

    It computes force constants, dielectric, Born charges, Raman and non-linear optical
    susceptibility tensors via finite displacements and finite fields. The output can then
    be used to quickly post-process and get the IR absorption/reflectivity and Raman spectra
    in different experimental settings.
    """
    from aiida.plugins import WorkflowFactory

    if overrides:
        overrides = yaml.safe_load(overrides)

    if pseudo_family:
        if overrides:
            overrides.setdefault('dielectric', {}).setdefault('scf', {})['pseudo_family'] = pseudo_family.label
            overrides.setdefault('phonon', {}).setdefault('scf', {})['pseudo_family'] = pseudo_family.label
        else:
            overrides = {
                'dielectric': {
                    'scf': {
                        'pseudo_family': pseudo_family.label
                    },
                },
                'phonon': {
                    'scf': {
                        'pseudo_family': pseudo_family.label
                    },
                },
            }

    builder = WorkflowFactory('vibroscopy.spectra.iraman').get_builder_from_protocol(
        code=pw_code,
        structure=structure,
        protocol=protocol,
        overrides=overrides,
    )

    if kpoints_mesh:
        builder.dielectric.pop('kpoints_parallel_distance')
        builder.dielectric.scf.pop('kpoints_distance')
        builder.dielectric.scf.kpoints = kpoints_mesh

        builder.phonon.scf.pop('kpoints_distance')
        builder.phonon.scf.kpoints = kpoints_mesh

    launch.launch_process(builder, daemon)
