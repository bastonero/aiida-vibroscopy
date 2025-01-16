# -*- coding: utf-8 -*-
"""Command line scripts to launch a `HarmonicWorkChain` for testing and demonstration purposes."""
from aiida.cmdline.utils import decorators
import click
import yaml

from .. import cmd_launch
from ...utils import defaults, launch, options


@cmd_launch.command('harmonic')
@options.PW_CODE()
@options.STRUCTURE(default=defaults.get_structure)
@options.PROTOCOL(type=click.Choice(['fast', 'moderate', 'precise']), default='moderate', show_default=True)
@options.PSEUDO_FAMILY()
@options.KPOINTS_MESH(show_default=False)
@options.OVERRIDES()
@options.PHONOPY_CODE(required=False)
@options.DAEMON()
@decorators.with_dbenv()
def launch_workflow(pw_code, structure, protocol, pseudo_family, kpoints_mesh, overrides, phonopy_code, daemon):
    """Run a `HarmonicWorkChain`.

    It computes force constants in the harmonic approximation,
    dielectric, Born charges, Raman and non-linear optical susceptibility tensors,
    to account for non-analytical behaviour of the dynamical matrix at small q-vectors.

    The output can then be used to quickly post-process and get phonons related properties,
    such as IR absorption/reflectivity and Raman spectra in different experimental settings,
    phonon dispersion .
    """
    from aiida.plugins import WorkflowFactory

    entry_point_name = 'vibroscopy.phonons.harmonic'

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

    builder = WorkflowFactory(entry_point_name).get_builder_from_protocol(
        pw_code=pw_code,
        structure=structure,
        protocol=protocol,
        overrides=overrides,
        phonopy_code=phonopy_code,
    )

    if kpoints_mesh:
        builder.dielectric.pop('kpoints_parallel_distance')
        builder.dielectric.scf.pop('kpoints_distance')
        builder.dielectric.scf.kpoints = kpoints_mesh

        builder.phonon.scf.pop('kpoints_distance')
        builder.phonon.scf.kpoints = kpoints_mesh

    launch.launch_process(builder, daemon)
