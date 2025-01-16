# -*- coding: utf-8 -*-
"""Pre-defined overridable options for commonly used command line interface parameters."""
# pylint: disable=too-few-public-methods,import-error
from aiida.cmdline.params import types
from aiida.cmdline.params.options import OverridableOption
from aiida.cmdline.utils import decorators
from aiida.common import exceptions
import click

from . import validate


class PseudoFamilyType(types.GroupParamType):
    """Subclass of `GroupParamType` in order to be able to print warning with instructions."""

    def __init__(self, pseudo_types=None, **kwargs):
        """Construct a new instance."""
        super().__init__(**kwargs)
        self._pseudo_types = pseudo_types

    @decorators.with_dbenv()
    def convert(self, value, param, ctx):
        """Convert the value to actual pseudo family instance."""
        try:
            group = super().convert(value, param, ctx)
        except click.BadParameter:
            try:
                from aiida.orm import load_group
                load_group(value)
            except exceptions.NotExistent:  # pylint: disable=try-except-raise
                raise

            raise click.BadParameter(  # pylint: disable=raise-missing-from
                f'`{value}` is not of a supported pseudopotential family type.\nTo install a supported '
                'pseudofamily, use the `aiida-pseudo` plugin. See the following link for detailed instructions:\n\n'
                '    https://github.com/aiidateam/aiida-quantumespresso#pseudopotentials'
            )

        if self._pseudo_types is not None and group.pseudo_type not in self._pseudo_types:
            pseudo_types = ', '.join(self._pseudo_types)
            raise click.BadParameter(
                f'family `{group.label}` contains pseudopotentials of the wrong type `{group.pseudo_type}`.\nOnly the '
                f'following types are supported: {pseudo_types}'
            )

        return group


PW_CODE = OverridableOption(
    '--pw',
    'pw_code',
    type=types.CodeParamType(entry_point='quantumespresso.pw'),
    required=True,
    help='The code to use for the pw.x executable.'
)

PHONOPY_CODE = OverridableOption(
    '--phonopy',
    'phonopy_code',
    type=types.CodeParamType(entry_point='phonopy.phonopy'),
    required=True,
    help='The code to use for the phonopy executable.'
)

PSEUDO_FAMILY = OverridableOption(
    '-F',
    '--pseudo-family',
    type=PseudoFamilyType(sub_classes=('aiida.groups:pseudo.family',), pseudo_types=('pseudo.upf',)),
    required=False,
    help='Select a pseudopotential family, identified by its label.'
)

STRUCTURE = OverridableOption(
    '-S',
    '--structure',
    type=types.DataParamType(sub_classes=('aiida.data:core.structure',)),
    help='A StructureData node identified by its ID or UUID.'
)

KPOINTS_MESH = OverridableOption(
    '-k',
    '--kpoints-mesh',
    'kpoints_mesh',
    nargs=6,
    type=click.Tuple([int, int, int, float, float, float]),
    show_default=True,
    callback=validate.validate_kpoints_mesh,
    help='The number of points in the kpoint mesh along each basis vector and the offset. '
    'Example: `-k 2 2 2 0 0 0`. Specify `0.5 0.5 0.5` for the offset if you want to result '
    'in the equivalent Quantum ESPRESSO pw.x `1 1 1` shift.'
)

PARENT_FOLDER = OverridableOption(
    '-P',
    '--parent-folder',
    'parent_folder',
    type=types.DataParamType(sub_classes=('aiida.data:core.remote',)),
    show_default=True,
    required=False,
    help='A parent remote folder node identified by its ID or UUID.'
)

DAEMON = OverridableOption(
    '-D',
    '--daemon',
    is_flag=True,
    default=True,
    show_default=True,
    help='Submit the process to the daemon instead of running it and waiting for it to finish.'
)

OVERRIDES = OverridableOption(
    '-o',
    '--overrides',
    type=click.File('r'),
    required=False,
    help='The filename or filepath containing the overrides, in YAML format.'
)

PROTOCOL = OverridableOption(
    '-p',
    '--protocol',
    type=click.STRING,
    required=False,
    help='Select the protocol that defines the accuracy of the calculation.'
)
