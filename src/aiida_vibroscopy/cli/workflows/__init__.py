# -*- coding: utf-8 -*-
"""Module with CLI commands for the various work chain implementations."""

from .. import cmd_root


@cmd_root.group('launch')
def cmd_launch():
    """Launch workflows."""


from .dielectric.base import launch_workflow
from .phonons.base import launch_workflow
from .phonons.harmonic import launch_workflow

# Import the sub commands to register them with the CLI
from .spectra.iraman import launch_workflow
