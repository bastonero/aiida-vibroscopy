# -*- coding: utf-8 -*-
"""
Physical and conversion factor constants.
Whenever possible, we try to use the constants defined in
:py:mod:qe_tools._constants, but some of them are missing.
"""
from math import pi, sqrt
from types import SimpleNamespace

from phonopy import units
from qe_tools import CONSTANTS

__all__ = ('DEFAULT',)

DEFAULT = SimpleNamespace(
    eVinv_to_ang=12398.487384539087,  # eV^-1 to angstrom
    efield_au_to_si=51.4220674763 / sqrt(2),  # Ry a.u. to Volt/angstrom
    forces_si_to_au=(CONSTANTS.ry_to_ev / CONSTANTS.bohr_to_ang)**-1,  # eV/ang to Ry/bohr
    nlo_conversion=0.16 * pi * CONSTANTS.ry_to_ev * CONSTANTS.bohr_to_ang,
    cm_to_kelvin=units.CmToEv / units.Kb,
    thz_to_cm=units.THzToCm,
    debey_ang=0.2081943
    # to be defined:
    # * kelvin to eV
    # * nm to eV
    # * THz to eV
)
