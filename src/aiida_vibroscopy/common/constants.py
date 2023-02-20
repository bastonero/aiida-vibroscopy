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

# We here refer to SI to the general convention in material science of:
# * Length ==> in Angstrom
# * Energy ==> in eV
# * Mass   ==> AMU

DEFAULT = SimpleNamespace(
    eVinv_to_ang=12398.487384539087,  # eV^-1 to angstrom
    efield_au_to_si=51.4220674763 / sqrt(2),  # Ry a.u. to Volt/angstrom
    forces_si_to_au=(CONSTANTS.bohr_to_ang / CONSTANTS.ry_to_ev),  # eV/ang to Ry/bohr
    # The nlo_conversion is for the correction to the Raman tensor dChi/dtau.
    # Since we store Chi(2) as pm/Volt, then we need to mulitply by 1/100 to get Angstrom/Volt.
    # Then, a pre-factor of 8*pi is present too, of which 4*pi is for the 1/epsilon0 term.
    # Lastly, we have the Born charge, corresponding to an electronic charge in atomic units.
    # This means that in order to obtain the units of Angstrom and Volt, we multiply by sqrt(Ha->eV * Bohr->Ang)
    nlo_conversion=0.08 * pi * sqrt(2 * CONSTANTS.ry_to_ev * CONSTANTS.bohr_to_ang),
    cm_to_kelvin=units.CmToEv / units.Kb,
    thz_to_cm=units.THzToCm,
    debey_ang=1.0 / 0.2081943  # 1 Debey = 0.2081943 e * Angstrom ==> e = (1/0.2081943) D/A
    # to be defined:
    # * kelvin to eV
    # * nm to eV
    # * THz to eV
)
