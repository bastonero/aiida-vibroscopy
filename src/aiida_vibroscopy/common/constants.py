# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Physical and conversion factor constants."""
# Whenever possible, we try to use the constants defined in
# :py:mod:`qe_tools._constants`, but some of them are missing.
from types import SimpleNamespace

import numpy as np
from phonopy import units
from qe_tools import CONSTANTS

__all__ = ('DEFAULT',)

# We here refer to SI to the general convention in material science of:
# * Length ==> in Angstrom
# * Energy ==> in eV
# * Mass   ==> AMU

DEFAULT = SimpleNamespace(
    eVinv_to_ang=12398.487384539087,  # eV^-1 to angstrom
    efield_au_to_si=51.4220674763 / np.sqrt(2),  # Ry a.u. to Volt/angstrom
    evang_to_rybohr=(CONSTANTS.bohr_to_ang / CONSTANTS.ry_to_ev),  # eV/ang to Ry/bohr
    # The nlo_conversion is for the correction to the Raman tensor dChi/dtau.
    # Since we store Chi(2) as pm/Volt, then we need to mulitply by 1/100 to get Angstrom/Volt.
    # Then, a pre-factor of 8*pi is present too, of which 4*pi is for the 1/epsilon0 term.
    # Lastly, we have the Born charge, corresponding to an electronic charge in atomic units.
    # This means that in order to obtain the units of Angstrom and Volt, we multiply by (Ha->eV * Bohr->Ang).
    # The exrta 'e' comes from the Ang/V = e * (Ang/eV) of Chi(2).
    nlo_conversion=0.02 *
    (2.0 * CONSTANTS.ry_to_ev * CONSTANTS.bohr_to_ang),  # we remove 4pi and put it back in the cross sections
    cm_to_kelvin=units.CmToEv / units.Kb,
    thz_to_cm=units.THzToCm,  # THz to cm^-1
    debey_ang=1.0 / 0.2081943,  # 1 Debey = 0.2081943 e * Angstrom ==> e = (1/0.2081943) D/A
    # The absolute theoretical Raman cross-section per unit volume is obtained with the
    # following conversion factor, expressing the intensities as:
    # > sterad^-1 cm^-2 <
    # The intensity must be computed using frequencies in cm^-1 and normalized eigenvectors
    # by atomic masses expressed in atomic mass unit (Dalton).
    # IMPORTANT: still misss the units from the Dirac delta
    raman_xsection=1.0e24 * 1.054571817e-34 /
    (2.0 * units.SpeedOfLight**4 * units.AMU *
     units.THzToCm**3),  # removed 1/4pi due to convention on Chi2 for the correction
    elementary_charge_si=1.602176634e-19,  # elementary charge in Coulomb
    electron_mass_si=units.Me,  # electron mass in kg
    atomic_mass_si=units.AMU,  # atomic mass unit in kg
    # to be defined:
    # * kelvin to eV
    # * nm to eV
    # * THz to eV
)
