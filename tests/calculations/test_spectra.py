# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Tests for :mod:`calculations.spectra_utils`."""
# yapf:disable
import numpy as np
import pytest

DEBUG = True


@pytest.fixture
def generate_phonopy_instance():
    """Return AlAs Phonopy instance.

    It contains:
    -  force constants in 2x2x2 supercell
    - born charges and dielectric tensors
    - symmetry info
    """

    def _generate_phonopy_instance():
        """Return AlAs Phonopy instance."""
        import os

        import phonopy

        filename = 'phonopy_AlAs.yaml'
        basepath = os.path.dirname(os.path.abspath(__file__))
        phyaml = os.path.join(basepath, filename)

        return phonopy.load(phyaml)

    return _generate_phonopy_instance


@pytest.fixture
def generate_third_rank_tensors():
    """Return tuple of AlAs third order tensors.

    Units:
    - Raman in 1/Angstrom
    - Chi2 in pm/V
    """

    def _generate_third_rank_tensors():
        """Return AlAs Phonopy instance."""
        c = 42.4621905  # pm/V
        chi2 = np.array([[
                [0, 0, 0],
                [0, 0, c],
                [0, 0, 0]],
            [
                [0, 0, c],
                [0, 0, 0],
                [0, 0, 0]
            ],
            [
                [0, c, 0],
                [c, 0, 0],
                [0, 0, 0]
            ]
        ])

        a = 3.52026291e-02  # 1/Ang
        raman = np.array([
            [
                [[0, 0, 0], [0, 0, -a], [0, -a, 0]],
                [[0, 0, -a],[0, 0, 0],[-a, 0, 0]],
                [[0, -a, 0], [-a, 0, 0],[0, 0, 0]],
            ],
            [
                [[0, 0, 0], [0, 0, a], [0, a, 0]],
                [[0, 0, a], [0, 0, 0], [a, 0, 0]],
                [[0, a, 0], [a, 0, 0], [0, 0, 0]],
            ]
        ])

        return raman, chi2

    return _generate_third_rank_tensors


def test_compute_raman_susceptibility_tensors(generate_phonopy_instance, generate_third_rank_tensors):
    """Test the `compute_raman_susceptibility_tensors` function.

    For cubic semiconductors AB, the Raman susceptibility for phonons polarized along the `l`
    direction can be written as sqrt(mu*Omega)*alpha_{12} = Omega dChi_{12}/dtau_{A,3},
    where A is the atom located at the origin, and the atom B in (1/4,1/4,1/4).

    As a consequence, we can test the implementation when specifying a q-direction (Cartesian).
    """
    from aiida_vibroscopy.calculations.spectra_utils import compute_raman_susceptibility_tensors
    from aiida_vibroscopy.common.constants import DEFAULT

    ph = generate_phonopy_instance()
    ph.symmetrize_force_constants()
    vol = ph.unitcell.volume
    raman, chi2 = generate_third_rank_tensors()

    reduced_mass = (26.981539 * 74.9216) / (26.981539 + 74.9216)
    prefactor = np.sqrt(reduced_mass * ph.unitcell.volume)

    alpha, _, _ = compute_raman_susceptibility_tensors(
        phonopy_instance=ph,
        raman_tensors=raman,
        nlo_susceptibility=chi2,
        nac_direction=[1, 0, 0],
    )

    alpha_comp = prefactor * alpha[2, 1, 2]
    alpha_theo = vol * raman[0, 0, 1, 2]
    if DEBUG:
        print('\n', '================================', '\n')
        print((prefactor * alpha).round(3))
        print('\t', 'DEBUG')
        print(alpha_comp, alpha_theo)
        print('\n', '================================', '\n')

    assert np.abs(abs(alpha_comp) - abs(alpha_theo)) < 1e-5

    alpha, _, _ = compute_raman_susceptibility_tensors(
        phonopy_instance=ph,
        raman_tensors=raman,
        nlo_susceptibility=chi2,
        nac_direction=[0, 0, 1],
    )
    diel = ph.nac_params['dielectric']
    borns = ph.nac_params['born']

    nlocorr = DEFAULT.nlo_conversion * borns[1, 2, 2] * chi2[0, 1, 2] / diel[2, 2]
    alpha_theo = vol * raman[1, 0, 1, 2] - nlocorr

    # we take the last, cause it is associated to the LO mode
    alpha_comp = prefactor * alpha[2, 0, 1]
    if DEBUG:
        print('\n', '================================', '\n')
        print('\t', 'DEBUG')
        print((prefactor * alpha).round(3))
        print('NLO corr. expected: ', nlocorr)
        print('Born corr. expected: ', -borns[1, 0, 0] / np.sqrt(reduced_mass))
        print('Conversion factor nlo: ', DEFAULT.nlo_conversion)
        print(alpha_comp, alpha_theo)
        print('\n', '================================', '\n')

    assert np.abs(abs(alpha_comp) - abs(alpha_theo)) < 1e-3


def test_compute_methods(generate_phonopy_instance, generate_third_rank_tensors, ndarrays_regression):
    """Test the post-processing methods with data regression techniques."""
    from aiida_vibroscopy.calculations.spectra_utils import (
        compute_active_modes,
        compute_complex_dielectric,
        compute_raman_space_average,
        compute_raman_susceptibility_tensors,
    )

    results = {}
    ph = generate_phonopy_instance()
    ph.symmetrize_force_constants()
    raman, chi2 = generate_third_rank_tensors()

    freqs, _, _ = compute_active_modes(phonopy_instance=ph)
    results['active_modes_freqs'] = freqs
    # results['active_modes_eigvecs'] = eigenvectors

    freqs, _ , _ = compute_active_modes(phonopy_instance=ph, nac_direction=[0,0,1])
    results['active_modes_nac_freqs'] = freqs
    # results['active_modes_nac_eigvecs'] = eigenvectors

    alpha, _, _ = compute_raman_susceptibility_tensors(ph, raman, chi2)
    ints_hh, ints_hv = compute_raman_space_average(alpha)
    # results['raman_susceptibility_tensors'] = alpha
    results['intensities_hh'] = ints_hh
    results['intensities_hv'] = ints_hv

    # alpha, _, _ = compute_raman_susceptibility_tensors(ph, raman, chi2, nac_direction=[0,0,1])
    # results['raman_susceptibility_tensors_nac'] = alpha

    # pols, _, _ = compute_polarization_vectors(ph)
    # results['polarization_vectors'] = pols

    # pols, _, _ = compute_polarization_vectors(ph, nac_direction=[0,0,1])
    # results['polarization_vectors_nac'] = pols

    freq_range = np.linspace(10,1000,900)
    eps = compute_complex_dielectric(ph, freq_range=freq_range)
    results['complex_dielectric'] = eps

    eps = compute_complex_dielectric(ph, nac_direction=[0,0,1], freq_range=freq_range)
    results['complex_dielectric_nac'] = eps

    ndarrays_regression.check(results, default_tolerance=dict(atol=1e-4, rtol=1e-4))


def test_generate_vibrational_data_from_forces(generate_vibrational_data_from_forces, ndarrays_regression):
    """Test `generate_vibrational_data_from_phonopy`."""
    vibro = generate_vibrational_data_from_forces()

    results = {
        'dielectric': vibro.dielectric,
        'raman': vibro.raman_tensors,
        'nlo': vibro.nlo_susceptibility,
        'becs': vibro.born_charges,
        'forces': vibro.forces,
    }
    ndarrays_regression.check(results, default_tolerance=dict(atol=1e-8, rtol=1e-8))
