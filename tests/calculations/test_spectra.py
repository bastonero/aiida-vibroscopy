# -*- coding: utf-8 -*-
"""Tests for :mod:`calculations.spectra_utils`."""
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

        ph = phonopy.load(phyaml)

        return ph

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
        chi2 = np.array([[[-1.42547451e-50, -4.81482486e-35, 1.36568821e-14],
                          [-4.81482486e-35, 0.00000000e+00, 4.24621905e+01],
                          [1.36568821e-14, 4.24621905e+01, 5.20011857e-15]],
                         [[-3.20988324e-35, 0.00000000e+00, 4.24621905e+01],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                          [4.24621905e+01, 0.00000000e+00, 5.20011857e-15]],
                         [[1.36568821e-14, 4.24621905e+01, 5.20011857e-15],
                          [4.24621905e+01, -2.40741243e-35, 5.20011857e-15],
                          [5.20011857e-15, 5.20011857e-15, 9.55246283e-31]]])

        raman = np.array([[[[7.82438427e-38, -1.38120586e-37, -1.13220290e-17],
                            [-1.37630797e-37, -5.64237288e-37, -3.52026291e-02],
                            [-1.13220290e-17, -3.52026291e-02, -4.31107870e-18]],
                           [[-4.23177966e-37, -7.99336159e-37, -3.52026291e-02],
                            [-5.48564030e-37, -4.99585099e-38, 1.47328625e-36],
                            [-3.52026291e-02, 1.77891478e-36, -4.31107870e-18]],
                           [[-1.13220290e-17, -3.52026291e-02, -4.31107870e-18],
                            [-3.52026291e-02, -2.66445386e-37, -4.31107870e-18],
                            [-4.31107870e-18, -4.31107870e-18, -7.91497448e-34]]],
                          [[[-1.01998624e-37, -1.60357021e-36, 1.13220290e-17],
                            [-1.45026616e-36, 2.31964219e-36, 3.52026291e-02],
                            [1.13220290e-17, 3.52026291e-02, 4.31107870e-18]],
                           [[7.67989643e-37, -7.36643127e-37, 3.52026291e-02],
                            [-4.23177966e-37, -1.59671316e-37, -7.20969869e-37],
                            [3.52026291e-02, -5.09380885e-37, 4.31107870e-18]],
                           [[1.13220290e-17, 3.52026291e-02, 4.31107870e-18],
                            [3.52026291e-02, -2.66445386e-37, 4.31107870e-18],
                            [4.31107870e-18, 4.31107870e-18, 7.91868953e-34]]]])
        return raman, chi2

    return _generate_third_rank_tensors


def test_compute_raman_susceptibility_tensors(generate_phonopy_instance, generate_third_rank_tensors):
    """Test the `compute_raman_susceptibility_tensors` function."""
    from aiida_vibroscopy.calculations.spectra_utils import compute_raman_susceptibility_tensors
    from aiida_vibroscopy.common.constants import DEFAULT

    ph = generate_phonopy_instance()
    vol = ph.unitcell.volume
    raman, chi2 = generate_third_rank_tensors()

    reduced_mass = (26.981539 * 74.9216) / (26.981539 + 74.9216)
    prefactor = np.sqrt(reduced_mass * ph.unitcell.volume)

    alpha, _, _ = compute_raman_susceptibility_tensors(
        phonopy_instance=ph,
        raman_tensors=raman,
        nlo_susceptibility=chi2,
        nac_direction=(0, 0, 0),
    )

    if DEBUG:
        print('\n', '================================', '\n')
        print('\t', 'DEBUG')
        print(prefactor * alpha[1, 1, 2], vol * raman[1, 0, 1, 2])
        print('\n', '================================', '\n')

    assert np.abs(prefactor * alpha[1, 1, 2] + vol * raman[1, 0, 1, 2]) < 0.01

    alpha, _, _ = compute_raman_susceptibility_tensors(
        phonopy_instance=ph,
        raman_tensors=raman,
        nlo_susceptibility=chi2,
        nac_direction=np.dot(ph.primitive.cell, [0, 0, 1]),
    )
    diel = ph.nac_params['dielectric']
    borns = ph.nac_params['born']

    dchivol = vol * raman[1, 0, 1, 2] - DEFAULT.nlo_conversion * borns[1, 0, 0] * chi2[0, 1, 2] / diel[0, 0]

    if DEBUG:
        print('\n', '================================', '\n')
        print('\t', 'DEBUG')
        print('NLO corr. expected: ', -DEFAULT.nlo_conversion * borns[1, 0, 0] * chi2[0, 1, 2] / diel[0, 0])
        print('Born corr. expected: ', -borns[1, 0, 0] / np.sqrt(reduced_mass))
        print('Conversion factor nlo: ', DEFAULT.nlo_conversion)
        # print(prefactor * alpha)
        print(dchivol, prefactor * np.abs(alpha).max())
        print('\n', '================================', '\n')

    assert np.abs(prefactor * alpha[2, 0, 1] - dchivol) < 0.01
