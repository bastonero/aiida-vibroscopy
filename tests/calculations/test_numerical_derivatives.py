# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Tests for :mod:`calculations.spectra_utils`."""
import numpy as np
import pytest

DEBUG = False


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
def generate_trajectory():
    """Return a `TrajectoryData` node."""

    def _generate_trajectory(scale=1, index=2):
        """Return a `TrajectoryData` with AlAs data."""
        from aiida.orm import TrajectoryData
        import numpy as np

        node = TrajectoryData()
        if index == 2:
            polarization = scale * np.array([[-4.88263729e-09, 6.84208048e-09, 1.67517339e-01]])
            forces = scale * np.array([[
                [-0.00000000e+00, -0.00000000e+00, 1.95259855e-02], [-0.00000000e+00, 0.00000000e+00, 1.95247000e-02],
                [-0.00000000e+00, -0.00000000e+00, 1.95247000e-02], [-0.00000000e+00, 0.00000000e+00, 1.95262427e-02],
                [-1.25984053e-05, -1.25984053e-05, -1.95383268e-02], [-1.31126259e-05, 1.31126259e-05, -1.95126158e-02],
                [1.28555156e-05, 1.25984053e-05, -1.95383268e-02], [1.31126259e-05, -1.31126259e-05, -1.95126158e-02]
            ]])
        if index == 3:
            polarization = scale * np.array([[9.55699034e-05, 1.18453183e-01, 1.18453160e-01]])
            forces = scale * np.array([
                [[-1.82548322e-05, 1.38068238e-02, 1.38068238e-02], [-1.82548322e-05, 1.38060524e-02, 1.38060524e-02],
                 [-1.82548322e-05, 1.38070809e-02, 1.38060524e-02], [-1.82548322e-05, 1.38060524e-02, 1.38070809e-02],
                 [5.39931655e-06, -1.38194222e-02, -1.38194222e-02], [5.14220624e-06, -1.37937111e-02, -1.37937111e-02],
                 [3.11103478e-05, -1.37937111e-02, -1.38194222e-02], [3.11103478e-05, -1.38194222e-02, -1.37937111e-02]]
            ])

        node.set_array('forces', forces)
        node.set_array('electronic_dipole_cartesian_axes', polarization)

        stepids = np.array([1])
        times = stepids * 0.0
        cells = np.array([5.62475444 * np.eye(3)])
        positions = np.array([[
            [
                0.,
                0.,
                0.,
            ],
            [
                0.,
                2.81237722,
                2.81237722,
            ],
            [
                2.81237722,
                0.,
                2.81237722,
            ],
            [
                2.81237722,
                2.81237722,
                0.,
            ],
            [
                1.40621634,
                1.40621634,
                1.40621634,
            ],
            [
                1.40621634,
                4.21853809,
                4.21853809,
            ],
            [
                4.21853809,
                4.21853809,
                1.40621634,
            ],
            [
                4.21853809,
                1.40621634,
                4.21853809,
            ],
        ]])
        symbols = ['Al', 'Al', 'Al', 'Al', 'As', 'As', 'As', 'As']
        node.set_trajectory(stepids=stepids, cells=cells, symbols=symbols, positions=positions, times=times)

        return node.store()

    return _generate_trajectory


def test_compute_tensors(generate_phonopy_instance, generate_trajectory):
    """Test the functions for computing tensors."""
    from aiida.orm import Float, Int
    from aiida_phonopy.data.preprocess import PreProcessData
    from qe_tools import CONSTANTS as C

    from aiida_vibroscopy.calculations.numerical_derivatives_utils import (
        compute_nac_parameters,
        compute_susceptibility_derivatives,
    )

    ph = generate_phonopy_instance()
    diagonal_scale = 1 / np.sqrt(2)
    volume = ph.unitcell.volume
    volume_au = ph.unitcell.volume / C.bohr_to_ang**3
    preprocess_data = PreProcessData(phonopy_atoms=ph.unitcell)

    accuracy = 2
    data = {'field_index_2': {'0': generate_trajectory(index=2)}, 'field_index_3': {'0': generate_trajectory(index=3)}}

    bec_ref = data['field_index_2']['0'].get_array('forces')[0, 0, 2] / 2.5e-4 / np.sqrt(2) * C.bohr_to_ang / C.ry_to_ev
    diel_ref = 4. * np.pi * data['field_index_2']['0'].get_array('electronic_dipole_cartesian_axes')[
        0, 2] / 2.5e-4 / volume_au + 1

    raman_ref = data['field_index_3']['0'].get_array('forces')[
        0, 0, 0] / (diagonal_scale * 2.5e-4)**2 / C.ry_to_ev / volume_au

    results = compute_nac_parameters(preprocess_data, Float(2.5e-4), Int(accuracy), **data)
    for i in range(3):
        bec = results['numerical_accuracy_2'].get_array('born_charges')[0, i, i]
        diel = results['numerical_accuracy_2'].get_array('dielectric')[i, i]
        assert np.abs(bec - bec_ref) < 1e-4
        assert np.abs(diel - diel_ref) < 1e-4
        if i != 2:
            bec = results['numerical_accuracy_2'].get_array('born_charges')[0, i, 2]
            diel = results['numerical_accuracy_2'].get_array('dielectric')[i, 2]
            assert np.abs(bec) < 1e-5
            assert np.abs(diel) < 1e-5

    results = compute_susceptibility_derivatives(
        preprocess_data, Float(2.5e-4), Float(diagonal_scale), Int(accuracy), **data
    )
    raman = results['numerical_accuracy_2'].get_array('raman_tensors')[0, 0, 1, 2]
    assert np.abs(raman - raman_ref) < 1e-4

    accuracy = 2
    data = {
        'field_index_0': {
            '0': generate_trajectory(index=2),
            '1': generate_trajectory(scale=-1, index=2)
        },
        'field_index_1': {
            '0': generate_trajectory(index=2),
            '1': generate_trajectory(scale=-1, index=2)
        },
        'field_index_2': {
            '0': generate_trajectory(index=2),
            '1': generate_trajectory(scale=-1, index=2)
        },
        'field_index_3': {
            '0': generate_trajectory(index=3),
            '1': generate_trajectory(scale=-1, index=3)
        },
        'field_index_4': {
            '0': generate_trajectory(index=3),
            '1': generate_trajectory(scale=-1, index=3)
        },
        'field_index_5': {
            '0': generate_trajectory(index=3),
            '1': generate_trajectory(scale=-1, index=3)
        },
    }

    preprocess_data = PreProcessData(phonopy_atoms=ph.unitcell, is_symmetry=False, symprec=1e-4)
    compute_nac_parameters(preprocess_data, Float(2.5e-4), Int(accuracy), **data)
    compute_susceptibility_derivatives(preprocess_data, Float(2.5e-4), Float(np.sqrt(2)), Int(accuracy), **data)


@pytest.mark.parametrize(
    'inputs,result', (
        ((2, 1), [1 / 2, 0.]),
        ((4, 1), [2 / 3, -1 / 12, 0.]),
        ((8, 1), [4 / 5, -1 / 5, 4 / 105, -1 / 280, 0.]),
        ((2, 2), [1., -2.]),
        ((4, 2), [4 / 3, -1 / 12, -5 / 2]),
        ((8, 2), [8 / 5, -1 / 5, 8 / 315, -1 / 560, -205 / 72]),
    )
)
def test_get_central_derivatives_coefficients(inputs, result):
    """Test the numerical derivatives coefficients.

    Exact values taken from Wikipedia: https://en.wikipedia.org/wiki/Finite_difference_coefficient
    """
    import numpy as np

    from aiida_vibroscopy.calculations.numerical_derivatives_utils import get_central_derivatives_coefficients

    array1 = np.array(get_central_derivatives_coefficients(*inputs))
    array2 = np.array(result)
    assert np.abs(array1 - array2).max() < 1.0e-10
