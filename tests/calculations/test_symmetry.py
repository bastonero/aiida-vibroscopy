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
def generate_trajectory():
    """Return a `TrajectoryData` node."""

    def _generate_trajectory(scale=1):
        """Return a `TrajectoryData` with AlAs data."""
        from aiida.orm import TrajectoryData
        import numpy as np

        # yapf: disable
        node = TrajectoryData()
        polarization = scale * np.array([[-4.88263729e-09, 6.84208048e-09, 1.67517339e-01]])
        node.set_array('electronic_dipole_cartesian_axes', polarization)
        forces = scale * np.array(
            [[[-0.00000000e+00, -0.00000000e+00, 1.95259855e-02],
              [-0.00000000e+00, 0.00000000e+00, 1.95247000e-02],
              [-0.00000000e+00, -0.00000000e+00, 1.95247000e-02],
              [-0.00000000e+00, 0.00000000e+00, 1.95262427e-02],
              [-1.25984053e-05, -1.25984053e-05, -1.95383268e-02],
              [-1.31126259e-05, 1.31126259e-05, -1.95126158e-02],
              [1.28555156e-05, 1.25984053e-05, -1.95383268e-02],
              [1.31126259e-05, -1.31126259e-05, -1.95126158e-02]]
        ])
        node.set_array('forces', forces)

        stepids = np.array([1])
        times = stepids * 0.0
        cells = np.array([5.62475444 * np.eye(3)])
        positions = np.array([[
            [ 0., 0., 0. ],
            [ 0., 2.81237722, 2.81237722 ],
            [ 2.81237722, 0., 2.81237722 ],
            [ 2.81237722, 2.81237722, 0. ],
            [ 1.40621634, 1.40621634, 1.40621634 ],
            [ 1.40621634, 4.21853809, 4.21853809 ],
            [ 4.21853809, 4.21853809, 1.40621634 ],
            [ 4.21853809, 1.40621634, 4.21853809 ],
        ]])
        # yapf: enable
        symbols = ['Al', 'Al', 'Al', 'Al', 'As', 'As', 'As', 'As']
        node.set_trajectory(stepids=stepids, cells=cells, symbols=symbols, positions=positions, times=times)

        return node.store()

    return _generate_trajectory


def test_transform_trajectory(generate_phonopy_instance, generate_trajectory):
    """Test the `compute_raman_susceptibility_tensors` function."""
    from aiida_vibroscopy.calculations.symmetry import transform_trajectory

    ph = generate_phonopy_instance()  # here z
    rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]])  # to -z
    trans = np.array([0, 0, 0])
    traj = generate_trajectory()
    forces = traj.get_array('forces')

    new_traj = transform_trajectory(traj, np.array([0, 0, 0]), rot, trans, ph.unitcell, 1e-5)
    new_forces = new_traj.get_array('forces')
    assert np.abs(forces + new_forces).max() < 5.0e-5  # eV/Ang

    rot = np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]])  # to -y
    new_traj = transform_trajectory(traj, np.array([0, 0, 0]), rot, trans, ph.unitcell, 1e-5)
    new_forces = new_traj.get_array('forces')
    assert np.abs(forces[-1, :, 2] + new_forces[-1, :, 1]).max() < 5.0e-5  # eV/Ang


def test_get_connected_field_with_operations(generate_phonopy_instance):
    """Test the `get_connected_fields_with_operations` method."""
    from aiida_vibroscopy.calculations.symmetry import get_connected_fields_with_operations

    ph = generate_phonopy_instance()
    direction = [0, 0, 1]
    eq_directions, _, _ = get_connected_fields_with_operations(ph, direction)

    for array in [[0, 0, -1], [0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]:
        assert array in eq_directions


def test_get_irreducible_numbers_and_signs(generate_phonopy_instance):
    """Test the `get_irreducible_numbers_and_signs` method."""
    from aiida_phonopy.data.preprocess import PreProcessData

    from aiida_vibroscopy.calculations.symmetry import get_irreducible_numbers_and_signs

    ph = generate_phonopy_instance()
    preprocess_data = PreProcessData(phonopy_atoms=ph.unitcell)

    irr_numbers, irr_signs = get_irreducible_numbers_and_signs(preprocess_data, 6)

    assert irr_numbers == [2, 3]
    assert irr_signs == [[True, False], [True, False]]


def test_get_trajectories_from_symmetries(generate_phonopy_instance, generate_trajectory):
    """Test the `get_trajectories_from_symmetries` method."""
    from aiida_phonopy.data.preprocess import PreProcessData

    from aiida_vibroscopy.calculations.symmetry import get_trajectories_from_symmetries

    ph = generate_phonopy_instance()
    preprocess_data = PreProcessData(phonopy_atoms=ph.unitcell)

    accuracy = 2
    data = {'field_index_2': {'0': generate_trajectory()}}
    data_0 = generate_trajectory(scale=0)

    trajs = get_trajectories_from_symmetries(preprocess_data, data, data_0, accuracy)

    for i in range(3):
        assert f'field_index_{i}' in trajs
        for j in range(2):
            assert str(j) in trajs[f'field_index_{i}']

    for i in range(3):
        forces_pz = trajs[f'field_index_{i}']['0'].get_array('forces')
        forces_mz = trajs[f'field_index_{i}']['1'].get_array('forces')
        assert np.abs(forces_pz + forces_mz).max() < 5.0e-5  # eV/Ang

    accuracy = 4
    data = {'fields_data_2': {'0': generate_trajectory(), '1': generate_trajectory(scale=2)}}
    data_0 = generate_trajectory(scale=0)

    trajs = get_trajectories_from_symmetries(preprocess_data, data, data_0, accuracy)

    for i in range(3):
        assert f'field_index_{i}' in trajs
        for j in range(2):
            assert str(j) in trajs[f'field_index_{i}']

    for i in range(3):
        forces_pz = trajs[f'field_index_{i}']['2'].get_array('forces')[-1, :, 2]
        forces_mz = trajs[f'field_index_{i}']['3'].get_array('forces')[-1, :, 2]
        assert np.abs(forces_pz + forces_mz).max() < 1.0e-4  # eV/Ang
        # Note: the higher thr for the diff is due to little distortion
