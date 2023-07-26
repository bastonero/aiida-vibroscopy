# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Calculation function to compute a k-point mesh for a structure with a directional k-point distance."""
from aiida.engine import calcfunction


@calcfunction
def create_directional_kpoints(structure, direction, parallel_distance, orthogonal_distance, force_parity):
    """Generate a `directional` spaced kpoint mesh for a given structure.

    The spacing between kpoints in reciprocal space is guaranteed to be at least the defined distance.

    :param structure: the StructureData to which the mesh should apply
    :param direction: a (3,) shape List representing the direction of the electric field
    :param parallel_distance: a Float with the desired distance between paralell kpoints to
        the electric field in reciprocal space
    :param orthogonal_distance: a Float with the desired distance between orthogonal kpoints
        to the electric field in reciprocal space
    :param force_parity: a Bool to specify whether the generated mesh should maintain parity
    :returns: a KpointsData with the generated mesh
    """
    from aiida.orm import KpointsData
    import numpy as np

    epsilon = 1E-5

    kpoints_ortho = KpointsData()
    kpoints_ortho.set_cell_from_structure(structure)
    kpoints_ortho.set_kpoints_mesh_from_density(orthogonal_distance.value, force_parity=force_parity.value)

    pymat = structure.get_pymatgen_structure()
    # Define weights upond projection lenght
    weigths = np.abs(pymat.lattice.get_vector_along_lattice_directions(direction))
    weigths /= weigths.max()

    mesh = kpoints_ortho.get_kpoints_mesh()[0]

    for i, weigth in enumerate(weigths):

        kpoints_para = KpointsData()
        kpoints_para.set_cell_from_structure(structure)
        distance = parallel_distance.value / weigth if weigth > epsilon else orthogonal_distance.value
        kpoints_para.set_kpoints_mesh_from_density(distance, force_parity=force_parity.value)
        mesh_para = kpoints_para.get_kpoints_mesh()[0]

        if mesh_para[i] > mesh[i]:
            mesh[i] = mesh_para[i]

    final_kpoints = KpointsData()
    final_kpoints.set_cell_from_structure(structure)
    final_kpoints.set_kpoints_mesh(mesh)

    return final_kpoints
