# -*- coding: utf-8 -*-
"""Symmetry utils for tensors."""
import numpy as np
from phonopy.harmonic.force_constants import similarity_transformation

__all__ = (
    'tensor_3rd_rank_transformation',
    'symmetrize_3nd_rank_tensor',
    'take_average_of_dph0',
    'symmetrize_susceptibility_derivatives',
)


def tensor_3rd_rank_transformation(rot, mat):
    """Tensor transformation."""
    return np.tensordot(
                rot, np.tensordot(
                    rot, np.tensordot(
                        rot, mat,
                        axes=[1,0]
                    ),
                    axes=[1,1]
                    ),
                axes=[1,2]
            )


def symmetrize_3nd_rank_tensor(tensor, symmetry_operations, lattice):
    """Symmetrize a 3rd rank tensor using symmetry operations in the lattice."""
    sym_cart = [similarity_transformation(lattice.T, r) for r in symmetry_operations]
    sum_tensor = np.zeros_like(tensor)
    for sym in sym_cart:
        sum_tensor += tensor_3rd_rank_transformation(sym, tensor)
    return sum_tensor / len(symmetry_operations)


def take_average_of_dph0(dchi_ph0, rotations, translations, cell, symprec):
    """Symmetrize :math:`\\frac{d \\Chi}{dr}` tensors in respect to space group symmetries
    and applies sum rules (as in *M. Veiten et al., PRB, 71, 125107, (2005)*)."""
    lattice = cell.cell
    positions = cell.scaled_positions
    dchi_ph0_ = np.zeros_like(dchi_ph0)

    for i in range(len(dchi_ph0)):
        for r, t in zip(rotations, translations):
            diff = np.dot(positions, r.T) + t - positions[i]
            diff -= np.rint(diff)
            dist = np.sqrt(np.sum(np.dot(diff, lattice) ** 2, axis=1))
            j = np.nonzero(dist < symprec)[0][0]
            r_cart = similarity_transformation(lattice.T, r)
            dchi_ph0_[i] += tensor_3rd_rank_transformation(r_cart, dchi_ph0[j])
        dchi_ph0_[i] /= len(rotations)

    # Apply simple sum rules as in `M. Veiten et al., PRB, 71, 125107 (2005)`
    sum_dchi = dchi_ph0_.sum(axis=0) / len(dchi_ph0_) # sum over atomic index
    dchi_ph0_ -= sum_dchi

    return dchi_ph0_


def symmetrize_susceptibility_derivatives(
    dph0_susceptibility,
    nlo_susceptibility,
    ucell,
    primitive_matrix=None,
    primitive=None,
    supercell_matrix=None,
    symprec=1e-5,
    is_symmetry=True,
):
    """Symmetrize susceptibility derivatives tensors (dChi/dr and Chi^2).

    :param dph0_susceptibility: array_like dChi/dr tensors, shape=(unitcell_atoms, 3, 3, 3).
        Convention is to have the symmetric elements over the 2 and 3 indices,
        i.e. [I,k,i,j] = [I,k,j,i] <==> k is the index of the displacement.
    :param nlo_susceptibility: array_like Chi^2 tensors, shape=(3, 3, 3)
    :param ucell: PhonopyAtoms unit cell
    :param primitive_matrix: array_like, optional
        Primitive matrix. This is used to select dChi/dr tensors in
        primitive cell. If None (default), dChi/dr tensors in unit cell
        are returned.
        shape=(3, 3)
    :param primitive: PhonopyAtoms
        This is an alternative of giving primitive_matrix (Mp). Mp is given as
            Mp = (a_u, b_u, c_u)^-1 * (a_p, b_p, c_p).
        In addition, the order of atoms is alined to those of atoms in this
        primitive cell for Born effective charges. No rigid rotation of
        crystal structure is assumed.
    :param supercell_matrix: array_like, optional
        Supercell matrix. This is used to select Born effective charges in
        **primitive cell**. Supercell matrix is needed because primitive
        cell is created first creating supercell from unit cell, then
        the primitive cell is created from the supercell. If None (defautl),
        1x1x1 supercell is created.
        shape=(3, 3)
    :param symprec: float, optional
        Symmetry tolerance. Default is 1e-5
    :param is_symmetry: bool, optinal
        By setting False, symmetrization can be switched off. Default is True.

    :return: symmetrized tensors (dph0, nlo)
    """
    from phonopy.structure.symmetry import Symmetry, _get_supercell_and_primitive, _get_mapping_between_cells

    lattice = ucell.cell
    u_sym = Symmetry(ucell, is_symmetry=is_symmetry, symprec=symprec)
    rotations = u_sym.symmetry_operations['rotations']
    translations = u_sym.symmetry_operations['translations']
    ptg_ops = u_sym.pointgroup_operations

    transpose_dph0 = np.transpose(dph0_susceptibility, axes=[0,2,1,3]) # transpose i <--> k

    nlo_ = symmetrize_3nd_rank_tensor(nlo_susceptibility, ptg_ops, lattice)
    tdph0_ = take_average_of_dph0(transpose_dph0, rotations, translations, ucell, symprec)
    dph0_ = np.transpose(tdph0_, axes=[0,2,1,3]) # transpose back i <--> k

    if (abs(dph0_susceptibility - dph0_) > 0.1).any():
        lines = [
            'Symmetry of dChi/dr tensors is largely broken. '
            'The max difference is:',
            f'{(dph0_susceptibility - dph0_).max()}',
        ]
        import warnings

        warnings.warn('\n'.join(lines))

    if primitive_matrix is None and primitive is None:
        return dph0_, nlo_
    else:
        if primitive is not None:
            pmat = np.dot(np.linalg.inv(ucell.cell.T), primitive.cell.T)
        else:
            pmat = primitive_matrix

        scell, pcell = _get_supercell_and_primitive(
            ucell,
            primitive_matrix=pmat,
            supercell_matrix=supercell_matrix,
            symprec=symprec,
        )

        idx = [scell.u2u_map[i] for i in scell.s2u_map[pcell.p2s_map]]
        dph0_in_prim = dph0_[idx].copy()

        if primitive is None:
            return dph0_in_prim, nlo_
        else:
            idx2 = _get_mapping_between_cells(pcell, primitive)
            return dph0_in_prim[idx2].copy(), nlo_
