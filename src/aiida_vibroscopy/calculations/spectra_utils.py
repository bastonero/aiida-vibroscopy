# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Calcfunctions utils for spectra workflows."""
from __future__ import annotations

from copy import deepcopy
from typing import Tuple, Union

from aiida import orm
from aiida.engine import calcfunction
from aiida.plugins import DataFactory
from aiida_phonopy.data.preprocess import PreProcessData
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
import numpy as np
from phonopy import Phonopy

from aiida_vibroscopy.common import UNITS

__all__ = (
    'boson_factor',
    'compute_active_modes',
    'compute_raman_space_average',
    'compute_raman_susceptibility_tensors',
    'compute_polarization_vectors',
    'compute_complex_dielectric',
    'compute_clamped_pockels_tensor',
    'get_supercells_for_hubbard',
    'elaborate_susceptibility_derivatives',
    'generate_vibrational_data_from_forces',
    'generate_vibrational_data_from_force_constants',
)


def boson_factor(frequency: float, temperature: float) -> float:
    """Return boson factor, i.e. (nb+1). Frequency in cm-1 and temperature in Kelvin."""
    return 1.0 / (1.0 - np.exp(-UNITS.cm_to_kelvin * frequency / temperature))


def compute_active_modes(
    phonopy_instance: Phonopy,
    degeneracy_tolerance: float = 1.e-5,
    nac_direction: list[float, float, float] | None = None,
    selection_rule: str | None = None,
    sr_thr: float = 1e-4,
    imaginary_thr: float = -5.0,
) -> tuple[list, list, list]:
    """Get frequencies, normalized eigenvectors and irreducible representation labels.

    Raman and infrared active modes can be extracted using `selection_rule`.

    :param nac_direction: (3,) shape list, indicating non analytical
        direction in Cartesian coordinates
    :param selection_rule: str, can be `raman` or `ir`, it uses symmetry in
        the selection of the modes for a specific type of process
    :param sr_thr: float, threshold for selection rule (the analytical value is 0)
    :param imaginary_thr: threshold for activating warnings on negative frequencies (in cm^-1)

    :return: tuple of (frequencies in cm-1, normalized eigenvectors, labels),
        normalized eigenvectors is an array of shape (num modes, num atoms, 3).
    """
    if selection_rule not in ('raman', 'ir', None):
        raise ValueError('`selection_rule` can only be `ir` or `raman`.')

    q_reduced = None

    # Notation: columns convention; k = reciprocal; x = direct space; (p) = primitive; (u) = unitcell
    # C(p) = C(u) * M  ==> M = C(u)^-1 * C(p)
    # x(p) = M^-1 x(u) ==> k(p) = M^T k(u);
    # q_reduced = np.dot(phonopy_instance.primitive_matrix.T, nac_dir)  # in reduced/crystal (PRIMITIVE) coordinates
    if nac_direction is not None:
        q_reduced = np.dot(phonopy_instance.primitive.cell,
                           nac_direction) / (2. * np.pi)  # in reduced/crystal (PRIMITIVE) coordinates

    # Step 1 - set the irreducible representations and the phonons
    phonopy_instance.set_irreps(q=[0, 0, 0], nac_q_direction=q_reduced, degeneracy_tolerance=degeneracy_tolerance)
    irreps = phonopy_instance.irreps

    phonopy_instance.run_qpoints(q_points=[0, 0, 0], nac_q_direction=q_reduced, with_eigenvectors=True)
    frequencies = phonopy_instance.qpoints.frequencies[0] * UNITS.thz_to_cm
    eigvectors = phonopy_instance.qpoints.eigenvectors.T.real

    # Step 2 - getting the active modes with eigenvectors
    Xr = []
    for mat in irreps.conventional_rotations:
        Xr.append(mat.trace())
    Xr = np.array(Xr)

    freq_active_modes = []
    eigvectors_active_modes = []
    labels_active_modes = []

    bands_indices = irreps.band_indices
    characters = irreps.characters.real
    labels = irreps._get_ir_labels()  #pylint: disable=protected-access

    mode_index = 0

    for band_indices, Xi, label in zip(bands_indices, characters, labels):
        degeneracy = len(band_indices)
        if mode_index > 2:  # excluding the acustic modes
            # Using selection rules (symmetry) constrains
            if selection_rule is not None:
                if selection_rule == 'raman':
                    condition = np.dot(Xr * Xr, Xi)
                elif selection_rule == 'ir':
                    condition = np.dot(Xr, Xi)
            else:
                condition = 10  # a number > 0

            if np.abs(condition) > sr_thr:  # selection rule (thr for inaccuracies)

                for band_index in band_indices:
                    freq_active_modes.append(frequencies[band_index])
                    eigvectors_active_modes.append(eigvectors[band_index])
                    labels_active_modes.append(label)
        else:
            message = f'negative frequencies detected below {imaginary_thr} cm-1 in the first 3 modes'
            for band_index in band_indices:
                if frequencies[band_index] < imaginary_thr:
                    import warnings
                    warnings.warn(message)

        mode_index += degeneracy

    freq_active_modes = np.array(freq_active_modes)

    # Step 3 - getting normalized eigenvectors
    masses = phonopy_instance.masses
    sqrt_masses = np.array([[np.sqrt(mass)] for mass in masses])

    eigvectors_active_modes = np.array(eigvectors_active_modes)
    shape = (len(freq_active_modes), len(masses), 3)
    eigvectors_active_modes = eigvectors_active_modes.reshape(shape)
    norm_eigvectors_active_modes = np.array([eigv / sqrt_masses for eigv in eigvectors_active_modes])

    return (freq_active_modes, norm_eigvectors_active_modes, labels_active_modes)


def compute_raman_space_average(raman_susceptibility_tensors: np.ndarray) -> tuple[list, list]:
    """Return the space average for the polarized (HH) and depolarized (HV) configurations.

    See e.g.:
        * `Light scattering in solides II, M. Cardona`
        * `S. A. Prosandeev et al., Phys. Rev. B, 71, 214307 (2005)`

    :return: tuple of numpy.ndarray (intensities HH, intensities HV)
    """
    intensities_hh = []
    intensities_hv = []
    for R in raman_susceptibility_tensors:
        #
        # Alternative representation.
        #
        # a = R.trace() / 3.0
        # a2 = a * a
        # b2 = (
        #     0.5 * ((R[0][0] - R[1][1])**2 + (R[0][0] - R[2][2])**2 + (R[1][1] - R[2][2])**2) + 3. *
        #     (R[0][1]**2 + R[0][2]**2 + R[1][2]**2)
        # )
        # intensities_hh.append(a2 + 4 * b2 / 45)
        # intensities_hv.append(3 * b2 / 45)
        #
        G0 = (R.trace()**2) / 3.0
        G1 = 0.5 * ((R[0][1] - R[1][0])**2 + (R[0][2] - R[2][0])**2 + (R[1][2] - R[2][1])**2)
        G2 = (
            0.5 * ((R[0][1] + R[1][0])**2 + (R[0][2] + R[2][0])**2 + (R[1][2] + R[2][1])**2) + (1. / 3.) *
            ((R[0][0] - R[1][1])**2 + (R[0][0] - R[2][2])**2 + (R[1][1] - R[2][2])**2)
        )

        intensities_hh.append((10 * G0 + 4 * G2) / 30)
        intensities_hv.append((5 * G1 + 3 * G2) / 30)

    return (np.array(intensities_hh), np.array(intensities_hv))


def compute_raman_susceptibility_tensors(
    phonopy_instance: Phonopy,
    raman_tensors: np.ndarray,
    nlo_susceptibility: np.ndarray = None,
    nac_direction: list[float, float, float] | None = None,
    use_irreps: bool = True,
    degeneracy_tolerance: float = 1e-5,
    sum_rules: bool = False,
) -> tuple[list, list, list]:
    """Return the Raman susceptibility tensors, frequencies (cm-1) and labels.

    .. note::
        * Units of Raman susceptibility tensor are (Angstrom/4pi/AMU)^(1/2)
        * Unitcell volume for Raman tensor as normalization (in case non-primitive cell was used).

    :param phonopy_instance: Phonopy instance with non-analytical constants included
    :param nac_direction: non-analytical direction in Cartesian coordinates
    :param raman_tensors: dChi/du in Cartesian coordinates (in 1/Angstrom)
    :param nlo_susceptibility: non linear optical susceptibility
        in Cartesian coordinates (in pm/V)
    :param use_irreps: whether to use irreducible representations
        in the selection of modes, defaults to True
    :param degeneracy_tolerance: degeneracy tolerance for
        irreducible representation
    :param sum_rules: whether to apply sum rules to the Raman tensors

    :return: tuple of numpy.ndarray (Raman susc. tensors, frequencies, labels)
    """
    raman_tensors = deepcopy(raman_tensors)

    volume = phonopy_instance.unitcell.volume
    sqrt_volume = np.sqrt(volume)
    raman_tensors *= volume

    # rcell = np.linalg.inv(phonopy_instance.unitcell.cell) # as columns
    # q_cartesian = np.dot(rcell, nac_direction)  # in Cartesian coordinates
    q_cartesian = np.zeros((3)) if nac_direction is None else np.array(nac_direction)

    selection_rule = 'raman' if use_irreps else None

    if sum_rules:
        sum_rule_correction = raman_tensors.sum(axis=0) / len(raman_tensors)  # sum over atomic index
        raman_tensors -= sum_rule_correction

    freqs, neigvs, labels = compute_active_modes(
        phonopy_instance=phonopy_instance,
        nac_direction=q_cartesian,
        degeneracy_tolerance=degeneracy_tolerance,
        selection_rule=selection_rule
    )

    # Here we check we do not have empty array.
    # E.g. happening in cubic crystals, such as MgO
    if not neigvs.tolist():
        return (np.array([]) for _ in range(3))

    # neigvs  shape|indices = (num modes, num atoms, 3) | (n, I, k)
    # dChi/du shape|indices = (num atoms, 3, 3, 3) | (I, k, a, b)
    # The contraction is performed over I and k, resulting in (n, a, b) Raman tensors.
    raman_susceptibility_tensors = np.tensordot(neigvs, raman_tensors, axes=([1, 2], [0, 1]))

    if nlo_susceptibility is not None and q_cartesian.nonzero()[0].tolist():
        borns = phonopy_instance.nac_params['born']
        dielectric = phonopy_instance.nac_params['dielectric']
        # -8 pi (Z.q/q.epsilon.q)[I,k] Chi(2).q [a,b] is the correction to dph0.
        # The indices I, k to do the scalar product with the eigenvectors run over the Borns term.
        # nac_direction shape|indices = (3) | (a)
        # borns  shape|indices = (num atoms, 3, 3) | (I, a, k)
        # nlo    shape|indices = (3, 3, 3) | (a, b, k)

        # q.epsilon.q
        # !!! ---------------------- !!!
        #    Here we can extend to 1/2D models.
        # !!! ---------------------- !!!
        dielectric_term = np.dot(np.dot(dielectric, q_cartesian), q_cartesian)

        ### DEBUG
        # print("\n", "================================", "\n")
        # print("DEBUG")
        # print("q dir cart: ", q_cartesian)
        # print("nac: ", nac_direction)
        ### DEBUG

        # Z*.q
        borns_term_dph0 = np.tensordot(borns, q_cartesian, axes=(1, 0))  # (num atoms, 3) | (I, k)
        borns_term = np.tensordot(borns_term_dph0, neigvs, axes=([0, 1], [1, 2]))  # (num modes) | (n)

        ### DEBUG
        # print("Born term: ", borns_term.round(5))
        ### DEBUG

        # Chi(2).q
        nlo_term = np.tensordot(nlo_susceptibility, q_cartesian, axes=(2, 0))  # (3, 3) | (a, b)

        ### DEBUG
        # print("Nlo term: ", nlo_term.round(5))
        # print("Tensordot B N: ", np.tensordot(borns_term, nlo_term, axes=0).round(5))
        ### DEBUG

        nlo_correction = -(UNITS.nlo_conversion / dielectric_term) * np.tensordot(
            borns_term, nlo_term, axes=0
        )  # (3,3,3) | (k,a,b)

        ### DEBUG
        # print("Correction: ", nlo_correction.round(5))
        ### DEBUG

        raman_susceptibility_tensors += nlo_correction

    return (raman_susceptibility_tensors / sqrt_volume, freqs, labels)


def compute_polarization_vectors(
    phonopy_instance: Phonopy,
    nac_direction: list[float, float, float] | None = None,
    use_irreps: bool = True,
    degeneracy_tolerance: float = 1e-5,
    sum_rules: bool = False,
    **kwargs
) -> tuple[list, list, list]:
    """Return the polarization vectors, frequencies (cm-1) and labels.

    .. note:: the unite for polarization vectors are in (debey/angstrom)/sqrt(AMU)

    :param phonopy_instance: Phonopy instance with non-analytical constants included
    :param nac_direction: non-analytical direction in Cartesian coordinates
    :param use_irreps: whether to use irreducible representations
        in the selection of modes, defaults to True
    :param degeneracy_tolerance: degeneracy tolerance
        for irreducible representation
    :param sum_rules: whether to apply charge neutrality to effective charges

    :return: tuple of numpy.ndarray (polarization vectors, frequencies, labels)
    """
    selection_rule = 'ir' if use_irreps else None

    freqs, neigvs, labels = compute_active_modes(
        phonopy_instance=phonopy_instance,
        nac_direction=nac_direction,
        degeneracy_tolerance=degeneracy_tolerance,
        selection_rule=selection_rule,
        **kwargs,
    )

    borns = phonopy_instance.nac_params['born']

    if not sum_rules:
        sum_rule_correction = borns.sum(axis=0) / len(borns)  # sum over atomic index
        borns -= sum_rule_correction

    # Here we check we do not have empty array.
    # E.g. happening in non-polar crystals, such as Si
    if not neigvs.tolist():
        return (np.array([]) for _ in range(3))

    # neigvs shape|indices = (num modes, num atoms, 3) | (n, I, k)
    # borns  shape|indices = (num atoms, 3, 3) | (I, a, k)
    # The contraction is performed over I and k, resulting in (n, a) polarization vectors.
    pol_vectors = UNITS.debey_ang * np.tensordot(neigvs, borns, axes=([1, 2], [0, 2]))  # in (D/ang)/sqrt(AMU)

    return (pol_vectors, freqs, labels)


def compute_complex_dielectric(
    phonopy_instance: Phonopy,
    freq_range: Union[str, np.ndarray] = 'auto',
    gammas: float | list[float] = 12.0,
    nac_direction: None | list[float, float, float] = None,
    use_irreps: bool = True,
    degeneracy_tolerance: float = 1e-5,
    sum_rules: bool = False,
) -> np.ndarray:
    """Return the frequency dependent complex dielectric function (tensor).

    :param freq_range: frequency range in cm^-1; set to `auto` for automatic choice
    :param gammas: list or single value of broadenings, i.e. full width at half maximum (FWHM)
    :param nac_direction: (3,) shape list, indicating non analytical
        direction in Cartesian coordinates
    :param use_irreps: whether to use irreducible representations
        in the selection of modes, defaults to True
    :param degeneracy_tolerance: degeneracy tolerance
        for irreducible representation
    :param sum_rules: whether to apply charge neutrality to effective charges

    :return: (3, 3, num steps) shape :class:`numpy.ndarray`, `num steps` refers to the
        number of frequency steps where the complex dielectric function is evaluated
    """
    from phonopy import units
    from qe_tools import CONSTANTS

    prefactor = 4 * np.pi * units.VaspToCm**2 * 2. * CONSTANTS.ry_to_ev * CONSTANTS.bohr_to_ang

    polarizations, frequencies, _ = compute_polarization_vectors(
        phonopy_instance=phonopy_instance,
        nac_direction=nac_direction,
        use_irreps=use_irreps,
        degeneracy_tolerance=degeneracy_tolerance,
        sum_rules=sum_rules
    )

    if isinstance(gammas, float):
        sigmas = [gammas for _ in frequencies]
    elif isinstance(gammas, (list, np.ndarray)):
        if len(gammas) != len(frequencies):
            raise ValueError("length of `gammas` and number of frequencies don't match")
        sigmas = deepcopy(gammas)
    else:
        sigmas = [float(gammas) for _ in frequencies]

    if isinstance(freq_range, str):
        xi = max(0, frequencies.min() - 200)
        xf = frequencies.max() + 200
        freq_range = np.arange(xi, xf, 1.)

    polarizations /= UNITS.debey_ang
    oscillator = np.zeros((3, 3))

    oscillators = []
    for pol, freq in zip(polarizations, frequencies):
        oscillators.append(np.outer(pol, pol))
        oscillator += np.outer(pol, pol) / (freq * freq)

    oscillators = np.array(oscillators)

    complex_diel = np.zeros((3, 3, freq_range.shape[0]), dtype=np.complex128)
    for n, osc in enumerate(oscillators):
        for i1 in range(3):
            for i2 in range(3):
                complex_diel[i1, i2] += (
                    np.array(osc[i1, i2]) / (
                        +np.array(frequencies[n])**2  # omega_n^2
                        - np.array(freq_range)**2  # - omega^2
                        - np.array(freq_range) * sigmas[n] * complex(1j)  # -i eta_n omega
                    )
                )

    diel = phonopy_instance.nac_params['dielectric']
    diel_infinity = np.tensordot(diel, np.ones(freq_range.shape[0]), axes=0)

    return diel_infinity + prefactor * (complex_diel) / phonopy_instance.unitcell.volume


def compute_clamped_pockels_tensor(
    phonopy_instance: Phonopy,
    raman_tensors: np.ndarray,
    nlo_susceptibility: np.ndarray,
    nac_direction: None | list[float, float, float] = None,
    imaginary_thr: float = -5.0 * 1.0e+12, # in Hz
    skip_frequencies: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the clamped Pockels tensor in Cartesian coordinates.

    .. note:: Units are in pm/V

    :param nac_direction: non-analytical direction in Cartesian coordinates;
        (3,) shape :class:`list` or :class:`numpy.ndarray`
    :param degeneracy_tolerance: degeneracy tolerance for irreducible representation
    :param imaginary_thr: threshold for activating warnings on negative frequencies (in Hz)
    :param skip_frequencies: number of frequencies to not include (i.e. the acoustic modes)

    :return: tuple of (r_ion + r_el, r_el, r_ion), each having (3, 3, 3) shape array
    """
    borns = phonopy_instance.nac_params['born'] * UNITS.elementary_charge_si  # convert to Coulomb
    dielectric = phonopy_instance.nac_params['dielectric']

    dielectric_inv = np.linalg.inv(dielectric)
    raman = raman_tensors * 1.0e10  # convert to 1/m

    q_reduced = None
    # Have a look in compute_active_modes for the notation
    if nac_direction is not None:
        q_reduced = np.dot(phonopy_instance.primitive.cell,
                           nac_direction) / (2. * np.pi)  # in reduced/crystal (PRIMITIVE) coordinates

    phonopy_instance.run_qpoints(q_points=[0, 0, 0], nac_q_direction=q_reduced, with_eigenvectors=True)
    frequencies = phonopy_instance.qpoints.frequencies[0] * 1.0e+12  # THz -> Hz
    eigvectors = phonopy_instance.qpoints.eigenvectors.T.real

    if frequencies.min() < imaginary_thr:
        raise ValueError('Negative frequencies detected.')

    masses = phonopy_instance.masses * UNITS.atomic_mass_si
    sqrt_masses = np.array([[np.sqrt(mass)] for mass in masses])

    shape = (len(frequencies), len(masses), 3)  # (modes, atoms, 3)
    eigvectors = eigvectors.reshape(shape)
    norm_eigvectors = np.array([eigv / sqrt_masses for eigv in eigvectors])

    # norm_eigvectors shape|indices = (modes, atoms, 3) | (m, a, p)
    # raman  shape|indices = (atoms, 3, 3, 3) | (a, p, i, j)
    # The contraction is performed over a and k, resulting in (m, i, j) raman susceptibility tensors.
    alpha = np.tensordot(norm_eigvectors, raman, axes=([1, 2], [0, 1]))

    # borns charges shape|indices = (atoms, 3, 3) | (a, k, p)
    # norm_eigvectors shape|indices = (modes, atoms, 3) | (m, a, p)
    # polarization vectors shape | indices = (3, modes) | (k, m)
    polvec = np.tensordot(borns, norm_eigvectors, axes=([0, 2], [1, 2]))  # (k, m)

    ir_contribution = polvec / frequencies**2  # (k, m)

    # sets the first `skip_frequencies` IR contributions to zero (i.e. the acoustic modes)
    for i in range(skip_frequencies):
        ir_contribution[:, i] = 0
    r_ion_inner = np.tensordot(alpha, ir_contribution, axes=([0], [1]))  # (i, j, k)

    r_ion_left = np.dot(dielectric_inv, np.transpose(r_ion_inner, axes=[1, 0, 2]))
    r_ion_transposed = np.dot(np.transpose(r_ion_left, axes=[0, 2, 1]), dielectric_inv)
    r_ion = -np.transpose(r_ion_transposed, axes=[0, 2, 1]) * 1.0e+12  # pm/V

    r_el_left = np.dot(dielectric_inv, np.transpose(nlo_susceptibility, axes=[1, 0, 2]))
    r_el_transposed = -2 * np.dot(np.transpose(r_el_left, axes=[0, 2, 1]), dielectric_inv)
    r_el = np.transpose(r_el_transposed, axes=[0, 2, 1])

    r = r_el + r_ion

    return r, r_el, r_ion


@calcfunction
def get_supercells_for_hubbard(
    preprocess_data: PreProcessData,
    ref_structure,
) -> dict:
    """Return a dictionary of supercells with displacements.

    The supercells are obtained from the reference structure,
    preventing the case scenario of folded atoms.
    This is essential for extended Hubbard functionals which depends
    upon explicit positions in the cell. An atom folded would mean
    losing the interaction between first neighbours.

    :return: a dict of :class:`~aiida.orm.StructureData` or
        :class:`~aiida_quantumespresso.data.hubbard_structure.HubbardStructureData`,
        labelled with `supercell_{}`, where {} is a number starting from 1.
    """
    displacements = preprocess_data.get_displacements()
    structures = {}

    if isinstance(ref_structure, HubbardStructureData):
        hubbard = ref_structure.hubbard

    for i, displacement in enumerate(displacements):
        translation = [[0., 0., 0.] for _ in ref_structure.sites]

        translation[displacement[0]] = displacement[1:4]
        ase = ref_structure.get_ase().copy()
        ase.translate(translation)
        ase_dict = ase.todict()
        cell = ase_dict['cell']
        positions = ase_dict['positions']
        kinds = [site.kind_name for site in ref_structure.sites]
        symbols = ase.get_chemical_symbols()

        structure = orm.StructureData(cell=cell)

        for position, symbol, name in zip(positions, symbols, kinds):
            structure.append_atom(position=position, symbols=symbol, name=name)

        if isinstance(ref_structure, HubbardStructureData):
            structure = HubbardStructureData.from_structure(structure, hubbard)

        structures.update({f'supercell_{i+1}': structure})

    return structures


@calcfunction
def elaborate_susceptibility_derivatives(
    preprocess_data: PreProcessData,
    raman_tensors=None,
    nlo_susceptibility=None,
) -> dict:
    """Return the susceptibility derivatives in the primitive cell.

    It uses the unique atoms referring to the supercell matrix.

    :return: dict with keyword `raman_tensors` and `nlo_susceptibility`
    """
    from aiida.orm import ArrayData

    from .symmetry import symmetrize_susceptibility_derivatives

    ref_dph0 = raman_tensors.get_array('raman_tensors')
    ref_nlo = nlo_susceptibility.get_array('nlo_susceptibility')

    dph0_, nlo_ = symmetrize_susceptibility_derivatives(
        # tensors
        raman_tensors=ref_dph0,
        nlo_susceptibility=ref_nlo,
        # preprocess info
        ucell=preprocess_data.get_phonopy_instance().unitcell,
        primitive_matrix=preprocess_data.primitive_matrix,
        supercell_matrix=preprocess_data.supercell_matrix,
        is_symmetry=preprocess_data.is_symmetry,
        symprec=preprocess_data.symprec,
    )

    dph0_data = ArrayData()
    dph0_data.set_array('raman_tensors', dph0_)
    nlo_data = ArrayData()
    nlo_data.set_array('nlo_susceptibility', nlo_)

    return {'raman_tensors': dph0_data, 'nlo_susceptibility': nlo_data}


@calcfunction
def elaborate_tensors(preprocess_data: PreProcessData, tensors: orm.ArrayData) -> orm.ArrayData:
    """Return second and third rank tensors in primitive cell.

    It uses the unique atoms referring to the supercell matrix.

    :return: :class:`~aiida.orm.ArrayData` with arraynames `born_charges`,
        `dielectric`, `raman_tensors`, `nlo_susceptibility`.
    """
    from phonopy.structure.symmetry import symmetrize_borns_and_epsilon

    from .symmetry import symmetrize_susceptibility_derivatives

    ref_dielectric = tensors.get_array('dielectric')
    ref_born_charges = tensors.get_array('born_charges')
    try:
        ref_dph0 = tensors.get_array('raman_tensors')
        ref_nlo = tensors.get_array('nlo_susceptibility')
        is_third = True
    except KeyError:
        is_third = False

    new_tensors = orm.ArrayData()

    # Non-analytical constants elaboration
    bec_, eps_ = symmetrize_borns_and_epsilon(
        # nac info
        borns=ref_born_charges,
        epsilon=ref_dielectric,
        # preprocess info
        ucell=preprocess_data.get_phonopy_instance().unitcell,
        primitive_matrix=preprocess_data.primitive_matrix,
        supercell_matrix=preprocess_data.supercell_matrix,
        is_symmetry=preprocess_data.is_symmetry,
        symprec=preprocess_data.symprec,
    )

    new_tensors.set_array('dielectric', eps_)
    new_tensors.set_array('born_charges', bec_)

    # Eventual susceptibility derivatives elaboration
    if is_third:
        dph0_, nlo_ = symmetrize_susceptibility_derivatives(
            # tensors
            raman_tensors=ref_dph0,
            nlo_susceptibility=ref_nlo,
            # preprocess info
            ucell=preprocess_data.get_phonopy_instance().unitcell,
            primitive_matrix=preprocess_data.primitive_matrix,
            supercell_matrix=preprocess_data.supercell_matrix,
            is_symmetry=preprocess_data.is_symmetry,
            symprec=preprocess_data.symprec,
        )

        new_tensors.set_array('raman_tensors', dph0_)
        new_tensors.set_array('nlo_susceptibility', nlo_)

    return new_tensors


@calcfunction
def generate_vibrational_data_from_forces(
    preprocess_data: PreProcessData, tensors: orm.ArrayData, forces_index: orm.Int = None, **forces_dict
):
    """Return a `VibrationalFrozenPhononData` node.

    Forces must be passed as **kwargs**, since we are calling a
    calcfunction with a variable number of supercells forces.

    :param tensors: :class:`~aiida.orm.ArrayData` with arraynames `dielectric`, `born_charges`
        and eventual `raman_tensors`, `nlo_susceptibility`
    :param forces_index: :class:`~aiida.orm.Int` if a :class:`~aiida.orm.TrajectoryData`
        is given, in order to get the correct slice of the array.
        In aiida-quantumespresso it should be 0 or -1.
    :param forces_dict: dictionary of supercells forces as :class:`~aiida.orm.ArrayData` stored
        as `forces`, each Data labelled in the dictionary in the format
        `forces_{suffix}`. The prefix is common and the suffix
        corresponds to the suffix number of the supercell with
        displacement label given from the
        `get_supercells_with_displacements` method. For example:
        ```
        {'forces_1':ArrayData, 'forces_2':ArrayData}
        <==>
        {'supercell_1':StructureData, 'supercell_2':StructureData}
        ```
        and forces in each ArrayData stored as 'forces',
        i.e. ArrayData.get_array('forces') must not raise error

    .. note:: if residual forces would be stored, label it with 0 as suffix.
    """
    VibrationalFrozenPhononData = DataFactory('vibroscopy.fp')
    prefix = 'forces'

    forces_0 = forces_dict.pop(f'{prefix}_0', None)
    # Setting the dictionary of forces
    dict_of_forces = {}

    for key, value in forces_dict.items():
        if key.startswith(prefix):
            dict_of_forces[key] = value.get_array('forces')

    if forces_index is not None:
        forces_index = forces_index.value

    # Setting data on a new PhonopyData
    vibrational_data = VibrationalFrozenPhononData(preprocess_data=preprocess_data.clone())
    vibrational_data.set_forces(dict_of_forces=dict_of_forces, forces_index=forces_index)

    if forces_0 is not None:
        vibrational_data.set_residual_forces(forces=forces_0.get_array('forces'))

    vibrational_data.set_dielectric(tensors.get_array('dielectric'))
    vibrational_data.set_born_charges(tensors.get_array('born_charges'))
    try:
        vibrational_data.set_raman_tensors(tensors.get_array('raman_tensors'))
        vibrational_data.set_nlo_susceptibility(tensors.get_array('nlo_susceptibility'))
    except KeyError:
        pass

    return vibrational_data


@calcfunction
def generate_vibrational_data_from_phonopy(phonopy_data, tensors: orm.ArrayData):
    """Return a `VibrationalData` node.

    .. note:: it computes the force constants naively; this will probably not work
    if random displacements have been used. Do use :class:`~aiida_phonopy.calculations.phonopy.PhonopyCalculation`
    to extract the force constants via e.g. HIPHIVE.
    Then use the :func:`~aiida_vibroscopy.calculations.spectra_utils.generate_vibrational_data_from_force_constants`

    :param tensors: :class:`~aiida.orm.ArrayData` with arraynames `dielectric`, `born_charges`
        and eventual `raman_tensors`, `nlo_susceptibility`
    """
    VibrationalData = DataFactory('vibroscopy.vibrational')

    # Getting the force constants
    ph = phonopy_data.get_phonopy_instance()
    ph.produce_force_constants()
    force_constants = ph.force_constants

    # Setting data on a new PhonopyData
    vibrational_data = VibrationalData(
        structure=phonopy_data.get_unitcell(),
        supercell_matrix=phonopy_data.supercell_matrix,
        primitive_matrix=phonopy_data.primitive_matrix,
        symprec=phonopy_data.symprec,
        is_symmetry=phonopy_data.is_symmetry,
    )

    vibrational_data.set_force_constants(force_constants)

    vibrational_data.set_dielectric(tensors.get_array('dielectric'))
    vibrational_data.set_born_charges(tensors.get_array('born_charges'))
    try:
        vibrational_data.set_raman_tensors(tensors.get_array('raman_tensors'))
        vibrational_data.set_nlo_susceptibility(tensors.get_array('nlo_susceptibility'))
    except KeyError:
        pass

    return vibrational_data


@calcfunction
def generate_vibrational_data_from_force_constants(preprocess_data, force_constants, tensors: orm.ArrayData):
    """Return a `VibrationalData` node.

    :param force_constants: ArrayData with arrayname `force_constants`
    :param tensors: ArrayData with arraynames `dielectric`, `born_charges`
        and eventual `raman_tensors`, `nlo_susceptibility`
    """
    VibrationalData = DataFactory('vibroscopy.vibrational')

    # Setting data on a new PhonopyData
    vibrational_data = VibrationalData(
        structure=preprocess_data.get_unitcell(),
        supercell_matrix=preprocess_data.supercell_matrix,
        primitive_matrix=preprocess_data.primitive_matrix,
        symprec=preprocess_data.symprec,
        is_symmetry=preprocess_data.is_symmetry,
    )

    vibrational_data.set_force_constants(force_constants.get_array('force_constants'))
    vibrational_data.set_dielectric(tensors.get_array('dielectric'))
    vibrational_data.set_born_charges(tensors.get_array('born_charges'))
    try:
        vibrational_data.set_raman_tensors(tensors.get_array('raman_tensors'))
        vibrational_data.set_nlo_susceptibility(tensors.get_array('nlo_susceptibility'))
    except KeyError:
        pass

    return vibrational_data


@calcfunction
def subtract_residual_forces(ref_meshes: orm.List, meshes_dict: orm.Dict, **kwargs) -> dict:
    """Return trajectories subtracting the residual forces.

    The forces related to of the finite electric fields are *normalized*
    subtracting the forces from the null electric fields calculations.

    :param ref_meshes: list containing the meshes of the null fields calculations
        in the order as they were called in the workflow
    :param meshes_dict: dic containing the meshes of the finite fields calculations,
        with keys as `field_index_{}`, {} an int, as in kwargs for `old_trajectories`
    :param kwargs: dict with keys `ref_trajectories` and `old_trajectories`,
        meaning the null electric fields and finite electric fiels trajectories,
    respectively. The structure of the two subdictionaries is:
        * `ref_trajectories`: {'0': TrajectoryData, ... }
        * `old_trajectories`: {'field_index_0': {'0': TrajectoryData, ...}, ...}

    :return: a dict with the same structure of `old_trajectories`, with *forces*
        rinormalized in each TrajectoryData.
    """
    ref_meshes_ = ref_meshes.get_list()
    meshes_dict_ = meshes_dict.get_dict()

    ref_trajectories = kwargs['ref_trajectories']
    old_trajectories = kwargs['old_trajectories']

    new_trajectories = {}

    for field_label, field_trajectories in old_trajectories.items():
        mesh = meshes_dict_[field_label]
        ref_index = ref_meshes_.index(mesh)
        ref_force = ref_trajectories[str(ref_index)].get_array('forces')

        new_trajectories[field_label] = {}

        for index, field_trajectory in field_trajectories.items():
            new_traj = field_trajectory.clone()
            old_force = new_traj.get_array('forces')

            new_traj.set_array('forces', old_force - ref_force)

            new_trajectories[field_label][str(index)] = new_traj

    return new_trajectories
