# -*- coding: utf-8 -*-
"""Calcfunctions utils for spectra workflows."""
import math

from aiida import orm
from aiida.engine import calcfunction
from aiida.plugins import DataFactory
import numpy as np

from aiida_vibroscopy.common import UNITS_FACTORS

PreProcessData = DataFactory('phonopy.preprocess')

__all__ = (
    'boson_factor',
    'compute_active_modes',
    'compute_raman_space_average',
    'compute_raman_tensors',
    'compute_polarization_vectors',
    'compute_polarization_vectors',
    'get_supercells_for_hubbard',
    'elaborate_susceptibility_derivatives',
    'generate_vibrational_data',
)


def boson_factor(frequency, temperature):
    """Return boson factor, i.e. (nb+1). Frequency in cm-1 and temperature in Kelvin."""
    return 1.0 / (1.0 - math.exp(-UNITS_FACTORS.cm_to_kelvin * frequency / temperature))


def compute_active_modes(
    phonopy_instance, degeneracy_tolerance=1.e-5, nac_direction=None, selection_rule=None, sr_thr=1e-4
):
    """Get frequencies, normalized eigenvectors and irreducible
    representation labels of active modes for calculation of
    polarization vectors and Raman tensors.

    :param nac_direction: (3,) shape list, indicating non analytical
        direction in fractional reciprocal space coordinates
    :param selection_rule: str, can be `raman` or `ir`;
        it uses symmetry in the selection of the modes
        for a specific type of process.
    :param sr_thr: float, threshold for selection
        rule (the analytical value is 0).

    :return: tuple of (frequencies in cm-1, normalized eigenvectors, labels);
        normalized eigenvectors is an
        array of shape (num modes, num atoms, 3).
    """

    if selection_rule not in ('raman', 'ir', None):
        raise ValueError('`selection_rule` can only be `ir` or `raman`.')

    # Step 1 - set the irreducible representations and the phonons
    phonopy_instance.set_irreps(q=[0, 0, 0], nac_q_direction=nac_direction, degeneracy_tolerance=degeneracy_tolerance)
    irreps = phonopy_instance.irreps

    phonopy_instance.run_qpoints(q_points=[0, 0, 0], nac_q_direction=nac_direction, with_eigenvectors=True)
    frequencies = phonopy_instance.qpoints.frequencies[0] * UNITS_FACTORS.thz_to_cm
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
    labels = irreps._get_ir_labels()

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

            if math.fabs(condition) > sr_thr:  # selection rule (thr for inaccuracies)

                for band_index in band_indices:
                    freq_active_modes.append(frequencies[band_index])
                    eigvectors_active_modes.append(eigvectors[band_index])
                    labels_active_modes.append(label)

        mode_index += degeneracy

    freq_active_modes = np.array(freq_active_modes)

    # Step 3 - getting normalized eigenvectors
    masses = phonopy_instance.masses
    sqrt_masses = np.array([[math.sqrt(mass)] for mass in masses])

    eigvectors_active_modes = np.array(eigvectors_active_modes)
    shape = (len(freq_active_modes), len(masses), 3)
    eigvectors_active_modes = eigvectors_active_modes.reshape(shape)
    norm_eigvectors_active_modes = np.array([eigv / sqrt_masses for eigv in eigvectors_active_modes])

    return (freq_active_modes, norm_eigvectors_active_modes, labels_active_modes)


def compute_raman_space_average(raman_tensors):
    """Return the space average for the HH and HV configuration
    (e.g. `Light scattering in solides II`, M. Cardona and
    `S. A. Prosandeev et al., Phys. Rev. B, 71, 214307 (2005) ).

    :return: (intensities HH, intensities HV)
    """
    intensities_hh = []
    intensities_hv = []
    for R in raman_tensors:
        # a = R.trace() / 3.0
        # a2 = a * a
        # b2 = (
        #     0.5 * ((R[0][0] - R[1][1])**2 + (R[0][0] - R[2][2])**2 + (R[1][1] - R[2][2])**2) + 3. *
        #     (R[0][1]**2 + R[0][2]**2 + R[1][2]**2)
        # )
        # intensities_hh.append(a2 + 4 * b2 / 45)
        # intensities_hv.append(3 * b2 / 45)
        G0 = (R.trace()**2) / 3.0
        G1 = 0.5 * ((R[0][1] - R[1][0])**2 + (R[0][2] - R[2][0])**2 + (R[1][2] - R[2][1])**2)
        G2 = (
            0.5 * ((R[0][1] + R[1][0])**2 + (R[0][2] + R[2][0])**2 + (R[1][2] + R[2][1])**2) + (1. / 3.) *
            ((R[0][0] - R[1][1])**2 + (R[0][0] - R[2][2])**2 + (R[1][1] - R[2][2])**2)
        )

        intensities_hh.append((10 * G0 + 4 * G2) / 30)
        intensities_hv.append((5 * G1 + 3 * G2) / 30)
    return (np.array(intensities_hh), np.array(intensities_hv))


def compute_raman_tensors(
    phonopy_instance,
    raman_susceptibility,
    nlo_susceptibility=None,
    nac_direction=lambda: [0, 0, 0],
    use_irreps=True,
    sum_rules=False,
    degeneracy_tolerance=1e-5,
):
    """Return the Raman tensors (in angstrom^2/sqrt(AMU)) along with
    each phonon mode with frequencies (cm-1) and labels.

    :param phonopy_instance: Phonopy instance with non-analytical constants included
    :param nac_direction: non-analytical direction in reciprocal
        space coordinates (only direction is meaningful)
    :param raman_susceptibility: derivatives of the susceptibility
        in respect to atomic positions in Cartesian coordinates and in angstrom^2
    :param nlo_susceptibility: non linear optical susceptibility
        in Cartesian coordinates and in pm/V
    :type nac_direction: (3,) shape list or numpy.ndarray
    :param use_irreps: whether to use irreducible representations
        in the selection of modes, defaults to True
    :type use_irreps: bool, optional
    :param degeneracy_tolerance: degeneracy tolerance for
        irreducible representation

    :return: tuple (Raman tensors, frequencies, labels)
    """
    try:
        nac_direction = nac_direction()
    except TypeError:
        pass

    if not isinstance(nac_direction, (list, np.ndarray)) or not isinstance(use_irreps, bool):
        raise TypeError('the input is not of the correct type')

    nac_direction = np.array(nac_direction)

    if nac_direction.shape != (3,):
        raise ValueError('the array is not of the correct shape')

    rcell = np.linalg.inv(phonopy_instance.primitive.cell)
    q_direction = np.dot(rcell, nac_direction)  # in Cartesian coordinates

    selection_rule = 'raman' if use_irreps else None

    if sum_rules:
        sum_rule_correction = raman_susceptibility.sum(axis=0) / len(raman_susceptibility)  # sum over atomic index
        raman_susceptibility -= sum_rule_correction

    freqs, neigvs, labels = compute_active_modes(
        phonopy_instance=phonopy_instance,
        nac_direction=nac_direction,
        degeneracy_tolerance=degeneracy_tolerance,
        selection_rule=selection_rule
    )

    # Here we check we do not have empty array.
    # E.g. happening in cubic crystals, such as MgO
    if not neigvs.tolist():
        return (np.array([]) for _ in range(3))

    # neigvs shape|indices = (num modes, num atoms, 3) | (n, I, k)
    # dph0   shape|indices = (num atoms, 3, 3, 3) | (I, k, i, j)
    # The contraction is performed over I and k, resulting in (n, i, j) Raman tensors.
    raman_tensors = np.tensordot(neigvs, raman_susceptibility, axes=([1, 2], [0, 1]))

    if nlo_susceptibility is not None and q_direction.nonzero()[0].tolist():
        borns = phonopy_instance.nac_params['born']
        dielectric = phonopy_instance.nac_params['dielectric']
        # -8 pi (Z.q/q.epsilon.q)[I,k] Chi(2).q [i,j] is the correction to dph0.
        # The indices I, k to do the scalar product with the eigenvectors run over the Borns term.
        # nac_direction shape|indices = (3) | (i)
        # borns  shape|indices = (num atoms, 3, 3) | (I, i, k)
        # nlo    shape|indices = (3, 3, 3) | (i, j, k)

        # q.epsilon.q
        # !!! ---------------------- !!!
        #    Here we can extend to 1/2D models.
        # !!! ---------------------- !!!
        dielectric_term = np.dot(np.dot(dielectric, q_direction), q_direction)
        # Z*.q
        borns_term_dph0 = np.tensordot(borns, q_direction, axes=(1, 0))  # (num atoms, 3) | (I, k)
        borns_term = np.tensordot(borns_term_dph0, neigvs, axes=([0, 1], [1, 2]))  # (num modes) | (n)
        # Chi(2).q
        nlo_term = np.dot(nlo_susceptibility, q_direction)  # (3, 3) | (i, j)

        nlo_correction = -(UNITS_FACTORS.nlo_conversion / dielectric_term) * np.tensordot(borns_term, nlo_term, axes=0)
        raman_tensors = raman_tensors + nlo_correction

    return (raman_tensors, freqs, labels)


def compute_polarization_vectors(
    phonopy_instance,
    nac_direction=lambda: [0, 0, 0],
    use_irreps=True,
    degeneracy_tolerance=1e-5,
    sum_rules=False,
    **kwargs
):
    """Return the polarization vectors (in (debey/angstrom)/sqrt(AMU))
    for each phonon mode with frequencies (cm-1) and labels.

    :param phonopy_instance: Phonopy instance with non-analytical constants included
    :param nac_direction: non-analytical direction in fractional coordinates
        in reciprocal space
    :type nac_direction: (3,) shape list or numpy.ndarray
    :param use_irreps: whether to use irreducible representations
        in the selection of modes, defaults to True
    :type use_irreps: bool, optional
    :param degeneracy_tolerance: degeneracy tolerance
        for irreducible representation

    :return: tuple (polarization vectors, frequencies, labels)
    """
    try:
        nac_direction = nac_direction()
    except TypeError:
        pass

    if not isinstance(nac_direction, (list, np.ndarray)) or not isinstance(use_irreps, bool):
        raise TypeError('the input is not of the correct type')

    nac_direction = np.array(nac_direction)

    if nac_direction.shape != (3,):
        raise ValueError('the array is not of the correct shape')

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
    if neigvs.tolist():
        return (np.array([]) for _ in range(3))

    # neigvs shape|indices = (num modes, num atoms, 3) | (n, I, k)
    # borns  shape|indices = (num atoms, 3, 3) | (I, i, k)
    # The contraction is performed over I and k, resulting in (n, i) polarization vectors.
    pol_vectors = UNITS_FACTORS.debey_ang * np.tensordot(neigvs, borns, axes=([1, 2], [0, 2]))  # in (D/ang)/AMU

    return (pol_vectors, freqs, labels)


@calcfunction
def get_supercells_for_hubbard(preprocess_data: PreProcessData, ref_structure: orm.StructureData):
    """Return the supercells with displacements applying the displacements to the
    unitcell structure, preventing the case scenario of folded atoms.

    This is essential for extended Hubbard functionals which depends
    upon explicit positions in the unitcell.
    """
    displacements = preprocess_data.get_displacements()
    structures = {}

    for i, displacement in enumerate(displacements):
        traslation = [[0., 0., 0.] for _ in ref_structure.sites]

        traslation[displacement[0]] = displacement[1:4]
        ase = ref_structure.get_ase().copy()
        ase.translate(traslation)
        ase_dict = ase.todict()
        cell = ase_dict['cell']
        positions = ase_dict['positions']
        kinds = [site.kind_name for site in ref_structure.sites]
        symbols = ase.get_chemical_symbols()

        structure = orm.StructureData(cell=cell)

        for position, symbol, name in zip(positions, symbols, kinds):
            structure.append_atom(position=position, symbols=symbol, name=name)

        structures.update({f'supercell_{i+1}': structure})

    return structures


@calcfunction
def elaborate_susceptibility_derivatives(
    preprocess_data: PreProcessData,
    raman_susceptibility=None,
    nlo_susceptibility=None,
):
    """Return the susceptibility derivatives in the primitive cell
    considering the supercell matrix, in order to maintain consistency.
    """
    from aiida.orm import ArrayData

    from .symmetry import symmetrize_susceptibility_derivatives

    ref_dph0 = raman_susceptibility.get_array('raman_susceptibility')
    ref_nlo = nlo_susceptibility.get_array('nlo_susceptibility')

    dph0_, nlo_ = symmetrize_susceptibility_derivatives(
        # tensors
        raman_susceptibility=ref_dph0,
        nlo_susceptibility=ref_nlo,
        # preprocess info
        ucell=preprocess_data.get_phonopy_instance().unitcell,
        primitive_matrix=preprocess_data.primitive_matrix,
        supercell_matrix=preprocess_data.supercell_matrix,
        is_symmetry=preprocess_data.is_symmetry,
        symprec=preprocess_data.symprec,
    )

    dph0_data = ArrayData()
    dph0_data.set_array('raman_susceptibility', dph0_)
    nlo_data = ArrayData()
    nlo_data.set_array('nlo_susceptibility', nlo_)

    return {'raman_susceptibility': dph0_data, 'nlo_susceptibility': nlo_data}


@calcfunction
def elaborate_tensors(preprocess_data: PreProcessData, tensors: orm.ArrayData) -> orm.ArrayData:
    """Return second and, eventual, third order tensor in primitive cell,
    considering the supercell matrix elaborating the tensors from the unitcell.
    """
    from phonopy.structure.symmetry import symmetrize_borns_and_epsilon

    from .symmetry import symmetrize_susceptibility_derivatives

    ref_dielectric = tensors.get_array('dielectric')
    ref_born_charges = tensors.get_array('born_charges')
    try:
        ref_dph0 = tensors.get_array('raman_susceptibility')
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
            raman_susceptibility=ref_dph0,
            nlo_susceptibility=ref_nlo,
            # preprocess info
            ucell=preprocess_data.get_phonopy_instance().unitcell,
            primitive_matrix=preprocess_data.primitive_matrix,
            supercell_matrix=preprocess_data.supercell_matrix,
            is_symmetry=preprocess_data.is_symmetry,
            symprec=preprocess_data.symprec,
        )

        new_tensors.set_array('raman_susceptibility', dph0_)
        new_tensors.set_array('nlo_susceptibility', nlo_)

    return new_tensors


@calcfunction
def generate_vibrational_data(preprocess_data: PreProcessData, tensors: orm.ArrayData, **forces_dict):
    """Create a VibrationalFrozenPhononData node from a PreProcessData node,
    storing forces and dielectric properties for spectra calculation.

    `Forces` must be passed as **kwargs**, since we are calling a
    calcfunction with a variable number of supercells forces.

    :param tensors: ArrayData with arraynames `dielectric`, `born_charges`
        and eventual `raman_susceptibility`, `nlo_susceptibility`
    :param forces_dict: dictionary of supercells forces as ArrayData stored
        as `forces`, each Data labelled in the dictionary in the format
        `{prefix}_{suffix}`. The prefix is common and the suffix
        corresponds to the suffix number of the supercell with
        displacement label given from the
        `get_supercells_with_displacements` method.

        For example:
            {'forces_1':ArrayData, 'forces_2':ArrayData}
            <==>
            {'supercell_1':StructureData, 'supercell_2':StructureData}
            and forces in each ArrayData stored as 'forces',
            i.e. ArrayData.get_array('forces') must not raise error

        .. note: if residual forces would be stored, label it with 0 as suffix.
    """
    VibrationalFrozenPhononData = DataFactory('vibroscopy.fp')

    # Getting the prefix
    for key in forces_dict:
        try:
            int(key.split('_')[-1])
            # dumb way of getting the prefix including cases of multiple `_`, e.g. `force_calculation_001`
            prefix = key[:-(len(key.split('_')[-1]) + 1)]
        except ValueError as err:
            raise ValueError(f'{key} is not an acceptable key, must finish with an int number.') from err

    forces_0 = forces_dict.pop(f'{prefix}_0', None)

    # Setting force sets array
    force_sets = [0 for _ in forces_dict]

    # Filling arrays in numeric order determined by the key label
    for key, value in forces_dict.items():
        index = int(key.split('_')[-1])
        force_sets[index - 1] = value.get_array('forces')

    # Finilizing force sets array
    sets_of_forces = np.array(force_sets)

    # Setting data on a new PhonopyData
    vibrational_data = VibrationalFrozenPhononData(preprocess_data=preprocess_data)
    vibrational_data.set_forces(sets_of_forces=sets_of_forces)

    if forces_0 is not None:
        vibrational_data.set_residual_forces(forces=forces_0.get_array('forces'))

    vibrational_data.set_dielectric(tensors.get_array('dielectric'))
    vibrational_data.set_born_charges(tensors.get_array('born_charges'))
    try:
        vibrational_data.set_raman_susceptibility(tensors.get_array('raman_susceptibility'))
        vibrational_data.set_nlo_susceptibility(tensors.get_array('nlo_susceptibility'))
    except KeyError:
        pass

    return vibrational_data
