# -*- coding: utf-8 -*-
"""Mixin for aiida-vibroscopy DataTypes."""
from __future__ import annotations

import numpy as np

from aiida_vibroscopy.calculations.spectra_utils import (
    compute_active_modes,
    compute_polarization_vectors,
    compute_raman_space_average,
    compute_raman_susceptibility_tensors,
)
from aiida_vibroscopy.utils.integration.lebedev import LebedevScheme

__all__ = ('VibrationalMixin',)


class VibrationalMixin:
    """Mixin class for vibrational DataTypes.

    This is meant to be used to extend aiida-phonopy DataTypes to include tensors
    and information for vibrational spectra.
    """

    @property
    def raman_tensors(self):
        r"""Get the Raman tensors in Cartesian coordinates.

        .. important:: with Raman tensors we mean
            :math:\\frac{1}{\\Omega}\\frac{\\partial \\chi}{\\partial u}

        .. note:
            * Units in 1/Angstrom, normalized using the UNIT cell volume.
            * The shape should match the primitive cell.
            * Indices are as follows:
                1. Atomic index.
                2. Atomic displacement index.
                3. Polarization index (i.e. referring to electric field derivative).
                4. Same as 3.

        :return: (number of atoms in the primitive cell, 3, 3, 3) shape array
        """
        try:
            value = self.get_array('raman_tensors')
        except (KeyError, AttributeError):
            value = None
        return value

    def set_raman_tensors(self, raman_tensors: list | np.ndarray):
        r"""Set the Raman tensors in Cartesian coordinates.

        .. important:: with Raman tensors we mean
            :math:\\frac{1}{\\Omega}\\frac{\\partial \\chi}{\\partial u}

        .. note:
            * Units in 1/Angstrom, normalized using the UNIT cell volume.
            * The shape should match the primitive cell.
            * Indices are as follows:
                1. Atomic index.
                2. Atomic displacement index.
                3. Polarization index (i.e. referring to electric field derivative).
                4. Same as 3.

        :param raman_tensors: (number of atoms in the primitive cell, 3, 3, 3) shape array

        :raises:
            * TypeError: if the format is not compatible or of the correct type
            * ValueError: if the format is not compatible or of the correct type
        """
        if not isinstance(raman_tensors, (list, np.ndarray)):
            raise TypeError('the input is not of the correct type')

        the_dchi = np.array(raman_tensors)
        n_atoms = len(self.get_primitive_cell().sites)

        if the_dchi.shape == (n_atoms, 3, 3, 3):
            self.set_array('raman_tensors', the_dchi)
        else:
            raise ValueError('the array is not of the correct shape')

    @property
    def nlo_susceptibility(self) -> np.ndarray:
        """Get the non linear optical susceptibility tensor in Cartesian coordinates.

        :return: (3,3,3) shape array
        """
        try:
            value = self.base.attributes.get('nlo_susceptibility')
            value = np.array(value)
        except (KeyError, AttributeError):
            value = None
        return value

    def set_nlo_susceptibility(self, nlo_susceptibility: list | np.ndarray):
        """Set the non linear optical susceptibility tensor in Cartesian coordinates.

        .. note: units in pm/V

        :param dielectric: (3, 3, 3) array like

        :raises:
            * TypeError: if the format is not compatible or of the correct type
            * ValueError: if the format is not compatible or of the correct type
        """
        self._if_can_modify()

        if not isinstance(nlo_susceptibility, (list, np.ndarray)):
            raise TypeError('the input is not of the correct type')

        the_chi2 = np.array(nlo_susceptibility)

        if the_chi2.shape == (3, 3, 3):
            self.base.attributes.set('nlo_susceptibility', the_chi2.tolist())
        else:
            raise ValueError('the array is not of the correct shape')

    def has_raman_parameters(self) -> bool:
        """Return wheter or not the Data has derivatives of susceptibility for Raman spectra."""
        return self.raman_tensors is not None

    def has_nlo(self) -> bool:
        """Return wheter or not the Data has non linear optical susceptibility tensor."""
        return self.nlo_susceptibility is not None

    def run_active_modes(
        self,
        degeneracy_tolerance: float = 1.e-5,
        nac_direction: None | list[float, float, float] = None,
        selection_rule: str | None = None,
        sr_thr: float = 1e-4,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get active modes frequencies, eigenvectors and irreducible representation labels.

        Inputs as in :func:`~aiida_virboscopy.calculations.spectra_utils.compute_active_modes`

        :param nac_direction: (3,) shape list, indicating non analytical
            direction in fractional reciprocal (primitive cell) space coordinates
        :param selection_rule: str, can be `raman` or `ir`;
            it uses symmetry in the selection of the modes
            for a specific type of process.
        :param sr_thr: float, threshold for selection
            rule (the analytical value is 0).
        :param kwargs: see also the :func:`~aiida_phonopy.data.phonopy.get_phonopy_instance` method
            * subtract_residual_forces: whether or not subract residual forces (if set);
                bool, defaults to False
            * symmetrize_nac: whether or not to symmetrize the nac parameters
                using point group symmetry; bool, defaults to self.is_symmetry

        :return: tuple of (frequencies in cm-1, normalized eigenvectors, labels);
            normalized eigenvectors is an array of shape (num modes, num atoms, 3).
        """
        phonopy_instance = self.get_phonopy_instance(**kwargs)

        if phonopy_instance.force_constants is None:
            phonopy_instance.produce_force_constants()

        return compute_active_modes(
            phonopy_instance=phonopy_instance,
            nac_direction=nac_direction,
            degeneracy_tolerance=degeneracy_tolerance,
            selection_rule=selection_rule,
            sr_thr=sr_thr
        )

    def run_raman_susceptibility_tensors(
        self,
        nac_direction: tuple[float, float, float] = lambda: [0, 0, 0],
        with_nlo: bool = True,
        use_irreps: bool = True,
        degeneracy_tolerance: float = 1e-5,
        sum_rules: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the Raman susceptibility tensors, frequencies and representation labels.

        .. note:: Units are:
                * Intensities: Anstrom/AMU
                * Frequencies: cm-1

        :param nac_direction: non-analytical direction in fractional coordinates (primitive cell)
            in reciprocal space; (3,) shape list or numpy.ndarray
        :param with_nlo: whether to use or not non-linear optical susceptibility
            correction (Froehlich term), defaults to True
        :type with_nlo: bool, optional
        :param use_irreps: whether to use irreducible representations
            in the selection of modes, defaults to True
        :type use_irreps: bool, optional
        :param sum_rules: whether to apply sum rules and symmetrize force constants
            by point and space group symmetries
        :type sum_rules: bool, optional
        :param kwargs: see also the :func:`~aiida_phonopy.data.phonopy.get_phonopy_instance` method
            * subtract_residual_forces: whether or not subract residual forces (if set);
                bool, defaults to False
            * symmetrize_nac: whether or not to symmetrize the nac parameters
                using point group symmetry; bool, defaults to self.is_symmetry

        :return: tuple (Raman tensors, frequencies, irreps labels)
        """
        try:
            nac_direction = nac_direction()
        except TypeError:
            pass

        if not isinstance(with_nlo, bool) or not isinstance(use_irreps, bool) or not isinstance(sum_rules, bool):
            raise TypeError('the input is not of the correct type')

        phonopy_instance = self.get_phonopy_instance(**kwargs)

        if phonopy_instance.force_constants is None:
            phonopy_instance.produce_force_constants()

        if sum_rules:
            phonopy_instance.symmetrize_force_constants()
            phonopy_instance.symmetrize_force_constants_by_space_group()

        nlo_susceptibility = self.nlo_susceptibility if with_nlo else None

        results = compute_raman_susceptibility_tensors(
            phonopy_instance=phonopy_instance,
            raman_tensors=self.raman_tensors,
            nlo_susceptibility=nlo_susceptibility,
            nac_direction=nac_direction,
            use_irreps=use_irreps,
            degeneracy_tolerance=degeneracy_tolerance,
            sum_rules=sum_rules,
        )

        return results

    def run_polarization_vectors(
        self,
        nac_direction: tuple[float, float, float] = lambda: [0, 0, 0],
        use_irreps: bool = True,
        degeneracy_tolerance: float = 1e-5,
        sum_rules: bool = False,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the polarization vectors, frequencies and representation labels.

        .. note:: Units are:
                * Intensities: (Debey/Angstrom)^2/AMU
                * Frequencies: cm-1

        :param nac_direction: non-analytical direction
        :type nac_direction: non-analytical direction in fractional coordinates (primitive cell)
            in reciprocal space; space(3,) shape list or numpy.ndarray
        :param use_irreps: whether to use irreducible representations in the
            selection of modes, defaults to True
        :type use_irreps: bool, optional
        :param sum_rules: whether to apply sum rules to the derivatives of the
            susceptibility in respect to atomic positions
        :type sum_rules: bool, optional
        :param kwargs: keys of :func:`~aiida_phonopy.data.phonopy.get_phonopy_instance` method
            * subtract_residual_forces: whether or not subract residual forces (if set);
                bool, defaults to False
            * symmetrize_nac: whether or not to symmetrize the nac parameters
                using point group symmetry; bool, defaults to self.is_symmetry

        :return: tuple (polarization vectors, frequencies, irreps labels)
        """
        try:
            nac_direction = nac_direction()
        except TypeError:
            pass
        if not isinstance(use_irreps, bool) or not isinstance(sum_rules, bool):
            raise TypeError('the input is not of the correct type')

        phonopy_instance = self.get_phonopy_instance(**kwargs)

        if phonopy_instance.force_constants is None:
            phonopy_instance.produce_force_constants()

        if sum_rules:
            phonopy_instance.symmetrize_force_constants()
            phonopy_instance.symmetrize_force_constants_by_space_group()

        results = compute_polarization_vectors(
            phonopy_instance=phonopy_instance,
            nac_direction=nac_direction,
            use_irreps=use_irreps,
            degeneracy_tolerance=degeneracy_tolerance,
            sum_rules=sum_rules,
        )

        return results

    def run_polarized_raman_intensities(
        self,
        pol_incoming: tuple[float, float, float],
        pol_outgoing: tuple[float, float, float],
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return polarized Raman intensities.

        .. note:: Units are:
                * Intensities: Anstrom/AMU
                * Frequencies: cm-1

        :param pol_incoming: light polarization vector of the incoming light
            (laser) in crystal/fractional coordinates
        :type pol_incoming: list or numpy.ndarray of shape (3,)
        :param pol_outgoing: light polarization vector of the outgoing light
            (scattered) in crystal/fractional coordinates
        :type pol_outgoing: list or numpy.ndarray of shape (3,)
        :param kwargs: keys of
        :func:`~aiida_vibroscopy.calculations.spectra_utils.compute_raman_susceptibility_tensors` method
            * nac_direction: non-analytical direction in reciprocal space coordinates (primitive cell)
            * use_irreps: whether to use irreducible representations
                in the selection of modes, defaults to True; bool, optional
            * degeneracy_tolerance: degeneracy tolerance for
                irreducible representation

        :return: tuple (Raman intensities, frequencies, labels)
        """
        if not isinstance(pol_incoming, (list, np.ndarray)) or not isinstance(pol_outgoing, (list, np.ndarray)):
            raise TypeError('the input is not of the correct type')

        pol_incoming_crystal = np.array(pol_incoming)
        pol_outgoing_crystal = np.array(pol_outgoing)

        cell = self.get_phonopy_instance().primitive.cell
        pol_incoming_cart = np.dot(cell, pol_incoming_crystal)  # in Cartesian coordinates
        pol_outgoing_cart = np.dot(cell, pol_outgoing_crystal)  # in Cartesian coordinates

        if pol_incoming_crystal.shape != (3,) or pol_outgoing_crystal.shape != (3,):
            raise ValueError('the array is not of the correct shape')

        raman_susceptibility_tensors, freqs, labels = self.run_raman_susceptibility_tensors(**kwargs)

        raman_intensities = [
            np.dot(pol_incoming_cart, np.dot(tensor, pol_outgoing_cart)) for tensor in raman_susceptibility_tensors
        ]
        raman_intensities = [intensity**2 for intensity in raman_intensities]

        return (raman_intensities, freqs, labels)

    def run_powder_raman_intensities(self,
                                     quadrature_order: int | None = None,
                                     **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return powder Raman intensities.

        ..important: it computes the common setups of polarized (HH) and unpolarized (HV)
        scattering. To obtain the total powder intensities, sum the two returned arrays of the intensities.

        .. note:: Units are:
                * Intensities: Anstrom/AMU
                * Frequencies: cm-1

        :param quadrature_order: algebraic order to perform the integration
            on the sphere of nac directions
        :param kwargs: keys of
        :func:`~aiida_vibroscopy.calculations.spectra_utils.compute_raman_susceptibility_tensors` method
            * nac_direction: non-analytical direction in reciprocal space coordinates (primitive cell)
            * use_irreps: whether to use irreducible representations
                in the selection of modes, defaults to True; bool, optional
            * degeneracy_tolerance: degeneracy tolerance for
                irreducible representation

        :return: (Raman intensities HH, Raman intensities HV, frequencies, labels)
        """
        raman_hh = []
        raman_hv = []
        if quadrature_order is None:
            raman_susceptibility_tensors, freqs, labels = self.run_raman_susceptibility_tensors(**kwargs)
            raman_hh, raman_hv = compute_raman_space_average(raman_susceptibility_tensors=raman_susceptibility_tensors)

        else:
            cell = self.get_phonopy_instance().primitive.cell

            scheme = LebedevScheme.from_order(quadrature_order)
            points = scheme.points.T
            weights = scheme.weights

            freqs = []
            labels = []

            kwargs.pop('nac_direction', None)

            for q, ws in zip(points, weights):
                q_crystal = np.dot(cell, q)  # in reciprocal fractional/crystal coordinates
                q_tensors, q_freqs, q_labels = self.run_raman_susceptibility_tensors(
                    nac_direction=q_crystal,
                    **kwargs,
                )

                q_raman_hh, q_raman_hv = compute_raman_space_average(raman_susceptibility_tensors=q_tensors)
                raman_hh.append(ws * q_raman_hh)
                raman_hv.append(ws * q_raman_hv)
                freqs += q_freqs.tolist()
                labels += q_labels

        return (np.array(raman_hh).flatten(), np.array(raman_hv).flatten(), np.array(freqs), labels)

    def run_polarized_ir_intensities(self, pol_incoming: tuple[float, float, float],
                                     **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return polarized IR intensities.

        .. note:: Units are:
            * Intensities: (Debey/Angstrom)^2/AMU
            * Frequencies: cm^-1

        :param pol_incoming: light polarization vector of the
            incident beam light in crystal coordinates
        :type pol_incoming: list or numpy.ndarray of shape (3,)
        :param kwargs: keys of :func:`~aiida_phonopy.data.phonopy.get_phonopy_instance` method
            * nac_direction: non-analytical direction in fractional coordinates (primitive cell)
                in reciprocal space; space(3,) shape list or numpy.ndarray
            * use_irreps: whether to use irreducible representations in the
                selection of modes, defaults to True; bool, optional
            * sum_rules: whether to apply sum rules to the derivatives of the
                susceptibility in respect to atomic positions; bool, optional
            * subtract_residual_forces: whether or not subract residual forces (if set);
                bool, defaults to False
            * symmetrize_nac: whether or not to symmetrize the nac parameters
                using point group symmetry; bool, defaults to self.is_symmetry
            * factor_nac: factor for non-analytical corrections;
                float, defaults to Hartree*Bohr

        :return: tuple(intensities, frequencies, labels).
            Units are:
                * Intensities: (Debey/Angstrom)^2/AMU
                * Frequencies: cm^-1
        """
        if not isinstance(pol_incoming, (list, np.ndarray)):
            raise TypeError('the input is not of the correct type')

        pol_incoming_crystal = np.array(pol_incoming)

        if pol_incoming_crystal.shape != (3,):
            raise ValueError('the array is not of the correct shape')

        cell = self.get_phonopy_instance().primitive.cell
        pol_incoming_cart = np.dot(cell, pol_incoming_crystal)  # in Cartesian coordinates

        pol_vectors, freqs, labels = self.run_polarization_vectors(**kwargs)

        ir_intensities = [np.dot(vector, pol_incoming_cart) for vector in pol_vectors]
        ir_intensities = [intensity**2 for intensity in ir_intensities]

        return (ir_intensities, freqs, labels)

    def run_powder_ir_intensities(self,
                                  quadrature_order: int | None = None,
                                  **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return powder IR intensities, frequencies, and labels.

        .. note:: Units are:
            * Intensities: (Debey/Angstrom)^2/AMU
            * Frequencies: cm^-1

        :param quadrature_order: algebraic order to perform the integration
            on the sphere of nac directions
        :param kwargs: keys of :func:`~aiida_phonopy.data.phonopy.get_phonopy_instance` method
            * nac_direction: non-analytical direction in fractional coordinates (primitive cell)
                in reciprocal space; space(3,) shape list or numpy.ndarray
            * use_irreps: whether to use irreducible representations in the
                selection of modes, defaults to True; bool, optional
            * sum_rules: whether to apply sum rules to the derivatives of the
                susceptibility in respect to atomic positions; bool, optional
            * subtract_residual_forces: whether or not subract residual forces (if set);
                bool, defaults to False
            * symmetrize_nac: whether or not to symmetrize the nac parameters
                using point group symmetry; bool, defaults to self.is_symmetry
            * factor_nac: factor for non-analytical corrections;
                float, defaults to Hartree*Bohr

        :return: tuple(intensities, frequencies, labels).
            Units are:
                * Intensities: (Debey/Angstrom)^2/AMU
                * Frequencies: cm^-1
        """
        ir_intensities = []

        if quadrature_order is None:
            pol_vectors, freqs, labels = self.run_polarization_vectors(**kwargs)

            for pol in pol_vectors:
                ir_intensities.append(np.dot(pol, pol))
        else:
            scheme = LebedevScheme.from_order(quadrature_order)
            points = scheme.points.T
            weights = scheme.weights
            freqs = []
            labels = []
            kwargs.pop('nac_direction', None)

            for q, ws in zip(points, weights):
                cell = self.get_phonopy_instance().primitive.cell
                q_crystal = np.dot(cell, q)  # in reciprocal fractional/Crystal coordinates
                q_pol, q_freqs, q_labels = self.run_polarization_vectors(**kwargs, **{'nac_direction': q_crystal})

                for pol, f, l in zip(q_pol, q_freqs, q_labels):
                    ir_intensities.append(ws * np.dot(pol, pol))
                    freqs.append(f)
                    labels.append(l)

        return (np.array(ir_intensities), np.array(freqs), labels)

    @staticmethod
    def get_available_quadrature_order_schemes():
        """Return the available orders for quadrature integration on the nac direction unitary sphere."""
        from aiida_vibroscopy.utils.integration.lebedev import get_available_quadrature_order_schemes
        get_available_quadrature_order_schemes()
