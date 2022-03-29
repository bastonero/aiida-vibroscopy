# -*- coding: utf-8 -*-
"""Mixin for aiida-quantumespresso-vibroscopy DataTypes."""

import numpy as np
from math import fabs, sqrt, pi, exp

from aiida import orm
from aiida_quantumespresso_vibroscopy.common import UNITS_FACTORS


__all__ = ('VibrationalMixin')

def boson_factor(frequency, temperature):
    """Return boson factor, i.e. (nb+1). Frequency in cm-1 and temperature in Kelvin."""
    return 1.0/( 1.0 -exp(-UNITS_FACTORS.cm_to_kelvin*frequency/temperature) )


def _get_active_modes(phonopy_instance, nac_direction, degeneracy_tolerance, selection_rule=None, sr_thr=1e-4):
    """Get frequencies, normalized eigenvectors and irreducible representation labels of active modes
    for calculation of polarization vectors and Raman tensors.

    :param selection_rule: str, can be `raman` or `ir`; it uses symmetry in the selection of the modes
        for a specific type of process.
    :param sr_thr: float, threshold for selection rule (the analytical value is 0).

    :return: tuple of (frequencies in cm-1, normalized eigenvectors, labels); normalized eigenvectors is an
        array of shape (num modes, num atoms, 3)."""

    if selection_rule not in ('raman', 'ir'):
        raise ValueError('`selection_rule` can only be `ir` or `raman`.')

    # Step 1 - set the irreducible representations and the phonons
    phonopy_instance.set_irreps(q=[0,0,0], nac_q_direction=nac_direction, degeneracy_tolerance=degeneracy_tolerance)
    irreps = phonopy_instance.irreps

    phonopy_instance.run_qpoints(q_points=[0,0,0], nac_q_direction=nac_direction, with_eigenvectors=True)
    frequencies = phonopy_instance.qpoints.frequencies[0]*UNITS_FACTORS.thz_to_cm
    eigvectors = phonopy_instance.qpoints.eigenvectors.T.real

    # Step 2 - getting the active modes with eigenvectors
    Xr = []
    for mat in irreps.get_rotations():
        Xr.append(mat.trace())
    Xr = np.array(Xr)

    freq_active_modes = []
    eigvectors_active_modes = []
    labels_active_modes = []

    bands_indices = irreps.get_band_indices()
    characters = irreps.get_characters().real
    labels = irreps._get_ir_labels()

    mode_index = 0

    for band_indices, Xi, label in zip(bands_indices, characters, labels):
        degeneracy = len(band_indices)
        if mode_index > 2: # excluding the acustic modes
            # Using selection rules (symmetry) constrains
            if selection_rule is not None:
                if selection_rule == 'raman':
                    condition =  np.dot(Xr*Xr, Xi)
                elif selection_rule == 'ir':
                    condition =  np.dot(Xr, Xi)
            else:
                condition = 10 # a number > 0

            if fabs(condition) > sr_thr: # selection rule (thr for inaccuracies)

                for band_index in band_indices:
                    freq_active_modes.append(frequencies[band_index])
                    eigvectors_active_modes.append(eigvectors[band_index])
                    labels_active_modes.append(label)

        mode_index += degeneracy

    freq_active_modes = np.array( freq_active_modes )

    # Step 3 - getting normalized eigenvectors
    masses = phonopy_instance.masses
    inv_sqrt_masses = np.array([ [1./sqrt(mass)] for mass in masses])
    eigvectors_active_modes = np.array( eigvectors_active_modes ).reshape(len(freq_active_modes), len(masses), 3)

    norm_eigvectors_active_modes = np.array( [eigv/inv_sqrt_masses for eigv in eigvectors_active_modes] )

    return (freq_active_modes, norm_eigvectors_active_modes, labels)


def _get_raman_tensors(phonopy_instance, dph0_susceptibility, nlo_susceptibility=None, nac_direction=[0,0,0], use_irreps=True, sum_rules=False, degeneracy_tolerance=1e-5, **kwargs):
        """Return the Raman tensors (in angstrom^2/sqrt(AMU)) along with each phonon mode with frequencies (cm-1) and labels.

        :param phonopy_instance: Phonopy instance with non-analytical constants included
        :param nac_direction: non-analytical direction
        :param dph0_susceptibility: derivatives of the susceptibility in respect to atomic positions in Cartesian coordinates and in angstrom^2
        :param nlo_susceptibility: non linear optical susceptibility in Cartesian coordinates and in pm/V
        :type nac_direction: (3,) shape list or numpy.ndarray
        :param use_irreps: whether to use irreducible representations in the selection of modes, defaults to True
        :type use_irreps: bool, optional
        :param degeneracy_tolerance: degeneracy tolerance for irreducible representation

        :return: tuple (Raman tensors, frequencies, labels)
        """
        if not isinstance(nac_direction, (list, np.ndarray)) or not isinstance(use_irreps, bool):
            raise TypeError('the input is not of the correct type')

        nac_direction = np.array(nac_direction)

        if not nac_direction.shape == (3,):
            raise ValueError('the array is not of the correct shape')

        selection_rule = 'raman' if use_irreps else None

        if sum_rules:
            sum_rule_correction = np.zeros((3,3,3))
            for tensor in dph0_susceptibility:
                sum_rule_correction += tensor
            dph0_susceptibility = dph0_susceptibility-sum_rule_correction

        freqs, neigvs, labels = _get_active_modes(
            phonopy_instance=phonopy_instance,
            nac_direction=nac_direction,
            degeneracy_tolerance=degeneracy_tolerance,
            selection_rule=selection_rule
        )

        # neigvs shape|indices = (num modes, num atoms, 3) | (n, I, k)
        # dph0   shape|indices = (num atoms, 3, 3, 3) | (I, k, i, j)
        # The contruction is performed over I and k, resulting in (n, i, j) Raman tensors.
        raman_tensors = np.tensordot(neigvs, dph0_susceptibility, axes=([1,2],[0,1]))

        if nlo_susceptibility is not None:
            borns = phonopy_instance.nac_params['born']
            dielectric = phonopy_instance.nac_params['dielectric']

            # q.epsilon.q
            dielectric_term = np.dot(np.dot(dielectric, nac_direction), nac_direction)
            # Z*.q
            borns_term = np.array( [np.dot(born, nac_direction) for born in borns])
            # Chi(2).q
            nlo_term = np.dot(nlo_susceptibility, nac_direction)
            # -8 pi (Z.q/q.epsilon.q) Chi(2).q is the correction to dph0.
            # The indices I, k to do the scalar product with the eigenvectors run over the Borns term.
            nlo_correction = -(8.*pi/(100.*dielectric_term))*np.tensordot(borns_term, neigvs, axes=([0,1],[0,1]))*nlo_term
            raman_tensors = raman_tensors + nlo_correction

        return (raman_tensors, freqs, labels)

def _get_polarization_vectors(phonopy_instance, nac_direction=[0,0,0], use_irreps=True, degeneracy_tolerance=1e-5, sum_rules=False, **kwargs):
        """Return the polarization vectors (in 1/sqrt(AMU)) for each phonon mode with frequencies (cm-1) and labels.

        :param phonopy_instance: Phonopy instance with non-analytical constants included
        :param nac_direction: non-analytical direction
        :type nac_direction: (3,) shape list or numpy.ndarray
        :param use_irreps: whether to use irreducible representations in the selection of modes, defaults to True
        :type use_irreps: bool, optional
        :param degeneracy_tolerance: degeneracy tolerance for irreducible representation

        :return: tuple (Raman tensors, frequencies, labels)
        """
        if not isinstance(nac_direction, (list, np.ndarray)) or not isinstance(use_irreps, bool):
            raise TypeError('the input is not of the correct type')

        nac_direction = np.array(nac_direction)

        if not nac_direction.shape == (3,):
            raise ValueError('the array is not of the correct shape')

        selection_rule = 'ir' if use_irreps else None

        freqs, neigvs, labels = _get_active_modes(
            phonopy_instance=phonopy_instance,
            nac_direction=nac_direction,
            degeneracy_tolerance=degeneracy_tolerance,
            selection_rule=selection_rule
        )

        borns = phonopy_instance.nac_params['born']

        if sum_rules:
            sum_rule_correction = np.zeros((3,3))
            for tensor in borns:
                sum_rule_correction += tensor
            borns = borns-sum_rule_correction

        # neigvs shape|indices = (num modes, num atoms, 3) | (n, I, k)
        # borns  shape|indices = (num atoms, 3, 3) | (I, k, i)
        # The contruction is performed over I and k, resulting in (n, i) polarization vectors.
        pol_vectors = np.tensordot(neigvs, borns, axes=([1,2],[0,1]))

        return (pol_vectors, freqs, labels)

class VibrationalMixin:
    """Mixin class for vibrational DataTypes.

    This is meant to be used to extend aiida-phonopy DataTypes to include tensors
    and information for vibrational spectra."""

    @property
    def dph0_susceptibility(self):
        """Get the derivatives of the susceptibility tensor in respect to atomic positions in Cartesian coordinates."""
        try:
            value = self.get_array('dph0_susceptibility')
        except (KeyError, AttributeError):
            value = None
        return value


    def set_dph0_susceptibility(self, dph0_susceptibility):
        """Set the derivatives of the susceptibility tensor in respect to atomic positions in Cartesian coordinates.

        .. note: it is assumed that the reference system is the same (if not the) of the primitive cell.

        :param dielectric: (number of atoms in the primitive cell, 3 3, 3) array like

        :raises:
            * TypeError: if the format is not compatible or of the correct type
            * ValueError: if the format is not compatible or of the correct type
        """
        self._if_can_modify()

        if not isinstance(dph0_susceptibility, (list, np.ndarray)):
            raise TypeError('the input is not of the correct type')

        the_dchi = np.array(dph0_susceptibility)
        nprimitive_atoms = len(self.get_primitive_cell().sites)

        if the_dchi.shape == (nprimitive_atoms, 3, 3, 3):
            self.set_array('dph0_susceptibility', the_dchi)
        else:
            raise ValueError('the array is not of the correct shape')

    @property
    def nlo_susceptibility(self):
        """Get the non linear optical susceptibility tensor in Cartesian coordinates."""
        try:
            value = self.get_array('nlo_susceptibility')
        except (KeyError, AttributeError):
            value = None
        return value


    def set_nlo_susceptibility(self, nlo_susceptibility):
        """Set the non linear optical susceptibility tensor in Cartesian coordinates.

        .. note: it is assumed that the reference system is the same (if not the) of the primitive cell.

        :param dielectric: (3 3, 3) array like

        :raises:
            * TypeError: if the format is not compatible or of the correct type
            * ValueError: if the format is not compatible or of the correct type
        """
        self._if_can_modify()

        if not isinstance(nlo_susceptibility, (list, np.ndarray)):
            raise TypeError('the input is not of the correct type')

        the_chi2 = np.array(nlo_susceptibility)

        if the_chi2.shape == (3, 3, 3):
            self.set_array('nlo_susceptibility', the_chi2)
        else:
            raise ValueError('the array is not of the correct shape')


    def has_raman_parameters(self):
        """Returns wheter or not the Data has derivatives of susceptibility for Raman spectra."""
        if self.dph0_susceptibility is not None:
            return True
        else:
            return False

    def has_nlo(self):
        """Returns wheter or not the Data has non linear optical susceptibility tensor."""
        if self.nlo_susceptibility is not None:
            return True
        else:
            return False


    def get_raman_tensors(self, nac_direction=[0,0,0], with_nlo=False, use_irreps=True, degeneracy_tolerance=1e-5, sum_rules=False, **kwargs):
        """Return the Raman tensors (in angstrom^2/sqrt(AMU)) for each phonon mode, along with frequencies (cm-1) and irreps labels.

        :param nac_direction: non-analytical direction
        :type nac_direction: (3,) shape list or numpy.ndarray
        :param with_nlo: whether to use or not non-linear optical susceptibility correction (Froehlich term), defaults to True
        :type with_nlo: bool, optional
        :param use_irreps: whether to use irreducible representations in the selection of modes, defaults to True
        :type use_irreps: bool, optional
        :param sum_rules: whether to apply sum rules to the derivatives of the susceptibility in respect to atomic positions
        :type sum_rules: bool, optional
        :param kwargs: kwargs of the respective class `get_phonopy_instance` method

        :return: tuple (Raman tensors, frequencies, irreps labels)
        """
        if not isinstance(with_nlo, bool) or not isinstance(use_irreps, bool)  or not isinstance(sum_rules, bool):
            raise TypeError('the input is not of the correct type')

        phonopy_instance = self.get_phonopy_instance(**kwargs)

        if phonopy_instance.force_constants is None:
            phonopy_instance.produce_force_constants()

        nlo_susceptibility = self.nlo_susceptibility if with_nlo else None

        results = _get_raman_tensors(
            phonopy_instance=phonopy_instance,
            dph0_susceptibility=self.dph0_susceptibility,
            nlo_susceptibility=nlo_susceptibility,
            nac_direction=nac_direction,
            use_irreps=use_irreps,
            degeneracy_tolerance=degeneracy_tolerance,
            sum_rules=sum_rules,
        )

        return results


    def get_polarization_vectors(self, nac_direction=[0,0,0], use_irreps=True, degeneracy_tolerance=1e-5, sum_rules=False, **kwargs):
        """Return the polarization vectors (in 1/sqrt(AMU)) for each phonon mode, along with frequencies (cm-1) and irreps labels.

        :param nac_direction: non-analytical direction
        :type nac_direction: (3,) shape list or numpy.ndarray
        :param use_irreps: whether to use irreducible representations in the selection of modes, defaults to True
        :type use_irreps: bool, optional
        :param sum_rules: whether to apply sum rules to the derivatives of the susceptibility in respect to atomic positions
        :type sum_rules: bool, optional
        :param kwargs: kwargs of the respective class `get_phonopy_instance` method

        :return: tuple (polarization vectors, frequencies, irreps labels)
        """
        if not isinstance(use_irreps, bool)  or not isinstance(sum_rules, bool):
            raise TypeError('the input is not of the correct type')

        phonopy_instance = self.get_phonopy_instance(**kwargs)

        if phonopy_instance.force_constants is None:
            phonopy_instance.produce_force_constants()

        results = _get_polarization_vectors(
            phonopy_instance=phonopy_instance,
            nac_direction=nac_direction,
            use_irreps=use_irreps,
            degeneracy_tolerance=degeneracy_tolerance,
            sum_rules=sum_rules,
        )

        return results


    def get_polarized_raman_intensities(self, pol_incoming, pol_outgoing, **kwargs):
        """Return polarized Raman intensities (in angstrom^4/AMU) with frequencies (cm-1) and irreps labels.

        :param pol_incoming: light polarization vector of the incoming light (laser)
        :type pol_incoming: list or numpy.ndarray of shape (3,)
        :param pol_outgoing: light polarization vector of the outgoing light
        :type pol_outgoing: list or numpy.ndarray of shape (3,)
        :param kwargs: keys of `get_raman_tensors` method
        """
        if not isinstance(pol_incoming, (list, np.ndarray)) or not isinstance(pol_outgoing, (list, np.ndarray)):
            raise TypeError('the input is not of the correct type')

        pol_incoming = np.array(pol_incoming)
        pol_outgoing = np.array(pol_outgoing)

        if not pol_incoming.shape == (3,) or not pol_outgoing.shape == (3,):
            raise ValueError('the array is not of the correct shape')

        raman_tensors, freqs, labels = self.get_raman_tensors(**kwargs)

        raman_intensities = [np.dot(np.dot(tensor, pol_incoming), pol_outgoing) for tensor in raman_tensors]
        raman_intensities = [intensity**2 for intensity in raman_intensities]

        return (raman_intensities, freqs, labels)


    def get_powder_raman_intensities(self,  quadrature_order=None, **kwargs):
        """Return unpolarized powder Raman intensities (in angstrom^4/AMU) in the two common setups
        of back scattering and 90 degress scattering, with frequencies (cm-1) and labels.

        .. note: to obtain the total unpolarized intensities, sum the two returned arrays of the intensities.

        :param quadrature_order: algebraic order to perform the integration on the sphere of nac directions
        :type pol_incoming: int, optional
        :param kwargs: keys of `get_raman_tensors` method

        :return: (Raman intensities HH, Raman intensities HV, frequencies, labels)
        """
        import quadpy

        raman_hh = []
        raman_hv = []
        if quadrature_order is None:
            raman_tensors, freqs, labels = self.get_raman_tensors(**kwargs)

            for R in raman_tensors:
                a = R.trace()/3.0
                a2 = a*a
                b2 = (
                    0.5*(
                        (R[0][0]-R[1][1])**2 +
                        (R[0][0]-R[2][2])**2 +
                        (R[1][1]-R[2][2])**2
                        )
                    +3.*(
                        R[0][1]**2+
                        R[0][2]**2+
                        R[1][2]**2
                    )
                )
                raman_hh.append(45*a2 +4*b2)
                raman_hv.append(3*b2)
        else:
            scheme = quadpy.u3.get_good_scheme(quadrature_order)
            points = scheme.points.transpose()
            weights = scheme.weights
            freqs = []
            labels = []
            kwargs.pop('nac_direction', None)

            for q, ws in zip (points, weights):
                q_tensors, q_freqs, q_labels = self.get_raman_tensors(**kwargs, **{'nac_direction':q})

                for R, f, l in zip(q_tensors, q_freqs, q_labels):
                    a = R.trace()/3.0
                    a2 = a*a
                    b2 = (
                        0.5*(
                            (R[0][0]-R[1][1])**2 +
                            (R[0][0]-R[2][2])**2 +
                            (R[1][1]-R[2][2])**2
                            )
                        +3.*(
                            R[0][1]**2+
                            R[0][2]**2+
                            R[1][2]**2
                        )
                    )
                    raman_hh.append(4*pi*ws*(45*a2 +4*b2))
                    raman_hv.append(4*pi*ws*(3*b2))
                    freqs.append(f)
                    labels.append(l)

        return (4*pi*np.array(raman_hh)/45., 4*pi*np.array(raman_hv)/45., np.array(freqs), labels)


    def get_polarized_ir_intensities(self, pol_incoming, **kwargs):
        """Return polarized IR intensities (in 1/AMU) with frequencies (cm-1) and irreps labels.

        :param pol_incoming: light polarization vector of the incident beam light
        :type pol_incoming: list or numpy.ndarray of shape (3,)
        :param kwargs: keys of `get_raman_tensors` method
        """
        if not isinstance(pol_incoming, (list, np.ndarray)):
            raise TypeError('the input is not of the correct type')

        pol_incoming = np.array(pol_incoming)

        if not pol_incoming.shape == (3,):
            raise ValueError('the array is not of the correct shape')

        pol_vectors, freqs, labels = self.get_polarization_vectors(**kwargs)

        ir_intensities = [ np.dot(tensor, pol_incoming)  for tensor in pol_vectors]
        ir_intensities = [intensity**2 for intensity in ir_intensities]

        return (ir_intensities, freqs, labels)


    def get_powder_ir_intensities(self,  quadrature_order=None, **kwargs):
        """Return unpolarized powder IR intensities (in 1/AMU) with frequencies (cm-1) and labels.

        :param quadrature_order: algebraic order to perform the integration on the sphere of nac directions
        :type pol_incoming: int, optional
        :param kwargs: keys of `get_polarization_vectors` method

        :return: (IR intensities, frequencies, labels)
        """
        import quadpy

        ir_intensities = []

        if quadrature_order is None:
            pol_vectors, freqs, labels = self.get_polarization_vectors(**kwargs)

            for pol in pol_vectors:
                ir_intensities.append(np.dot(pol,pol))
        else:
            scheme = quadpy.u3.get_good_scheme(quadrature_order)
            points = scheme.points.transpose()
            weights = scheme.weights
            freqs = []
            labels = []
            kwargs.pop('nac_direction', None)

            for q, ws in zip (points, weights):
                q_pol, q_freqs, q_labels = self.get_polarization_vectors(**kwargs, **{'nac_direction':q})

                for pol, f, l in zip(q_pol, q_freqs, q_labels):
                    ir_intensities.append(ws*np.dot(pol,pol))
                    freqs.append(f)
                    labels.append(l)

        return (4*pi*np.array(ir_intensities), np.array(freqs), labels)
