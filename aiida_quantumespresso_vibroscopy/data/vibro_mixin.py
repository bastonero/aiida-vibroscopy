# -*- coding: utf-8 -*-
"""Mixin for aiida-quantumespresso-vibroscopy DataTypes."""

import numpy as np
from math import pi

from aiida_quantumespresso_vibroscopy.calculations.spectra_utils import (
    compute_active_modes,
    compute_raman_space_average,
    compute_raman_tensors,
    compute_polarization_vectors,
    compute_polarization_vectors,
)

__all__ = [
    'VibrationalMixin'
]

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

        .. note: it is assumed that the reference system is the same (if not the one) of the primitive cell.

        :param dph0_susceptibility: (number of atoms in the primitive cell, 3, 3, 3) array like

        :raises:
            * TypeError: if the format is not compatible or of the correct type
            * ValueError: if the format is not compatible or of the correct type
        """
        self._if_can_modify()

        if not isinstance(dph0_susceptibility, (list, np.ndarray)):
            raise TypeError('the input is not of the correct type')

        the_dchi = np.array(dph0_susceptibility)
        n_atoms = len(self.get_primitive_cell().sites)

        if the_dchi.shape == (n_atoms, 3, 3, 3):
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

        .. note: it is assumed that the reference system is the same (if not the one) of the primitive cell.

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

    def run_active_modes(self, degeneracy_tolerance=1.e-5, nac_direction=None, selection_rule=None, sr_thr=1e-4, **kwargs):
        """Get active modes frequencies, eigenvectors and labels."""
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

    def run_raman_tensors(self, nac_direction=[0,0,0], with_nlo=False, use_irreps=True, degeneracy_tolerance=1e-5, sum_rules=False, **kwargs):
        """Return the Raman tensors (in angstrom^2/sqrt(AMU)) for each phonon mode, along with frequencies (cm-1) and irreps labels.

        :param nac_direction: non-analytical direction
        :type nac_direction: non-analytical direction in fractional coordinates in reciprocal space; (3,) shape list or numpy.ndarray
        :param with_nlo: whether to use or not non-linear optical susceptibility correction (Froehlich term), defaults to True
        :type with_nlo: bool, optional
        :param use_irreps: whether to use irreducible representations in the selection of modes, defaults to True
        :type use_irreps: bool, optional
        :param sum_rules: whether to apply sum rules to the derivatives of the susceptibility in respect to atomic positions
        :type sum_rules: bool, optional
        :param kwargs: keys of :func:`~aiida_phonopy.data.phonopy.get_phonopy_instance` method

        :return: tuple (Raman tensors, frequencies, irreps labels)
        """
        if not isinstance(with_nlo, bool) or not isinstance(use_irreps, bool)  or not isinstance(sum_rules, bool):
            raise TypeError('the input is not of the correct type')

        phonopy_instance = self.get_phonopy_instance(**kwargs)

        if phonopy_instance.force_constants is None:
            phonopy_instance.produce_force_constants()

        if sum_rules:
            phonopy_instance.symmetrize_force_constants()
            phonopy_instance.symmetrize_force_constants_by_space_group()

        nlo_susceptibility = self.nlo_susceptibility if with_nlo else None

        results = compute_raman_tensors(
            phonopy_instance=phonopy_instance,
            dph0_susceptibility=self.dph0_susceptibility,
            nlo_susceptibility=nlo_susceptibility,
            nac_direction=nac_direction,
            use_irreps=use_irreps,
            degeneracy_tolerance=degeneracy_tolerance,
            sum_rules=sum_rules,
        )

        return results


    def run_polarization_vectors(self, nac_direction=[0,0,0], use_irreps=True, degeneracy_tolerance=1e-5, sum_rules=False, **kwargs):
        """Return the polarization vectors (in (debey/angstrom)/sqrt(AMU) ) for each phonon mode, along with frequencies (cm-1) and irreps labels.

        :param nac_direction: non-analytical direction
        :type nac_direction: non-analytical direction in fractional coordinates in reciprocal space; space(3,) shape list or numpy.ndarray
        :param use_irreps: whether to use irreducible representations in the selection of modes, defaults to True
        :type use_irreps: bool, optional
        :param sum_rules: whether to apply sum rules to the derivatives of the susceptibility in respect to atomic positions
        :type sum_rules: bool, optional
        :param kwargs: keys of :func:`~aiida_phonopy.data.phonopy.get_phonopy_instance` method

        :return: tuple (polarization vectors, frequencies, irreps labels)
        """
        if not isinstance(use_irreps, bool)  or not isinstance(sum_rules, bool):
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


    def run_polarized_raman_intensities(self, pol_incoming, pol_outgoing, **kwargs):
        """Return polarized Raman intensities (in angstrom^4/AMU) with frequencies (cm-1) and irreps labels.

        :param pol_incoming: light polarization vector of the incoming light (laser)
        :type pol_incoming: list or numpy.ndarray of shape (3,)
        :param pol_outgoing: light polarization vector of the outgoing light
        :type pol_outgoing: list or numpy.ndarray of shape (3,)
        :param kwargs: keys of :func:`~aiida_quantumespresso_vibroscopy.calculations.spectra_utils.compute_raman_tensors` method
        """
        if not isinstance(pol_incoming, (list, np.ndarray)) or not isinstance(pol_outgoing, (list, np.ndarray)):
            raise TypeError('the input is not of the correct type')

        pol_incoming = np.array(pol_incoming)
        pol_outgoing = np.array(pol_outgoing)

        if not pol_incoming.shape == (3,) or not pol_outgoing.shape == (3,):
            raise ValueError('the array is not of the correct shape')

        raman_tensors, freqs, labels = self.run_raman_tensors(**kwargs)

        raman_intensities = [np.dot(np.dot(tensor, pol_outgoing), pol_incoming) for tensor in raman_tensors]
        raman_intensities = [intensity**2 for intensity in raman_intensities]

        return (raman_intensities, freqs, labels)


    def run_powder_raman_intensities(self,  quadrature_order=None, **kwargs):
        """Return unpolarized powder Raman intensities (in angstrom^4/AMU) in the two common setups
        of back scattering and 90 degress scattering, with frequencies (cm-1) and labels.

        .. note: to obtain the total unpolarized intensities, sum the two returned arrays of the intensities.

        :param quadrature_order: algebraic order to perform the integration on the sphere of nac directions
        :type pol_incoming: int, optional
        :param kwargs: keys of :func:`~aiida_quantumespresso_vibroscopy.calculations.spectra_utils.compute_raman_tensors` method

        :return: (Raman intensities HH, Raman intensities HV, frequencies, labels)
        """
        raman_hh = []
        raman_hv = []
        if quadrature_order is None:
            raman_tensors, freqs, labels = self.run_raman_tensors(**kwargs)
            raman_hh, raman_hv = compute_raman_space_average(raman_tensors=raman_tensors)

        else:
            scheme = self._get_scheme_from_order(quadrature_order)
            points = scheme.points.transpose()
            weights = scheme.weights
            freqs = []
            labels = []
            kwargs.pop('nac_direction', None)

            for q, ws in zip (points, weights):
                q_tensors, q_freqs, q_labels = self.run_raman_tensors(**kwargs, **{'nac_direction':q})

                q_raman_hh, q_raman_hv = compute_raman_space_average(raman_tensors=q_tensors)
                raman_hh.append(4*pi*ws*q_raman_hh)
                raman_hv.append(4*pi*ws*q_raman_hv)
                freqs  += q_freqs.tolist()
                labels += q_labels

        return (np.array(raman_hh).flatten(), np.array(raman_hv).flatten(), np.array(freqs), labels)


    def run_polarized_ir_intensities(self, pol_incoming, **kwargs):
        """Return polarized IR intensities (in (debey/angstrom)^2/AMU) with frequencies (cm-1) and irreps labels.

        :param pol_incoming: light polarization vector of the incident beam light
        :type pol_incoming: list or numpy.ndarray of shape (3,)
        :param kwargs: keys of `.run_polarization_vectors` method
        """
        if not isinstance(pol_incoming, (list, np.ndarray)):
            raise TypeError('the input is not of the correct type')

        pol_incoming = np.array(pol_incoming)

        if not pol_incoming.shape == (3,):
            raise ValueError('the array is not of the correct shape')

        pol_vectors, freqs, labels = self.run_polarization_vectors(**kwargs)

        ir_intensities = [ np.dot(vector, pol_incoming)  for vector in pol_vectors]
        ir_intensities = [intensity**2 for intensity in ir_intensities]

        return (ir_intensities, freqs, labels)


    def run_powder_ir_intensities(self,  quadrature_order=None, **kwargs):
        """Return unpolarized powder IR intensities (in (debey/angstrom)^2/AMU) with frequencies (cm-1) and labels.

        :param quadrature_order: algebraic order to perform the integration on the sphere of nac directions
        :type pol_incoming: int, optional
        :param kwargs: keys of `.run_polarization_vectors` method

        :return: (IR intensities, frequencies, labels)
        """
        ir_intensities = []

        if quadrature_order is None:
            pol_vectors, freqs, labels = self.run_polarization_vectors(**kwargs)

            for pol in pol_vectors:
                ir_intensities.append(np.dot(pol,pol))
        else:
            scheme = self._get_scheme_from_order(quadrature_order)
            points = scheme.points.transpose()
            weights = scheme.weights
            freqs = []
            labels = []
            kwargs.pop('nac_direction', None)

            for q, ws in zip (points, weights):
                q_pol, q_freqs, q_labels = self.run_polarization_vectors(**kwargs, **{'nac_direction':q})

                for pol, f, l in zip(q_pol, q_freqs, q_labels):
                    ir_intensities.append(4*pi*ws*np.dot(pol,pol))
                    freqs.append(f)
                    labels.append(l)

        return (4*pi*np.array(ir_intensities), np.array(freqs), labels)

    @staticmethod
    def get_available_quadrature_order_schemes():
        """Return the available orders for quadrature integration on the nac direction unitary sphere."""
        import quadpy
        from quadpy.u3  import _lebedev

        print('Max order guaranteed is: 131.')
        print('Good schemes are available for the following quadrature orders:')

        orders = ''
        last = 0 # the last good scheme
        for i in range(132):
            if quadpy.u3.get_good_scheme(i) is not None:
                orders += f'{i} '
                last = i
        print(orders)

        print(f'Other quadrature orders above {last} are available within the Lebedev scheme:')

        orders = ''
        for i in range(last+1, 132):
            key = str(i).zfill(3)
            try:
                getattr(_lebedev, f'lebedev_{key}')
                orders += f'{int(key)} '
            except (KeyError, AttributeError):
                pass
        print(orders)

    def _get_scheme_from_order(self, order):
        """Get scheme for U3 integration."""
        import quadpy
        from quadpy.u3  import _lebedev

        scheme = quadpy.u3.get_good_scheme(order)

        if scheme is None:
            try:
                key = str(order).zfill(3)
                scheme = getattr(_lebedev, f'lebedev_{key}')()
            except (KeyError, AttributeError):
                raise NotImplementedError('quadrature order  not implemented.')

        return scheme