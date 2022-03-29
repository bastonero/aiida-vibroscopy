# -*- coding: utf-8 -*-
"""Tests for the :mod:`data` module."""

import pytest
import numpy

@pytest.fixture
def generate_vibrational_data(generate_structure):
    """Generate a `VibrationalFrozenPhononData`."""

    def _generate_vibrational_data(dielectric=None, born_charges=None, dph0=None, nlo=None, forces=None):
        """Return a `VibrationalFrozenPhononData` with bulk silicon as structure."""
        from aiida.plugins import DataFactory
        # from aiida_quantumespresso_vibroscopy.data.vibro_fp import VibrationalFrozenPhononData

        VibrationalFrozenPhononData = DataFactory('quantumespresso.vibroscopy.fp')
        PreProcessData = DataFactory('phonopy.preprocess')

        structure = generate_structure()

        supercell_matrix = [1,1,1]

        preprocess_data =  PreProcessData(
            structure=structure,
            supercell_matrix=supercell_matrix,
            primitive_matrix='auto'
        )

        preprocess_data.set_displacements()

        vibrational_data = VibrationalFrozenPhononData(preprocess_data=preprocess_data)

        if dielectric is not None:
            vibrational_data.set_dielectric(dielectric)
        else:
            vibrational_data.set_dielectric(numpy.eye(3))

        if born_charges is not None:
            vibrational_data.set_born_charges(born_charges)
        else:
            becs = numpy.array([numpy.eye(3), -1*numpy.eye(3)])
            vibrational_data.set_born_charges(becs)

        if dph0 is not None:
            vibrational_data.set_dph0_susceptibility(dph0_susceptibility=dph0)
        else:
            dph0 = numpy.zeros((2,3,3,3))
            dph0[0][0][0][0] = +1
            dph0[1][0][0][0] = -1
            vibrational_data.set_dph0_susceptibility(dph0_susceptibility=dph0)

        if nlo is not None:
            vibrational_data.set_nlo_susceptibility(nlo_susceptibility=nlo)
        else:
            vibrational_data.set_nlo_susceptibility(nlo_susceptibility=numpy.zeros((3,3,3)))

        if forces is not None:
            vibrational_data.set_forces()
        else:
            vibrational_data.set_forces( [ [[1,0,0],[-1,0,0]], [[2,0,0],[-2,0,0]] ] )

        return vibrational_data

    return _generate_vibrational_data


@pytest.mark.usefixtures('aiida_profile')
def test_methods(generate_vibrational_data):
    """Test `VibrationalMixin` method."""
    vibrational_data = generate_vibrational_data()

    vibrational_data.get_raman_tensors()
    vibrational_data.get_polarization_vectors()
    vibrational_data.get_polarized_raman_intensities([1,0,0],[-1,0,0])
    vibrational_data.get_powder_raman_intensities(3)
    vibrational_data.get_polarized_ir_intensities([1,0,0])
    vibrational_data.get_powder_ir_intensities(3)
