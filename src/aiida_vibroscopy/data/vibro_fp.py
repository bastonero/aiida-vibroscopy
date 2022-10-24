# -*- coding: utf-8 -*-
"""Mixin for aiida-vibroscopy DataTypes."""

from aiida.plugins import DataFactory

from .vibro_mixin import VibrationalMixin

PhonopyData = DataFactory('phonopy.phonopy')

__all__ = ('VibrationalFrozenPhononData',)


class VibrationalFrozenPhononData(PhonopyData, VibrationalMixin):
    """Vibrational data for IR and Raman spectra from frozen phonons
    (i.e. finite displacements and fields).
    """
