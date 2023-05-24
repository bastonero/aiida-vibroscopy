# -*- coding: utf-8 -*-
"""Mixin for forzen phonons vibrational data."""
from aiida_phonopy.data.phonopy import PhonopyData

from .vibro_mixin import VibrationalMixin

__all__ = ('VibrationalFrozenPhononData',)


class VibrationalFrozenPhononData(PhonopyData, VibrationalMixin):  # pylint: disable=too-many-ancestors
    """Vibrational data for IR and Raman spectra from frozen phonons."""
