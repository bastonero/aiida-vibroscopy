# -*- coding: utf-8 -*-
"""Mixin for vibrational data."""
from aiida_phonopy.data.force_constants import ForceConstantsData

from .vibro_mixin import VibrationalMixin

__all__ = ('VibrationalData',)


class VibrationalData(ForceConstantsData, VibrationalMixin):  # pylint: disable=too-many-ancestors
    """Vibrational data for IR and Raman spectra."""
