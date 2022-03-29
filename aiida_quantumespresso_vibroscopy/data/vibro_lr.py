# -*- coding: utf-8 -*-
"""Mixin for aiida-quantumespresso-vibroscopy DataTypes."""

from aiida.plugins import DataFactory
from .vibro_mixin import VibrationalMixin

ForceConstantsData = DataFactory('phonopy.force_constants')

__all__ = ('VibrationalLinearResponseData')


class VibrationalLinearResponseData(ForceConstantsData, VibrationalMixin):
    """Vibrational data for IR and Raman spectra from linear response."""
    pass
