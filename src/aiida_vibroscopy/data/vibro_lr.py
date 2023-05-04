# -*- coding: utf-8 -*-
"""Mixin for vibrational data."""

from aiida.plugins import DataFactory

from .vibro_mixin import VibrationalMixin

ForceConstantsData = DataFactory('phonopy.force_constants')

__all__ = ('VibrationalData',)


class VibrationalData(ForceConstantsData, VibrationalMixin):
    """Vibrational data for IR and Raman spectra."""
