# -*- coding: utf-8 -*-
"""Test the :mod:`utils.plotting`."""


def test_get_spectra_plot():
    """Test the inputs for `get_spectra_plot`."""
    from aiida_vibroscopy.utils.plotting import get_spectra_plot

    get_spectra_plot([50, 100, 200], [1, 1, 1])
