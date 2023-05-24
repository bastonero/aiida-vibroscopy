# -*- coding: utf-8 -*-
"""Tests for :mod:`aiida_vibroscopy.utils.integration`."""
import pytest

from aiida_vibroscopy.utils.integration.lebedev import LebedevScheme, available_orders


@pytest.mark.parametrize('order', tuple(available_orders))
def test_lebedev_schemas(order):
    """Test all lebedev schemas weights sum up to 1 within threshold."""
    thr = 1e-12
    scheme = LebedevScheme.from_order(order=order)
    assert abs(scheme.weights.sum() - 1) < thr


def test_invalid_order():
    """Test `LebedevScheme.from_order` raising error for not available order."""
    match = 'the requested order is not tabulated'
    with pytest.raises(NotImplementedError, match=match):
        LebedevScheme.from_order(order=1000)
