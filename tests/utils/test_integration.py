# -*- coding: utf-8 -*-
"""Tests for :mod:`aiida_vibroscopy.utils.integration`."""


def test_lebedev_schemas():
    """Test all lebedev schemas weights sum up to 1 within threshold."""
    from aiida_vibroscopy.utils.integration.lebedev import LebedevScheme, available_orders
    thr = 1e-12

    for order in available_orders:
        scheme = LebedevScheme.from_order(order=order)
        assert abs(scheme.weights.sum() - 1) < thr
