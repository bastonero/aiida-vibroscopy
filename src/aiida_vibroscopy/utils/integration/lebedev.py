# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Module implementing a general Lebedev integration scheme on sphere."""
from __future__ import annotations

import json

import numpy as np
import sympy

# yapf: disable
available_orders = [
    3, 5, 7, 9, 11, 13, 15, 17,
    19, 21, 23, 25, 27, 29, 31,
    35, 41, 47, 53, 59, 65, 71,
    77, 83, 89, 95, 101, 107,
    113, 119, 125, 131,
]
# yapf: enable


class LebedevScheme:
    """Class for handling a generic Lebedev integration scheme."""

    def __init__(
        self,
        name: str,
        symmetry_data: dict,
        order: int,
    ):
        """Instantiate the class for Lebedev scheme.

        :param name: name of the Lebedev scheme
        :param symmetry_data: dict containing the symmetry data of the scheme
        :order: the degree of the integration scheme
        """
        self._name = name
        self._order = order
        self._symmetry_data = symmetry_data

        points, weights = expand_symmetries(symmetry_data)

        self._points = points
        self._weights = weights

    @property
    def points(self):
        """Return the explicit points for integration in Cartesian coordinates."""
        return self._points

    @property
    def weights(self):
        """Return the explicit weights for integration in Cartesian coordinates."""
        return self._weights

    @staticmethod
    def from_order(order: int):
        """Get a `LebedevScheme` instance from precison order."""
        from importlib_resources import files

        from . import schemas

        if order not in available_orders:
            raise NotImplementedError('the requested order is not tabulated')

        filepath = files(schemas) / f'lebedev_{str(order).zfill(3)}.json'

        with open(filepath, mode='r', encoding='utf-8') as handle:
            content = json.load(handle)

        data = content['data']
        if 'weight factor' in content:
            w = content['weight factor']
            for value in data.values():
                value[0] = [v * w for v in value[0]]

        return LebedevScheme(content['name'], data, order=content['degree'])


def get_available_quadrature_order_schemes() -> list[int]:
    """Print the available orders for numerical integration.

    :return: list of available orders
    """
    print('Available quadrature orders in Lebedev scheme:')
    orders = ' '.join(map(str, available_orders))
    print(orders)

    return available_orders


def _a1(vals):
    """Return the expanded symmetry points."""
    symbolic = np.asarray(vals).dtype == sympy.Basic
    a = 1 if symbolic else 1.0
    points = np.array([[+a, 0, 0], [-a, 0, 0], [0, +a, 0], [0, -a, 0], [0, 0, +a], [0, 0, -a]]).T
    return points


def _a2(vals):
    """Return the expanded symmetry points."""
    symbolic = np.asarray(vals).dtype == sympy.Basic
    a = 1 / sympy.sqrt(2) if symbolic else 1 / np.sqrt(2)
    points = np.array([
        [+a, +a, 0],
        [+a, -a, 0],
        [-a, +a, 0],
        [-a, -a, 0],
        #
        [+a, 0, +a],
        [+a, 0, -a],
        [-a, 0, +a],
        [-a, 0, -a],
        #
        [0, +a, +a],
        [0, +a, -a],
        [0, -a, +a],
        [0, -a, -a],
    ]).T
    return points


def _a3(vals):
    """Return the expanded symmetry points."""
    symbolic = np.asarray(vals).dtype == sympy.Basic
    a = 1 / sympy.sqrt(3) if symbolic else 1 / np.sqrt(3)
    points = np.array([
        [+a, +a, +a],
        [+a, +a, -a],
        [+a, -a, +a],
        [+a, -a, -a],
        [-a, +a, +a],
        [-a, +a, -a],
        [-a, -a, +a],
        [-a, -a, -a],
    ]).T
    return points


def _pq0(vals):
    """Return the expanded symmetry points."""
    return _pq02([np.sin(vals[0] * np.pi), np.cos(vals[0] * np.pi)])


def _pq02(vals):
    """Return the expanded symmetry points."""
    if len(vals) == 1:
        a = vals[0]
        b = np.sqrt(1 - a**2)
    else:
        assert len(vals) == 2
        a, b = vals

    if isinstance(a, sympy.Basic):
        zero = 0
    else:
        zero = np.zeros_like(a)

    points = np.array([
        [+a, +b, zero],
        [-a, +b, zero],
        [-a, -b, zero],
        [+a, -b, zero],
        #
        [+b, +a, zero],
        [-b, +a, zero],
        [-b, -a, zero],
        [+b, -a, zero],
        #
        [+a, zero, +b],
        [-a, zero, +b],
        [-a, zero, -b],
        [+a, zero, -b],
        #
        [+b, zero, +a],
        [-b, zero, +a],
        [-b, zero, -a],
        [+b, zero, -a],
        #
        [zero, +a, +b],
        [zero, -a, +b],
        [zero, -a, -b],
        [zero, +a, -b],
        #
        [zero, +b, +a],
        [zero, -b, +a],
        [zero, -b, -a],
        [zero, +b, -a],
    ])
    points = np.moveaxis(points, 0, 1)
    return points


def _rs0(vals):
    """Return the expanded symmetry points."""
    if len(vals) == 1:
        a = vals[0]
        b = np.sqrt(1 - a**2)
    else:
        assert len(vals) == 2
        a, b = vals

    if isinstance(a, sympy.Basic):
        zero = 0
    else:
        zero = np.zeros_like(a)

    points = np.array([
        [+a, +b, zero],
        [-a, +b, zero],
        [-a, -b, zero],
        [+a, -b, zero],
        #
        [+b, zero, +a],
        [-b, zero, +a],
        [-b, zero, -a],
        [+b, zero, -a],
        #
        [zero, +a, +b],
        [zero, -a, +b],
        [zero, -a, -b],
        [zero, +a, -b],
    ])
    points = np.moveaxis(points, 0, 1)
    return points


def _llm(vals):
    """Return the expanded symmetry points."""
    # translate the point into cartesian coords; note that phi=pi/4.
    beta = vals[0] * np.pi
    L = np.sin(beta) / np.sqrt(2)
    m = np.cos(beta)
    return _llm2([L, m])


def _llm2(vals):
    """Return the expanded symmetry points."""
    if len(vals) == 1:
        L = vals[0]
        m = np.sqrt(1 - 2 * L**2)
    else:
        assert len(vals) == 2
        L, m = vals

    points = np.array([
        [+L, +L, +m],
        [-L, +L, +m],
        [+L, -L, +m],
        [-L, -L, +m],
        [+L, +L, -m],
        [-L, +L, -m],
        [+L, -L, -m],
        [-L, -L, -m],
        #
        [+L, +m, +L],
        [-L, +m, +L],
        [+L, +m, -L],
        [-L, +m, -L],
        [+L, -m, +L],
        [-L, -m, +L],
        [+L, -m, -L],
        [-L, -m, -L],
        #
        [+m, +L, +L],
        [+m, -L, +L],
        [+m, +L, -L],
        [+m, -L, -L],
        [-m, +L, +L],
        [-m, -L, +L],
        [-m, +L, -L],
        [-m, -L, -L],
    ])
    points = np.moveaxis(points, 0, 1)
    return points


def _rsw(vals):
    """Return the expanded symmetry points."""
    # translate the point into cartesian coords; note that phi=pi/4.
    phi_theta = vals * np.pi

    sin_phi, sin_theta = np.sin(phi_theta)
    cos_phi, cos_theta = np.cos(phi_theta)

    return _rsw2([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])


def _rsw2(vals):
    """Return the expanded symmetry points."""
    if len(vals) == 2:
        r, s = vals
        w = np.sqrt(1 - r**2 - s**2)
    else:
        assert len(vals) == 3
        r, s, w = vals

    points = np.array([
        [+r, +s, +w],
        [+w, +r, +s],
        [+s, +w, +r],
        [+s, +r, +w],
        [+w, +s, +r],
        [+r, +w, +s],
        #
        [-r, +s, +w],
        [+w, -r, +s],
        [+s, +w, -r],
        [+s, -r, +w],
        [+w, +s, -r],
        [-r, +w, +s],
        #
        [+r, -s, +w],
        [+w, +r, -s],
        [-s, +w, +r],
        [-s, +r, +w],
        [+w, -s, +r],
        [+r, +w, -s],
        #
        [+r, +s, -w],
        [-w, +r, +s],
        [+s, -w, +r],
        [+s, +r, -w],
        [-w, +s, +r],
        [+r, -w, +s],
        #
        [-r, -s, +w],
        [+w, -r, -s],
        [-s, +w, -r],
        [-s, -r, +w],
        [+w, -s, -r],
        [-r, +w, -s],
        #
        [-r, +s, -w],
        [-w, -r, +s],
        [+s, -w, -r],
        [+s, -r, -w],
        [-w, +s, -r],
        [-r, -w, +s],
        #
        [+r, -s, -w],
        [-w, +r, -s],
        [-s, -w, +r],
        [-s, +r, -w],
        [-w, -s, +r],
        [+r, -w, -s],
        #
        [-r, -s, -w],
        [-w, -r, -s],
        [-s, -w, -r],
        [-s, -r, -w],
        [-w, -s, -r],
        [-r, -w, -s],
    ])
    points = np.moveaxis(points, 0, 1)
    return points


def _rst(vals):
    """Return the expanded symmetry points."""
    if len(vals) == 2:
        r, s = vals
        w = np.sqrt(1 - r**2 - s**2)
    else:
        assert len(vals) == 3
        r, s, w = vals

    points = np.array([
        [+r, +s, +w],
        [+w, +r, +s],
        [+s, +w, +r],
        #
        [-r, +s, +w],
        [+w, -r, +s],
        [+s, +w, -r],
        #
        [+r, -s, +w],
        [+w, +r, -s],
        [-s, +w, +r],
        #
        [+r, +s, -w],
        [-w, +r, +s],
        [+s, -w, +r],
        #
        [-r, -s, +w],
        [+w, -r, -s],
        [-s, +w, -r],
        #
        [-r, +s, -w],
        [-w, -r, +s],
        [+s, -w, -r],
        #
        [+r, -s, -w],
        [-w, +r, -s],
        [-s, -w, +r],
        #
        [-r, -s, -w],
        [-w, -r, -s],
        [-s, -w, -r],
    ])
    points = np.moveaxis(points, 0, 1)
    return points


def _rst_weird(vals):
    """Return the expanded symmetry points."""
    if len(vals) == 2:
        r, s = vals
        t = np.sqrt(1 - r**2 - s**2)
    else:
        assert len(vals) == 3
        r, s, t = vals

    points = np.array([
        [+r, +s, +t],
        [-r, +t, +s],
        [+s, +t, +r],
        [-s, +r, +t],
        [+t, +r, +s],
        [-t, +s, +r],
        #
        [+r, -s, -t],
        [-r, -t, -s],
        [+s, -t, -r],
        [-s, -r, -t],
        [+t, -r, -s],
        [-t, -s, -r],
        #
        [+r, +t, -s],
        [-r, +s, -t],
        [+s, +r, -t],
        [-s, +t, -r],
        [+t, +s, -r],
        [-t, +r, -s],
        #
        [+r, -t, +s],
        [-r, -s, +t],
        [+s, -r, +t],
        [-s, -t, +r],
        [+t, -s, +r],
        [-t, -r, +s],
    ])
    points = np.moveaxis(points, 0, 1)
    return points


def expand_symmetries_points_only(data):
    """Return expanded points from symmetries."""
    points = []
    counts = []

    for key, points_raw in data.items():
        fun = {
            'a1': _a1,
            'a2': _a2,
            'a3': _a3,
            'llm': _llm,
            'llm2': _llm2,
            'pq0': _pq0,
            'pq02': _pq02,
            'rs0': _rs0,
            'rsw': _rsw,
            'rsw2': _rsw2,
            'rst': _rst,
            'rst_weird': _rst_weird,
            'plain': lambda vals: vals.reshape(3, 1, -1),
        }[key]
        pts = fun(np.asarray(points_raw))

        counts.append(pts.shape[1])
        pts = pts.reshape(pts.shape[0], -1)
        points.append(pts)

    points = np.ascontiguousarray(np.concatenate(points, axis=1))
    return points, counts


def expand_symmetries(data) -> tuple[np.ndarray, np.ndarray]:
    """Return Cartesian points and associated weights."""
    points_raw = {}
    weights_raw = []
    for key, values in data.items():
        weights_raw.append(values[0])
        points_raw[key] = values[1:]

    points, counts = expand_symmetries_points_only(points_raw)
    weights = np.concatenate([np.tile(values, count) for count, values in zip(counts, weights_raw)])
    return points, weights
