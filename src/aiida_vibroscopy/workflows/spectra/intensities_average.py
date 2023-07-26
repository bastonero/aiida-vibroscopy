# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Workflow for numerical averaging of vibrational intensities."""
from __future__ import annotations

from aiida import orm
from aiida.engine import WorkChain, calcfunction
import numpy as np

from aiida_vibroscopy.data import VibrationalData, VibrationalFrozenPhononData


@calcfunction
def compute_ir_average(**kwargs) -> orm.ArrayData:
    """Return infrared average.

    :param kwargs: it must contain the following key-value pairs:
        * `vibro_data`: a :class:`~aiida_vibroscopy.data.VibrationalData` node
        * `parameters`: dict containing the parameters for the averaging;
        see also :func:`~aiida_vibroscopy.data.VibrationalMixin.run_powder_ir_intensities`


    :return: :class:`~aiida.orm.ArrayData` with arraynames `intensities`, `frequencies`, `labels`
    """
    vibro_data = kwargs['vibro_data']
    parameters = kwargs['parameters']
    intensities, freqs, labels = vibro_data.run_powder_ir_intensities(**parameters)
    array = orm.ArrayData()

    array.set_array('intensities', intensities)
    array.set_array('frequencies', freqs)
    array.set_array('labels', np.array(labels, dtype='U'))

    return array


@calcfunction
def compute_raman_average(**kwargs):
    """Return Raman average.

    :param kwargs: it must contain the following key-value pairs:
        * `vibro_data`: a :class:`~aiida_vibroscopy.data.VibrationalData` node
        * `parameters`: dict containing the parameters for the averaging;
        see also :func:`~aiida_vibroscopy.data.VibrationalMixin.run_powder_raman_intensities`

    :return: :class:`~aiida.orm.ArrayData` with arraynames `intensities`, `frequencies`, `labels`
    """
    vibro_data = kwargs['vibro_data']
    parameters = kwargs['parameters']
    intensities_hh, intensities_hv, freqs, labels = vibro_data.run_powder_raman_intensities(**parameters)
    array = orm.ArrayData()

    array.set_array('intensities_HH', intensities_hh)
    array.set_array('intensities_HV', intensities_hv)
    array.set_array('frequencies', freqs)
    array.set_array('labels', np.array(labels, dtype='U'))

    return array


def validate_vibrational_data(value, _):
    """Validate vibrational data with having at least the NAC parameters."""
    if not value.has_nac_parameters():
        return 'vibrational data does not contain non-analytical parameters.'


def validate_positive(value, _):
    """Validate the value of the electric field."""
    if value.value < 0:
        return 'specified value is negative.'


class IntensitiesAverageWorkChain(WorkChain):
    """Workchain that computes IR and Raman spatial and q-direction average spectra."""

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)

        spec.input(
            'vibrational_data',
            valid_type=(VibrationalData, VibrationalFrozenPhononData),
            required=True,
            validator=validate_vibrational_data,
            help=(
                'Vibrational data containing force constants or '
                'frozen phonons forces, nac parameters and/or '
                'susceptibility derivatives.'
            ),
        )
        spec.input(
            'parameters',
            valid_type=orm.Dict,
            default=lambda: orm.Dict({'quadrature_order': 41}),
            help='Options for averaging on the non-analytical directions.'
        )

        spec.outline(cls.run_results,)

        spec.output(
            'ir_averaged',
            valid_type=orm.ArrayData,
            help='Contains high frequency dielectric tensor computed in Cartesian coordinates.'
        )
        spec.output(
            'raman_averaged',
            valid_type=orm.ArrayData,
            required=False,
            help='Contains Born effective charges tensors computed in Cartesian coordinates.'
        )
        spec.output('units', valid_type=orm.Dict, required=False, help='Units of intensities and frequencies.')

    def run_results(self):
        """Run averaging procedure."""
        vibrational_data = self.inputs.vibrational_data
        kwargs = {'vibro_data': vibrational_data, 'parameters': self.inputs.parameters}

        self.report('IR averaging calcfunction started')
        ir_average = compute_ir_average(**kwargs)

        self.out('ir_averaged', ir_average)

        if vibrational_data.has_raman_parameters():
            parameters = kwargs['parameters'].get_dict()

            try:
                parameters['with_nlo'] = parameters.pop('with_nlo')
            except KeyError:
                if vibrational_data.has_nlo():
                    parameters['with_nlo'] = True

            kwargs['parameters'] = orm.Dict(parameters)

            self.report('Raman averaging calcfunction started')
            raman_average = compute_raman_average(**kwargs)

            self.out('raman_averaged', raman_average)
