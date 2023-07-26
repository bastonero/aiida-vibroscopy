# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
"""Workflow for numerical derivatives."""
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain
from aiida_phonopy.data import PreProcessData
import numpy as np

from aiida_vibroscopy.calculations.numerical_derivatives_utils import (
    compute_nac_parameters,
    compute_susceptibility_derivatives,
    join_tensors,
)
from aiida_vibroscopy.utils.validation import validate_positive


def validate_data(data, _):
    """Validate the `data` namespace inputs."""
    control_null_namespace = 0  # must be 1

    for label in data:
        # first, control if `null`
        if label.startswith('null'):
            control_null_namespace += 1
        elif not label[-1] in ['0', '1', '2', '3', '4', '5']:
            return f'`{label[-1]}` is an invalid label ending for field labels`'


class NumericalDerivativesWorkChain(WorkChain):
    r"""Workchain carrying out numerical derivatives.

    It computes the first and second order derivatives
    of forces and polarization in respect to electric field,
    to obtain dielectric tensor, Born effective charges,
    non linear optical susceptibility and Raman tensors.

    Forces and polarization must be passed as TrajectoryData
    as a dictionary in `data`. Numerical derivatives can have
    different number of evaluation points, depending on order and accuracy.
    The price to pay is the standardization of the structure of
    the dictionary to pass to this namespace.

    To understand, let's review the approach.In central differencs approach
    we need the evaluation of the function at the value we want
    the derivative (in our case at :math:`\mathcal{E}=0`,
    E is the electric field), and at
    displaced positions from this value.
    The evaluation of the function at these points will
    have weights (or coefficients), which depend on order and accuracy.
    For example:

    - :math:`\frac{df}{dx} = \frac{ 0.5 \cdot f(+1.0 \cdot h) -0.5 \cdot f(-1.0 \cdot h) }{h} +\mathcal{O}(h^2)`
    - :math:`\frac{d^2 f}{dx^2} = \frac{ 1.0 \cdot f(+1.0 \cdot h) -2.0 \cdot f(0. \cdot h) +1.0 \cdot f(-1.0 \cdot h) }{h^2} +\mathcal{O}(h^2)`

    Referring to the coefficients for each step as :math:`c_i`,
    where `i` is an integer, our convention is
    to put in sequence the Trajectory data with increasing
    numbers as labels, for example:

    | '0': TrajectoryData for :math:`c_1`,
    | '1': TrajectoryData for :math:`c_{-1}`,
    | '2': TrajectoryData for :math:`c_2`,
    | '3': TrajectoryData for :math:`c_{-2}`,
    | ...

    This way to creating an analogous of an array with
    coefficients :math:`[c_1,c_{-1},c_2,c_{-2}, \dots]`.

    These dictionaries are going to be put as sub-dictionary
    in a general `data` dictionary. Each sub-dict
    has to be put with a key with suffix a number indicating
    which tensor component is referring to.
    In our case, we use a similar Voigt notation.
    Namely we have two cases:

    * first order derivatives: keys suffices are 0,1,2;
        0 for :math:`[i,x]`, 1 for :math:`[i,y]`, 2 for
        :math:`[i,z]` (with :math:`i={x,y,z}`)
    * second order derivatives: keys suffices are 0,...5;
        0 for :math:`[i,x,x]`, :math:`\dots` (as in Voigt),
        5 for :math:`[i,x,y]` (with :math:`i={x,y,z}`)

    The prefix can be anything. Best practice is using ``field_``
    with and underscorre as prefix. The Trajectory data for the
    :math:`c_0` coefficient (i.e. the one with :math:`\mathcal{E}=0`)
    must be passed with a different key, namely ``null_field``.
    This is to avoid errors and due to the fact that is common
    to the all derivatives.
    """  # pylint: disable=line-too-long

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        # yapf: disable
        spec.input('structure', valid_type=orm.StructureData)
        spec.input_namespace(
            'data', validator=validate_data,
            help='Namespace for passing TrajectoryData containing forces and polarization.',
        )
        spec.input('data.null_field', valid_type=orm.TrajectoryData, required=False)
        for i in range(6):
            spec.input_namespace(f'data.field_index_{i}', valid_type=orm.TrajectoryData, required=False)
        spec.input_namespace(
            'central_difference',
            help='The inputs for the central difference scheme.'
        )
        spec.input(
            'central_difference.electric_field_step', valid_type=orm.Float, required=False,
            help=(
                'Electric field step in Ry atomic units used in the numerical differenciation. '
                'Only positive values. If not specified, an NSCF is run to evaluate the critical '
                'electric field; an electric field step is then extracted to secure a stable SCF.'
            ),
            validator=validate_positive,
        )
        spec.input(
            'central_difference.diagonal_scale', valid_type=orm.Float, default=lambda: orm.Float(1/np.sqrt(2)),
            help='Scaling factor for electric fields non parallel to cartesiaan axis (i.e. E --> scale*E).',
            validator=validate_positive,
        )
        spec.input(
            'central_difference.accuracy', valid_type=orm.Int, required=False,
            help=('Central difference scheme accuracy to employ (i.e. number of points for derivative evaluation). '
                  'This must be an EVEN positive integer number. If not specified, an automatic '
                  'choice is made upon the intensity of the critical electric field.'),
        )
        spec.input(
            'symmetry.symprec', valid_type=orm.Float, default=lambda:orm.Float(1e-5),
            help='Symmetry tolerance for space group analysis on the input structure.',
        )
        spec.input(
            'symmetry.distinguish_kinds', valid_type=orm.Bool, default=lambda:orm.Bool(False),
            help='Whether or not to distinguish atom with same species but different names with symmetries.',
        )
        spec.input(
            'symmetry.is_symmetry', valid_type=orm.Bool, default=lambda:orm.Bool(True),
            help='Whether using or not the space group symmetries.',
        )

        spec.outline(cls.run_results,)

        spec.output_namespace(
            'tensors',
            valid_type=orm.ArrayData,
            help=(
                'Contains high frequency dielectric and Born effective'
                'charges tensors computed in Cartesian coordinates. '
                'Depending on the inputs, it can also contain the '
                'derivatives of the susceptibility in respect '
                'to the atomic positions (called `Raman tensors`) '
                'and the non linear optical susceptibility, '
                'always expressed in Cartesian coordinates.'
            ),
        )
        spec.output(
            'units', valid_type=orm.Dict, required=False,
            help='Units of the susceptibility derivatives tensors.')
        # yapf: enable

    def run_results(self):
        """Wrap up results from previous calculations."""
        preprocess_data = PreProcessData.generate_preprocess_data(
            structure=self.inputs.structure,
            symprec=self.inputs.symmetry.symprec,
            is_symmetry=self.inputs.symmetry.is_symmetry,
            distinguish_kinds=self.inputs.symmetry.distinguish_kinds,
        )

        kwargs = AttributeDict(self.inputs.data)

        # Non analytical constants
        out_nac_parameters = compute_nac_parameters(
            preprocess_data=preprocess_data,
            electric_field=self.inputs.central_difference.electric_field_step,
            accuracy_order=self.inputs.central_difference.accuracy,
            **kwargs
        )

        # Derivatives of the susceptibility
        mixed_indecis = [f'field_index_{i}' for i in ['3', '4', '5']]
        which_mixed = [mixed_index in self.inputs.data for mixed_index in mixed_indecis]

        if any(which_mixed):
            out_dchis = compute_susceptibility_derivatives(
                preprocess_data=preprocess_data,
                electric_field=self.inputs.central_difference.electric_field_step,
                diagonal_scale=self.inputs.central_difference.diagonal_scale,
                accuracy_order=self.inputs.central_difference.accuracy,
                **kwargs,
            )

            self.out('units', out_dchis['units'])

        keys = out_nac_parameters.keys()
        for key in keys:
            if any(which_mixed):
                tensors = join_tensors(out_nac_parameters[key], out_dchis[key])
            else:
                tensors = out_nac_parameters[key]

            self.out(f'tensors.{key}', tensors)
