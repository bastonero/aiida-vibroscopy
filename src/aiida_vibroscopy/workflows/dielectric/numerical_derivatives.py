# -*- coding: utf-8 -*-
"""Workflow for numerical derivatives."""
from aiida import orm
from aiida.engine import WorkChain
from aiida_phonopy.data import PreProcessData

from aiida_vibroscopy.calculations.numerical_derivatives_utils import (
    compute_nac_parameters,
    compute_susceptibility_derivatives,
    join_tensors,
)
from aiida_vibroscopy.utils.validation import validate_positive


def validate_data(data, _):
    """
    Validate the `data` namespace inputs.
    """
    control_null_namespace = 0  # must be 1

    for label in data:
        # first, control if `null`
        if label.startswith('null'):
            control_null_namespace += 1
        elif not label[-1] in ['0', '1', '2', '3', '4', '5']:
            return f'`{label[-1]}` is an invalid label ending for field labels`'
        # else:
        #     if not len(trajectory) % 2 == 0:
        #         return 'field index data must contains even number of key:TrajectoryData pairs'

    if not control_null_namespace == 1:
        return f'invalid number of `null_field` namespaces: expected 1, given {control_null_namespace}'


class NumericalDerivativesWorkChain(WorkChain):
    r"""
    Workchain that computes first and second order derivatives
    of forces and polarization in respect to
    polarization, to obtain dielectric tensor, Born effective charges,
    susceptibility derivatives in respect to atomic position and
    polarization (i.e. non linear optical susceptibility) to be used
    to compute Raman susceptibility tensors.

    Forces and polarization must passed as TrajectoryData
    as a dictionary in `data`. Numerical derivatives can have
    different number of evaluation points, depending on order and accuracy.
    The price to pay is the standardization of the structure of
    the dictionary to pass to this namespace.

    To understand, let's review the approach.In central differencs approach
    we need the evaluation of the function at the value we want
    the derivative (in our case at :math:`\\mathcal{E}=0`,
    E is the electric field), and at
    displaced positions from this value.
    The evaluation of the function at these points will
    have weights (or coefficients), which depend on order and accuracy.
    For example:

        * :math:`\\frac{df}{dx}   = \\frac{ 0.5 \\cdot f(+1.0 \\cdot h) -0.5
            \\cdot f(-1.0 \\cdot h) }{h} +\mathcal{O}(h^2)`
        * :math:`\\frac{d^2 f}{dx^2} = \\frac{ 1.0 \\cdot f(+1.0 \\cdot h) -2.0
            \\cdot f(0. \\cdot h) +1.0 \\cdot f(-1.0 \\cdot h) }{h^2} +\mathcal{O}(h^2)`

    Referring to the coefficients for each step as :math:`c_i`,
    where `i` is an integer, our convention is
    to put in sequence the Trajectory data with increasing
    numbers as labels, for example:

    | {
    |   '0': TrajectoryData for :math:`c_1`,
    |   '1': TrajectoryData for :math:`c_{-1}`,
    |   '2': TrajectoryData for :math:`c_2`,
    |   '3': TrajectoryData for :math:`c_{-2}`,
    |   ...
    | }

    This way to creating an analogous of an array with
    coefficients :math:`[c_1,c_{-1},c_2,c_{-2}, \\dots]`.

    These dictionaries are going to be put as sub-dictionary
    in a general `data` dictionary. Each sub-dict
    has to be put with a key with suffix a number indicating
    which tensor component is referring to.
    In our case, we use a similar Voigt notation.
    Namely we have two cases:

        * first order derivatives: keys suffices are 0,1,2; 0
            for [i,x], 1 for [i,y], 2 for [i,z] (with i={x,y,z})
        * second order derivatives: keys suffices are 0,...5; 0
            for [i,x,x], ... (as in Voigt), 5 for [i,x,y] (with i={x,y,z})

    The prefix can be anything. Best practice is using ``field_``
    with and underscorre as prefix. The Trajectory data for the
    c0 coefficient (i.e. the one with :math:`\\mathcal{E}=0`)
    must be passed with a different key, namely ``null_field``.
    This is to avoid errors and due to the fact that is common
    to the all derivatives.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input_namespace(
            'data',
            validator=validate_data,
            help='Namespace for passing TrajectoryData containing forces and polarization.',
        )
        spec.input('data.null_field', valid_type=orm.TrajectoryData, required=True)
        spec.input_namespace('data.field_index_0', valid_type=orm.TrajectoryData, required=False)
        spec.input_namespace('data.field_index_1', valid_type=orm.TrajectoryData, required=False)
        spec.input_namespace('data.field_index_2', valid_type=orm.TrajectoryData, required=False)
        spec.input_namespace('data.field_index_3', valid_type=orm.TrajectoryData, required=False)
        spec.input_namespace('data.field_index_4', valid_type=orm.TrajectoryData, required=False)
        spec.input_namespace('data.field_index_5', valid_type=orm.TrajectoryData, required=False)

        spec.input('accuracy_order', valid_type=orm.Int, validator=validate_positive)
        spec.input(
            'electric_field_step', valid_type=orm.Float, validator=validate_positive, help='The electric field step.'
        )
        spec.input('diagonal_scale', valid_type=orm.Float, required=False, validator=validate_positive)
        spec.input('structure', valid_type=orm.StructureData)
        spec.input_namespace(
            'options',
            help='Symmetry analysis options.',
        )
        spec.input(
            'options.symprec',
            valid_type=orm.Float,
            default=lambda: orm.Float(1e-5),
            help='Symmetry tolerance for space group analysis on the input structure.',
        )
        spec.input(
            'options.distinguish_kinds',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help=('Whether or not to distinguish atom with '
                  'same species but different names with symmetries.'),
        )
        spec.input(
            'options.is_symmetry',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
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
                'to the atomic positions (often called `Raman tensors`) '
                'and the non linear optical susceptibility, '
                'always expressed in Cartesian coordinates.'
            ),
        )
        spec.output('units', valid_type=orm.Dict, help='Units of the susceptibility derivatives tensors.')

    def run_results(self):
        """Wrap up results from previous calculations."""
        preprocess_data = PreProcessData.generate_preprocess_data(
            structure=self.inputs.structure,
            symprec=self.inputs.options.symprec,
            is_symmetry=self.inputs.options.is_symmetry,
            distinguish_kinds=self.inputs.options.distinguish_kinds,
        )

        # Non analytical constants
        out_nac_parameters = compute_nac_parameters(
            preprocess_data=preprocess_data,
            electric_field=self.inputs.electric_field_step,
            accuracy_order=self.inputs.accuracy_order,
            **self.inputs.data
        )

        # Derivatives of the susceptibility
        mixed_indecis = [f'field_index_{i}' for i in ['3', '4', '5']]
        which_mixed = [mixed_index in self.inputs.data for mixed_index in mixed_indecis]

        if any(which_mixed):
            out_dchis = compute_susceptibility_derivatives(
                preprocess_data=preprocess_data,
                electric_field=self.inputs.electric_field_step,
                diagonal_scale=self.inputs.diagonal_scale,
                accuracy_order=self.inputs.accuracy_order,
                **self.inputs.data,
            )

            self.out('units', out_dchis['units'])

        keys = out_nac_parameters.keys()
        for key in keys:
            if any(which_mixed):
                tensors = join_tensors(out_nac_parameters[key], out_dchis[key])
            else:
                tensors = out_nac_parameters[key]

            self.out(f'tensors.{key}', tensors)
