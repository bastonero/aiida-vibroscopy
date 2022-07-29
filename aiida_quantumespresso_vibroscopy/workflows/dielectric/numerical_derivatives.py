# -*- coding: utf-8 -*-
"""Workflow for numerical derivatives."""
from aiida import orm
from aiida.engine import WorkChain

from aiida_quantumespresso_vibroscopy.calculations.numerical_derivatives_utils import (
    compute_nac_parameters, compute_susceptibility_derivatives
)
from aiida_quantumespresso_vibroscopy.utils.validation import validate_positive


def validate_data(data, _):
    """
    Validate the `data` namespace inputs.
    """
    length = len(data)  # must be 4 or 7
    control_null_namespace = 0  # must be 1

    if not length in [4, 7]:
        return f'invalid total number of inputs for namespace `data`: expected 4 or 7, given {length}'

    for label, trajectory in data.items():
        # first, control if `null`
        if label.startswith('null'):
            control_null_namespace += 1
        elif not label[-1] in ['0', '1', '2', '3', '4', '5']:
            return f'`{label[-1]}` is an invalid label ending for field labels`'
        else:
            if not len(trajectory) % 2 == 0:
                return 'field index data must contains even number of key:TrajectoryData pairs'

    if not control_null_namespace == 1:
        return f'invalid number of `null_field` namespaces: expected 1, given {control_null_namespace}'


class NumericalDerivativesWorkChain(WorkChain):
    """
    Workchain that computes first and second order derivatives of forces and polarization in respect to
    polarization, to obtain dielectric tensor, Born effective charges, susceptibility derivatives in
    respect to atomic position and polarization (i.e. non linear optical susceptibility) to be used
    to compute Raman susceptibility tensors.

    Forces and polarization must passed as TrajectoryData as a dictionary in `data`.
    Numerical derivatives can have different number of evaluation points, depending on order and accuracy.
    The price to pay is the standardization of the structure of the dictionary to pass to this namespace.

    To understand, let's review the approach.In central differencs approach we need the evaluation
    of the function at the value we want the derivative (in our case at :math:`\\mathcal{E}=0`, E is the electric field),
    and at displaced positions from this value. The evaluation of the function at these points will
    have weights (or coefficients), which depend on order and accuracy. For example:

        * :math:`\\frac{df}{dx}   = \\frac{ 0.5 \\cdot f(+1.0 \\cdot h) -0.5 \\cdot f(-1.0 \\cdot h) }{h} +\mathcal{O}(h^2)`
        * :math:`\\frac{d^2 f}{dx^2} = \\frac{ 1.0 \\cdot f(+1.0 \\cdot h) -2.0 \\cdot f(0. \\cdot h) +1.0 \\cdot f(-1.0 \\cdot h) }{h^2} +\mathcal{O}(h^2)`

    Referring to the coefficients for each step as :math:`c_i`, where `i` is an integer, our convention is
    to put in sequence the Trajectory data with increasing numbers as labels, for example:

    | {
    |   '0': TrajectoryData for :math:`c_1`,
    |   '1': TrajectoryData for :math:`c_{-1}`,
    |   '2': TrajectoryData for :math:`c_2`,
    |   '3': TrajectoryData for :math:`c_{-2}`,
    |   ...
    | }

    This way to creating an analogous of an array with coefficients :math:`[c_1,c_{-1},c_2,c_{-2}, \\dots]`.

    These dictionaries are going to be put as sub-dictionary in a general `data` dictionary. Each sub-dict
    has to be put with a key with suffix a number indicating which tensor component is referring to.
    In our case, we use a similar Voigt notation. Namely we have two cases:

        * first order derivatives: keys suffices are 0,1,2; 0 for [i,x], 1 for [i,y], 2 for [i,z] (with i={x,y,z})
        * second order derivatives: keys suffices are 0,...5; 0 for [i,x,x], ... (as in Voigt), 5 for [i,x,y] (with i={x,y,z})

    The prefix can be anything. Best practice is using ``field_`` with and underscorre as prefix.
    The Trajectory data for the c0 coefficient (i.e. the one with :math:`\\mathcal{E}=0`) must be passed with a different key,
    namely ``null_field``. This is to avoid errors and due to the fact that is common to the all derivatives."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input_namespace(
            'data',
            validator=validate_data,
            help='Namespace for passing TrajectoryData containing forces and polarization.',
        )
        spec.input('data.null_field', valid_type=orm.TrajectoryData, required=True)
        spec.input_namespace('data.field_index_0', valid_type=orm.TrajectoryData, required=True)
        spec.input_namespace('data.field_index_1', valid_type=orm.TrajectoryData, required=True)
        spec.input_namespace('data.field_index_2', valid_type=orm.TrajectoryData, required=True)
        spec.input_namespace('data.field_index_3', valid_type=orm.TrajectoryData, required=False)
        spec.input_namespace('data.field_index_4', valid_type=orm.TrajectoryData, required=False)
        spec.input_namespace('data.field_index_5', valid_type=orm.TrajectoryData, required=False)

        spec.input('electric_field', valid_type=orm.Float, validator=validate_positive)
        spec.input('diagonal_scale', valid_type=orm.Float, required=False, validator=validate_positive)
        spec.input('structure', valid_type=orm.StructureData)

        spec.outline(
            cls.run_results,
        )

        spec.output(
            'dielectric',
            valid_type=orm.ArrayData,
            help='Contains high frequency dielectric tensor computed in Cartesian coordinates.',
        )
        spec.output(
            'born_charges',
            valid_type=orm.ArrayData,
            help='Contains Born effective charges tensors computed in Cartesian coordinates.',
        )
        spec.output(
            'dph0_susceptibility',
            valid_type=orm.ArrayData,
            help=('Contains the derivatives of the susceptibility in respect'
                'to the atomic positions in Cartesian coordinates.'),
        )
        spec.output(
            'nlo_susceptibility',
            valid_type=orm.ArrayData,
            help='Contains the non linear optical susceptibility tensor in Cartesian coordinates.',
        )
        spec.output(
            'units',
            valid_type=orm.Dict,
            help='Units of the susceptibility derivatives tensors.'
        )

    def run_results(self):
        """Wrap up results from previous calculations."""

        # Non analytical constants
        out_nac_parameters = compute_nac_parameters(
            structure=self.inputs.structure, electric_field=self.inputs.electric_field, **self.inputs.data
        )
        for key, output in out_nac_parameters.items():
            self.out(key, output)

        # Derivatives of the susceptibility
        if len(self.inputs.data) == 7:
            out_dchis = compute_susceptibility_derivatives(
                structure=self.inputs.structure,
                electric_field=self.inputs.electric_field,
                diagonal_scale=self.inputs.diagonal_scale,
                **self.inputs.data,
            )
            for key, output in out_dchis.items():
                self.out(key, output)
