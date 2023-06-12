(topics-conventions)=

# Conventions

The package provides different data types. As it works with tensorial data embedded in a particular environment (or *metric*), i.e. defined by the atomic structure, a number of conventions had to be taken. In particular, here we give an overview of the units and tensorial indices conventions used for the quantities computed and then stored in the {{ vibrational_data }}.

All the tensors are always given in Cartesian coordinates, so that they are always commensurate with the structure reference system. In
parenthesis we give the corresponding attributes to access quantity via the {{ vibrational_data }}.

- Dielectric tensors (`dielectric`): no particular convention, unitless.
- Born effective charges (`born_charges`): in units of the elemental charge; the are given as an array with 3 indices, in the following order:

    1. Atomic index.
    2. Polarization index.
    3. Atomic displacements index.

- Non-linear optical susceptibility (`nlo_susceptibility`): in units of pm/V; 3rd rank tensors, with equivalent indices.
- Raman tensors (`raman_tensors`): in units of 1/Angstrom; array with 4 indices, respectively:

    1. Atomic index.
    2. Atomic displacements index.
    3. Polarization index.
    4. Polarization index.

    It is important to notice that the Raman tensors are dependent on the unitcell volume used. In the package, by convention,
    we make use of the unit cell volume, i.e. the one used in inputs to compute the tensors, even if with symmetries one can
    find a smaller, primitive, cell.

Regarding the conventions to carry out the numerical differentiations:

- For phonons, please refer to the [`Phonopy`](https://phonopy.github.io/phonopy/) documentation.
- For derivatives using the electric field, please have a look at the description of {class}`~aiida_vibroscopy.workflows.dielectric.numerical_derivatives.NumericalDerivativesWorkChain`.
