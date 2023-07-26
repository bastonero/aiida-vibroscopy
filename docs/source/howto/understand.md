(howto-understand)=

# How-to understand the input/builder structure

In AiiDA the CalcJobs and WorkChains have usually nested inputs and different options on how to run the calculation
and/or workflows. To understand the nested input structure of CalcJobs/Workflows, we made them available in a clickable
fashion in the topics section.

Moreover, it could be useful to understand the
[_expose inputs/outputs_](https://aiida.readthedocs.io/projects/aiida-core/en/latest/topics/workflows/usage.html#modular-workflow-design)
mechanism used in AiiDA for workflows, which guarantees a __modular design__.
This means that the workflows can use the inputs of other workflows or calculations, and specify them under a new namespace.

This is the case for many workflows in this package. For example, the {class}`~aiida_vibroscopy.workflows.dielectric.base.DielectricWorkChain` makes use of the `PwBaseWorkChain` for the scf calculation, for which the inputs can be specified in the namespace `scf` (have a look at its inputs structure [here](topics-workflows-dielectric)). The same happens also for the {class}`~aiida_vibroscopy.workflows.phonons.base.PhononWorkChain`

Then, one can go further with a next level on nesting, i.e. we can use the `DielectricWorkChain` in an other workflow. This happens
for the `HarmonicWorkChain` and the `IRamanSpectraWorkChain`, for which its inputs are defined via the `dielectric` namespace.

This "trick" can continue over and over, allowing one to always be able to reuse written workflows in other more complex workflows.
