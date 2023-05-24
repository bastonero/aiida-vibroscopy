(topics-overrides)=

# On protocols and overrides

The `get_builder_from_protocol` method is an additional method that we added to make the filling of
the inputs of the builder easier and smoother. In fact, many parameters that should be defined are
somewhat always the same or generalisable, or you may not even need to specify them, such as the number of atoms in the `SYSTEM`
card of quantum espresso. Already `aiida-quantumespresso` does it for you, as you probably already know.

Nevertheless, it is important to know that inputs got in this way __should be considered as starting point, and not as good for any calculation__.
Infact, especially for phonons and Raman (i.e. third order derivatives) many parameters should be converged.

For example, one should check that the tensors or phonon modes are converged (within desired thresholds) in respect with:

- Energy cutoff(s)
- K point sampling
- Convergence of the electronic charge (here we mean the `ELECTRONS.conv_thr`)

That's why you should use the protocols as a starting point to get started with your simulations.

::: {important} The `get_builder_from_protocol` does not raise errors
The `get_builder_from_protocol` _does not perform_ any kind of checking on the overrides you are giving.
That's why is important that you always check your builder inputs before submitting important simulations.
:::
