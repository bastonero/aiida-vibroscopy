# General information

All the examples make use of the `get_builder_from_protocol` to help filling all the necessary information.
This means you need to install the compatible SSSP library via `aiida-pseudo`.

For instance:

```console
aiida-pseudo install sssp -v 1.3 -x PBEsol
```

If you cannot download, and/or you want to use your own pseudo-potentials, you can still download separately the files on the [_Materials Cloud Archive_](https://archive.materialscloud.org/record/2023.65) (search for _SSSP_ if this link doesn't work). Then, you can proceed by installing a `CutoffsPseudoPotentialFamily`, i.e. a family of pseudo-potentials with pre-defined cutoffs.

See [here](https://aiida-pseudo.readthedocs.io/en/latest/howto.html#adding-recommended-cutoffs) for further instructions.

Then you will need to specify the `pseudo_family` you created in the `overrides`. Where to specify it depends on the `WorkChain` you are using. For example, in the case of the `IRamanSpectraWorkChain` the two places are:

```
dielectric:
    scf:
        pseudo_family: 'YourPseudoFamilyWithCutoff' # replace here with the label you created your family
phonon:
    scf:
        pseudo_family: 'YourPseudoFamilyWithCutoff' # replace here with the label you created your family
```

In general, everywhere where the inputs of the `PwBaseWorkChain` are _exposed_. In the previous example, these inputs are exposed under the `scf` namespace.
