# General information

You need to install first a pseudpotential library via `aiida-pseudo` (see examples/README.md).

You can run workflows using the command `aiida-vibroscopy launch WORKFLOW_NAME`.

To know the available implemented workflows that can be used via CLI:

```console
> aiida-vibroscopy launch --help
Usage: aiida-vibroscopy launch [OPTIONS] COMMAND [ARGS]...

  Launch workflows.

Options:
  -v, --verbosity [notset|debug|info|report|warning|error|critical]
                                  Set the verbosity of the output.
  -h, --help                      Show this message and exit.

Commands:
  dielectric      Run an `DielectricWorkChain`.
  harmonic        Run a `HarmonicWorkChain`.
  iraman-spectra  Run an `IRamanSpectraWorkChain`.
  phonon          Run an `PhononWorkChain`.
```

For furhter information, for instance:

```console
> aiida-vibroscopy launch dielectric --help
Usage: aiida-vibroscopy launch dielectric [OPTIONS]

  Run an `DielectricWorkChain`.

  It computes dielectric, Born charges, Raman and non-linear optical
  susceptibility tensors for a given structure.

Options:
  --pw CODE                       The code to use for the pw.x executable.
                                  [required]
  -S, --structure DATA            A StructureData node identified by its ID or
                                  UUID.
  -p, --protocol [fast|moderate|precise]
                                  Select the protocol that defines the
                                  accuracy of the calculation.  [default:
                                  moderate]
  -F, --pseudo-family GROUP       Select a pseudopotential family, identified
                                  by its label.
  -k, --kpoints-mesh <INTEGER INTEGER INTEGER FLOAT FLOAT FLOAT>...
                                  The number of points in the kpoint mesh
                                  along each basis vector and the offset.
                                  Example: `-k 2 2 2 0 0 0`. Specify `0.5 0.5
                                  0.5` for the offset if you want to result in
                                  the equivalent Quantum ESPRESSO pw.x `1 1 1`
                                  shift.
  -o, --overrides FILENAME        The filename or filepath containing the
                                  overrides, in YAML format.
  -D, --daemon                    Submit the process to the daemon instead of
                                  running it and waiting for it to finish.
                                  [default: True]
  -v, --verbosity [notset|debug|info|report|warning|error|critical]
                                  Set the verbosity of the output.
  -h, --help                      Show this message and exit.
```

## Example

```console
> aiida-vibroscopy launch dielectric \
    --pw pw@localhost \ # change here with your installed code
    -S 12345 \ # replace with you StructureData
    --protocol moderate \ # (optional) choose between fast, moderate, precise
    --pseudo-family SSSP/1.3/PBEsol/efficiency \ # (optional) change with your favorite
    --kpoints-mesh 4 4 4 \ # (optional) it overrides the overrides, in particular removes kpoints_distance and kpoints_parallel_distance
    --overrides overrides.yaml \ # (optional) filepath to overrides
    --deamon True # (optional) submit to deamon
```
