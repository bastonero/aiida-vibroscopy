# aiida-vibroscopy

AiiDA plugin that uses finite displacements and fields
to compute phonon properties, dielectric, Born effective charges,
 Raman and non-linear optical susceptibility tensors,
coming with lots of post-processing tools to compute infrared and
Raman spectra in different settings.

|    | |
|-----|----------------------------------------------------------------------------|
|Latest release| [![PyPI version](https://badge.fury.io/py/aiida-vibroscopy.svg)](https://badge.fury.io/py/aiida-vibroscopy)[![PyPI pyversions](https://img.shields.io/pypi/pyversions/aiida-vibroscopy.svg)](https://pypi.python.org/pypi/aiida-vibroscopy) |
|References| [![Static Badge](https://img.shields.io/badge/npj%20comp.%20mat.%20-%20implementation%20-%20purple?style=flat)](https://www.nature.com/articles/s41524-024-01236-3) |
|Getting help| [![Docs status](https://readthedocs.org/projects/aiida-vibroscopy/badge)](http://aiida-vibroscopy.readthedocs.io/) [![Discourse status](https://img.shields.io/discourse/status?server=https%3A%2F%2Faiida.discourse.group%2F)](https://aiida.discourse.group/)
|Build status| [![Build Status](https://github.com/bastonero/aiida-vibroscopy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bastonero/aiida-vibroscopy/actions) [![Coverage Status](https://codecov.io/gh/bastonero/aiida-vibroscopy/branch/main/graph/badge.svg)](https://codecov.io/gh/bastonero/aiida-vibroscopy) |
|Activity| [![PyPI-downloads](https://img.shields.io/pypi/dm/aiida-vibroscopy.svg?style=flat)](https://pypistats.org/packages/aiida-vibroscopy) [![Commit Activity](https://img.shields.io/github/commit-activity/m/bastonero/aiida-vibroscopy.svg)](https://github.com/bastonero/aiida-vibroscopy/pulse)
|Community|  [![Discourse](https://img.shields.io/discourse/topics?server=https%3A%2F%2Faiida.discourse.group%2F&logo=discourse)](https://aiida.discourse.group/)

## Installation
To install from PyPI, simply execute:

    pip install aiida-vibroscopy

or when installing from source:

    git clone https://github.com/bastonero/aiida-vibrosopy
    pip install .

## Command line interface tool
The plugin comes with a builtin CLI tool: `aiida-vibroscopy`.
For example, the following command should print:

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

## How to cite

If you use this plugin for your research, please cite the following works:

* L. Bastonero and N. Marzari, [*Automated all-functionals infrared and Raman spectra*](https://doi.org/10.1038/s41524-024-01236-3), npj Computational Materials **10**, 55 (2024)

* S. P. Huber _et al._, [*AiiDA 1.0, a scalable computational infrastructure for automated reproducible workflows and data provenance*](https://doi.org/10.1038/s41597-020-00638-4), Scientific Data **7**, 300 (2020)

* M. Uhrin _et al._, [*Workflows in AiiDA: Engineering a high-throughput, event-based engine for robust and modular computational workflows*](https://www.sciencedirect.com/science/article/pii/S0010465522001746), Computational Materials Science **187**, 110086 (2021)

Please, also cite the underlying Quantum ESPRESSO and Phonopy codes references.

## License
The `aiida-vibroscopy` plugin package is released under a special academic license.
See the `LICENSE.txt` file for more details.


## Acknowlegements
We acknowledge support from:
* the [U Bremen Excellence Chairs](https://www.uni-bremen.de/u-bremen-excellence-chairs) program funded within the scope of the [Excellence Strategy of Germany’s federal and state governments](https://www.dfg.de/en/research_funding/excellence_strategy/index.html);
* the [MAPEX](https://www.uni-bremen.de/en/mapex) Center for Materials and Processes.

<img src="https://raw.githubusercontent.com/aiida-phonopy/aiida-phonopy/main/docs/source/images/UBREMEN.png" width="300px" height="108px"/>
<img src="https://raw.githubusercontent.com/aiida-phonopy/aiida-phonopy/main/docs/source/images/MAPEX.jpg" width="300px" height="99px"/>
