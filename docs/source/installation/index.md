---
myst:
  substitutions:
    SSSP: Standard Solid-State Pseudopotentials (SSSP)
    aiida-pseudo: '`aiida-pseudo`'
    pip: '`pip`'
---

# Get started

(installation-requirements)=

## Requirements

To work with `aiida-vibroscopy`, you should have:

- installed `aiida-core`
- configured an AiiDA profile.

Please refer to the [documentation](https://aiida.readthedocs.io/projects/aiida-core/en/latest/intro/get_started.html) of `aiida-core` for detailed instructions.

(installation-installation)=

## Installation

The Python package can be installed from the Python Package index [PyPI](https://pypi.org/) or directly from the source:

::::{tab-set}

:::{tab-item} PyPI
The recommended method of installation is to use the Python package manager `pip`:

```console
$ pip install aiida-vibroscopy
```

This will install the latest stable version that was released to PyPI.
:::

:::{tab-item} Source
To install the package from source, first clone the repository and then install using `pip`:

```console
$ git clone https://github.com/bastonero/aiida-vibroscopy
$ pip install -e aiida-vibroscopy
```

The ``-e`` flag will install the package in editable mode, meaning that changes to the source code will be automatically picked up.
:::

::::

(installation-setup)=

## Setup

(installation-setup-computer)=

### Computer

To run the workflows, you need Quantum ESPRESSO on a compute resource. Such computer should first be set up in AiiDA.
This can be done from the command line interface (CLI) or the Python application programming interface (API).
In this example, we will set up the `localhost`, the computer where AiiDA itself is running:

::::{tab-set}

:::{tab-item} CLI

To set up a computer, use the ``verdi`` CLI of ``aiida-core``.

```console
$ verdi computer setup -n -L localhost -H localhost -T core.local -S core.direct
```

After creating the localhost computer, configure it using:

```console
$ verdi computer configure core.local localhost -n --safe-interval 0
```

Verify that the computer was properly setup by running:

```console
$ verdi computer test localhost
```
:::

:::{tab-item} API

To setup a computer using the Python API, run the following code in a Python script or interactive shell:

```python

from aiida.orm import Computer

computer = Computer(
    label='localhost',
    hostname='localhost',
    transport_type='core.local',
    scheduler_type='core.direct'
).store()
computer.configure()
```
:::
::::

For more detailed information, please refer to the documentation [on setting up compute resources](https://aiida.readthedocs.io/projects/aiida-core/en/latest/howto/run_codes.html#how-to-set-up-a-computer).

(installation-setup-code)=

### Code

The Quantum ESPRESSO ``pw.x`` routine is needed for all the workflows, and it should be setup in AiiDA.
This can be done from the command line interface (CLI) or the Python application programming interface (API).
Here we will setup it up supposing that is installed on the computer where AiiDA is running:

::::{tab-set}

:::{tab-item} CLI

To setup a particular Quantum ESPRESSO code, use the ``verdi`` CLI of ``aiida-core``.

```console
$ verdi code create core.code.installed -n --computer localhost --label hp --default-calc-job-plugin quantumespresso.pw --filepath-executable /path/to/pw.x
```
:::

:::{tab-item} API

To setup particular Quantum ESPRESSO code using the Python API, run the following code in a Python script or interactive shell:

```python

from aiida.orm import InstalledCode

computer = load_computer('localhost')
code = InstalledCode(
label='pw',
computer=computer,
filepath_executable='/path/to/pw.x',
default_calc_job_plugin='quantumespresso.pw',
).store()
```
:::

::::

:::{important}
Make sure to replace `/path/to/pw.x` with the actual absolute path to the `pw.x` binary.
:::

For more detailed information, please refer to the documentation [on setting up codes](https://aiida.readthedocs.io/projects/aiida-core/en/latest/howto/run_codes.html#how-to-setup-a-code).

(installation-setup-pseudopotentials)=

### Pseudopotentials

The `pw.x` code used in this plugin requires pseudo potentials.
The simplest way of installing these is through the `aiida-pseudo` plugin package.
This should come as a dependency of `aiida-vibroscopy` and should already be installed.
If this is not the case, it can be installed using:

```console
$ pip install aiida-pseudo
```

At a minimum, at least one pseudo potential family should be installed.
We recommend using the [SSSP] with the PBEsol functional:

```console
$ aiida-pseudo install sssp -x PBEsol
```

For more detailed information on installing other pseudo potential families, please refer to the documentation of [aiida-pseudo].
