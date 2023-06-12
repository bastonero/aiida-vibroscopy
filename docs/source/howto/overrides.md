(howto-overrides)=

# How-to use protocols and overrides

:::{important}
The following how-to assume that you are familiar with submitting the workflows of the package.
At the very least, make sure to have completed the [tutorials](../tutorials/index).

Please, make also sure to have a look at the [dedicated topic section](topics-overrides),
where we explain more conceptually what this is about and why/how/when you should use it.
:::

To simplify the filling of the inputs, you can use the `get_builder_from_protocol` of a WorkChain, specifying eventually a protocol and some overrides,
as you probably noticed in the [tutorials](tutorials).

::: {note}
Not necesseraly all the WorkChains have the `get_builder_from_protocol`. E.g. The {class}`~aiida_vibroscopy.workflows.dielectric.numerical_derivatives`.
:::

## How to use in general

Depending on the WorkChain, you will need to specify some minimal, _but required_, inputs. Here is an example:

```python
from aiida.orm import load_code
from aiida.plugins import WorkflowFactory
from aiida.engine import submit

WorkChain = WorkflowFactory("vibroscopy.some.workchain")

code = load_code(IDENTIFIER)
structure = load_node(IDENTIFIER)
overrides = {...}

builder = WorkChain.get_builder_from_protocol(code=code, structure=structure, protocol="fast", overrides=overrides)

submit(builder)
```

To get to know the available protocols:

```python
In [1]: WorkChain.get_available_protocols()
Out[1]:
{'moderate': {'description': 'Protocol to perform an IR/Raman spectra calculation at normal precision at moderate computational cost.'},
 'precise': {'description': 'Protocol to perform an IR/Raman spectra calculation at high precision at higher computational cost.'},
 'fast': {'description': 'Protocol to perform an IR/Raman spectra calculation at low precision at minimal computational cost for testing purposes.'}}

```

## How to use overrides (beginner)

As stated at the beginning, you might need to tweak your builder inputs before submitting. Instead of modifying the inputs via the builder, you can do it via `overrides`. The way the overrides must be secified __is WorkChain dependent__. The structure should be the same as the input specification of the specific WorkChain, so always refer to the inputs of the particular workflow, which you can find [here](topics-workflows).

Here an example on how to use it for the `IRamanSpectraWorkChain`.

```python
from aiida.plugins import WorkflowFactory
IRamanSpectraWorkChain = WorkflowFactory("vibroscopy.spectra.iraman")

code = load_code(IDENTIFIER)
structure = load_node(IDENTIFIER)
overrides = {
    "phonon": {
        "displacement_generator":{
            "distance": 0.01,
        },
        "scf":{
            "pseudo_family": "SSSP/1.2/PBEsol/efficiency",
            "kpoints_distance": 0.4,
            "pw":{
                "parameters":{
                    "SYSTEM":{
                        "ecutwfc": 40.0,
                        "ecutrho": 320.0,
                    },
                },
                "metadata":{
                    "options":{
                        "resources":{"num_machines":1, "num_mpiprocs_per_machine":2,}
                    },
                },
                "settings":{
                    "cmdline": ["-nk", "2", "-ndiag", "1"]
                },
            }
        },
    },
    "dielectric":{
        "kpoints_parallel_distance": 0.2,
        "central_difference":{
            "electric_field_step": 0.0005,
            "accuracy": 2,
        },
        "scf":{
            "pseudo_family": "SSSP/1.2/PBEsol/efficiency",
            "kpoints_distance": 0.4,
            "pw":{
                "parameters":{
                    "SYSTEM":{
                        "ecutwfc": 40.0,
                        "ecutrho": 320.0,
                    },
                },
                "metadata":{
                    "options":{
                        "resources":{"num_machines":1, "num_mpiprocs_per_machine":2,}
                    },
                },
                "settings":{
                    "cmdline": ["-nk", "1", "-ndiag", "1"]
                },
            }
        },
    },
    "symmetry":{
        "symprec": 1e-5,
        "distinguish_kinds": True,
        "is_symmetry": True
    }
    "settings":{
        "run_parallel": True
    }
}

builder = IRamanSpectraWorkChain.get_builder_from_protocol(code=code, structure=structure, overrides=overrides)
```

To make you understand, as the `IRamanSpectraWorkChain` uses the `DielectricWorkChain`, the inputs of this workchain are specified
under the name `dielectric`. The same overrides for `DielectricWorkChain`, if run alone, would be:

```python
from aiida.plugins import WorkflowFactory
DielectricWorkChain = WorkflowFactory("vibroscopy.dielectric")

code = load_code(IDENTIFIER)
structure = load_node(IDENTIFIER)

overrides = {
    "kpoints_parallel_distance": 0.2,
    "central_difference":{
        "electric_field_step": 0.0005,
        "accuracy": 2,
        },
    "scf":{
        "pseudo_family": "SSSP/1.2/PBEsol/efficiency",
        "kpoints_distance": 0.4,
        "pw":{
            "parameters":{
                "SYSTEM":{
                    "ecutwfc": 40.0,
                    "ecutrho": 320.0,
                },
            },
            "metadata":{
                "options":{
                    "resources":{"num_machines":1, "num_mpiprocs_per_machine":2,}
                },
            },
            "settings":{
                "cmdline": ["-nk", "1", "-ndiag", "1"]
            },
        }
    }
}

builder = DielectricWorkChain.get_builder_from_protocol(code=code, structure=structure, overrides=overrides)
```

You got the gist, so it will be the same for the `phonon` inputs, which corresponds to the `PhononWorkChain`.

## How to overrides (advanced)

Once you got used to using the overrides, you will notice your python script will become quite a mess.
The idea is to write the overrides in a separate file, using for example the convenient `YAML` format,
and then load them in the script.

```python
import pathlib
from aiida.plugins import WorkflowFactory
IRamanSpectraWorkChain = WorkflowFactory("vibroscopy.spectra.iraman")

code = load_code(IDENTIFIER)
structure = load_node(IDENTIFIER)

builder = IRamanSpectraWorkChain.get_builder_from_protocol(
    code=code,
    structure=structure,
    overrides=pathlib.Path("/path/to/overrides.yml"),
)
```

And your overrides YAML will much cleaner as well:

```
phonon:
    displacement_generator:
        distance: 0.01
    scf:
        pw:
            kpoints_distance: 0.2
            pw:
                ...
dielectric:
    central_difference:
        accuracy: 2
    scf:
        pw:
            ...
```

## Get automated inputs for insulators, metals, and magnetism

Inputs might change depending on the nature of the material: insulating, metallic, a form of magnetism.
This can be the need of specifying a smearing or a magnetic configuration.

To do this via the `get_builder_from_protocol`. For specifying it is an insulator:

```python
from aiida_quantumespresso.common.types import ElectronicType

kwargs = {'electronic_type': ElectronicType.INSULATOR}
builder = WorkChain.get_builder_from_protocol(code, structure, **kwargs)
```

Metals are the default, but you can specify it via `ElectronicType.METAL` if you want.

To specify a __collinear ferromagnetic__ structure:

```python
from aiida_quantumespresso.common.types import SpinType

kwargs = {'spin_type': SpinType.COLLINEAR}
builder = WorkChain.get_builder_from_protocol(code, structure, **kwargs)
```

If you have a magnetic sublattice, e.g. antiferromagnetic, then you can still easily do it:

```python
from aiida_quantumespresso.common.types import SpinType

kwargs = {
    'spin_type': SpinType.COLLINEAR,
    'initial_magnetic_moments': {
        'Fe1': +8.0,
        'Fe2': -8.0,
    }
}
builder = WorkChain.get_builder_from_protocol(code, structure, **kwargs)
```

::: {important}
The initial magnetic moments are defined through the valence electrons of the pseupotential in use. So for the case of iron and SSSP pseudo,
the valence electrons are 16. Thus, using 8 translates in the following pw parameters:

```python
builder.pw.parameters = Dict({
    "SYSTEM":{
        "starting_magnetization": {
            "Fe1": 0.5,
            "Fe2": -0.5,
        }
    }
})
```
:::

Again, we stress the fact of always checking the inputs before submitting your calculations!

::: {note}
The magnetic sublattice is defined using the `name` in the `StructureData`, as usual in Quantum ESPRESSO.
:::
