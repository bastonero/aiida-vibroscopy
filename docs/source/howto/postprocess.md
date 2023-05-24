(howto-postprocess)=

# How-to post-process data

Here you will find some _how-tos_ on how to post-process the `VibrationalData` and `PhonopyData`.
These can be generated via the `HarmonicWorkChain`, the `IRamanSpectraWorkChain` or the `PhononWorkChain`.
Please, have a look at the [tutorials](tutorials) for an overview, and at the specific [topics](topics) sections.

::: {hint}
You can always access to the detailed description of a function/method putting a `?` question mark in front
of the function and press enter. This will print the description of the function, with detailed information
on inputs, outputs and purpose.

::: {note}
A rendered version can also be found in the documentation referring to
the dedicated [API section](reference).
:::

:::

## How to get powder spectra

You can get the powder infrared and/or Raman spectra from a computed {{ vibrational_data }}. For Raman:

```python
from aiida.orm import load_code

vibro = load_node(IDENTIFIER) # a VibrationalData node

polarized_intensities, unpolarized_intensities, frequencies, labels = vibro.run_powder_raman_intensities(frequency_laser=532, temperature=300)
```

The total powder intensity will be the some of the polarized and unpolarized intensities:

```python
total = polarized_intensities + unpolarized_intensities
```

The infrared in a similar way, but no distinction between polarized and unpolarized, and no laser frequency and temperature inputs:

```python
intensities, frequencies, labels = vibro.run_powder_ir_intensities()
```

The `labels` output is referring to the irreducible representation labels of the modes.


## How to get single crystal spectra

::: {important}
It is extremely important that you match the experimental conditions, meaning the direction of
incoming/outgoing light should be given as in the experimental setup. So, pay extremely attention
on the convention used, both in the code and in the experiments.
:::

::: {note} Convention
In the following methods, the direction of the light must be given in crystal/fractional coordinates.
:::

You can get the single crystal infrared and/or Raman spectra from a computed {{ vibrational_data }}. For Raman:

```python
from aiida.orm import load_code

vibro = load_node(IDENTIFIER) # a VibrationalData node

incoming = [0,0,1] # light polatization of the incident laser beam
outgoing = [0,0,1] # light polatization of the emitted laser beam

intensities, frequencies, labels = vibro.run_single_crystal_raman_intensities(
    pol_incoming=incoming,
    pol_outgoing=outgoing,
    frequency_laser=532,
    temperature=300,
)
```

Infrared in a similar fashion:

```python
from aiida.orm import load_code

vibro = load_node(IDENTIFIER) # a VibrationalData node

incoming = [0,0,1] # light polatization of the incident laser beam

intensities, frequencies, labels = vibro.run_single_crystal_ir_intensities(pol_incoming=incoming)
```

::: {hint} Cartesian coordinates
To get the direction in Cartesian coordinates expressed in crystal coordinates of the primitive cell, you can use the following
snippet

```python
import numpy as np

cell = vibro.get_primitive_cell().cell

incoming_cartesian = [0,0,1]
inv_cell = np.linalg.inv(cell)
incoming = np.dot(invcell, incoming_cartesian)
```
:::

## How to get the IR/Raman active modes

To get the active modes from a computed {{ vibrational_data }}:

```python
from aiida.orm import load_code

vibro = load_node(IDENTIFIER) # a VibrationalData node

frequencies, _, labels = vibro.run_active_modes(selectrion_rules='ir')
```

This will return the IR active frequencies and the corresponding irreducible representation labels.
For Raman active modes, specify `'raman'` instead.
