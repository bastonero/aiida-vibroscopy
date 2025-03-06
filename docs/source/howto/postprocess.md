(howto-postprocess)=

# Post-process data

Here you will find some _how-tos_ on how to post-process the `VibrationalData` and `PhonopyData`.
These can be generated via the `HarmonicWorkChain`, the `IRamanSpectraWorkChain` or the `PhononWorkChain`.
Please, have a look at the [tutorials](tutorials) for an overview, and at the specific topics section.

::: {hint}
You can always access to the detailed description of a function/method putting a `?` question mark in front
of the function and press enter. This will print the description of the function, with detailed information
on inputs, outputs and purpose.

A rendered version can also be found in the documentation referring to
the dedicated [API section](reference).
:::

## Powder spectra

You can get the powder infrared and/or Raman spectra from a computed {{ vibrational_data }}. For Raman:

```python
from aiida.orm import load_code

vibro = load_node(IDENTIFIER) # a VibrationalData node

polarized_intensities, depolarized_intensities, frequencies, labels = vibro.run_powder_raman_intensities(frequency_laser=532, temperature=300)
```

The total powder intensity will be the some of the polarized (backscattering geometry, HH/VV setup)
and depolarized (90º geometry, HV setup) intensities:

```python
total = polarized_intensities + depolarized_intensities
```

The infrared in a similar way, but no distinction between polarized and depolarized, and no laser frequency and temperature inputs:

```python
intensities, frequencies, labels = vibro.run_powder_ir_intensities()
```

The `labels` output is referring to the irreducible representation labels of the modes.


## Single crystal polarized spectra

::: {important}
It is extremely important that you match the experimental conditions, meaning the direction of
incoming/outgoing light should be given as in the experimental setup. So, pay extremely attention
on the convention used, both in the code and in the experiments.
:::

::: {note} Convention
In the following methods, the direction of the light must be given in **Cartesian coordinates**.
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

<!-- ::: {admonition} Cartesian coordinates
:class: hint
To get the direction in Cartesian coordinates expressed in crystal coordinates of the primitive cell, you can use the following
snippet
```python
import numpy as np

cell = vibro.get_unitcell().cell

incoming_cartesian = [0,0,1]
inv_cell = np.linalg.inv(cell)
incoming = np.dot(invcell.T, incoming_cartesian)
```
::: -->

## IR/Raman active modes

To get the active modes from a computed {{ vibrational_data }}:

```python
from aiida.orm import load_code

vibro = load_node(IDENTIFIER) # a VibrationalData node

frequencies, _, labels = vibro.run_active_modes(selectrion_rules='ir')
```

This will return the IR active frequencies and the corresponding irreducible representation labels.
For Raman active modes, specify `'raman'` instead.

## Clamped Pockels tensor

You can get individual variables consituting the Pockels tensor from a computed {{ vibrational_data }}. For instance:

```python
from aiida.orm import load_node

vibro = load_node(IDENTIFIER) # a VibrationalData node

r_tot, r_el, r_ion = vibro.run_clamped_pockels_tensor()
```

The total Pockels tensor will be the sum of the electronic and ionic tensors:

```python
r_tot, r_el, r_ion
```
