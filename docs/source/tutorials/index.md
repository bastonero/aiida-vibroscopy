(tutorials)=

# Tutorials

:::{important}
Before you get started, make sure that you have:

- installed the `aiida-quantumespresso` and `aiida-phonopy` packages ([see instructions](installation-installation))
- configured the `pw.x` code ([see instructions](installation-setup-code))
- installed the SSSP pseudopotential family ([see instructions](installation-setup-pseudopotentials))
:::

In this section you will find some tutorials that you will guide you through how to use the aiida-vibroscopy plugin, from **zero** to **hero**! We strongly recommend to start from the first one and work your way up with the other ones.

Go to one of the tutorials!

#. [Phonon band structure](../1_phonon.ipynb): get started with predicting the phonon dispersion of silicon.
#. [Dielectric properties](../2_dielectric.ipynb): compute the dielectric and Raman tensors of silicon.
#. [Raman spectra](../3_iraman.ipynb): learn the automated calculation of Raman spectra of silicon.
#. [Polar materials](../4_polar.ipynb): predict the phonon and dielectric properties of AlAs with LO-TO splitting.
#. [Spectra using different functionals](../5_iraman_functionals.ipynb): compute the vibrational spectra of LiCo{sub}`2` using DFT and __DFT+U+V__, and understand the power of Hubbard corrections comparing the results to experiments!

Here below the estimated time to run each tutorial (jupyter notebook):

```{nb-exec-table}
```

```{toctree}
:maxdepth: 1
:hidden: true

../1_phonon.ipynb
../2_dielectric.ipynb
../3_iraman.ipynb
../4_polar.ipynb
../5_iraman_functionals.ipynb
```
