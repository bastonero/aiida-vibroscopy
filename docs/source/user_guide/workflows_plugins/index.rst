
.. _sec.workflows:

Available workflows
-------------------

Description
^^^^^^^^^^^
The available workflows are the results of the combination of `aiida-quantumespresso`_ and
`aiida-phonopy`_,the automated versions of `Quantum Espresso`_ and `Phonopy`_.

The package has the main focus on providing **vibrational spectra** automated workflows, from which the name *vibroscopy* comes from!
Since the underlying theory and packages allow for the computation of many other properties (e.g. thermal expansion),
the plugin provides the capabilities for the computation of many other vibrational/phonon properties.

Workflows
^^^^^^^^^

* **IR/Raman spectra**: the :py:class:`~aiida_vibroscopy.workflows.spectra.iraman.IRamanSpectraWorkChain` is the core the package. It computes using finite displacements and fields the *infra-red* and *Raman* spectra. The workflow can be used for computing either only the IR spectra (less computationally demanding), or for both IR and Raman at the same time. The computation of the Raman spectra makes the IR freely available!
* **Dielectric properties**: the :py:class:`~aiida_vibroscopy.workflows.dielectric.base.DielectricWorkChain` is meant to compute the dielectric properties in the static limit. The workflows can then be used alone to compute *Born effective charges*, *susceptibility (or dielectric) tensor*, *Raman tensors* and the *non-linear optical susceptibility*.
* **Frozen phonons**: the :py:class:`~aiida_vibroscopy.workflows.phonons.harmonic.HarmonicWorkChain` can be used for the computation of the *interatomic force constants* (IFC), which constitutes the base for phonon properties in the *harmonic approximation*. The workflow returns a **PhonopyData** as outputs, which can then be used for post-processing via the calculations (CalcJobs) provided in `aiida-phonopy`_.

.. _aiida-quantumespresso: https://github.com/aiidateam/aiida-quantumespresso
.. _aiida-phonopy: https://github.com/aiida-phonopy/aiida-phonopy
.. _Quantum Espresso: http://www.quantum-espresso.org/
.. _Phonopy: http://phonopy.github.io/phonopy/
