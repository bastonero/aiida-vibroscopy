(topics-formulation)=

# Formulation

In this section we briefly explore the underlying theory and formulation made used in the workflows.
For a more in depth explanation, please refer to the [main paper of the package]().

Considered good sources are:

- Theoretical backgroud books:

    - Max Born and Kun Huang, _Dynamical Theory of Crystal Lattices_ (__1954__)
    - Peter Brueescvh, _Phonons: Theory and Experiments II_ (__1982__)

- Ab-initio related articles:

    - Stefano Baroni _et al._, _Phonons and related crystal properties from density-functional perturbation theory_, Rev. Modern Phys., __73__, 515 (__2001__)
    - Paolo Umari and Alfredo Pasquarello, _Infrared and Raman spectra of disordered materials from first principles_, Diamond and Rel. Mat., __14, 9 (__2005__)

In the code, all properties are computed within the __Born-Oppenheimer__ and __harmonic__ approximation.
The vibrational spectra are computed in the __first-order non-resonant__ regime: the infrared using the __dipole-dipole__ approximation, and the Raman using the __Placzek__ approximation.

:::{important}
These are considered __good approximations for insulators__. Nevertheless, a frequency dependent solution form is usually used also for the __resonant__ case and for __metals__. _Nevertheless_, one must be aware that in such cases (resonance, metals) these approximations might not hold, as multiphonon processes, non-adiabaticity, excitonic effects (i.e. electronic excitations), or even exciton-phonon interactions might be non negligible, thus comparison with experiments could result poor. If these effects are important for your case, you can refer to [S. Reichardt and L. Wirtz, _Science Advances_, __7__, 32 (__2020__)](https://www.science.org/doi/10.1126/sciadv.abb5915).

Moreover, temperature effects can also play a crucial role, as __anharmonic__ effects (of ions) should be incorporate to the phonons. A state-of-the-art approach, which differs from the classical molecular dynamics solutions, can be found using the [time-dependent self-consistent harmonic approximation](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.103.104305).
:::

Thus, for insulators, one needs to evaluate the following (static) tensors:

:::{list-table}
:widths: 15 15 15 15
:header-rows: 1

*   -
    - $\partial/\partial \tau_{K,k}$
    - $\partial/\partial \mathcal{E}_i$
    - $\partial^2/\partial \mathcal{E}_i \partial \mathcal{E}_j$
*   - $\partial/\partial \tau{L,l}$
    - $\Phi_{KL,kl}$
    - $Z^*_{K,ki}$
    - $\partial \chi_{ij}/\partial \tau{K,k}$
*   - $\partial/\partial \mathcal{E}_i$
    - $Z^*_{K,ik}$
    - $\epsilon^{\infty}_{ij}$
    - $\chi^{(2)}_{ijk}$
*   - $\partial^2/\partial \mathcal{E}_i \partial \mathcal{E}_j$
    - $\partial \chi_{ij}/\partial \tau{K,k}$
    - $\chi^{(2)}_{ijk}$
    - '-'
:::
