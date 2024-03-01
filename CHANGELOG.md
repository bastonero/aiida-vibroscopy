## v1.1.0

This minor release includes new post-processing utilities, a small breaking change in [[42503f3]](https://github.com/bastonero/aiida-vibroscopy/commit/42503f312d9a812cfc46d4c4a03a78641201e1d3) with regards to reference system for non-analytical and polarization directions. Some examples providing
a unique python script to run the `IRamanSpectraWorkChain` are also added to help new users to get started. The license terms are also updated.
A CHANEGELOG file is finally added to keep track in a pretty format of the changes among releases of the code.

The new post-processing utilities can be used directly through a `VibrationalData` node, in a similar fashion to the other methods.
For instance, to compute the complex dielectric matrix and the normal reflectivity in the infrared regime:

```python
node = load_node(PK) # PK to a VibrationalData node

complex_dielectric = node.run_complex_dielectric_function() # (3,3,num_steps) shape complex array; num_steps are the number of frequency points where the function is evaluated
reflectivity = node.run_normal_reflectivity_spectrum([0,0,1]) # (frequency points, reflectance value), [0,0,1] is the orthogonal direction index probed via q.eps.q
```

Now, the polarization and non-analytical directions in _all_ methods in aiida-vibroscopy should be given in Cartesian coordinates:

```python
node = load_node(PK) # PK to a VibrationalData node

scattering_geometry = dict(pol_incoming=[1,0,0], pol_outgoing=[1,0,0], nac_direction=[0,0,1]) # corresponding to ZXXZ scattering setup
intensities, frequencies, mode_symmetry_labels = node.run_single_crystal_raman_intensities(**scattering_geometry)
```

### ‼️ Breaking changes

* Post-processing: polarization and nac directions in Cartesian coordinates [[42503f3]](https://github.com/bastonero/aiida-vibroscopy/commit/42503f312d9a812cfc46d4c4a03a78641201e1d3)

### 👌 Improvements

* Post-processing: computation of complex dielectric function and normal reflectivity in the infrared [[42503f3]](https://github.com/bastonero/aiida-vibroscopy/commit/42503f312d9a812cfc46d4c4a03a78641201e1d3)
* `Examples`: new folder with working examples for different use cases to get new users started [[7deb31b]](https://github.com/bastonero/aiida-vibroscopy/commit/7deb31b5f547ca16e4522be960b4aa5bbe13fccf)
* CI: add codecov step [[f36e8a1]](https://github.com/bastonero/aiida-vibroscopy/commit/f36e8a10566af68843546bae428560dff393aaf1)

### 🐛 Bug Fixes

* `Docs`: fix typos [[85b1830]](https://github.com/bastonero/aiida-vibroscopy/commit/85b18305be6e7e76efce35d9e4ae4c5a3547f9bc), [[e924b3d]](https://github.com/bastonero/aiida-vibroscopy/commit/e924b3dd436a67192f6c0780ff3a318581ab1fc5)
* Post-processing: fix coordinates and units [[42503f3]](https://github.com/bastonero/aiida-vibroscopy/commit/42503f312d9a812cfc46d4c4a03a78641201e1d3)

### 📚 Documentation

* Set correct hyperlink for AiiDA paper [[c92994d]](https://github.com/bastonero/aiida-vibroscopy/commit/c92994de36c336a265ac262eea2dc8d77fb11f08)

### 🔧 Maintenance

* Adapt tests also for other changes [[be3a6b7]](https://github.com/bastonero/aiida-vibroscopy/commit/be3a6b7d67926816957634fd7b520cd021532f0f)
* Add loads of tests [[42503f3]](https://github.com/bastonero/aiida-vibroscopy/commit/42503f312d9a812cfc46d4c4a03a78641201e1d3)
* `README`: add more information and badges [[c92994d]](https://github.com/bastonero/aiida-vibroscopy/commit/c92994de36c336a265ac262eea2dc8d77fb11f08)
* Docs: Remove aiida.manage.configuration.load_documentation_profile [[f914cbb]](https://github.com/bastonero/aiida-vibroscopy/commit/f914cbb5460d4f988dd117628890a8f53f1c976a)
* DevOps: update docs dependencies [[a0d124e]](https://github.com/bastonero/aiida-vibroscopy/commit/a0d124ee24cb287f9d90583b389f38d6b6265b9e)
* Bump SSSP version to 1.3 in tests [[94c72e5]](https://github.com/bastonero/aiida-vibroscopy/commit/94c72e5183584af08d9874fe2b6fc2ad41fce1b5)

### ⬆️ Update dependencies

* DevOps: update docs dependencies [[a0d124e]](https://github.com/bastonero/aiida-vibroscopy/commit/a0d124ee24cb287f9d90583b389f38d6b6265b9e)
