# Examples

This section is the Phase 2 entry point for example-driven documentation.

The notebooks under `examples/notebooks/` remain the canonical rendered tutorial artifacts, synced from the source scripts in `examples/scripts/` plus `experiments/spectral_filter_optimization.py`. During docs builds, the notebooks are mirrored into the docs tree and rendered directly as pages below.

Recommended order:

1. `basic_propagation` for the core optics objects and a first forward pass
2. `lens_optimization` for the main differentiable-optics workflow
3. One domain-specific branch:
   `4f_correlator` for Fourier optics,
   `hologram_coherent_logo` for phase-only design, or
   `incoherent_camera` for incoherent imaging
4. Then the family extensions and more specialized examples

Example families:

- `lens_optimization` -> `sensitivity_analysis`, `metaatom_optimization`
- `4f_correlator` -> `4f_edge_optimization`
- `hologram_coherent_logo` -> `holography_polarized_dual`

```{toctree}
:maxdepth: 1

notebooks/basic_propagation.ipynb
notebooks/lens_optimization.ipynb
notebooks/4f_correlator.ipynb
notebooks/hologram_coherent_logo.ipynb
notebooks/incoherent_camera.ipynb
notebooks/spectral_filter_optimization.ipynb
notebooks/onn_mnist.ipynb
notebooks/4f_edge_optimization.ipynb
notebooks/holography_polarized_dual.ipynb
notebooks/metaatom_optimization.ipynb
notebooks/sensitivity_analysis.ipynb
```
