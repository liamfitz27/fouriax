# TODO

## Example Notebooks

- [ ] Finalize notebook format and text for `examples/notebooks/4f_correlator.ipynb`
- [ ] Finalize notebook format and text for `examples/notebooks/4f_edge_optimization.ipynb`
- [ ] Finalize notebook format and text for `examples/notebooks/hologram_coherent_logo.ipynb`
- [ ] Finalize notebook format and text for `examples/notebooks/holography_polarized_dual.ipynb`
- [ ] Finalize notebook format and text for `examples/notebooks/incoherent_camera.ipynb`
- [ ] Finalize notebook format and text for `examples/notebooks/lens_optimization.ipynb`
- [ ] Finalize notebook format and text for `examples/notebooks/metaatom_optimization.ipynb`
- [ ] Finalize notebook format and text for `examples/notebooks/onn_mnist.ipynb`
- [ ] Finalize notebook format and text for `examples/notebooks/sensitivity_analysis.ipynb`
- [ ] Finalize notebook format and text for `examples/notebooks/spectral_filter_optimization.ipynb`

## Examples And Benchmarks

- [ ] Add a dedicated performance benchmarking script
- [ ] Add performance comparisons across propagators, grids, and batch sizes
- [ ] Profile and document optimization opportunities in the main optics paths
- [ ] Add another hybrid module optimization example

## Potential Follow-Up TODOs

- [ ] Add full test coverage for batched Jones fields and batched meta-atom layers
- [ ] Add batch-aware convenience constructors to `Field` (for example `batch_shape` support)
- [ ] Add end-to-end example smoke tests for the scripts in `examples/scripts/`
- [ ] Add a benchmark for memory scaling versus batch size and wavelength count
- [ ] Add a developer note on shape conventions (`*batch, wavelength, ...`) across the optics API
- [ ] Add notebook execution checks to CI for a small curated subset of examples
