# fouriax Documentation

This page is the GitHub-friendly entry point for browsing documentation in the repository.

The `docs/` tree is authored for Sphinx, so some files in this directory contain MyST and Sphinx directives that render best in the built docs site. For direct GitHub browsing, start from the plain Markdown and notebook links below.

## Core Docs

- [Project README](../README.md)
- [User Guide](./USER_GUIDE.md)
- [Architecture](./ARCHITECTURE.md)
- [Development Workflow](./DEVELOPMENT_WORKFLOW.md)
- [TODO](./TODO.md)

## API Reference Source

These pages define the Sphinx API structure, but GitHub will show the raw MyST content rather than the built reference site:

- [API Index](./api/index.md)
- [Core Data Model](./api/core.md)
- [Layers](./api/layers.md)
- [Propagation and Planning](./api/propagation.md)
- [Advanced Utilities](./api/advanced.md)

## Examples

The runnable scripts are the source of truth. The notebooks are synced tutorial artifacts.

### Scripts

- [4f Correlator](../examples/scripts/4f_correlator.py)
- [4f Edge Optimization](../examples/scripts/4f_edge_optimization.py)
- [Coherent Hologram](../examples/scripts/hologram_coherent_logo.py)
- [Polarized Holography](../examples/scripts/holography_polarized_dual.py)
- [Incoherent Camera](../examples/scripts/incoherent_camera.py)
- [Lens Optimization](../examples/scripts/lens_optimization.py)
- [Meta-Atom Optimization](../examples/scripts/metaatom_optimization.py)
- [ONN MNIST](../examples/scripts/onn_mnist.py)
- [Sensitivity Analysis](../examples/scripts/sensitivity_analysis.py)
- [Spectral Filter Optimization](../examples/scripts/spectral_filter_optimization.py)

### Notebooks

- [4f Correlator](../examples/notebooks/4f_correlator.ipynb)
- [4f Edge Optimization](../examples/notebooks/4f_edge_optimization.ipynb)
- [Coherent Hologram](../examples/notebooks/hologram_coherent_logo.ipynb)
- [Polarized Holography](../examples/notebooks/holography_polarized_dual.ipynb)
- [Incoherent Camera](../examples/notebooks/incoherent_camera.ipynb)
- [Lens Optimization](../examples/notebooks/lens_optimization.ipynb)
- [Meta-Atom Optimization](../examples/notebooks/metaatom_optimization.ipynb)
- [ONN MNIST](../examples/notebooks/onn_mnist.ipynb)
- [Sensitivity Analysis](../examples/notebooks/sensitivity_analysis.ipynb)
- [Spectral Filter Optimization](../examples/notebooks/spectral_filter_optimization.ipynb)

## Building the Docs Locally

```bash
pip install -e ".[dev]"
PYTHONPATH=src .venv/bin/python -m sphinx -W -b html docs docs/_build/html
```
