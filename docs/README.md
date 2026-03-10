# fouriax Documentation

This page is the GitHub-friendly entry point for browsing documentation in the repository.

The `docs/` tree is authored for Sphinx, so some files in this directory contain MyST and Sphinx directives that render best in the built docs site. For direct GitHub browsing, start from the plain Markdown and notebook links below.

## Core Docs

- [Project README](https://github.com/liamfitz27/fouriax/blob/main/README.md)
- [User Guide](https://github.com/liamfitz27/fouriax/blob/main/docs/USER_GUIDE.md)
- [Architecture](https://github.com/liamfitz27/fouriax/blob/main/docs/ARCHITECTURE.md)
- [Development Workflow](https://github.com/liamfitz27/fouriax/blob/main/docs/DEVELOPMENT_WORKFLOW.md)
- [TODO](https://github.com/liamfitz27/fouriax/blob/main/docs/TODO.md)

## API Reference Source

These pages define the Sphinx API structure, but GitHub will show the raw MyST content rather than the built reference site:

- [API Index](https://github.com/liamfitz27/fouriax/blob/main/docs/api/index.md)
- [Core Data Model](https://github.com/liamfitz27/fouriax/blob/main/docs/api/core.md)
- [Layers](https://github.com/liamfitz27/fouriax/blob/main/docs/api/layers.md)
- [Propagation and Planning](https://github.com/liamfitz27/fouriax/blob/main/docs/api/propagation.md)
- [Advanced Utilities](https://github.com/liamfitz27/fouriax/blob/main/docs/api/advanced.md)

## Examples

The runnable scripts are the source of truth. The notebooks are synced tutorial artifacts.

### Scripts

- [4f Correlator](https://github.com/liamfitz27/fouriax/blob/main/examples/scripts/4f_correlator.py)
- [4f Edge Optimization](https://github.com/liamfitz27/fouriax/blob/main/examples/scripts/4f_edge_optimization.py)
- [Coherent Hologram](https://github.com/liamfitz27/fouriax/blob/main/examples/scripts/hologram_coherent_logo.py)
- [Polarized Holography](https://github.com/liamfitz27/fouriax/blob/main/examples/scripts/holography_polarized_dual.py)
- [Incoherent Camera](https://github.com/liamfitz27/fouriax/blob/main/examples/scripts/incoherent_camera.py)
- [Lens Optimization](https://github.com/liamfitz27/fouriax/blob/main/examples/scripts/lens_optimization.py)
- [Meta-Atom Optimization](https://github.com/liamfitz27/fouriax/blob/main/examples/scripts/metaatom_optimization.py)
- [ONN MNIST](https://github.com/liamfitz27/fouriax/blob/main/examples/scripts/onn_mnist.py)
- [Sensitivity Analysis](https://github.com/liamfitz27/fouriax/blob/main/examples/scripts/sensitivity_analysis.py)
- [Spectral Filter Optimization](https://github.com/liamfitz27/fouriax/blob/main/examples/scripts/spectral_filter_optimization.py)

### Notebooks

- [4f Correlator](https://github.com/liamfitz27/fouriax/blob/main/examples/notebooks/4f_correlator.ipynb)
- [4f Edge Optimization](https://github.com/liamfitz27/fouriax/blob/main/examples/notebooks/4f_edge_optimization.ipynb)
- [Coherent Hologram](https://github.com/liamfitz27/fouriax/blob/main/examples/notebooks/hologram_coherent_logo.ipynb)
- [Polarized Holography](https://github.com/liamfitz27/fouriax/blob/main/examples/notebooks/holography_polarized_dual.ipynb)
- [Incoherent Camera](https://github.com/liamfitz27/fouriax/blob/main/examples/notebooks/incoherent_camera.ipynb)
- [Lens Optimization](https://github.com/liamfitz27/fouriax/blob/main/examples/notebooks/lens_optimization.ipynb)
- [Meta-Atom Optimization](https://github.com/liamfitz27/fouriax/blob/main/examples/notebooks/metaatom_optimization.ipynb)
- [ONN MNIST](https://github.com/liamfitz27/fouriax/blob/main/examples/notebooks/onn_mnist.ipynb)
- [Sensitivity Analysis](https://github.com/liamfitz27/fouriax/blob/main/examples/notebooks/sensitivity_analysis.ipynb)
- [Spectral Filter Optimization](https://github.com/liamfitz27/fouriax/blob/main/examples/notebooks/spectral_filter_optimization.ipynb)

## Building the Docs Locally

```bash
pip install -e ".[dev]"
PYTHONPATH=src .venv/bin/python -m sphinx -W -b html docs docs/_build/html
```
