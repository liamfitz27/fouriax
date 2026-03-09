# fouriax

Differentiable free-space optics for JAX.

**fouriax** is a JAX library for simulating and optimizing coherent and incoherent optical systems with automatic differentiation. It provides composable layers for wave propagation, phase/amplitude modulation, polarization (Jones calculus), meta-atom lookup tables, and sensor readout — all fully traceable through JAX for gradient-based inverse design.

## Features

- **Composable optical stack** — build systems from `OpticalLayer` transforms composed in an `OpticalModule`, with explicit spatial ↔ k-space domain transitions via `FourierTransform` / `InverseFourierTransform`.
- **Propagation** — Angular Spectrum Method (`ASMPropagator`), Rayleigh–Sommerfeld (`RSPropagator`), and k-space transfer function (`KSpacePropagator`). Automatic method selection via `plan_propagation()`.
- **Modulation layers** — `PhaseMask`, `AmplitudeMask`, `ComplexMask`, `ThinLens`, plus k-space equivalents (`KSpacePhaseMask`, `KSpaceAmplitudeMask`, `KSpaceComplexMask`).
- **Jones polarization** — full 2×2 Jones matrix layers (`JonesMatrixLayer`, `KJonesMatrixLayer`) and per-channel field diagnostics.
- **Meta-atom libraries** — wavelength- and geometry-parameterized lookup tables with multilinear interpolation (`MetaAtomLibrary`, `MetaAtomInterpolationLayer`).
- **Incoherent imaging** — shift-invariant PSF/OTF filtering via `IncoherentImager`.
- **Detectors** — `Detector` for region-integrated intensity readout and `DetectorArray` for grid readout with optional noise.
- **NA planning** — numerical-aperture schedule utilities for multi-segment systems.
- **End-to-end differentiable** — every layer traces through JAX for gradient-based optimization with Optax.

## Installation

```bash
git clone <repo-url>
cd fouriax
pip install -e ".[dev]"
```

Requires Python ≥ 3.12. Core dependencies: JAX, Flax, Optax, NumPy, Matplotlib.

## Quickstart

```python
import jax
import fouriax as fx

# Define grid, spectrum, and input field
grid = fx.Grid.from_extent(nx=64, ny=64, dx_um=1.0, dy_um=1.0)
spectrum = fx.Spectrum.from_scalar(0.532)  # 532 nm
field = fx.Field.plane_wave(grid=grid, spectrum=spectrum)

# Build an optical system: random phase mask → propagate 1 mm
key = jax.random.PRNGKey(0)
phase = jax.random.uniform(key, (1, 64, 64)) * 2 * 3.14159
propagator = fx.plan_propagation(
    mode="auto", grid=grid, spectrum=spectrum, distance_um=1000.0,
)
module = fx.OpticalModule(
    layers=(fx.PhaseMask(phase_map_rad=phase), propagator)
)

# Forward pass — fully differentiable
output = module.forward(field)
intensity = output.intensity()  # unbatched: (num_wavelengths, ny, nx)
```

## Examples

Scripts live in `examples/scripts/` with matching notebooks in `examples/notebooks/`.

| Example | Script | Notebook | Description |
|---|---|---|---|
| Lens Optimization | `lens_optimization.py` | `lens_optimization.ipynb` | Learn a phase mask to focus a plane wave |
| 4f Correlator | `4f_correlator.py` | `4f_correlator.ipynb` | Matched-filter cross-correlation in a 4f system |
| 4f Edge Optimization | `4f_edge_optimization.py` | `4f_edge_optimization.ipynb` | Optimize a spiral phase filter for edge detection |
| Hybrid Super-Resolution | `hybrid_super_resolution.py` | `hybrid_super_resolution.ipynb` | Co-design a coded optical front-end and CNN decoder for single-shot super-resolution |
| Coherent Hologram | `hologram_coherent_logo.py` | `hologram_coherent_logo.ipynb` | Phase-only hologram via GS-style optimization |
| Polarized Holography | `holography_polarized_dual.py` | `holography_polarized_dual.ipynb` | Dual-pattern holography with Jones polarization |
| Incoherent Camera | `incoherent_camera.py` | `incoherent_camera.ipynb` | Shift-invariant incoherent imaging simulation |
| Meta-Atom Optimization | `metaatom_optimization.py` | `metaatom_optimization.ipynb` | Optimize meta-atom geometries for focusing |
| ONN MNIST | `onn_mnist.py` | `onn_mnist.ipynb` | Optical neural network for MNIST classification |
| Spectral Filter | `spectral_filter_optimization.py` | `spectral_filter_optimization.ipynb` | Multi-wavelength spectral filter design |

## Project Structure

```
src/fouriax/
  fft.py           # FFT convolution utilities
  optics/          # Core: Field, Grid, layers, propagation, sensors, meta-atoms, NA planning
  optim/           # Optimization loops, batching, and loss helpers for examples
  analysis/        # Fisher information, sensitivity analysis, design optimality
examples/
  scripts/         # Runnable Python scripts
  notebooks/       # Jupyter notebooks with explanations
tests/             # pytest suite (propagation, layers, polarization, meta-atoms, …)
docs/              # Architecture, dev workflow, planned additions
```

## Development

```bash
scripts/dev_setup.sh      # create .venv, install deps, set up pre-commit hooks
scripts/tests_local.sh    # run ruff + mypy + pytest
```

See `docs/DEVELOPMENT_WORKFLOW.md` for the full git flow and CI expectations.

## Architecture

See `docs/ARCHITECTURE.md` for the complete design document covering the `Field` data model, layer system, propagation architecture, sensors, meta-atom layers, and execution patterns.
