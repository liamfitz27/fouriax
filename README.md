# fouriax

Differentiable free-space optics for JAX.

**fouriax** is a JAX library for simulating and optimizing coherent and incoherent optical systems with automatic differentiation. It provides composable layers for wave propagation, phase/amplitude modulation, polarization (Jones calculus), meta-atom lookup tables, and sensor readout — all fully traceable through JAX for gradient-based inverse design.

Documentation: <https://liamfitz27.github.io/fouriax/>

> [!WARNING]
> This project is almost entirely vibecoded. Expect rough edges, inconsistent design decisions, incomplete validation, and APIs that may change substantially.

## Features

- **Composable optical stack** — build coherent systems from `OpticalLayer` transforms composed in an `OpticalModule`, with explicit spatial ↔ k-space domain transitions via `FourierTransform` / `InverseFourierTransform`.
- **Propagation** — Angular Spectrum Method (`ASMPropagator`), Rayleigh–Sommerfeld (`RSPropagator`), and k-space transfer function (`KSpacePropagator`). `plan_propagation()` is the recommended high-level entry point: it selects the propagator, plans the working grid, and precomputes reusable diagonal transfer data for ASM and k-space propagation.
- **Modulation layers** — `PhaseMask`, `AmplitudeMask`, `ComplexMask`, `ThinLens`, plus k-space equivalents (`KSpacePhaseMask`, `KSpaceAmplitudeMask`, `KSpaceComplexMask`).
- **Jones polarization** — full 2×2 Jones matrix layers (`JonesMatrixLayer`, `KJonesMatrixLayer`) and per-channel field diagnostics.
- **Meta-atom libraries** — wavelength- and geometry-parameterized lookup tables with multilinear interpolation (`MetaAtomLibrary`, `MetaAtomInterpolationLayer`).
- **Incoherent imaging** — explicit `Intensity` data model and shift-invariant PSF/OTF filtering via `IncoherentImager`.
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

For most workflows, start with `plan_propagation(...)` rather than constructing
`ASMPropagator` or `RSPropagator` directly. It is the intended entry point when
the simulation setup is fixed across repeated forward passes, especially in
optimization loops.

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

Why `plan_propagation(...)`:

- It chooses between ASM and RS from the sampling regime when `mode="auto"`.
- It applies the propagation-grid planning needed for stable sampled propagation.
- For ASM and k-space propagation, it precomputes the diagonal transfer stack once and reuses it across repeated calls.

So even when you know you want ASM, prefer:

```python
propagator = fx.plan_propagation(
    mode="asm",
    grid=grid,
    spectrum=spectrum,
    distance_um=1000.0,
)
```

and reach for `ASMPropagator(...)` or `RSPropagator(...)` directly only when
you specifically need manual low-level control.

## Examples

Most tutorial source scripts live in `examples/scripts/` with matching notebooks in
`examples/notebooks/`. The spectral filter optimization source lives in
`experiments/spectral_filter_optimization.py` and syncs to the same notebook set.

Recommended example order:

```text
basic_propagation

lens_optimization

4f_correlator
or
hologram_coherent_logo
or
incoherent_camera

then the family extensions and more specialized examples
```

Example families:

```text
lens_optimization
├── sensitivity_analysis
└── metaatom_optimization

4f_correlator
└── 4f_edge_optimization

hologram_coherent_logo
└── holography_polarized_dual

incoherent_camera
spectral_filter_optimization
onn_mnist
```

Suggested progression:

- Start with `basic_propagation` to learn the core objects: `Grid`, `Spectrum`, `Field`, `OpticalModule`, `Detector`, and `plan_propagation()`.
- Move to `lens_optimization` for the main differentiable-optics workflow used across the library.
- Then choose a direction: `4f_correlator` for Fourier optics, `hologram_coherent_logo` for phase-only design, or `incoherent_camera` for incoherent imaging.
- After that, use the child notebooks and more specialized examples as extensions.

| Example | Script | Notebook | Description |
|---|---|---|---|
| Basic Propagation | [`basic_propagation.py`](examples/scripts/basic_propagation.py) | [`basic_propagation.ipynb`](examples/notebooks/basic_propagation.ipynb) | Minimal optical stack using `Grid`, `Field`, `ThinLens`, and `plan_propagation()` |
| Lens Optimization | [`lens_optimization.py`](examples/scripts/lens_optimization.py) | [`lens_optimization.ipynb`](examples/notebooks/lens_optimization.ipynb) | Learn a phase mask to focus a plane wave |
| 4f Correlator | [`4f_correlator.py`](examples/scripts/4f_correlator.py) | [`4f_correlator.ipynb`](examples/notebooks/4f_correlator.ipynb) | Matched-filter cross-correlation in a 4f system |
| 4f Edge Optimization | [`4f_edge_optimization.py`](examples/scripts/4f_edge_optimization.py) | [`4f_edge_optimization.ipynb`](examples/notebooks/4f_edge_optimization.ipynb) | Optimize a spiral phase filter for edge detection |
| Coherent Hologram | [`hologram_coherent_logo.py`](examples/scripts/hologram_coherent_logo.py) | [`hologram_coherent_logo.ipynb`](examples/notebooks/hologram_coherent_logo.ipynb) | Phase-only hologram via GS-style optimization |
| Polarized Holography | [`holography_polarized_dual.py`](examples/scripts/holography_polarized_dual.py) | [`holography_polarized_dual.ipynb`](examples/notebooks/holography_polarized_dual.ipynb) | Dual-pattern holography with Jones polarization |
| Incoherent Camera | [`incoherent_camera.py`](examples/scripts/incoherent_camera.py) | [`incoherent_camera.ipynb`](examples/notebooks/incoherent_camera.ipynb) | Shift-invariant incoherent imaging simulation |
| Meta-Atom Optimization | [`metaatom_optimization.py`](examples/scripts/metaatom_optimization.py) | [`metaatom_optimization.ipynb`](examples/notebooks/metaatom_optimization.ipynb) | Optimize meta-atom geometries for focusing |
| ONN MNIST | [`onn_mnist.py`](examples/scripts/onn_mnist.py) | [`onn_mnist.ipynb`](examples/notebooks/onn_mnist.ipynb) | Optical neural network for MNIST classification |
| Spectral Filter | [`spectral_filter_optimization.py`](experiments/spectral_filter_optimization.py) | [`spectral_filter_optimization.ipynb`](examples/notebooks/spectral_filter_optimization.ipynb) | Multi-wavelength spectral filter design |

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
pip install -e ".[dev,docs]"
python -m sphinx -b html docs docs/_build/html
```

See `docs/DEVELOPMENT_WORKFLOW.md` for the full git flow and CI expectations.

## Architecture

See `docs/ARCHITECTURE.md` for the complete design document covering the `Field` data model, layer system, propagation architecture, sensors, meta-atom layers, and execution patterns.
