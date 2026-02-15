# fouriax Developer Status

## Purpose

Build a JAX-native, differentiable free-space optics library with reliable numerical primitives, testing, and reproducible development workflows.

## High-Level Plan

1. Foundation and scaffolding

- Repository structure and package boundaries
- Tooling: tests, linting, docs, CI
- Version-control conventions on GitHub

2. Core numerical layer

- FFT-based convolution and related low-level tensor operations
- Validation against reference libraries
- Early gradient and dtype correctness checks

3. Optics pipeline baseline

- Field representation
- Core propagation primitives
- End-to-end example pipeline for a minimal FSO simulation

4. Stabilization and hardening

- API cleanup and consistency
- Performance baselines
- Expanded examples and docs

## Current Progress (as of 2026-02-15)

### Completed

- Initialized repository and `main` branch.
- Added short project `README.md`.
- Created feature branch `feat/core-fft-convolution-tests`.
- Added package scaffold:
  - `src/fouriax/__init__.py`
  - `src/fouriax/core/__init__.py`
  - `src/fouriax/core/fft.py`
- Ported `fftconvolve` from previous project reference.
- Added tests comparing against `scipy.signal.fftconvolve` for:
  - real-valued full mode
  - real-valued same mode
  - complex-valued convolution
  - selected-axes convolution
  - `sum_fft_axis` channel-sum optimization behavior
  - invalid mode error path
- Verified current test suite passes locally (`6 passed`).
- Added `pyproject.toml` with:
  - core package metadata
  - `test` and `dev` dependency groups
  - minimal `pytest`, `ruff`, and `mypy` configuration
- Added CI workflow at `.github/workflows/ci.yml` to run lint, type checks, and tests on PRs/pushes to `main`.
- Added educational workflow documentation at `docs/DEVELOPMENT_WORKFLOW.md`.
- Added core optics data model and interfaces:
  - `Grid`, `Spectrum`, `Field`
  - `OpticalLayer`, `Sensor`, `PropagationModel`
- Added propagators and optical layer baseline:
  - `RSPropagator`
  - `ASMPropagator`
  - `ThinLensLayer`
- Added planner-driven propagation resampling:
  - automatic propagation-grid planning from spectrum/grid settings
  - interpolation-based resampling to planned grid before propagation
  - resampling back to original output grid after propagation
- Added method-selection policy and auto-dispatch:
  - `PropagationPolicy` with `fast` / `balanced` / `accurate` modes
  - `AutoPropagator` that dispatches to ASM/RS from policy decisions
  - balanced-mode cost and tolerance overrides
- Added optics/regime validation tests:
  - lens-focus benchmark against Airy profile
  - policy selection tests (critical-distance rule, mode behavior, risk override)
  - near/far regime test ensuring selected method has lower Airy error
- Added analysis scripts:
  - `scripts/propagator_regime_comparison_plot.py`
  - `scripts/propagator_sweep_report.py`

### In Progress

- None.

### Next Up

- Open PR for `feat/sampling-planner-policy` and merge after CI passes.
- Add stricter performance benchmarking methodology:
  - separate compile time vs steady-state runtime
  - log planned propagation grid (`nx`, `ny`, `dx`, `dy`) per method case
- Add first concrete ASM/RS production policy calibration table from sweep outputs.

## Open TODO Backlog

- Add advanced `fftconvolve` tests:
  - broadcasting behavior across non-convolution dimensions
  - gradient checks (`jax.grad`) against finite differences
  - dtype coverage (`float32`, `float64`, `complex64`, `complex128`)
  - explicit `fft_shape` override behavior
  - JIT static-argument behavior and recompilation expectations
  - edge-shape coverage (odd/even kernels, singleton axes, higher dimensions)
- Add benchmark harness for FFT conv sizes relevant to optics use cases.
- Add deterministic numeric tolerance policy by dtype/backend.
- Add propagation planner tests for interpolation fidelity and energy drift bounds.
- Add optional band-limited ASM variant for long-distance stability.

## Version-Control Workflow

- Mainline branch: `main` (always releasable).
- Short-lived branches: `feat/*`, `fix/*`, `docs/*`, `chore/*`.
- Conventional commit messages.
- PR-only merge policy for GitHub once remote protections are enabled.
