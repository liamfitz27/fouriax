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

## Current Progress (as of 2026-02-11)

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

### In Progress

- None.

### Next Up

- Establish minimal packaging/dev tooling (`pyproject.toml`, test/dependency groups, pre-commit, CI workflow).
- Start optics-focused core primitives after numerical baseline is stable.

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

## Version-Control Workflow

- Mainline branch: `main` (always releasable).
- Short-lived branches: `feat/*`, `fix/*`, `docs/*`, `chore/*`.
- Conventional commit messages.
- PR-only merge policy for GitHub once remote protections are enabled.
