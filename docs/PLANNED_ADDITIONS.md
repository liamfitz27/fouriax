# Planned Additions

## Scope

This document defines the next feature track after current domain-explicit propagation and NA-planning extraction:

1. Rectangular-pillar meta-atom workflow derived from existing square-pillar LUT data.
2. Dual-pattern holography example using the polarized pipeline.
3. Arbitrary per-pixel transmission filter support.
4. Sensor quantum-efficiency (QE) curve integration.

## Notes on Already Implemented Items

- Coherent single-target holography is already implemented in:
  - `scripts/hologram_coherent_logo_example.py`
- Incoherent shift-invariant imaging is already implemented via:
  - `IncoherentImager`
  - `scripts/incoherent_camera_example.py`

## Goals

- Keep the scalar-path API working unchanged.
- Add new sensing/filter options as explicit opt-in paths.
- Preserve differentiability and JAX tracing safety.
- Keep planner/selection logic outside hot differentiable paths when possible.

## Phase 1: Rectangular-Pillar Meta-Atom Workflow from Square LUT

### Source Data

- Reuse existing square-pillar LUT:
  - side length `s` -> complex transmission `t(s, wavelength)`.

### First Model (Diagonal Channel Approximation)

- Build rectangular parameterization `(sx, sy)` from square LUT lookup:
  - `Jxx = t(sx, wavelength)`
  - `Jyy = t(sy, wavelength)`
  - `Jxy = 0`, `Jyx = 0` (no cross-polarization in first approximation)
- This yields an effective diagonal per-channel response using independent LUT lookups.

### API Direction

- Prefer the existing generalized `MetaAtomLibrary` + `MetaAtomInterpolationLayer`.
- Reuse the square-pillar LUT and supply two parameter maps/sets for independent channel responses.
- Add helper loaders/utilities only if they reduce duplication, but avoid rectangular-specific core types
  unless simulation data demands them.

### Tests (Phase 1)

- interpolation consistency:
  - `sx == sy` recovers square-library scalar behavior.
- bounded-parameter checks and gradients w.r.t. both `sx` and `sy`.
- channel-separated response tests for x/y inputs.

## Phase 2: Dual-Pattern Holography

### Example

- Script target: `scripts/holography_polarized_dual_example.py`
- Objective:
  - one device creates pattern A for x-polarized input and pattern B for y-polarized input.
- Current placeholder uses independent phase maps per channel.
- Future variant will use meta-atom-constrained optimization.
- Loss:
  - weighted sum: `L = Lx(pattern A) + Ly(pattern B)` + regularizers.

### Outputs

- Save artifacts:
  - optimized parameters
  - reconstructed images per channel
  - training curves
  - cross-talk metrics

## Phase 3: Arbitrary Per-Pixel Transmission Filters

### Motivation

- Current modulation layers support amplitude and/or phase maps, but device/filter workflows often need direct complex transmission maps from measured/calibrated data.

### Scope

- Add explicit transmission-filter support for spatial and k-space paths:
  - spatial: `TransmissionFilterLayer(transmission_map=...)`
  - k-space: `KTransmissionFilterLayer(transmission_map=...)`
- Support scalar and wavelength-dependent maps:
  - `(ny, nx)` or `(wavelength, ny, nx)`
- Optionally support channel-aware map shapes in future extension.

### Tests

- shape validation and wavelength broadcasting tests.
- equivalence tests vs `ComplexMask` for matching inputs.
- gradient tests through transmission maps.

## Phase 4: QE Curve Integration

### Motivation

- Sensor output currently integrates intensity without wavelength-dependent detector efficiency.

### Scope

- Extend `IntensitySensor` with optional spectral QE weighting:
  - e.g. `qe_curve` sampled over wavelengths or callable response.
- Define deterministic interpolation behavior against `Spectrum.wavelengths_um`.
- Apply QE weights before wavelength summation and detector-mask integration.

### Tests

- scalar/spectral consistency tests.
- interpolation tests for off-grid wavelengths.
- regression tests showing equivalence to current behavior when QE is uniform (`qe=1`).

## Risks and Constraints

- Need strict shape validation to avoid silent channel misuse.
- Diagonal channel approximation ignores cross-coupling terms.
- Must keep scalar-path performance and API ergonomics intact.
- Transmission maps and QE curves need clear precedence rules with existing sensor/layer options.

## Done Criteria

- Rectangular library behaves consistently and is documented in code/docstrings.
- Dual-channel holography example runs in `.venv` and generates reproducible artifacts.
- Arbitrary transmission filters are available with parity/gradient tests.
- QE-aware sensor integration is available with spectral weighting tests.
