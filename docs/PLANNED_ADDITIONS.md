# Planned Additions

## Scope

This document defines the next feature track after current NA/hybrid-domain stabilization:

1. Polarization support with Jones vectors and Jones-matrix layers.
2. Rectangular-pillar meta-atom library derived from existing square-pillar LUT data.
3. Holography examples for:
   - unpolarized target reconstruction
   - polarized dual-pattern reconstruction (different output per polarization channel)
4. Coherence-aware imaging mode using shift-invariant PSF/OTF models.

## Goals

- Keep the current scalar-path API working unchanged.
- Add polarization as an explicit opt-in mode.
- Preserve differentiability and JAX tracing safety.
- Keep planning/selection logic outside hot differentiable paths when possible.

## Phase 1: Polarization / Jones Support

### Data Model

- Extend `Field` to support polarized representation:
  - scalar mode: existing shape `(wavelength, ny, nx)`
  - Jones mode: complex shape `(wavelength, 2, ny, nx)` where axis 1 is `(Ex, Ey)`
- Add `polarization_mode: Literal["scalar", "jones"]` metadata.
- Keep existing scalar constructors; add Jones constructors:
  - `Field.plane_wave_jones(..., ex: complex | array, ey: complex | array)`

### Layer Contracts

- Scalar-only layers:
  - continue to work for scalar mode.
  - for Jones mode, either:
    - broadcast identically to both channels, or
    - raise explicit `ValueError` if not physically meaningful.
- New Jones layers:
  - `JonesMatrixLayer`: per-pixel 2x2 complex matrix multiplication.
  - `KJonesMatrixLayer`: same operation in k-space domain.
- Propagators:
  - propagate each Jones component identically for isotropic media in first release.
  - future extension: birefringent/an-isotropic propagation operators.

### Sensors and Readout

- `IntensitySensor` in Jones mode defaults to total intensity:
  - `|Ex|^2 + |Ey|^2`
- Add optional channel-resolved output:
  - per-channel intensity map for diagnostics/training losses.

### Tests (Phase 1)

- shape/validation tests for scalar and Jones modes.
- Jones matrix identity/regression tests.
- gradient tests through mixed scalar + Jones-compatible stacks.
- domain conversion tests (`to_kspace`, `to_spatial`) preserving polarization axes.

## Phase 2: Rectangular-Pillar Meta-Atom Library from Square LUT

### Source Data

- Reuse existing square-pillar LUT:
  - side length `s` -> complex transmission `t(s, wavelength)`.

### First Model (Diagonal Jones Approximation)

- Build rectangular parameterization `(sx, sy)` from square LUT lookup:
  - `Jxx = t(sx, wavelength)`
  - `Jyy = t(sy, wavelength)`
  - `Jxy = 0`, `Jyx = 0` (no cross-polarization in first approximation)
- This yields an effective Jones matrix:
  - `J = [[Jxx, 0], [0, Jyy]]`

### API Additions

- `RectangularMetaAtomLibrary.from_square_library(square_library)`
- `RectangularMetaAtomInterpolationLayer` with geometry params `(sx_map, sy_map)`.
- Backward-compatible scalar projection helper:
  - optional equivalent scalar transmission for legacy scripts.

### Tests (Phase 2)

- interpolation consistency:
  - `sx == sy` recovers square-library scalar behavior.
- bounded-parameter checks and gradients w.r.t. both `sx` and `sy`.
- Jones response tests for x/y-polarized inputs.

## Phase 3: Holography Examples

### Example A: Unpolarized Holography

- Script target: `scripts/holography_unpolarized_example.py`
- Objective:
  - optimize phase (and optional amplitude) to reconstruct one target pattern.
- Input model:
  - unpolarized approximation as average of orthogonal linear polarizations.
- Loss:
  - image MSE + optional regularizers (TV/smoothness/power normalization).

### Example B: Polarized Dual-Pattern Holography

- Script target: `scripts/holography_polarized_dual_example.py`
- Objective:
  - one device creates pattern A for x-polarized input and pattern B for y-polarized input.
- Uses Jones-capable field/layers and rectangular meta-atom layer.
- Loss:
  - weighted sum: `L = Lx(pattern A) + Ly(pattern B)` + regularizers.

### Example Outputs

- save artifacts:
  - optimized parameters
  - reconstructed images per polarization
  - training curves
  - optional polarization cross-talk metrics

## Phase 4: Coherence-Aware Imaging (Shift-Invariant PSF/OTF)

### Motivation

- Current stack is coherent field propagation with intensity readout.
- Add a planned imaging mode for incoherent or partially coherent approximations
  via shift-invariant convolution operators.

### Scope (First Release)

- In scope:
  - real-space imaging path with shift-invariant PSF convolution.
  - equivalent OTF implementation in Fourier domain for efficiency.
  - single-plane forward model for image formation tasks.
- Out of scope:
  - full mutual-coherence / cross-spectral-density transport.
  - strongly space-variant imaging systems.
  - polarization-coherence coupling.
  - dedicated k-space-domain layer stack for incoherent transport.

### API Direction

- New imaging operator/layer family:
  - `IncoherentImagingLayer(psf=..., normalize=True)` for direct convolution.
  - `OTFImagingLayer(otf=...)` for FFT-domain implementation.
- Optional source coherence presets:
  - `coherence_mode: Literal["coherent", "incoherent"]`
  - later extension: `"partial"` with parameterized kernels.
- Keep explicit contract:
  - this mode acts on intensity-domain imaging semantics, not coherent phase transport.

### k-Space Support Decision

- Decision: defer dedicated incoherent k-space pipeline in phase 4.
- Rationale:
  - OTF implementation already uses FFT internally for speed.
  - avoids mixing coherent k-space propagation semantics with incoherent image transfer in v1.
- Revisit later as phase 4b:
  - expose standalone k-space-oriented API only if a clear use case emerges.

### Tests (Phase 4)

- PSF/OTF numerical equivalence tests.
- delta-input response test (`input=impulse` reproduces PSF).
- non-negativity and energy/normalization checks.
- gradient tests through PSF/OTF parameters and upstream phase masks.
- regression tests vs analytic diffraction-limited incoherent benchmarks where available.

## Risks and Constraints

- Memory/runtime increase from extra polarization axis.
- Need strict shape validation to avoid silent channel misuse.
- Phase-2 diagonal Jones approximation ignores coupling terms (`Jxy`, `Jyx`).
- Must keep scalar-path performance and API ergonomics intact.
- Coherence mode naming must stay explicit to avoid confusion with coherent field propagation.
- PSF boundary conditions and padding choices can bias optimization if not standardized.

## Done Criteria

- Jones path is fully differentiable and covered by tests.
- Rectangular library behaves consistently and is documented in code/docstrings.
- Both holography examples run in `.venv` and generate reproducible artifacts.
- Coherence-aware imaging mode passes PSF/OTF parity tests and has clear in/out-of-scope docs.
