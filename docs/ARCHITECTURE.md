# Fouriax Architecture

This document describes the current implementation of `fouriax` in `src/fouriax/optics`.

## Scope

The optics package is organized around:

- `Field` state representation (`model.py`)
- composable `OpticalLayer` transforms (`interfaces.py`, `layers.py`, `propagation.py`)
- readout `Sensor`s (`sensors.py`)
- objective and matrix-analysis utilities (`losses.py`)

All physical transforms are implemented as explicit layers. Domain transitions between spatial and k-space are explicit in layer stacks.

## Core Data Model

### `Grid`

`Grid` defines a 2D Cartesian sampling lattice in micrometers:

- `nx`, `ny`
- `dx_um`, `dy_um`

Key methods:

- `spatial_grid()`: centered `(x, y)` grids
- `frequency_grid()`: `(fx, fy)` grids in cycles/um
- `kspace_pixel_size_cyc_per_um()`

### `Spectrum`

`Spectrum` stores wavelengths as a 1D array in micrometers:

- `wavelengths_um`
- `size`

### `Field`

`Field` is the tensor passed between layers:

- `data`:
  - scalar mode: complex array `(num_wavelengths, ny, nx)`
  - Jones mode: complex array `(num_wavelengths, 2, ny, nx)` with channels `(Ex, Ey)`
- `grid`: `Grid`
- `spectrum`: `Spectrum`
- `polarization_mode`: `"scalar"` or `"jones"`
- `domain`: `"spatial"` or `"kspace"`
- k-space pixel metadata

Operations:

- `intensity()`, `phase()`, `power()`, `normalize_power()`
- `component_intensity()` for channel-resolved Jones diagnostics
- `apply_phase()`, `apply_amplitude()`
- `to_kspace()`, `to_spatial()` for explicit FFT/IFFT conversion

## Interfaces

### `OpticalLayer`

Abstract base class for field-to-field transforms:

- `forward(field) -> Field`
- `validate_for(field)`
- `parameters() -> dict[str, jnp.ndarray]`

### `Sensor`

Abstract base class for measurement readout:

- `measure(field) -> jnp.ndarray`
- `validate_for(field)`

## Layer System

### `OpticalModule`

`OpticalModule` composes an ordered `tuple[OpticalLayer, ...]` with optional `sensor`.

Execution methods:

- `forward(field) -> Field`
- `measure(field) -> jnp.ndarray`
- `trace(field, include_input=True) -> list[Field]`
- `parameters()`

`OpticalModule` is a thin sequential executor. It does not perform NA planning, domain inference, or propagation-method resolution.

### NA Planning (`na_planning.py`)

Numerical aperture (NA) geometry logic is implemented in `optics/na_planning.py` as external planning utilities:

- `propagation_layer_view(...)`
- `collect_stops(...)`
- `collect_propagation_segments(...)`
- `local_na_for_segment(...)`
- `effective_na(...)`
- `na_schedule(...)`
- `layer_with_na_if_supported(...)`
- `apply_na_limits(...)`

Typical use:

1. Build a schedule with `na_schedule(layers, field, ...)`.
2. Produce modified layers via `apply_na_limits(layers, schedule)`.
3. Construct `OpticalModule(layers=...)` with the resulting stack.

### Domain-Explicit Transform Layers

`layers.py` includes explicit domain conversion layers:

- `FourierTransform`: spatial -> k-space
- `InverseFourierTransform`: k-space -> spatial

Domain-specific layers validate input domain and raise if incorrect.

### Spatial-Domain Modulation Layers

- `PhaseMask`
- `AmplitudeMask`
- `ComplexMask`
- `JonesMatrixLayer`
- `ThinLens`

These require `field.domain == "spatial"`.

`PhaseMask`/`AmplitudeMask`/`ComplexMask`/`ThinLens` broadcast over Jones channels in Jones mode.
`JonesMatrixLayer` is the polarization-mixing (full 2x2) spatial-domain layer.

### K-Space Modulation Layers

- `KSpacePhaseMask`
- `KSpaceAmplitudeMask`
- `KSpaceComplexMask`
- `KJonesMatrixLayer`

These require `field.domain == "kspace"`.

`KJonesMatrixLayer` is the k-space analog of `JonesMatrixLayer`.

### `IncoherentImager`

`IncoherentImager` models shift-invariant incoherent imaging by:

1. generating a coherent calibration source (`impulse` or `plane_wave_focus`)
2. propagating through `optical_layer` and `propagator`
3. constructing PSF from output intensity
4. applying PSF/OTF filtering to input intensity

Modes:

- `mode="psf"`, `mode="otf"`, or `mode="auto"`

Normalization controls:

- `normalization_reference`: `"near_1um"`, `"near_wavelength"`, `"at_imaging_distance"`
- optional explicit `normalization_reference_distance_um`

Current limitation:

- `IncoherentImager` currently supports scalar fields only (Jones mode rejected explicitly).

## Propagation Architecture

Implemented in `propagation.py`.

### Concrete Propagators

- `RSPropagator`
  - spatial-domain input
  - convolution with Rayleigh-Sommerfeld delta response
  - optional NA mask
  - optional sampling planner / precomputed grid
  - Jones mode support via independent propagation of `Ex`/`Ey` in isotropic media (v1)

- `ASMPropagator`
  - spatial-domain input
  - explicitly composes:
    - `FourierTransform`
    - `KSpacePropagator(include_evanescent=True)`
    - `InverseFourierTransform`
  - optional sampling planner / precomputed grid
  - optional NA mask
  - Jones mode support via FFT/k-space propagation of both channels

- `KSpacePropagator`
  - k-space input
  - diagonal transfer-function multiplication
  - `include_evanescent` controls whether evanescent terms are retained
  - Jones mode support via identical per-channel transfer multiplication

### Propagator Planning

`plan_propagation(...)` is the propagation selector function.

Inputs include:

- `mode`: `"auto" | "asm" | "rs" | "kspace"`
- `grid`, `spectrum`, `distance_um`
- `input_domain`
- planner and NA parameters

Output:

- one concrete layer instance: `ASMPropagator | RSPropagator | KSpacePropagator`

In `mode="auto"`:

- k-space input selects `KSpacePropagator`
- spatial input selects ASM or RS using `select_propagator_method(...)`

Supporting utilities:

- `critical_distance_um(...)`
- `select_propagator_method(...)`
- `recommend_nyquist_grid(...)`

## Sensors

Implemented in `sensors.py`.

- `IntensitySensor`
  - computes intensity, optional wavelength sum
  - optional detector integration via `detector_masks`
  - optional `channel_resolved=True` for Jones per-channel intensity output

- `FieldReadout`
  - returns `"complex"`, `"real_imag"`, or `"amplitude_phase"` representations
  - preserves Jones channel axis when present

Current sensor behavior converts inputs to spatial domain internally (`field.to_spatial()`).

## Meta-Atom Layers

Implemented in `meta_atoms.py`.

- `MetaAtomLibrary`
  - regular-grid LUT over wavelength + arbitrary geometry parameter axes
  - multilinear interpolation over geometry and 1D interpolation over wavelength

- `MetaAtomInterpolationLayer`
  - general parameterized modulation layer using `MetaAtomLibrary`
  - `polarization_mode="scalar"` applies one complex transmission to scalar/Jones fields
  - `polarization_mode="jones_diagonal"` applies independent LUT lookups to `Ex` and `Ey`
    (diagonal Jones approximation, no cross-polarization coupling)

This keeps meta-atom support as a specialized layer built on top of the generic `Field` /
`OpticalLayer` abstractions.

## FFT Utilities

Implemented in `src/fouriax/core/fft.py`.

- `fftconvolve(...)`: generic FFT convolution
- `fftconvolve_same_with_otf(...)`: same-mode linear convolution using a precomputed OTF
  (used by `IncoherentImager` OTF path)

## Losses and Matrix Utilities

Implemented in `losses.py`:

- `focal_spot_loss`
- DCT synthesis matrix helpers
- sensing matrix construction
- randomized SVD
- coherence objectives
- mutual-information objective

## Execution Pattern

Typical coherent stack with explicit domain transitions:

1. spatial layers (e.g., lens/masks)
2. `FourierTransform` when entering k-space operations
3. k-space layers (e.g., k-space masks/propagator)
4. `InverseFourierTransform` when returning to spatial domain
5. optional sensor readout

Propagation selection is explicit at construction/planning time through `plan_propagation(...)`, producing a concrete `OpticalLayer` used directly in `OpticalModule.layers`.
