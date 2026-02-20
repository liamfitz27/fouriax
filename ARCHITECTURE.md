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

- `data`: complex array shaped `(num_wavelengths, ny, nx)`
- `grid`: `Grid`
- `spectrum`: `Spectrum`
- `domain`: `"spatial"` or `"kspace"`
- k-space pixel metadata

Operations:

- `intensity()`, `phase()`, `power()`, `normalize_power()`
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
- `ThinLens`

These require `field.domain == "spatial"`.

### K-Space Modulation Layers

- `KSpacePhaseMask`
- `KSpaceAmplitudeMask`
- `KSpaceComplexMask`

These require `field.domain == "kspace"`.

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

## Propagation Architecture

Implemented in `propagation.py`.

### Concrete Propagators

- `RSPropagator`
  - spatial-domain input
  - convolution with Rayleigh-Sommerfeld delta response
  - optional NA mask
  - optional sampling planner / precomputed grid

- `ASMPropagator`
  - spatial-domain input
  - explicitly composes:
    - `FourierTransform`
    - `KSpacePropagator(include_evanescent=True)`
    - `InverseFourierTransform`
  - optional sampling planner / precomputed grid
  - optional NA mask

- `KSpacePropagator`
  - k-space input
  - diagonal transfer-function multiplication
  - `include_evanescent` controls whether evanescent terms are retained

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

- `FieldReadout`
  - returns `"complex"`, `"real_imag"`, or `"amplitude_phase"` representations

Current sensor behavior converts inputs to spatial domain internally (`field.to_spatial()`).

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
