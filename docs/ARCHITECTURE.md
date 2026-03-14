# Fouriax Architecture

This document describes the current implementation of `fouriax` in `src/fouriax/optics`.

## Scope

The optics package is organized around:

- `Field` state representation (`model.py`)
- composable `OpticalLayer` transforms (`interfaces.py`, `layers.py`, `propagation.py`)
- readout `Sensor`s (`sensors.py`)
- optics-domain field/layer/sensor primitives (with optimization utilities in
  `src/fouriax/optim`)

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
  - scalar mode: complex array `(*batch, num_wavelengths, ny, nx)`
  - Jones mode: complex array `(*batch, num_wavelengths, 2, ny, nx)` with channels `(Ex, Ey)`
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

### `Intensity`

`Intensity` is the tensor passed between incoherent layers and sensors:

- `data`: real array `(*batch, num_wavelengths, ny, nx)`
- `grid`: `Grid`
- `spectrum`: `Spectrum`

Operations:

- `power()`
- `sum_wavelengths()`

## Interfaces

### `OpticalLayer`

Abstract base class for field-to-field transforms:

- `forward(field) -> Field`
- `validate_for(field)`
- `parameters() -> dict[str, jnp.ndarray]`

### `IncoherentLayer`

Abstract base class for intensity-to-intensity transforms:

- `forward(intensity) -> Intensity`
- `validate_for(intensity)`
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

`OpticalModule.forward(...)` remains `Field -> Field`. The optional sensor is an attached terminal readout used by `measure(...)`; it is not part of the coherent layer composition itself.

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

1. generating a coherent calibration source (`impulse`, `plane_wave_focus`, or `point_source`)
2. propagating through `optical_layer` and `propagator`
3. constructing PSF from output intensity
4. applying PSF/OTF filtering to input intensity

It implements `IncoherentLayer`, not `OpticalLayer`:

- `build_psf(field_template: Field) -> Intensity`
- `forward(intensity: Intensity) -> Intensity`
- `for_far_field(...)` for plane-wave calibration at the sensor/image distance
- `for_finite_distance(...)` for point-source calibration with explicit object and image distances
- `infer_from_paraxial_limit(sensor_grid, paraxial_max_angle_rad) -> Grid`
  for same-shape input-grid planning from a detector grid

`infer_from_paraxial_limit(...)` is a planning helper. It does not change
`forward(...)` semantics. For finite-distance imagers it returns an object-plane
grid scaled by magnification; for far-field imagers it returns an image-
equivalent input grid with the same pitch as the sensor grid. Pair the inferred
input grid with `DetectorArray(detector_grid=...)` when you want distinct input
and detector coordinate metadata in a same-shape simulation.

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

- `Detector`
  - integrates intensity over one masked region
  - accepts either `Field` or `Intensity`
  - preserves Jones per-channel output only when given a Jones `Field` with `channel_resolved=True`

- `DetectorArray`
  - integrates onto a detector grid with optional QE weighting, filter mask, and noise
  - accepts either `Field` or `Intensity`
  - `resample_method="linear"` uses conservative pixel-overlap redistribution
    when source and detector grids differ
  - `resample_method="nearest"` keeps nearest-bin accumulation semantics

Current detector behavior converts `Field` inputs via `Field.to_intensity()` and consumes `Intensity` inputs directly.

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

Implemented in `src/fouriax/fft.py`.

- `fftconvolve(...)`: generic FFT convolution
- `fftconvolve_same_with_otf(...)`: same-mode linear convolution using a precomputed OTF
  (used by `IncoherentImager` OTF path)

## Loss Utilities

Implemented in `src/fouriax/optim/losses.py`:

- `focal_spot_loss`

`focal_spot_loss` is an optimization helper kept in the `optim` package alongside other
training utilities. It is not part of the core optics API.

## Optimization Utilities

Implemented in `src/fouriax/optim`.

These utilities are example-facing training helpers layered on top of JAX/Optax. They are
intentionally outside the core optics API (`src/fouriax/optics`) and exist to reduce repeated
optimization loop code in scripts/notebooks.

### `optim/data.py`

Array-based batching and split helpers used by example training loops:

- `batch_slices(...)`
- `iter_minibatches(...)`
- `random_batch_indices(...)`
- `shuffled_arrays(...)`
- `train_val_split(...)`

### `optim/optim.py`

Contains both low-level and high-level optimization wrappers:

- `optimize_optical_module(...)`
  - single-loss, non-dataset Optax loop for optimizing one optical parameter pytree
  - returns `ModuleOptResult`

- `optimize_dataset_params(...)`
  - generic minibatch optimizer engine with optional validation tracking
  - used as the internal engine for higher-level wrappers
  - supports arbitrary param pytrees and custom batch/eval functions

- `optimize_dataset_optical_module(...)`
  - simplified dataset wrapper for optics-only training
  - caller provides:
    - `build_module(params) -> OpticalModule`
    - one shared `loss_fn(params, batch)`
    - `train_data`, optional `val_data`, `batch_size`, `epochs`
  - internal behavior:
    - derives minibatching via `iter_minibatches(...)`
    - computes validation as mean `val_loss` using the same `loss_fn`
    - uses built-in uniform console reporting
  - returns `ModuleDatasetOptResult`

- `optimize_dataset_hybrid_module(...)`
  - simplified dataset wrapper for joint optimization of optical parameters plus a decoder/NN
  - separates parameters into `{"optical": ..., "decoder": ...}`
  - supports either:
    - a single combined `optimizer`, or
    - separate `optical_optimizer` and `decoder_optimizer` (internally combined with
      `optax.multi_transform(...)`)
  - uses the same minibatching/validation/reporting conventions as
    `optimize_dataset_optical_module(...)`
  - returns `HybridModuleDatasetOptResult`

Default dataset-wrapper reporting cadence (both simplified wrappers):

- `val_every_epochs=1`
- `val_every_steps=0`
- `log_every_steps=0`

Practical note for JIT-compiled optimization:

- any propagation/planning decisions (for example, ASM vs RS selection) should be resolved at
  module-construction time before entering the JIT-traced loss, so the traced path uses a fixed
  concrete propagator layer.

## Execution Pattern

Typical coherent stack with explicit domain transitions:

1. spatial layers (e.g., lens/masks)
2. `FourierTransform` when entering k-space operations
3. k-space layers (e.g., k-space masks/propagator)
4. `InverseFourierTransform` when returning to spatial domain
5. optional sensor readout

Propagation selection is explicit at construction/planning time through `plan_propagation(...)`, producing a concrete `OpticalLayer` used directly in `OpticalModule.layers`.

## Analysis Utilities

Implemented in `src/fouriax/analysis`.

### `analysis/fisher.py`

Fisher information computation with two tiers:

**Closed-form FIM** for known noise models:

- `jacobian_matrix(forward_fn, params)`: Jacobian `d(output)/d(params)`, pytree-aware via internal flatten
- `fisher_information(forward_fn, params, noise_model=...)`: closed-form FIM
  - Gaussian: `J^T diag(1/σ²) J`
  - Poisson: `J^T diag(1/μ) J`
- `cramer_rao_bound(fim)`: diagonal of FIM inverse (minimum variance per parameter)
- `d_optimality(fim)`: log-determinant criterion for experimental design

**Score-based Monte Carlo FIM** for general distributions:

- `score_fisher_information(log_prob_fn, params, samples)`: general FIM via
  `(1/N) Σᵢ ∇_θ log p(yᵢ|θ) ∇_θ log p(yᵢ|θ)^T`. Works with arbitrary differentiable
  log-likelihoods.

All functions accept arbitrary JAX pytrees as parameters.

### `analysis/sensitivity.py`

Design sensitivity and fabrication tolerance analysis:

- `sensitivity_map(forward_fn, params)`: per-parameter L2 norm of the Jacobian column
- `parameter_tolerance(forward_fn, params, target_change=...)`: allowable perturbation
  per parameter for a given output change (linear approximation)

Both return pytrees matching the input parameter structure.
