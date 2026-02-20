# Fouriax Architecture Overview

This document provides a comprehensive overview of the `fouriax` differentiable free-space optics library. It covers the core abstractions, the layer hierarchy, propagation models, sensor interfaces, and explains how the three major regimes (coherent imaging, incoherent imaging, and k-space operators) are unified under a common computational framework.

## 1. Core Data Models (`optics/model.py`)

The foundational data structure passed between optical layers is the `Field`.

- **`Grid`**: Represents a 2D uniform spatial grid.
  - Attributes: `nx`, `ny`, `dx_um`, `dy_um`.
  - Methods: `frequency_grid()`, `spatial_grid()`, `kspace_pixel_size_cyc_per_um()`.
- **`Spectrum`**: Represents a collection of processing wavelengths.
  - Attributes: `wavelengths_um` (1D array).
- **`Field`**: A complex optical field defined over a `Grid` and `Spectrum`.
  - Shape: `(num_wavelengths, ny, nx)`.
  - Attributes: `data`, `grid`, `spectrum`, `domain` (either `"spatial"` or `"kspace"`).
  - Methods: `intensity()`, `phase()`, `power()`, `normalize_power()`, `apply_phase()`, `apply_amplitude()`.
  - Domain Operations: `to_kspace()`, `to_spatial()` seamlessly handle FFT/IFFT calls and metadata alignment.

## 2. Interface Hierarchy (`optics/interfaces.py`)

The library leverages simple Abstract Base Classes to define composable optical transformations. All components are natively compatible with JAX transformations (like `vmap`, `grad`, and `jit`).

- **`OpticalLayer`**: The base class for all operators, including components and propagation models.
  - `forward(field: Field) -> Field`
  - `validate_for(field: Field)`
  - `parameters() -> dict[str, jnp.ndarray]`
- **`Sensor`**: Computes measured readouts from complex fields.
  - `measure(field: Field) -> jnp.ndarray`

## 3. Optical Layers (`optics/layers.py`)

Layers implement the `OpticalLayer` interface and perform specific physical operations. They can be freely combined.

### Spatial Modulation Layers (Coherent)
- **`PhaseMask`**: Applies a configurable 2D phase modulation array (`jnp.exp(1j * phase)`).
- **`AmplitudeMask`**: Applies a configurable 2D amplitude modulation array.
- **`ComplexMask`**: Combines both amplitude and phase modulation.
- **`ThinLens`**: An ideal quadratic phase mask representing a lens of `focal_length_um`, with an optional `aperture_diameter_um`.

### K-Space Layers
- **`KSpacePhaseMask`, `KSpaceAmplitudeMask`, `KSpaceComplexMask`**: These layers enforce execution strictly in the `"kspace"` domain. They are useful for spatial frequency filtering (e.g., surrogate 4f systems).

### Incoherent Layers
- **`IncoherentImager`**: Converts a coherent optical subsystem into an incoherent, shift-invariant imager.
  - It works by automatically probing a defined coherent subsystem (an `OpticalLayer` serving as a propagator) with an impulse (or plane wave focus) to extract a coherent Point Spread Function (PSF).
  - It then computes an incoherent response via spatial convolution (`mode="psf"`) or frequency-domain filtering (`mode="otf"`) applied to the *intensity* (rather than complex amplitude) of the incoming field.

### System Composition
- **`OpticalModule`**: The primary sequence orchestrator. It holds an ordered sequence of `layers` and an optional `sensor`. 
  - Resolves domain transitions (automatically tracks when a layer forces `"spatial"` vs `"kspace"` representation and invokes transforms seamlessly).
  - Recursively collects cascaded aperture stops across the system (`effective_na()`) and automatically injects safe Numerical Aperture bounds (`na_schedule()`) to prevent high numerical frequency aliasing inside propagators.
  - Exposes `forward()`, `measure()`, `trace()` (for returning intermediate fields) and `parameters()`.

## 4. Propagation Models (`optics/propagation.py`)

Free-space optical propagation requires solving the wave equation. Rather than forcing the user to pick equations blindly, Fouriax implements multiple FFT-based regimes:

- **`RSPropagator`**: Employs spatial convolution with a Rayleigh-Sommerfeld spherical impulse response. Ideal for near-field regimes.
- **`ASMPropagator`**: Employs the Angular Spectrum Method (multiplying by an exact phase transfer function directly in frequency space). Ideal for accurate paraxial to far-field propagation.
- **`KSpacePropagator`**: Applies a direct phase advance on a field natively inside the `kspace` domain.
- **`AutoPropagator` / `CoherentPropagator`**: Intelligent meta-propagators that dynamically select between the exact RS, ASM, or K-Space methods based on simulation inputs (`distance_um`, wavelength, and `z_crit`).

## 5. Sensors (`optics/sensors.py`)

Transforming infinite precision physics to a real-world digital signal is done at sensor edges.

- **`IntensitySensor`**: Computes the squared magnitude of the field (intensity). Can optionally integrate power over specific `detector_masks` regions (such as binning photons in optical neural networks).
- **`FieldReadout`**: Exposes raw underlying complex layers via configurable representations (`complex`, `real_imag`, `amplitude_phase`) for mathematical analysis rather than physical detection.

---

## Integrating Under a Common Framework

The central challenge in modeling differentiable optics is harmonizing differing physical behaviors—coherent complex wavefronts, incoherent intensities, and frequency-domain masking—into a singular autodiff pipeline. Fouriax achieves this seamlessly because the components speak a unified language via `Field` and `OpticalModule`.

### 1. The `Field` as the Common Currency
Everything passed forward or backwards through the system is an instance of `Field`. 
- By explicitly tracking the `domain` (`"spatial"` vs `"kspace"`), the framework delays executing explicit Fast Fourier Transforms until strictly necessary. 
- By tracking `spectrum`, every layer natively broadcasts JAX operations across multi-band wavelengths automatically (hyperspectral imaging).

### 2. Bridging Coherent and K-Space Operations
In physical optics (such as a 4f correlator system), a spatial component acts as a physical analog Fourier transform. You have a lens, then an aperture, then another lens. 
- **The Structural Approach (Coherent):** `[ThinLens, CoherentPropagator, AmplitudeMask, CoherentPropagator, ThinLens]`.
- **The Idealized Surrogate Approach (K-Space):** Alternatively, you can drop physical lenses entirely and use `KSpaceAmplitudeMask` directly. 

`OpticalModule` reconciles these approaches automatically. When the stack encounters a `KSpace` layer, it ensures `Field.to_kspace()` runs (performing one global FFT), applies the surrogate element-wise in frequency blocks, and leaves the sequence in `"kspace"`. It only transitions back (`Field.to_spatial()`) when the next physical lens explicitly demands it, guaranteeing minimal FFT execution.

### 3. Bridging Coherent and Incoherent Operations
Coherent imaging processes Electric fields ($E_{out} \approx E * h$); Incoherent imaging deals purely in scattered intensities ($I_{out} \approx I * |h|^2$). Typically, simulation tools build two entirely discrete codebases.

In `fouriax`, this gap is cleanly bridged using the `IncoherentImager` layer.
- `IncoherentImager` takes your existing coherent layout elements and analytically computes its continuous PSF. 
- Because it honors the exact same `OpticalLayer` base class (accepting a `Field` input and returning a non-negative derived pseudo-`Field` amplitude back out), it drops immediately into existing `OpticalModule` pipelines alongside normal phase masks or propagators. 

### Summary: Execution Inside `OpticalModule`
Whenever a user calls `module.forward(field)` or `module.measure(field)`, the framework undergoes structural compilation:
1. NA limits are gathered system-wide from objects like `ThinLens` and passed to bounded segments.
2. `AutoPropagators` compile to static `RS` or `ASM` kernels depending on `z_crit` limits.
3. As the simulated pipeline traverses layer $N$, if layer $N$ requires `kspace` but previous output was `spatial`, a single batched `jnp.fft.fftn` is injected.
4. Intermediate states are resolved dynamically without compromising gradient backpropagation (using standard `jax.grad`).
5. A batched `Sensor` flattens the terminating complex arrays to physical measurements.
