# Project Status

## Purpose
Build a JAX-native differentiable free-space optics library for optical + NN workflows.

## Current State (2026-02-17)
- Core optics primitives and composition stack are stable (`Field`, `Grid`, `Spectrum`, `OpticalModule`).
- Hybrid domain support is in progress: spatial layers, k-space mask layers, domain-aware field conversions.
- Propagators now include spatial (`ASMPropagator`, `RSPropagator`, `AutoPropagator`) and k-domain (`KSpacePropagator`) paths.
- NA/bandlimit work is in progress via module-level scheduling and propagator NA masking.

## Current Priorities
1. Finalize NA policy architecture across the optics stack:
   - decide whether NA is represented as explicit stop/pupil layers, propagator-level limits, or both
   - resolve current asymmetry (`ASM`/`KSpace` NA support vs `RS`)
2. Stabilize hybrid-domain API contracts:
   - define explicit expectations at module and sensor boundaries (`spatial` vs `kspace`)
   - keep conversions predictable and avoid surprising implicit behavior
3. Validate parity envelopes between physical 4f spatial models and k-space surrogate filtering.
4. Harden integration tests/examples for hybrid spatial+k-space workflows.

## In Progress: Fourier + Hybrid Modules
- `Field` tracks domain (`spatial` / `kspace`) and provides internal transforms (`to_kspace`, `to_spatial`).
- k-space diagonal layers exist (`KPhaseMaskLayer`, `KAmplitudeMaskLayer`, `KComplexMaskLayer`).
- `KSpacePropagator` supports k-domain inter-layer propagation (propagating spectrum).
- `OpticalModule` can compute per-segment local NA schedules and inject limits into NA-capable propagators.
- 4f comparison/sweep scripts are being used to validate regime-dependent agreement and throughput behavior.
- Sensor behavior is being standardized so readout interfaces remain spatial-domain consistent.

## Known Risks
- Throughput and boundary effects can differ between full spatial 4f models and ideal FFT-filter surrogates.
- NA handling is currently asymmetric across propagator families (`ASM`/`KSpace` vs `RS`).
- Large apertures/high NA on fixed grids can re-introduce edge artifacts without sufficient padding/window margin.
- Hybrid-domain auto-conversion can obscure where expensive FFT transitions occur if contract boundaries are not explicit.
