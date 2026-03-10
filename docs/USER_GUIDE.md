# Fouriax User Guide

Welcome to **fouriax**, a differentiable free-space optics library for JAX. 
This guide will walk you through building a system, running an optimization, using polarization (Jones mode), using meta-atoms, and reading sensor output.

## 1. Basic Setup & Differentiable Forward Pass

An optical stack in `fouriax` consists of creating a spatial `Grid`, assigning a `Spectrum` (wavelengths), initializing an input `Field`, passing it through an `OpticalModule`, and collecting the result with a `Sensor`.

```python
import jax
import jax.numpy as jnp
import fouriax as fx

# 1. Define grid (micrometers) and spectrum (532 nm)
grid = fx.Grid.from_extent(nx=64, ny=64, dx_um=1.0, dy_um=1.0)
spectrum = fx.Spectrum.from_scalar(0.532)

# 2. Setup initial plane wave field
field = fx.Field.plane_wave(grid=grid, spectrum=spectrum)

# 3. Create components: Phase mask and propagator
rng = jax.random.PRNGKey(0)
initial_phase = jnp.zeros((64, 64))
phase_layer = fx.PhaseMask(phase_map_rad=initial_phase)

# Automatic propagator selection (Angular Spectrum or Rayleigh-Sommerfeld)
propagator = fx.plan_propagation(
    mode="auto", grid=grid, spectrum=spectrum, distance_um=1000.0,
)

# 4. Compose the module and sensor
module = fx.OpticalModule(layers=(phase_layer, propagator))
detector = fx.Detector()

# 5. Forward pass to get intensity
field_out = module.forward(field)
intensity = detector.measure(field_out)
```

## 2. Gradient-Based Optimization

Since every layer is fully traceable in JAX, you can define a loss function and compute gradients with respect to the system parameters. Fouriax provides lightweight wrappers in `fx.optim` for Optax training loops.

```python
import optax
from fouriax.optim import optimize_optical_module

def build_module(params):
    # params is a dictionary mapping parameter names to JAX arrays
    return fx.OpticalModule(layers=(
        fx.PhaseMask(phase_map_rad=params["phase"]),
        propagator
    ))

def loss_fn(params):
    mod = build_module(params)
    out = mod.forward(field)
    return jnp.mean((out.intensity() - target_intensity)**2)

# Initial parameters
init_params = {"phase": jnp.zeros((64, 64))}
optimizer = optax.adam(learning_rate=0.1)

# Optimize for 100 steps
result = fx.optim.optimize_optical_module(
    init_params=init_params,
    build_module=build_module,
    loss_fn=loss_fn,
    optimizer=optimizer,
    steps=100
)

print(f"Final loss: {result.final_loss}")
optimized_phase = result.best_params["phase"]
```

## 3. Polarization (Jones Mode)

To track arbitrary polarization states, initialize a Jones field. Modulation layers will correctly broadcast or mix the `Ex` and `Ey` channels.

```python
# Initialize a linearly polarized input (45 degrees)
jones_field = fx.Field.plane_wave_jones(
    grid=grid, spectrum=spectrum, ex=1.0, ey=1.0
)

# Apply a spatial-domain Jones matrix (e.g., quarter waveplate or polarizer)
jones_matrix = jnp.array([
    [1.0, 0.0],
    [0.0, 1j]
])
polarizer_layer = fx.JonesMatrixLayer(jones_matrix=jones_matrix)

# Trace through the layer
jones_field_out = polarizer_layer.forward(jones_field)
```

## 4. Meta-Atom Look-up Tables

Fouriax supports parameterized `MetaAtomLibrary` models, mapping physical meta-atom dimensions to complex transmission coefficients across a wavelength spectrum.

```python
# Load a precomputed LUT 
library = fx.MetaAtomLibrary.from_file("my_meta_atoms.npz")

# Use it in a layer by providing spatial geometry maps 
# (e.g., width of a pillar at each grid point)
meta_layer = fx.MetaAtomInterpolationLayer(
    library=library,
    geometry_maps={"width": jnp.ones((64, 64)) * 0.25}
)

field_out = meta_layer.forward(field)
```

## 5. Noise and Sensors

You can model physical camera properties using realistic noise distributions and masks via a `DetectorArray`.

```python
sensor = fx.DetectorArray(
    detector_grid=fx.Grid.from_extent(nx=32, ny=32, dx_um=2.0, dy_um=2.0),
    noise_model=fx.PoissonGaussianNoise(
        count_scale=100.0, # scales expected intensity to expected photons
        read_noise_std=2.5 # additive Gaussian read noise standard deviation
    )
)

# Returns noisy samples over the specified detector_grid
noisy_measurement = sensor.measure(field_out, key=rng)
```
