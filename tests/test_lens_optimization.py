import jax
import jax.numpy as jnp

from fouriax.optics import (
    AmplitudeMaskLayer,
    AutoPropagator,
    Field,
    Grid,
    OpticalModule,
    PhaseMaskLayer,
    PropagationLayer,
    Spectrum,
    focal_spot_loss,
)


def _circular_aperture(grid: Grid, diameter_um: float) -> jnp.ndarray:
    x, y = grid.spatial_grid()
    r2 = x * x + y * y
    radius = diameter_um / 2.0
    return (r2 <= radius * radius).astype(jnp.float32)


def test_focal_spot_loss_decreases_over_short_optimization():
    grid = Grid.from_extent(nx=32, ny=32, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field_in = Field.plane_wave(grid=grid, spectrum=spectrum)
    aperture = _circular_aperture(grid, diameter_um=20.0)

    distance_um = 200.0
    propagator = AutoPropagator(
        policy_mode="balanced",
        setup_grid=grid,
        setup_spectrum=spectrum,
        setup_distance_um=distance_um,
    )

    target_xy = (grid.nx // 2, grid.ny // 2)

    def make_module(phase_2d: jnp.ndarray) -> OpticalModule:
        return OpticalModule(
            layers=(
                PhaseMaskLayer(phase_map_rad=phase_2d[None, :, :]),
                AmplitudeMaskLayer(amplitude_map=aperture[None, :, :]),
                PropagationLayer(model=propagator, distance_um=distance_um),
            )
        )

    def loss_fn(phase_map: jnp.ndarray) -> jnp.ndarray:
        phase_limited = 2 * jnp.pi * jax.nn.sigmoid(phase_map)
        out = make_module(phase_limited).forward(field_in)
        return focal_spot_loss(out.intensity(), target_xy=target_xy, window_px=1)

    value_and_grad = jax.value_and_grad(loss_fn)
    key = jax.random.PRNGKey(0)
    phase_map = 0.1 * jax.random.normal(key, (grid.ny, grid.nx))
    lr = 0.1
    steps = 10

    loss0 = float(loss_fn(phase_map))
    for _ in range(steps):
        loss, grad = value_and_grad(phase_map)
        phase_map = phase_map - lr * grad
    loss_final = float(loss_fn(phase_map))

    assert loss_final < loss0
