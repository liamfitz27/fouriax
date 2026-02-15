import jax
import jax.numpy as jnp

from fouriax.optics import (
    ASMPropagator,
    AutoPropagator,
    Field,
    Grid,
    PropagationPolicy,
    RSPropagator,
    SamplingPlanner,
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
    planner = SamplingPlanner(safety_factor=2.0, min_padding_factor=2.0)
    plan = planner.recommend_grid(mask_grid=grid, spectrum=spectrum)
    policy = PropagationPolicy(mode="balanced")
    decision = policy.choose(grid=grid, spectrum=spectrum, distance_um=distance_um, plan=plan)
    propagator = AutoPropagator(
        asm=ASMPropagator(use_sampling_planner=False, precomputed_plan=plan),
        rs=RSPropagator(use_sampling_planner=False, precomputed_plan=plan),
        policy=policy,
        precomputed_plan=plan,
        precomputed_method=decision.method,
    )

    target_xy = (grid.nx // 2, grid.ny // 2)

    def loss_fn(phase_map: jnp.ndarray) -> jnp.ndarray:
        phase_limited = 2 * jnp.pi * jax.nn.sigmoid(phase_map)
        field = field_in.apply_phase(phase_limited[None, :, :]).apply_amplitude(aperture[None, :, :])
        out = propagator.propagate(field, distance_um=distance_um)
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
