#!/usr/bin/env python3
"""Simple end-to-end lens optimization example (script-first)."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxopt import OptaxSolver

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


def circular_aperture(grid: Grid, diameter_um: float) -> jnp.ndarray:
    x, y = grid.spatial_grid()
    r2 = x * x + y * y
    radius = diameter_um / 2.0
    return (r2 <= radius * radius).astype(jnp.float32)


def main() -> None:
    grid = Grid.from_extent(nx=64, ny=64, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field_in = Field.plane_wave(grid=grid, spectrum=spectrum)

    distance_um = 1000.0
    aperture = circular_aperture(grid, diameter_um=48.0)
    target_xy = (grid.nx // 2, grid.ny // 2)
    window_px = 2

    propagator = AutoPropagator(
        setup_grid=grid,
        setup_spectrum=spectrum,
        setup_distance_um=distance_um,
    )

    def loss_fn(phase_map: jnp.ndarray) -> jnp.ndarray:
        phase_limited = 2 * jnp.pi * jax.nn.sigmoid(phase_map)
        module = OpticalModule(
            layers=(
                PhaseMaskLayer(phase_map_rad=phase_limited[None, :, :]),
                AmplitudeMaskLayer(amplitude_map=aperture[None, :, :]),
                PropagationLayer(model=propagator, distance_um=distance_um),
            )
        )
        intensity = module.forward(field_in).intensity()
        return focal_spot_loss(
            intensity=intensity,
            target_xy=target_xy,
            window_px=window_px,
        )

    key = jax.random.PRNGKey(0)
    phase_map = 0.1 * jax.random.normal(key, (grid.ny, grid.nx))
    lr = 0.05
    steps = 60
    solver = OptaxSolver(fun=loss_fn, opt=optax.adam(lr))
    opt_state = solver.init_state(phase_map)

    history: list[float] = []
    for step in range(steps):
        phase_map, opt_state = solver.update(phase_map, opt_state)
        loss = loss_fn(phase_map)
        history.append(float(loss))
        if step % 20 == 0 or step == steps - 1:
            print(f"step={step:03d} loss={float(loss):.6f}")

    initial_loss = history[0]
    final_loss = history[-1]

    final_phase_limited = 2 * jnp.pi * jax.nn.sigmoid(phase_map)
    final_module = OpticalModule(
        layers=(
            PhaseMaskLayer(phase_map_rad=final_phase_limited[None, :, :]),
            AmplitudeMaskLayer(amplitude_map=aperture[None, :, :]),
            PropagationLayer(model=propagator, distance_um=distance_um),
        )
    )
    final_intensity = np.asarray(final_module.forward(field_in).intensity())[0]
    center_intensity = float(final_intensity[target_xy[1], target_xy[0]])
    optimized_profile = final_intensity[target_xy[1], :]

    # Reference: phase profile for ideal spherical wavefront convergence at distance_um.
    x_um, y_um = grid.spatial_grid()
    wavelength_um = float(spectrum.wavelengths_um[0])
    k = 2.0 * jnp.pi / wavelength_um
    hyperbolic_phase = -k * (jnp.sqrt(x_um * x_um + y_um * y_um + distance_um**2) - distance_um)
    reference_module = OpticalModule(
        layers=(
            PhaseMaskLayer(phase_map_rad=hyperbolic_phase[None, :, :]),
            AmplitudeMaskLayer(amplitude_map=aperture[None, :, :]),
            PropagationLayer(model=propagator, distance_um=distance_um),
        )
    )
    reference_out = reference_module.forward(field_in)
    reference_intensity = np.asarray(reference_out.intensity())[0]
    reference_profile = reference_intensity[target_xy[1], :]

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "steps": steps,
        "optimizer": "jaxopt+optax_adam",
        "learning_rate": lr,
        "precomputed_method": propagator.precomputed_method,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "improvement": initial_loss - final_loss,
        "target_xy": target_xy,
        "window_px": window_px,
        "center_intensity": center_intensity,
    }
    with (out_dir / "lens_opt_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("saved:", out_dir / "lens_opt_summary.json")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    optimized_phase = np.asarray(2 * jnp.pi * jax.nn.sigmoid(phase_map))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    axes[0, 0].plot(history)
    axes[0, 0].set_title("Loss History")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(alpha=0.3)

    phase_im = axes[0, 1].imshow(optimized_phase, cmap="twilight")
    axes[0, 1].set_title("Optimized Phase (rad)")
    plt.colorbar(phase_im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    focus_im = axes[1, 0].imshow(final_intensity, cmap="inferno")
    axes[1, 0].set_title("Optimized 2D Focal Spot")
    plt.colorbar(focus_im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].plot(optimized_profile, label="Optimized")
    axes[1, 1].plot(reference_profile, label="Hyperbolic-phase reference", linestyle="--")
    axes[1, 1].set_title("Center Row Profile")
    axes[1, 1].set_xlabel("x pixel")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "lens_opt_overview.png", dpi=160)
    print("saved:", out_dir / "lens_opt_overview.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
