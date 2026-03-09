"""Simple end-to-end lens optimization example."""

# %% Imports
from __future__ import annotations

# %% Paths and Parameters
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from fouriax.optics import (
    AmplitudeMask,
    Field,
    Grid,
    OpticalModule,
    PhaseMask,
    Spectrum,
    plan_propagation,
)
from fouriax.optim import focal_spot_loss, optimize_optical_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lens Optimization Example")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-n", type=int, default=64)
    parser.add_argument("--grid-dx-um", type=float, default=1.0)
    parser.add_argument("--wavelength-um", type=float, default=0.532)
    parser.add_argument("--distance-um", type=float, default=1000.0)
    parser.add_argument("--aperture-diameter-um", type=float, default=48.0)
    parser.add_argument("--window-px", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    return parser.parse_args()


ARGS = parse_args()

ARTIFACTS_DIR = Path(ARGS.artifacts_dir)
PLOT_PATH = ARTIFACTS_DIR / "lens_optimization_overview.png"
SUMMARY_PATH = ARTIFACTS_DIR / "lens_opt_summary.json"

SEED = ARGS.seed
GRID_N = ARGS.grid_n
GRID_DX_UM = ARGS.grid_dx_um
WAVELENGTH_UM = ARGS.wavelength_um
DISTANCE_UM = ARGS.distance_um
APERTURE_DIAMETER_UM = ARGS.aperture_diameter_um
WINDOW_PX = ARGS.window_px
LR = ARGS.lr
STEPS = ARGS.steps
PLOT = not ARGS.no_plot


# %% Helper functions
def circular_aperture(grid: Grid, diameter_um: float) -> jnp.ndarray:
    x, y = grid.spatial_grid()
    r2 = x * x + y * y
    radius = diameter_um / 2.0
    return (r2 <= radius * radius).astype(jnp.float32)


def main() -> None:
    # %% Setup
    grid = Grid.from_extent(nx=GRID_N, ny=GRID_N, dx_um=GRID_DX_UM, dy_um=GRID_DX_UM)
    spectrum = Spectrum.from_scalar(WAVELENGTH_UM)
    field_in = Field.plane_wave(grid=grid, spectrum=spectrum)

    aperture = circular_aperture(grid, diameter_um=APERTURE_DIAMETER_UM)
    target_xy = (grid.nx // 2, grid.ny // 2)
    propagator = plan_propagation(
        mode="auto",
        grid=grid,
        spectrum=spectrum,
        distance_um=DISTANCE_UM,
    )

    def build_module(raw_phase_map: jnp.ndarray) -> OpticalModule:
        phase_limited = 2.0 * jnp.pi * jax.nn.sigmoid(raw_phase_map)
        return OpticalModule(
            layers=(
                PhaseMask(phase_map_rad=phase_limited[None, :, :]),
                AmplitudeMask(amplitude_map=aperture[None, :, :]),
                propagator,
            )
        )

    # %% Loss Function and Optimization
    def loss_fn(raw_phase_map: jnp.ndarray) -> jnp.ndarray:
        module = build_module(raw_phase_map)
        intensity = module.forward(field_in).intensity()
        return focal_spot_loss(
            intensity=intensity,
            target_xy=target_xy,
            window_px=WINDOW_PX,
        )

    key = jax.random.PRNGKey(SEED)
    phase_map = 0.1 * jax.random.normal(key, (grid.ny, grid.nx))

    optimizer = optax.adam(LR)
    result = optimize_optical_module(
        init_params=phase_map,
        build_module=build_module,
        loss_fn=loss_fn,
        optimizer=optimizer,
        steps=STEPS,
        log_every=20,
    )
    final_phase_limited = 2.0 * jnp.pi * jax.nn.sigmoid(result.best_params)
    final_intensity = np.asarray(result.best_module.forward(field_in).intensity())[0]
    optimized_profile = final_intensity[target_xy[1], :]

    # %% Evaluation
    x_um, y_um = grid.spatial_grid()
    wavelength_um = float(spectrum.wavelengths_um[0])
    k = 2.0 * jnp.pi / wavelength_um
    hyperbolic_phase = -k * (jnp.sqrt(x_um * x_um + y_um * y_um + DISTANCE_UM**2) - DISTANCE_UM)
    reference_module = OpticalModule(
        layers=(
            PhaseMask(phase_map_rad=hyperbolic_phase[None, :, :]),
            AmplitudeMask(amplitude_map=aperture[None, :, :]),
            propagator,
        )
    )
    reference_intensity = np.asarray(reference_module.forward(field_in).intensity())[0]
    reference_profile = reference_intensity[target_xy[1], :]

    # %% Plot Results
    if PLOT:
        optimized_phase = np.asarray(final_phase_limited)
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

        axes[0, 0].plot(result.history)
        axes[0, 0].set_title("Loss History")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(alpha=0.3)

        phase_im = axes[0, 1].imshow(optimized_phase, cmap="twilight")
        axes[0, 1].set_title("Optimized Phase (rad)")
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])
        plt.colorbar(phase_im, ax=axes[0, 1], fraction=0.046, pad=0.04)

        focus_im = axes[1, 0].imshow(final_intensity, cmap="inferno")
        axes[1, 0].set_title("Optimized 2D Focal Spot")
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
        plt.colorbar(focus_im, ax=axes[1, 0], fraction=0.046, pad=0.04)

        axes[1, 1].plot(optimized_profile, label="Optimized")
        axes[1, 1].plot(reference_profile, label="Hyperbolic-phase reference", linestyle="--")
        axes[1, 1].axvline(
            target_xy[0], color="black", linestyle=":", linewidth=1.2, label="Target x"
        )
        axes[1, 1].set_title("Center Row Profile")
        axes[1, 1].set_xlabel("x pixel")
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].legend()

        fig.tight_layout()
        fig.savefig(PLOT_PATH, dpi=150)
        plt.close(fig)
        print(f"saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
