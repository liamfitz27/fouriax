"""Sensitivity analysis and Fisher information for a lens optimization problem."""

# %% Imports
from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import fouriax as fx

# %% Paths and Parameters

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sensitivity Analysis Example")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-n", type=int, default=32)
    parser.add_argument("--grid-dx-um", type=float, default=1.0)
    parser.add_argument("--wavelength-um", type=float, default=0.532)
    parser.add_argument("--distance-um", type=float, default=150.0)
    parser.add_argument("--nominal-angle-rad", type=float, default=0.005)
    parser.add_argument("--nominal-direction-deg", type=float, default=0.0)
    parser.add_argument("--stretch-factor", type=float, default=1.5)
    parser.add_argument("--input-count-scale", type=float, default=1000.0)
    parser.add_argument(
        "--sensitivity-pool",
        type=int,
        default=4,
        help=(
            "Average-pooling factor used before design-sensitivity Jacobians are formed. "
            "Higher values reduce memory usage substantially."
        ),
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    return parser.parse_args()


ARGS = parse_args()

ARTIFACTS_DIR = Path(ARGS.artifacts_dir)
PLOT_PATH = ARTIFACTS_DIR / "sensitivity_analysis_overview.png"

SEED = ARGS.seed
GRID_N = ARGS.grid_n
GRID_DX_UM = ARGS.grid_dx_um
WAVELENGTH_UM = ARGS.wavelength_um
DISTANCE_UM = ARGS.distance_um
NOMINAL_ANGLE_RAD = ARGS.nominal_angle_rad
NOMINAL_DIRECTION_DEG = ARGS.nominal_direction_deg
STRETCH_FACTOR = ARGS.stretch_factor
INPUT_COUNT_SCALE = ARGS.input_count_scale
SENSITIVITY_POOL = ARGS.sensitivity_pool
PLOT = not ARGS.no_plot


# %% Helper Functions
def stretched_hyperbolic_phase(
    grid: fx.Grid,
    distance_um: float,
    wavelength_um: float,
    stretch_factor: float = 2.0,
) -> jnp.ndarray:
    x, y = grid.spatial_grid()
    k = 2.0 * jnp.pi / wavelength_um
    return -k * (jnp.sqrt(stretch_factor * x * x + y * y + distance_um**2) - distance_um)


def pooled_metric(image: jnp.ndarray, pool: int) -> jnp.ndarray:
    if pool <= 1:
        return image.ravel()
    if image.shape[0] % pool != 0 or image.shape[1] % pool != 0:
        raise ValueError(
            "sensitivity_pool must divide both image dimensions; "
            f"got pool={pool} for shape={image.shape}"
        )
    pooled = image.reshape(
        image.shape[0] // pool,
        pool,
        image.shape[1] // pool,
        pool,
    ).mean(axis=(1, 3))
    return pooled.ravel()


# %% Setup
def main() -> None:
    if SENSITIVITY_POOL <= 0:
        raise ValueError("sensitivity_pool must be strictly positive")
    grid = fx.Grid.from_extent(nx=GRID_N, ny=GRID_N, dx_um=GRID_DX_UM, dy_um=GRID_DX_UM)
    spectrum = fx.Spectrum.from_scalar(WAVELENGTH_UM)
    propagator = fx.plan_propagation(
        mode="auto",
        grid=grid,
        spectrum=spectrum,
        distance_um=DISTANCE_UM,
    )

    # Use a hyperbolic lens phase as the "optimized" design
    phase = stretched_hyperbolic_phase(grid, DISTANCE_UM, WAVELENGTH_UM, STRETCH_FACTOR)

    def make_module(phase_map: jnp.ndarray) -> fx.OpticalModule:
        return fx.OpticalModule(
            layers=(
                fx.PhaseMask(phase_map_rad=phase_map[None, :, :]),
                propagator,
            )
        )

    input_amp = jnp.sqrt(jnp.asarray(INPUT_COUNT_SCALE, dtype=jnp.float32))
    field_in = fx.Field.plane_wave(grid=grid, spectrum=spectrum, amplitude=input_amp)

    def forward_intensity(phase_map: jnp.ndarray) -> jnp.ndarray:
        return make_module(phase_map).forward(field_in).intensity()[0]

    # %% Observation Fisher Information (angle of incidence)
    k = 2.0 * jnp.pi / WAVELENGTH_UM
    x_grid, y_grid = grid.spatial_grid()

    def forward_angle(angles: jnp.ndarray) -> jnp.ndarray:
        """Map (θ_x, θ_y) → focal-plane intensity."""
        theta_x, theta_y = angles[0], angles[1]
        tilt_phase = k * (theta_x * x_grid + theta_y * y_grid)
        field_data = (input_amp * jnp.exp(1j * tilt_phase)).astype(jnp.complex64)
        field_tilted = fx.Field(
            data=field_data[None, :, :],
            grid=grid,
            spectrum=spectrum,
            domain="spatial",
        )
        return make_module(phase).forward(field_tilted).intensity()[0].ravel()

    # Off-axis nominal angle at the configured direction
    angle_dir = jnp.deg2rad(NOMINAL_DIRECTION_DEG)
    angles_nominal = NOMINAL_ANGLE_RAD * jnp.array(
        [jnp.cos(angle_dir), jnp.sin(angle_dir)],
    )
    print("Computing observation FIM (angle of incidence)...")
    fim_angle = fx.analysis.fisher_information(
        forward_angle,
        angles_nominal,
        noise_model=fx.PoissonNoise(count_scale=1.0),
    )
    crb_angle = fx.analysis.cramer_rao_bound(fim_angle, regularize=1e-15)
    d_opt = fx.analysis.d_optimality(fim_angle)

    fim_np = np.asarray(fim_angle)
    crb_np = np.asarray(crb_angle)
    print(f"FIM (2×2):\n{fim_np}")
    print(f"CRB (θ_x, θ_y): ({crb_np[0]:.3e}, {crb_np[1]:.3e}) rad²")
    print(
        f"Angular precision: σ_θx={np.sqrt(max(crb_np[0], 0)):.3e}, "
        f"σ_θy={np.sqrt(max(crb_np[1], 0)):.3e} rad"
    )
    print(f"D-optimality: {float(d_opt):.2f}")

    # %% Design Sensitivity Analysis
    print("Computing design sensitivity (per-pixel phase sensitivity)...")
    def metric_fn(output: jnp.ndarray) -> jnp.ndarray:
        return pooled_metric(output, SENSITIVITY_POOL)

    sens = fx.analysis.sensitivity_map(forward_intensity, phase, metric_fn=metric_fn)
    sens_np = np.asarray(sens)

    print("Computing fabrication tolerance map...")
    tol = 0.01 / jnp.maximum(sens, 1e-12)
    tol_np = np.asarray(tol)

    # %% Plot Results
    if PLOT:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))

        phase_wrapped = np.angle(np.exp(1j * np.asarray(phase)))
        im0 = axes[0, 0].imshow(phase_wrapped, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[0, 0].set_title("Phase Mask (wrapped, rad)")
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04, label="Phase (rad)")

        intensity = np.asarray(
            forward_angle(angles_nominal).reshape(GRID_N, GRID_N),
        )
        im1 = axes[0, 1].imshow(intensity, cmap="inferno")
        theta_mrad = float(NOMINAL_ANGLE_RAD * 1e3)
        axes[0, 1].set_title(
            f"Focal Spot (θ = {theta_mrad:.1f} mrad, dir = {NOMINAL_DIRECTION_DEG:.0f}°)",
        )
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Intensity")
        axes[0, 1].text(
            0.03,
            0.97,
            (
                f"Fisher Information Matrix (2×2):\n"
                f"  FIM_θxθx = {fim_np[0, 0]:.4e}\n"
                f"  FIM_θyθy = {fim_np[1, 1]:.4e}\n"
                f"  FIM_θxθy = {fim_np[0, 1]:.4e}\n\n"
                f"Cramér–Rao Bound:\n"
                f"  σ_θx ≥ {np.sqrt(max(crb_np[0], 0)):.3e} rad\n"
                f"  σ_θy ≥ {np.sqrt(max(crb_np[1], 0)):.3e} rad\n\n"
                f"D-optimality: {float(d_opt):.2f}"
            ),
            transform=axes[0, 1].transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            family="monospace",
            color="white",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "black",
                "alpha": 0.70,
                "edgecolor": "white",
                "linewidth": 0.8,
            },
        )

        im2 = axes[1, 0].imshow(sens_np, cmap="magma")
        axes[1, 0].set_title("Phase Sensitivity Map")
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04, label="‖∂I/∂φᵢ‖")

        tol_clipped = np.clip(tol_np, 0, np.nanpercentile(tol_np, 95))
        im3 = axes[1, 1].imshow(tol_clipped, cmap="viridis")
        axes[1, 1].set_title("Fabrication Tolerance (rad)")
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Δφ for 1% ΔI")

        fig.tight_layout()
        fig.savefig(PLOT_PATH, dpi=150)
        plt.close(fig)
        print(f"saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
