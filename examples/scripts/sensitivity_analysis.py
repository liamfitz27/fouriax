"""Sensitivity analysis and Fisher information for a lens optimization problem."""

#%% Imports
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from fouriax.analysis import (
    cramer_rao_bound,
    d_optimality,
    fisher_information,
    parameter_tolerance,
    sensitivity_map,
)
from fouriax.optics import (
    Field,
    Grid,
    OpticalModule,
    PhaseMask,
    PoissonNoise,
    Spectrum,
    plan_propagation,
)

#%% Paths and Parameters
ARTIFACTS_DIR = Path("artifacts")
PLOT_PATH = ARTIFACTS_DIR / "sensitivity_analysis_overview.png"

SEED = 0
GRID_N = 32
GRID_DX_UM = 1.0
WAVELENGTH_UM = 0.532
DISTANCE_UM = 150.0
NOMINAL_ANGLE_RAD = 0.005  # small off-axis tilt
NOMINAL_DIRECTION_DEG = 0  # tilt direction
STRETCH_FACTOR = 1.5
INPUT_COUNT_SCALE = 1_000.0  # Poisson mean-count scale (variance equals mean)


#%% Helper Functions
def stretched_hyperbolic_phase(
    grid: Grid, distance_um: float, wavelength_um: float,
    stretch_factor: float = 2.0,
) -> jnp.ndarray:
    x, y = grid.spatial_grid()
    k = 2.0 * jnp.pi / wavelength_um
    return -k * (jnp.sqrt(stretch_factor * x * x + y * y + distance_um**2) - distance_um)


#%% Setup
def main() -> None:
    grid = Grid.from_extent(nx=GRID_N, ny=GRID_N, dx_um=GRID_DX_UM, dy_um=GRID_DX_UM)
    spectrum = Spectrum.from_scalar(WAVELENGTH_UM)
    propagator = plan_propagation(
        mode="auto", grid=grid, spectrum=spectrum, distance_um=DISTANCE_UM,
    )

    # Use a hyperbolic lens phase as the "optimized" design
    phase = stretched_hyperbolic_phase(grid, DISTANCE_UM, WAVELENGTH_UM, STRETCH_FACTOR)

    def make_module(phase_map: jnp.ndarray) -> OpticalModule:
        return OpticalModule(
            layers=(
                PhaseMask(phase_map_rad=phase_map[None, :, :]),
                propagator,
            )
        )

    input_amp = jnp.sqrt(jnp.asarray(INPUT_COUNT_SCALE, dtype=jnp.float32))
    field_in = Field.plane_wave(grid=grid, spectrum=spectrum, amplitude=input_amp)

    def forward_intensity(phase_map: jnp.ndarray) -> jnp.ndarray:
        return make_module(phase_map).forward(field_in).intensity()[0]

    #%% Observation Fisher Information (angle of incidence)
    k = 2.0 * jnp.pi / WAVELENGTH_UM
    x_grid, y_grid = grid.spatial_grid()

    def forward_angle(angles: jnp.ndarray) -> jnp.ndarray:
        """Map (θ_x, θ_y) → focal-plane intensity."""
        theta_x, theta_y = angles[0], angles[1]
        tilt_phase = k * (theta_x * x_grid + theta_y * y_grid)
        field_data = (input_amp * jnp.exp(1j * tilt_phase)).astype(jnp.complex64)
        field_tilted = Field(
            data=field_data[None, :, :], grid=grid,
            spectrum=spectrum, domain="spatial",
        )
        return make_module(phase).forward(field_tilted).intensity()[0].ravel()

    # Off-axis nominal angle at the configured direction
    angle_dir = jnp.deg2rad(NOMINAL_DIRECTION_DEG)
    angles_nominal = NOMINAL_ANGLE_RAD * jnp.array(
        [jnp.cos(angle_dir), jnp.sin(angle_dir)],
    )
    print("Computing observation FIM (angle of incidence)...")
    fim_angle = fisher_information(
        forward_angle, angles_nominal,
        noise_model=PoissonNoise(count_scale=1.0),
    )
    crb_angle = cramer_rao_bound(fim_angle, regularize=1e-15)
    d_opt = d_optimality(fim_angle)

    fim_np = np.asarray(fim_angle)
    crb_np = np.asarray(crb_angle)
    print(f"FIM (2×2):\n{fim_np}")
    print(f"CRB (θ_x, θ_y): ({crb_np[0]:.3e}, {crb_np[1]:.3e}) rad²")
    print(f"Angular precision: σ_θx={np.sqrt(max(crb_np[0], 0)):.3e}, "
          f"σ_θy={np.sqrt(max(crb_np[1], 0)):.3e} rad")
    print(f"D-optimality: {float(d_opt):.2f}")

    #%% Design Sensitivity Analysis
    print("Computing design sensitivity (per-pixel phase sensitivity)...")
    sens = sensitivity_map(forward_intensity, phase)
    sens_np = np.asarray(sens)

    print("Computing fabrication tolerance map...")
    tol = parameter_tolerance(forward_intensity, phase, target_change=0.01)
    tol_np = np.asarray(tol)

    #%% Plot Results
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
        0.03, 0.97,
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
        va="top", ha="left", fontsize=8.5, family="monospace",
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
