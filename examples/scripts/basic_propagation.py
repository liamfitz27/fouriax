"""Basic propagation example.

Build a minimal optical stack from a plane-wave source, a thin lens, and a
planned free-space propagator. This is the simplest entry point for learning
the core `fouriax` objects before moving on to optimization examples.
"""

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
    parser = argparse.ArgumentParser(description="Basic Propagation Example")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--grid-n", type=int, default=128)
    parser.add_argument("--grid-dx-um", type=float, default=0.5)
    parser.add_argument("--wavelength-um", type=float, default=0.532)
    parser.add_argument("--distance-um", type=float, default=50.0)
    parser.add_argument("--aperture-fraction", type=float, default=0.35)
    parser.add_argument("--focus-radius-px", type=float, default=3.0)
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    return parser.parse_args()


ARGS = parse_args()

ARTIFACTS_DIR = Path(ARGS.artifacts_dir)
PLOT_PATH = ARTIFACTS_DIR / "basic_propagation_overview.png"

GRID_N = ARGS.grid_n
GRID_DX_UM = ARGS.grid_dx_um
WAVELENGTH_UM = ARGS.wavelength_um
DISTANCE_UM = ARGS.distance_um
APERTURE_FRACTION = ARGS.aperture_fraction
FOCUS_RADIUS_PX = ARGS.focus_radius_px
PLOT = not ARGS.no_plot


# %% Concept and Helper Functions
def circular_region_mask(
    grid: fx.Grid,
    *,
    radius_um: float,
    center_xy: tuple[int, int] | None = None,
) -> jnp.ndarray:
    x_um, y_um = grid.spatial_grid()
    if center_xy is None:
        cx_um = 0.0
        cy_um = 0.0
    else:
        cx_px, cy_px = center_xy
        cx_um = (cx_px - (grid.nx - 1) / 2.0) * grid.dx_um
        cy_um = (cy_px - (grid.ny - 1) / 2.0) * grid.dy_um
    r2 = (x_um - cx_um) ** 2 + (y_um - cy_um) ** 2
    return (r2 <= radius_um * radius_um).astype(jnp.float32)


def main() -> None:
    # %% Setup
    grid = fx.Grid.from_extent(nx=GRID_N, ny=GRID_N, dx_um=GRID_DX_UM, dy_um=GRID_DX_UM)
    spectrum = fx.Spectrum.from_scalar(WAVELENGTH_UM)
    field_in = fx.Field.plane_wave(grid=grid, spectrum=spectrum)

    aperture_diameter_um = APERTURE_FRACTION * grid.nx * grid.dx_um
    lens = fx.ThinLens(
        focal_length_um=DISTANCE_UM,
        aperture_diameter_um=aperture_diameter_um,
    )
    propagator = fx.plan_propagation(
        mode="auto",
        grid=grid,
        spectrum=spectrum,
        distance_um=DISTANCE_UM,
    )
    module = fx.OpticalModule(layers=(lens, propagator))

    field_out = module.forward(field_in)
    intensity = np.asarray(field_out.intensity())[0]
    center_xy = (grid.nx // 2, grid.ny // 2)
    focus_mask = circular_region_mask(
        grid,
        radius_um=FOCUS_RADIUS_PX * grid.dx_um,
        center_xy=center_xy,
    )

    total_detector = fx.Detector()
    focus_detector = fx.Detector(region_mask=focus_mask)

    # %% Evaluation
    total_power = float(np.asarray(total_detector.measure(field_out)))
    focus_power = float(np.asarray(focus_detector.measure(field_out)))
    focus_fraction = focus_power / total_power if total_power > 0 else 0.0
    peak_intensity = float(np.max(intensity))
    center_row = intensity[center_xy[1], :]
    x_um = (np.arange(grid.nx) - (grid.nx - 1) / 2.0) * grid.dx_um
    aperture_mask = np.asarray(
        circular_region_mask(grid, radius_um=aperture_diameter_um / 2.0),
        dtype=np.float32,
    )

    print(f"planned propagator: {propagator.__class__.__name__}")
    print(f"grid: {grid.nx} x {grid.ny}, dx = {grid.dx_um:.3f} um")
    print(f"wavelength: {WAVELENGTH_UM:.3f} um")
    print(f"distance: {DISTANCE_UM:.3f} um")
    print(f"aperture diameter: {aperture_diameter_um:.3f} um")
    print(f"peak intensity: {peak_intensity:.6f}")
    print(f"focus power fraction: {focus_fraction:.6f}")

    # %% Plot Results
    if PLOT:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))

        axes[0].imshow(aperture_mask, cmap="gray")
        axes[0].set_title("Lens Aperture")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        focus_im = axes[1].imshow(intensity, cmap="inferno")
        axes[1].set_title(f"Focal Intensity\n{propagator.__class__.__name__}")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        plt.colorbar(focus_im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].plot(x_um, center_row / np.maximum(np.max(center_row), 1e-12))
        axes[2].set_title("Center-Row Profile")
        axes[2].set_xlabel("x (um)")
        axes[2].set_ylabel("Normalized intensity")
        axes[2].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(PLOT_PATH, dpi=150)
        plt.close(fig)
        print(f"saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
