#!/usr/bin/env python3
"""Visual benchmark for thin-lens focus with RS propagation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import j1

from fouriax.optics import Field, Grid, RSPropagator, Spectrum, ThinLens


def airy_profile(
    r_um: np.ndarray,
    wavelength_um: float,
    focal_um: float,
    diameter_um: float,
) -> np.ndarray:
    alpha = np.pi * diameter_um * r_um / (wavelength_um * focal_um)
    profile = np.ones_like(alpha, dtype=np.float64)
    nonzero = alpha != 0.0
    profile[nonzero] = (2.0 * j1(alpha[nonzero]) / alpha[nonzero]) ** 2
    return profile


def run_benchmark() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    wavelength_um = 0.532
    focal_um = 20_000.0
    aperture_diameter_um = 400.0

    grid = Grid.from_extent(nx=256, ny=256, dx_um=2.0, dy_um=2.0)
    spectrum = Spectrum.from_scalar(wavelength_um)
    field_in = Field.plane_wave(grid=grid, spectrum=spectrum)

    lens = ThinLens(
        focal_length_um=focal_um,
        aperture_diameter_um=aperture_diameter_um,
    )
    propagator = RSPropagator()

    field_focus = propagator.propagate(lens.forward(field_in), distance_um=focal_um)

    intensity = np.asarray(field_focus.intensity()[0])
    center_y = grid.ny // 2
    center_x = grid.nx // 2
    row = intensity[center_y, :]
    row = row / np.max(row)

    x_um = (np.arange(grid.nx) - (grid.nx - 1) / 2.0) * grid.dx_um
    r_um = np.abs(x_um)

    expected_first_zero_um = 1.22 * wavelength_um * focal_um / aperture_diameter_um
    search_lo = int(max(1, np.floor(0.7 * expected_first_zero_um / grid.dx_um)))
    search_hi = int(np.ceil(1.3 * expected_first_zero_um / grid.dx_um))
    window = row[center_x + search_lo : center_x + search_hi]
    min_idx_local = int(np.argmin(window))
    first_zero_px = search_lo + min_idx_local
    measured_first_zero_um = first_zero_px * grid.dx_um

    r_max = 0.9 * expected_first_zero_um
    n_compare = int(np.floor(r_max / grid.dx_um))
    sim_profile = row[center_x : center_x + n_compare]
    r_compare = r_um[center_x : center_x + n_compare]
    airy = airy_profile(
        r_compare,
        wavelength_um=wavelength_um,
        focal_um=focal_um,
        diameter_um=aperture_diameter_um,
    )
    mean_abs_err = float(np.mean(np.abs(sim_profile - airy)))
    return (
        r_compare,
        sim_profile,
        airy,
        expected_first_zero_um,
        measured_first_zero_um,
        mean_abs_err,
    )


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    (
        r_um,
        sim_profile,
        airy,
        expected_first_zero_um,
        measured_first_zero_um,
        mean_abs_err,
    ) = run_benchmark()

    rel_err = abs(measured_first_zero_um - expected_first_zero_um) / expected_first_zero_um
    print(f"Expected first zero (um): {expected_first_zero_um:.4f}")
    print(f"Measured first zero (um): {measured_first_zero_um:.4f}")
    print(f"Relative first-zero error: {rel_err:.4%}")
    print(f"Mean absolute profile error: {mean_abs_err:.6f}")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(r_um, sim_profile, label="RS simulation", linewidth=2.0)
    ax.plot(r_um, airy, label="Airy theory", linewidth=2.0, linestyle="--")
    ax.axvline(expected_first_zero_um, color="black", linestyle=":", label="Expected first zero")
    ax.set_title("Thin Lens Focus: RS vs Airy")
    ax.set_xlabel("Radius (um)")
    ax.set_ylabel("Normalized intensity")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()

    out = Path("artifacts")
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "lens_benchmark_profile.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"Saved plot: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
