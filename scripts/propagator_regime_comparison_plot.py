#!/usr/bin/env python3
"""Compare ASM vs RS against Airy theory in two propagation regimes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import j1

from fouriax.optics import (
    ASMPropagator,
    Field,
    Grid,
    PropagationPolicy,
    RSPropagator,
    Spectrum,
    ThinLensLayer,
)


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


def radial_row_profile(
    intensity_2d: np.ndarray,
    dx_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = intensity_2d.shape
    cy = ny // 2
    cx = nx // 2
    row = intensity_2d[cy, :]
    row = row / np.max(row)
    x_um = (np.arange(nx) - (nx - 1) / 2.0) * dx_um
    r_um = np.abs(x_um)
    return r_um[cx:], row[cx:]


def run_case(
    grid: Grid,
    spectrum: Spectrum,
    aperture_diameter_um: float,
    distance_um: float,
    policy: PropagationPolicy,
) -> dict[str, float | str | np.ndarray]:
    wavelength_um = float(spectrum.wavelengths_um[0])
    field_in = Field.plane_wave(grid=grid, spectrum=spectrum)
    lens = ThinLensLayer(
        focal_length_um=distance_um,
        aperture_diameter_um=aperture_diameter_um,
    )
    field_lens = lens.forward(field_in)

    asm = ASMPropagator()
    rs = RSPropagator()
    out_asm = asm.propagate(field_lens, distance_um=distance_um)
    out_rs = rs.propagate(field_lens, distance_um=distance_um)

    r_um, profile_asm = radial_row_profile(np.asarray(out_asm.intensity()[0]), grid.dx_um)
    _, profile_rs = radial_row_profile(np.asarray(out_rs.intensity()[0]), grid.dx_um)

    expected_first_zero_um = 1.22 * wavelength_um * distance_um / aperture_diameter_um
    r_max = 0.9 * expected_first_zero_um
    n_compare = max(8, int(np.floor(r_max / grid.dx_um)))
    r_compare = r_um[:n_compare]
    profile_asm = profile_asm[:n_compare]
    profile_rs = profile_rs[:n_compare]
    airy = airy_profile(r_compare, wavelength_um, distance_um, aperture_diameter_um)

    mae_asm = float(np.mean(np.abs(profile_asm - airy)))
    mae_rs = float(np.mean(np.abs(profile_rs - airy)))
    best_method = "asm" if mae_asm <= mae_rs else "rs"
    decision = policy.choose(grid=grid, spectrum=spectrum, distance_um=distance_um)

    return {
        "r_um": r_compare,
        "airy": airy,
        "asm": profile_asm,
        "rs": profile_rs,
        "mae_asm": mae_asm,
        "mae_rs": mae_rs,
        "z_um": distance_um,
        "z_crit_um": decision.critical_distance_um,
        "policy": decision.method,
        "best": best_method,
    }


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    grid = Grid.from_extent(nx=256, ny=256, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    aperture_diameter_um = 30.0
    policy = PropagationPolicy()
    z_crit_um = policy.critical_distance_um(grid=grid, spectrum=spectrum)

    cases = [
        ("Near-field regime (z < z_crit)", 0.5 * z_crit_um),
        ("Far-field regime (z > z_crit)", 2.0 * z_crit_um),
    ]
    results = [
        run_case(
            grid=grid,
            spectrum=spectrum,
            aperture_diameter_um=aperture_diameter_um,
            distance_um=z_um,
            policy=policy,
        )
        for _, z_um in cases
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    for ax, (title, _), result in zip(axes, cases, results, strict=True):
        r_um = result["r_um"]
        airy = result["airy"]
        asm = result["asm"]
        rs = result["rs"]
        method = str(result["policy"])
        chosen = asm if method == "asm" else rs

        ax.plot(r_um, airy, label="Airy theory", linewidth=2.2, color="black")
        ax.plot(r_um, asm, label="ASM", linewidth=2.0, color="#1f77b4")
        ax.plot(r_um, rs, label="RS", linewidth=2.0, color="#d62728")
        ax.plot(
            r_um,
            chosen,
            label=f"Policy choice ({method.upper()})",
            linewidth=2.5,
            color="#2ca02c",
            linestyle="--",
            alpha=0.85,
        )

        text = (
            f"z={result['z_um']:.1f} um, z_crit={result['z_crit_um']:.1f} um\n"
            f"MAE(ASM)={result['mae_asm']:.4f}, MAE(RS)={result['mae_rs']:.4f}\n"
            f"Policy={result['policy'].upper()}, Best={result['best'].upper()}"
        )
        ax.text(
            0.03,
            0.04,
            text,
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )
        ax.set_title(title)
        ax.set_xlabel("Radius (um)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Normalized intensity")
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].legend(loc="upper right")
    fig.suptitle("Propagation Regime Comparison: ASM vs RS vs Airy", fontsize=13)
    fig.tight_layout()

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "propagator_regime_comparison.png"
    fig.savefig(out_path, dpi=160)
    print(f"Saved plot: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
