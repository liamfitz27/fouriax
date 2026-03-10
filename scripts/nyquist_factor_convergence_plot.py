#!/usr/bin/env python3
"""Sweep nyquist_factor and show convergence toward a high-factor reference."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import j1

from fouriax.optics import ASMPropagator, Field, Grid, RSPropagator, Spectrum, ThinLens
from fouriax.optics.propagation import critical_distance_um, recommend_nyquist_grid


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


def radial_row_profile(intensity_2d: np.ndarray, dx_um: float) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = intensity_2d.shape
    cy = ny // 2
    cx = nx // 2
    row = intensity_2d[cy, :]
    row = row / np.max(row)
    x_um = (np.arange(nx) - (nx - 1) / 2.0) * dx_um
    r_um = np.abs(x_um)
    return r_um[cx:], row[cx:]


def mae_vs_airy(
    intensity_2d: np.ndarray,
    wavelength_um: float,
    focal_um: float,
    aperture_diameter_um: float,
    dx_um: float,
) -> float:
    r_um, profile = radial_row_profile(intensity_2d, dx_um)
    expected_first_zero_um = 1.22 * wavelength_um * focal_um / aperture_diameter_um
    r_max = 0.9 * expected_first_zero_um
    n_compare = max(8, int(np.floor(r_max / dx_um)))
    r_compare = r_um[:n_compare]
    sim_profile = profile[:n_compare]
    airy = airy_profile(
        r_compare,
        wavelength_um=wavelength_um,
        focal_um=focal_um,
        diameter_um=aperture_diameter_um,
    )
    return float(np.mean(np.abs(sim_profile - airy)))


def rel_l2_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    denom = np.linalg.norm(reference.ravel())
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm((candidate - reference).ravel()) / denom)


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def build_propagator(method: str):
    if method == "asm":
        return ASMPropagator
    if method == "rs":
        return RSPropagator
    raise ValueError(f"unsupported method: {method}")


def timed_propagate(propagator, field: Field, repeats: int, warmup: int) -> tuple[Field, float]:
    out = None
    for _ in range(warmup):
        out = propagator.forward(field)
        out.data.block_until_ready()

    times_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = propagator.forward(field)
        out.data.block_until_ready()
        times_ms.append(1e3 * (time.perf_counter() - t0))

    assert out is not None
    return out, float(np.median(times_ms))


def run_sweep(args: argparse.Namespace) -> tuple[list[dict[str, float | int | str]], np.ndarray]:
    spectrum = Spectrum.from_scalar(args.wavelength_um)
    grid = Grid.from_extent(nx=args.n, ny=args.n, dx_um=args.dx_um, dy_um=args.dx_um)
    z_crit_um = critical_distance_um(grid=grid, spectrum=spectrum)
    distance_um = args.z_ratio * z_crit_um
    aperture_diameter_um = args.aperture_fraction * (args.n * args.dx_um)

    field_in = Field.plane_wave(grid=grid, spectrum=spectrum)
    lens = ThinLens(
        focal_length_um=distance_um,
        aperture_diameter_um=aperture_diameter_um,
    )
    field_lens = lens.forward(field_in)

    propagator_cls = build_propagator(args.method)
    reference_layer = propagator_cls(
        distance_um=distance_um,
        use_sampling_planner=True,
        nyquist_factor=args.reference_factor,
        min_padding_factor=args.padding_factor,
        warn_on_regime_mismatch=False,
    )
    reference_out = reference_layer.forward(field_lens)
    reference_out.data.block_until_ready()
    reference_intensity = np.asarray(reference_out.intensity()[0])

    records: list[dict[str, float | int | str]] = []
    for factor in args.nyquist_factors:
        target_grid = recommend_nyquist_grid(
            grid=grid,
            spectrum=spectrum,
            nyquist_factor=factor,
            min_padding_factor=args.padding_factor,
        )
        layer = propagator_cls(
            distance_um=distance_um,
            use_sampling_planner=True,
            nyquist_factor=factor,
            min_padding_factor=args.padding_factor,
            warn_on_regime_mismatch=False,
        )
        out, runtime_ms = timed_propagate(
            propagator=layer,
            field=field_lens,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        intensity = np.asarray(out.intensity()[0])
        records.append(
            {
                "method": args.method,
                "n": args.n,
                "dx_um": args.dx_um,
                "wavelength_um": args.wavelength_um,
                "z_ratio": args.z_ratio,
                "z_um": distance_um,
                "z_crit_um": z_crit_um,
                "aperture_fraction": args.aperture_fraction,
                "aperture_diameter_um": aperture_diameter_um,
                "padding_factor": args.padding_factor,
                "nyquist_factor": factor,
                "reference_factor": args.reference_factor,
                "runtime_ms_median": runtime_ms,
                "mae_vs_airy": mae_vs_airy(
                    intensity_2d=intensity,
                    wavelength_um=args.wavelength_um,
                    focal_um=distance_um,
                    aperture_diameter_um=aperture_diameter_um,
                    dx_um=args.dx_um,
                ),
                "rel_l2_vs_reference": rel_l2_error(intensity, reference_intensity),
                "planned_nx": target_grid.nx,
                "planned_ny": target_grid.ny,
                "planned_dx_um": target_grid.dx_um,
                "planned_dy_um": target_grid.dy_um,
            }
        )
    return records, reference_intensity


def print_summary(records: list[dict[str, float | int | str]]) -> None:
    print("Nyquist convergence sweep (lower error is better):")
    print(" factor  runtime_ms  rel_l2_vs_ref  mae_vs_airy  planned_grid   planned_dx")
    for r in records:
        print(
            f" {float(r['nyquist_factor']):6.2f}  "
            f"{float(r['runtime_ms_median']):10.3f}  "
            f"{float(r['rel_l2_vs_reference']):13.6e}  "
            f"{float(r['mae_vs_airy']):11.6f}  "
            f"{int(r['planned_ny']):4d}x{int(r['planned_nx']):4d}   "
            f"{float(r['planned_dx_um']):9.5f}"
        )

    best = min(records, key=lambda r: float(r["rel_l2_vs_reference"]))
    print(
        "\nBest agreement with reference: "
        f"nyquist_factor={float(best['nyquist_factor']):.2f}, "
        f"rel_l2_vs_reference={float(best['rel_l2_vs_reference']):.6e}"
    )


def save_plot(
    records: list[dict[str, float | int | str]],
    out_path: Path,
    method: str,
    reference_factor: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    factors = np.asarray([float(r["nyquist_factor"]) for r in records], dtype=np.float64)
    mae = np.asarray([float(r["mae_vs_airy"]) for r in records], dtype=np.float64)
    runtime = np.asarray([float(r["runtime_ms_median"]) for r in records], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))

    # axes[0].plot(factors, rel_l2, marker="o", linewidth=2.0, label="Rel. L2 vs reference")
    axes[0].plot(factors, mae, marker="s", linewidth=2.0, label="MAE vs Airy")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Nyquist factor")
    axes[0].set_ylabel("Error")
    axes[0].set_title(f"{method.upper()} convergence")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(factors, runtime, marker="o", linewidth=2.0, color="#d62728")
    axes[1].set_xlabel("Nyquist factor")
    axes[1].set_ylabel("Median runtime (ms)")
    axes[1].set_title(f"Runtime vs factor (reference={reference_factor:g})")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"Saved plot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("asm", "rs"), default="rs")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--dx-um", type=float, default=1.0)
    parser.add_argument("--wavelength-um", type=float, default=0.532)
    parser.add_argument("--z-ratio", type=float, default=1.0)
    parser.add_argument("--aperture-fraction", type=float, default=0.12)
    parser.add_argument("--padding-factor", type=float, default=2.0)
    parser.add_argument(
        "--nyquist-factors",
        type=parse_float_list,
        default=parse_float_list("0.5,1.0,2.0,4.0,8.0"),
    )
    parser.add_argument("--reference-factor", type=float, default=8.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    records, _reference_intensity = run_sweep(args)
    print_summary(records)

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"nyquist_factor_convergence_{args.method}"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    plot_path = out_dir / f"{stem}.png"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    save_plot(records, plot_path, args.method, args.reference_factor)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
