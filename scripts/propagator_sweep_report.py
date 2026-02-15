#!/usr/bin/env python3
"""Sweep propagation/grid settings and report ASM vs RS speed-accuracy trade-offs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import j1

from fouriax.optics import (
    ASMPropagator,
    Field,
    Grid,
    PropagationPolicy,
    RSPropagator,
    SamplingPlanner,
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


def timed_propagate(
    propagator,
    field: Field,
    distance_um: float,
    repeats: int,
    warmup: int,
) -> tuple[Field, float]:
    out = None
    for _ in range(warmup):
        out = propagator.propagate(field, distance_um=distance_um)
        out.data.block_until_ready()

    times_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = propagator.propagate(field, distance_um=distance_um)
        out.data.block_until_ready()
        times_ms.append(1e3 * (time.perf_counter() - t0))

    assert out is not None
    return out, float(np.median(times_ms))


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def run_sweep(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    policy = PropagationPolicy()
    wavelength_um = args.wavelength_um
    spectrum = Spectrum.from_scalar(wavelength_um)
    records: list[dict[str, float | int | str]] = []

    method_factories = {
        "asm_base": lambda: ASMPropagator(use_sampling_planner=False),
        "asm_planned_s2": lambda: ASMPropagator(
            use_sampling_planner=True,
            sampling_planner=SamplingPlanner(safety_factor=2.0, min_padding_factor=2.0),
        ),
        "rs_base": lambda: RSPropagator(use_sampling_planner=False),
    }
    for s in args.rs_safety_factors:
        key = f"rs_planned_s{s:g}"
        method_factories[key] = (
            lambda sf=s: RSPropagator(
                use_sampling_planner=True,
                sampling_planner=SamplingPlanner(
                    safety_factor=sf,
                    min_padding_factor=args.padding_factor,
                ),
            )
        )

    for n in args.n_values:
        for dx_um in args.dx_values_um:
            grid = Grid.from_extent(nx=n, ny=n, dx_um=dx_um, dy_um=dx_um)
            z_crit_um = policy.critical_distance_um(grid=grid, spectrum=spectrum)
            aperture_diameter_um = args.aperture_fraction * (n * dx_um)

            for ratio in args.z_ratios:
                distance_um = ratio * z_crit_um
                lens = ThinLensLayer(
                    focal_length_um=distance_um,
                    aperture_diameter_um=aperture_diameter_um,
                )
                field_in = Field.plane_wave(grid=grid, spectrum=spectrum)
                field_lens = lens.forward(field_in)
                policy_decision = policy.choose(
                    grid=grid,
                    spectrum=spectrum,
                    distance_um=distance_um,
                )

                method_results: dict[str, tuple[float, float]] = {}
                for method_name, factory in method_factories.items():
                    propagator = factory()
                    out, runtime_ms = timed_propagate(
                        propagator=propagator,
                        field=field_lens,
                        distance_um=distance_um,
                        repeats=args.repeats,
                        warmup=args.warmup,
                    )
                    mae = mae_vs_airy(
                        intensity_2d=np.asarray(out.intensity()[0]),
                        wavelength_um=wavelength_um,
                        focal_um=distance_um,
                        aperture_diameter_um=aperture_diameter_um,
                        dx_um=dx_um,
                    )
                    method_results[method_name] = (runtime_ms, mae)

                best_accuracy_method = min(method_results, key=lambda k: method_results[k][1])
                fastest_method = min(method_results, key=lambda k: method_results[k][0])

                for method_name, (runtime_ms, mae) in method_results.items():
                    records.append(
                        {
                            "n": n,
                            "dx_um": dx_um,
                            "z_ratio": ratio,
                            "z_um": distance_um,
                            "z_crit_um": z_crit_um,
                            "policy_method": policy_decision.method,
                            "method": method_name,
                            "runtime_ms_median": runtime_ms,
                            "mae_vs_airy": mae,
                            "is_best_accuracy": int(method_name == best_accuracy_method),
                            "is_fastest": int(method_name == fastest_method),
                        }
                    )
    return records


def print_summary(records: list[dict[str, float | int | str]]) -> None:
    print("Sweep summary (lower MAE and lower runtime are better):")
    print(
        " n   dx_um  z/z_crit  method         runtime_ms  mae       policy  best_acc  fastest"
    )
    for r in records:
        print(
            f"{int(r['n']):3d}  "
            f"{float(r['dx_um']):5.2f}   "
            f"{float(r['z_ratio']):6.2f}   "
            f"{str(r['method']):13s} "
            f"{float(r['runtime_ms_median']):10.3f} "
            f"{float(r['mae_vs_airy']):8.5f}  "
            f"{str(r['policy_method']):6s}    "
            f"{int(r['is_best_accuracy']):1d}        "
            f"{int(r['is_fastest']):1d}"
        )

    grouped: dict[tuple[int, float, float], list[dict[str, float | int | str]]] = {}
    for r in records:
        key = (int(r["n"]), float(r["dx_um"]), float(r["z_ratio"]))
        grouped.setdefault(key, []).append(r)
    matches = 0
    for rows in grouped.values():
        policy = str(rows[0]["policy_method"])
        best = next(row for row in rows if int(row["is_best_accuracy"]) == 1)
        if str(best["method"]).startswith(policy):
            matches += 1
    total = len(grouped)
    print(f"\nPolicy matched best-accuracy family in {matches}/{total} cases.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavelength-um", type=float, default=0.532)
    parser.add_argument("--n-values", type=parse_int_list, default=parse_int_list("128,256"))
    parser.add_argument(
        "--dx-values-um",
        type=parse_float_list,
        default=parse_float_list("1.0,2.0"),
    )
    parser.add_argument(
        "--z-ratios",
        type=parse_float_list,
        default=parse_float_list("0.25,0.5,1.0,2.0,4.0"),
    )
    parser.add_argument(
        "--rs-safety-factors",
        type=parse_float_list,
        default=parse_float_list("1.5,2.0,3.0"),
    )
    parser.add_argument("--padding-factor", type=float, default=2.0)
    parser.add_argument("--aperture-fraction", type=float, default=0.12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    records = run_sweep(args)
    print_summary(records)

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "propagator_sweep_report.csv"
    json_path = out_dir / "propagator_sweep_report.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
