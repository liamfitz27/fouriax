#!/usr/bin/env python3
"""Benchmark library ASM PSF generation against a handwritten JAX kernel.

This benchmark isolates the ASM propagation path itself. The source field
after the thin lens is built once outside the timed region, then reused for:

- the library path: ``ASMPropagator(use_sampling_planner=False)``
- the manual path: precomputed transfer stack + ``jax.jit`` + ``jax.vmap``

The manual implementation is intentionally optimized for best-case throughput,
not structural parity with the library API.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.special import j1

from fouriax.optics import Field, Grid, Spectrum, ThinLens
from fouriax.optics.propagation import critical_distance_um, plan_propagation
from fouriax.optim import focal_spot_loss


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
    row = row / np.maximum(np.max(row), 1e-12)
    x_um = (np.arange(nx) - (nx - 1) / 2.0) * dx_um
    r_um = np.abs(x_um)
    return r_um[cx:], row[cx:]


def mean_mae_vs_airy(
    intensity_stack: np.ndarray,
    wavelengths_um: np.ndarray,
    focal_um: float,
    aperture_diameter_um: float,
    dx_um: float,
) -> float:
    maes: list[float] = []
    for i, wavelength_um in enumerate(wavelengths_um):
        r_um, profile = radial_row_profile(intensity_stack[i], dx_um)
        expected_first_zero_um = 1.22 * wavelength_um * focal_um / aperture_diameter_um
        r_max = 0.9 * expected_first_zero_um
        n_compare = max(8, int(np.floor(r_max / dx_um)))
        r_compare = r_um[:n_compare]
        sim_profile = profile[:n_compare]
        airy = airy_profile(
            r_compare,
            wavelength_um=float(wavelength_um),
            focal_um=focal_um,
            diameter_um=aperture_diameter_um,
        )
        maes.append(float(np.mean(np.abs(sim_profile - airy))))
    return float(np.mean(maes))


def rel_l2_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    denom = np.linalg.norm(reference.ravel())
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm((candidate - reference).ravel()) / denom)


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_spectrum(count: int, center_um: float, span_um: float) -> Spectrum:
    if count <= 0:
        raise ValueError("wavelength count must be positive")
    if count == 1:
        return Spectrum.from_scalar(center_um)
    half_span = span_um / 2.0
    wavelengths_um = np.linspace(center_um - half_span, center_um + half_span, count)
    return Spectrum.from_array(wavelengths_um)


def build_manual_transfer_stack(
    *,
    grid: Grid,
    wavelengths_um: jnp.ndarray,
    distance_um: float,
) -> jnp.ndarray:
    fx, fy = grid.frequency_grid()
    z = jnp.asarray(distance_um, dtype=jnp.float32)
    wavelengths_um = jnp.asarray(wavelengths_um, dtype=jnp.float32)

    def transfer_for_wavelength(wavelength_um: jnp.ndarray) -> jnp.ndarray:
        wl = jnp.asarray(wavelength_um, dtype=jnp.float32)
        k = 2.0 * jnp.pi / wl
        argument = 1.0 - (wl * fx) ** 2 - (wl * fy) ** 2
        kz = k * jnp.sqrt(argument.astype(jnp.complex64))
        return jnp.exp(1j * kz * z).astype(jnp.complex64)

    return jax.vmap(transfer_for_wavelength)(wavelengths_um)


def build_manual_psf_fn(
    *,
    field_after_lens: jnp.ndarray,
    grid: Grid,
    wavelengths_um: jnp.ndarray,
    distance_um: float,
):
    transfer_stack = build_manual_transfer_stack(
        grid=grid,
        wavelengths_um=wavelengths_um,
        distance_um=distance_um,
    )
    field_after_lens = jnp.asarray(field_after_lens, dtype=jnp.complex64)

    def propagate_one(field_2d: jnp.ndarray, transfer_2d: jnp.ndarray) -> jnp.ndarray:
        field_k = jnp.fft.fftn(field_2d, axes=(-2, -1))
        propagated = jnp.fft.ifftn(field_k * transfer_2d, axes=(-2, -1))
        return jnp.abs(propagated) ** 2

    return jax.jit(lambda: jax.vmap(propagate_one)(field_after_lens, transfer_stack))


def first_call_and_median_ms(fn, repeats: int) -> tuple[jnp.ndarray, float, float]:
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    first_call_ms = 1e3 * (time.perf_counter() - t0)

    times_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times_ms.append(1e3 * (time.perf_counter() - t0))
    return out, first_call_ms, float(np.median(times_ms))


def circular_aperture(grid: Grid, diameter_um: float) -> jnp.ndarray:
    x, y = grid.spatial_grid()
    radius_um = diameter_um / 2.0
    return ((x * x + y * y) <= radius_um * radius_um).astype(jnp.float32)


def dx_um_for_nyquist(*, wavelength_um: float, nyquist_factor: float) -> float:
    if wavelength_um <= 0.0:
        raise ValueError("wavelength_um must be positive")
    if nyquist_factor <= 0.0:
        raise ValueError("nyquist_factor must be positive")
    return wavelength_um / (2.0 * nyquist_factor)


def aperture_diameter_for_fraction(*, grid_width_um: float, aperture_fraction: float) -> float:
    if grid_width_um <= 0.0:
        raise ValueError("grid_width_um must be positive")
    if not 0.0 < aperture_fraction <= 1.0:
        raise ValueError("aperture_fraction must be in (0, 1]")
    return aperture_fraction * grid_width_um


def run_psf_case(
    *,
    n: int,
    dx_um: float,
    wavelength_count: int,
    center_wavelength_um: float,
    wavelength_span_um: float,
    z_ratio: float,
    aperture_fraction: float,
    repeats: int,
) -> dict[str, float | int | str]:
    grid = Grid.from_extent(nx=n, ny=n, dx_um=dx_um, dy_um=dx_um)
    spectrum = build_spectrum(wavelength_count, center_wavelength_um, wavelength_span_um)
    z_crit_um = critical_distance_um(grid=grid, spectrum=spectrum)
    distance_um = z_ratio * z_crit_um
    aperture_diameter_um = aperture_fraction * (n * dx_um)

    field_in = Field.plane_wave(grid=grid, spectrum=spectrum)
    lens = ThinLens(
        focal_length_um=distance_um,
        aperture_diameter_um=aperture_diameter_um,
    )
    field_lens = lens.forward(field_in)

    propagator = plan_propagation(
        mode="asm",
        grid=grid,
        spectrum=spectrum,
        distance_um=distance_um,
        use_sampling_planner=False,
        warn_on_regime_mismatch=False,
    )

    def library_psf():
        return propagator.forward(field_lens).intensity()

    manual_psf = build_manual_psf_fn(
        field_after_lens=field_lens.data,
        grid=grid,
        wavelengths_um=field_lens.spectrum.wavelengths_um,
        distance_um=distance_um,
    )

    library_out, library_first_ms, library_median_ms = first_call_and_median_ms(
        library_psf,
        repeats=repeats,
    )
    manual_out, manual_first_ms, manual_median_ms = first_call_and_median_ms(
        manual_psf,
        repeats=repeats,
    )

    library_intensity = np.asarray(library_out)
    manual_intensity = np.asarray(manual_out)
    wavelengths_um = np.asarray(field_lens.spectrum.wavelengths_um, dtype=np.float64)

    return {
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "n": n,
        "dx_um": dx_um,
        "wavelength_count": wavelength_count,
        "center_wavelength_um": center_wavelength_um,
        "wavelength_span_um": wavelength_span_um,
        "z_ratio": z_ratio,
        "z_um": distance_um,
        "z_crit_um": z_crit_um,
        "aperture_fraction": aperture_fraction,
        "aperture_diameter_um": aperture_diameter_um,
        "library_first_call_ms": library_first_ms,
        "library_runtime_ms_median": library_median_ms,
        "manual_first_call_ms": manual_first_ms,
        "manual_runtime_ms_median": manual_median_ms,
        "manual_speedup_vs_library": library_median_ms / manual_median_ms,
        "rel_l2_intensity_manual_vs_library": rel_l2_error(manual_intensity, library_intensity),
        "max_abs_intensity_diff": float(np.max(np.abs(manual_intensity - library_intensity))),
        "library_mae_vs_airy": mean_mae_vs_airy(
            library_intensity,
            wavelengths_um,
            distance_um,
            aperture_diameter_um,
            dx_um,
        ),
        "manual_mae_vs_airy": mean_mae_vs_airy(
            manual_intensity,
            wavelengths_um,
            distance_um,
            aperture_diameter_um,
            dx_um,
        ),
    }


def print_psf_summary(records: list[dict[str, float | int | str]]) -> None:
    print("ASM library vs manual JAX PSF benchmark:")
    print(
        " n  wl  z/z_crit  lib_ms  manual_ms  speedup  rel_l2_intensity  "
        "lib_mae  manual_mae"
    )
    for r in records:
        print(
            f"{int(r['n']):3d}  "
            f"{int(r['wavelength_count']):2d}  "
            f"{float(r['z_ratio']):8.2f}  "
            f"{float(r['library_runtime_ms_median']):7.3f}  "
            f"{float(r['manual_runtime_ms_median']):9.3f}  "
            f"{float(r['manual_speedup_vs_library']):7.2f}  "
            f"{float(r['rel_l2_intensity_manual_vs_library']):16.6e}  "
            f"{float(r['library_mae_vs_airy']):8.6f}  "
            f"{float(r['manual_mae_vs_airy']):10.6f}"
        )


def run_optimization(
    *,
    init_params: jnp.ndarray,
    optimizer: optax.GradientTransformation,
    step_fn,
    eval_intensity_fn,
    steps: int,
) -> tuple[jnp.ndarray, np.ndarray, np.ndarray, float, float, float]:
    params = init_params
    opt_state = optimizer.init(params)

    t0 = time.perf_counter()
    params, opt_state, loss = step_fn(params, opt_state)
    jax.block_until_ready((params, opt_state, loss))
    first_step_ms = 1e3 * (time.perf_counter() - t0)

    history = [float(loss)]
    step_times_ms: list[float] = []
    for _ in range(steps - 1):
        t0 = time.perf_counter()
        params, opt_state, loss = step_fn(params, opt_state)
        jax.block_until_ready((params, opt_state, loss))
        step_times_ms.append(1e3 * (time.perf_counter() - t0))
        history.append(float(loss))

    intensity = np.asarray(eval_intensity_fn(params))
    median_step_ms = float(np.median(step_times_ms)) if step_times_ms else 0.0
    total_runtime_ms = first_step_ms + float(np.sum(step_times_ms))
    return params, intensity, np.asarray(history), first_step_ms, median_step_ms, total_runtime_ms


def run_lens_optimization_case(
    *,
    n: int,
    wavelength_um: float,
    distance_um: float,
    window_px: int,
    lr: float,
    steps: int,
    seed: int,
    dx_um: float | None = None,
    nyquist_factor: float = 2.0,
    aperture_diameter_um: float | None = None,
    aperture_fraction: float = 0.35,
) -> dict[str, float | int | str]:
    dx_um = (
        dx_um
        if dx_um is not None
        else dx_um_for_nyquist(
            wavelength_um=wavelength_um,
            nyquist_factor=nyquist_factor,
        )
    )
    grid_width_um = n * dx_um
    aperture_diameter_um = (
        aperture_diameter_um
        if aperture_diameter_um is not None
        else aperture_diameter_for_fraction(
            grid_width_um=grid_width_um,
            aperture_fraction=aperture_fraction,
        )
    )
    grid = Grid.from_extent(nx=n, ny=n, dx_um=dx_um, dy_um=dx_um)
    spectrum = Spectrum.from_scalar(wavelength_um)
    field_in = Field.plane_wave(grid=grid, spectrum=spectrum)
    aperture = circular_aperture(grid, aperture_diameter_um)
    aperture_broadcast = aperture[None, :, :]
    target_xy = (grid.nx // 2, grid.ny // 2)
    key = jax.random.PRNGKey(seed)
    init_params = 0.1 * jax.random.normal(key, (grid.ny, grid.nx))

    propagator = plan_propagation(
        mode="asm",
        grid=grid,
        spectrum=spectrum,
        distance_um=distance_um,
        use_sampling_planner=False,
        warn_on_regime_mismatch=False,
    )

    def phase_from_raw(raw_phase_map: jnp.ndarray) -> jnp.ndarray:
        return 2.0 * jnp.pi * jax.nn.sigmoid(raw_phase_map)

    def library_intensity(raw_phase_map: jnp.ndarray) -> jnp.ndarray:
        phase_limited = phase_from_raw(raw_phase_map)
        field_mod = field_in.apply_phase(phase_limited[None, :, :]).apply_amplitude(
            aperture_broadcast
        )
        return propagator.forward(field_mod).intensity()

    def library_loss(raw_phase_map: jnp.ndarray) -> jnp.ndarray:
        return focal_spot_loss(
            intensity=library_intensity(raw_phase_map),
            target_xy=target_xy,
            window_px=window_px,
        )

    transfer_stack = build_manual_transfer_stack(
        grid=grid,
        wavelengths_um=field_in.spectrum.wavelengths_um,
        distance_um=distance_um,
    )
    base_field = jnp.asarray(field_in.data, dtype=jnp.complex64)

    def manual_intensity(raw_phase_map: jnp.ndarray) -> jnp.ndarray:
        phase_limited = phase_from_raw(raw_phase_map)
        modulated = base_field * aperture_broadcast * jnp.exp(1j * phase_limited)[None, :, :]
        field_k = jnp.fft.fftn(modulated, axes=(-2, -1))
        propagated = jnp.fft.ifftn(field_k * transfer_stack, axes=(-2, -1))
        return jnp.abs(propagated) ** 2

    def manual_loss(raw_phase_map: jnp.ndarray) -> jnp.ndarray:
        return focal_spot_loss(
            intensity=manual_intensity(raw_phase_map),
            target_xy=target_xy,
            window_px=window_px,
        )

    optimizer = optax.adam(lr)

    @jax.jit
    def library_step(raw_phase_map: jnp.ndarray, opt_state):
        loss, grads = jax.value_and_grad(library_loss)(raw_phase_map)
        updates, opt_state = optimizer.update(grads, opt_state, raw_phase_map)
        raw_phase_map = optax.apply_updates(raw_phase_map, updates)
        return raw_phase_map, opt_state, loss

    @jax.jit
    def manual_step(raw_phase_map: jnp.ndarray, opt_state):
        loss, grads = jax.value_and_grad(manual_loss)(raw_phase_map)
        updates, opt_state = optimizer.update(grads, opt_state, raw_phase_map)
        raw_phase_map = optax.apply_updates(raw_phase_map, updates)
        return raw_phase_map, opt_state, loss

    library_eval = jax.jit(library_intensity)
    manual_eval = jax.jit(manual_intensity)

    (
        library_params,
        library_final_intensity,
        library_history,
        library_first_step_ms,
        library_step_ms_median,
        library_total_runtime_ms,
    ) = run_optimization(
        init_params=init_params,
        optimizer=optimizer,
        step_fn=library_step,
        eval_intensity_fn=library_eval,
        steps=steps,
    )
    (
        manual_params,
        manual_final_intensity,
        manual_history,
        manual_first_step_ms,
        manual_step_ms_median,
        manual_total_runtime_ms,
    ) = run_optimization(
        init_params=init_params,
        optimizer=optimizer,
        step_fn=manual_step,
        eval_intensity_fn=manual_eval,
        steps=steps,
    )

    library_final_loss = float(library_history[-1])
    manual_final_loss = float(manual_history[-1])

    return {
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "n": n,
        "dx_um": dx_um,
        "nyquist_factor": nyquist_factor,
        "wavelength_um": wavelength_um,
        "pixels_per_wavelength": wavelength_um / dx_um,
        "grid_width_um": grid_width_um,
        "grid_width_wavelengths": grid_width_um / wavelength_um,
        "aperture_fraction_of_grid": aperture_diameter_um / grid_width_um,
        "aperture_diameter_wavelengths": aperture_diameter_um / wavelength_um,
        "distance_um": distance_um,
        "aperture_diameter_um": aperture_diameter_um,
        "window_px": window_px,
        "lr": lr,
        "steps": steps,
        "seed": seed,
        "library_first_step_ms": library_first_step_ms,
        "library_step_ms_median": library_step_ms_median,
        "library_total_runtime_ms": library_total_runtime_ms,
        "manual_first_step_ms": manual_first_step_ms,
        "manual_step_ms_median": manual_step_ms_median,
        "manual_total_runtime_ms": manual_total_runtime_ms,
        "manual_speedup_vs_library_total": library_total_runtime_ms / manual_total_runtime_ms,
        "manual_speedup_vs_library_step": library_step_ms_median / manual_step_ms_median,
        "library_final_loss": library_final_loss,
        "manual_final_loss": manual_final_loss,
        "final_loss_abs_diff": abs(library_final_loss - manual_final_loss),
        "rel_l2_final_intensity_manual_vs_library": rel_l2_error(
            manual_final_intensity,
            library_final_intensity,
        ),
        "rel_l2_final_phase_manual_vs_library": rel_l2_error(
            np.asarray(manual_params),
            np.asarray(library_params),
        ),
    }


def print_lens_optimization_summary(records: list[dict[str, float | int | str]]) -> None:
    print("ASM library vs manual JAX lens optimization benchmark:")
    print(
        " n  grid/lambda  lib_total_ms  manual_total_ms  total_speedup  "
        "lib_final_loss  manual_final_loss  rel_l2_final_intensity"
    )
    for record in records:
        print(
            f"{int(record['n']):3d}  "
            f"{float(record['grid_width_wavelengths']):11.2f}  "
            f"{float(record['library_total_runtime_ms']):12.3f}  "
            f"{float(record['manual_total_runtime_ms']):15.3f}  "
            f"{float(record['manual_speedup_vs_library_total']):13.2f}  "
            f"{float(record['library_final_loss']):14.6f}  "
            f"{float(record['manual_final_loss']):17.6f}  "
            f"{float(record['rel_l2_final_intensity_manual_vs_library']):22.6e}"
        )


def write_records(path: Path, records: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("psf", "lens-opt", "both"), default="psf")
    parser.add_argument("--n-values", type=parse_int_list, default=parse_int_list("64,128,256"))
    parser.add_argument("--dx-um", type=float, default=1.0)
    parser.add_argument(
        "--wavelength-counts",
        type=parse_int_list,
        default=parse_int_list("1,4,8"),
    )
    parser.add_argument("--center-wavelength-um", type=float, default=0.532)
    parser.add_argument("--wavelength-span-um", type=float, default=0.08)
    parser.add_argument("--z-ratios", type=parse_float_list, default=parse_float_list("0.5"))
    parser.add_argument("--aperture-fraction", type=float, default=0.12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--lens-grid-n", type=int, default=None)
    parser.add_argument(
        "--lens-n-values",
        type=parse_int_list,
        default=parse_int_list("64,128,256,512,1024"),
    )
    parser.add_argument(
        "--lens-grid-dx-um",
        type=float,
        default=None,
        help="Override lens benchmark sampling pitch in um. Defaults to wavelength/(2*nyquist).",
    )
    parser.add_argument("--lens-nyquist-factor", type=float, default=2.0)
    parser.add_argument("--lens-wavelength-um", type=float, default=0.532)
    parser.add_argument("--lens-distance-um", type=float, default=100.0)
    parser.add_argument(
        "--lens-aperture-diameter-um",
        type=float,
        default=None,
        help=(
            "Override lens benchmark aperture diameter in um. "
            "Defaults to a fixed fraction of grid width."
        ),
    )
    parser.add_argument("--lens-aperture-fraction", type=float, default=0.35)
    parser.add_argument("--lens-window-px", type=int, default=2)
    parser.add_argument("--lens-lr", type=float, default=0.05)
    parser.add_argument("--lens-steps", type=int, default=100)
    parser.add_argument("--lens-seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmark in ("psf", "both"):
        psf_records: list[dict[str, float | int | str]] = []
        for n in args.n_values:
            for wavelength_count in args.wavelength_counts:
                for z_ratio in args.z_ratios:
                    psf_records.append(
                        run_psf_case(
                            n=n,
                            dx_um=args.dx_um,
                            wavelength_count=wavelength_count,
                            center_wavelength_um=args.center_wavelength_um,
                            wavelength_span_um=args.wavelength_span_um,
                            z_ratio=z_ratio,
                            aperture_fraction=args.aperture_fraction,
                            repeats=args.repeats,
                        )
                    )
        print_psf_summary(psf_records)
        psf_csv_path = out_dir / "asm_psf_library_vs_manual.csv"
        psf_json_path = out_dir / "asm_psf_library_vs_manual.json"
        write_records(psf_csv_path, psf_records)
        with psf_json_path.open("w", encoding="utf-8") as f:
            json.dump(psf_records, f, indent=2)
        print(f"\nSaved CSV: {psf_csv_path}")
        print(f"Saved JSON: {psf_json_path}")

    if args.benchmark in ("lens-opt", "both"):
        lens_n_values = [args.lens_grid_n] if args.lens_grid_n is not None else args.lens_n_values
        lens_records = [
            run_lens_optimization_case(
                n=n,
                wavelength_um=args.lens_wavelength_um,
                distance_um=args.lens_distance_um,
                window_px=args.lens_window_px,
                lr=args.lens_lr,
                steps=args.lens_steps,
                seed=args.lens_seed,
                dx_um=args.lens_grid_dx_um,
                nyquist_factor=args.lens_nyquist_factor,
                aperture_diameter_um=args.lens_aperture_diameter_um,
                aperture_fraction=args.lens_aperture_fraction,
            )
            for n in lens_n_values
        ]
        print_lens_optimization_summary(lens_records)
        lens_csv_path = out_dir / "asm_lens_optimization_library_vs_manual.csv"
        lens_json_path = out_dir / "asm_lens_optimization_library_vs_manual.json"
        write_records(lens_csv_path, lens_records)
        with lens_json_path.open("w", encoding="utf-8") as f:
            json.dump(lens_records, f, indent=2)
        print(f"\nSaved CSV: {lens_csv_path}")
        print(f"Saved JSON: {lens_json_path}")


if __name__ == "__main__":
    main()
