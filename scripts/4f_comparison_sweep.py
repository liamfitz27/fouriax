"""Parameter sweep for 4f physical-vs-FFT comparison."""

from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from fouriax.optics import (
    ASMPropagator,
    AmplitudeMaskLayer,
    Field,
    Grid,
    IntensitySensor,
    KAmplitudeMaskLayer,
    OpticalModule,
    PhaseMaskLayer,
    PropagationLayer,
    Spectrum,
    ThinLensLayer,
)

ARTIFACTS_DIR = Path("artifacts")
SWEEP_DIR = ARTIFACTS_DIR / "4f_sweep"

WAVELENGTH_UM = 0.532
N_MEDIUM = 1.0
GRID_N = 256
GRID_DX_UM = 1.0
NA_VALUES = [0.02, 0.04, 0.06, 0.08, 0.10]
APERTURE_VALUES_UM = [80.0, 120.0, 180.0, 240.0, 320.0, 500.0]


def _make_object_field(grid: Grid, spectrum: Spectrum) -> Field:
    x, y = grid.spatial_grid()
    r = jnp.sqrt(x * x + y * y)
    disk = (r <= 35.0).astype(jnp.float32)
    grating = (jnp.cos(2.0 * jnp.pi * x / 6.0) > 0.0).astype(jnp.float32)
    envelope = jnp.exp(-(r / 70.0) ** 2)
    obj = jnp.clip(0.25 + 0.55 * disk + 0.25 * grating * envelope, 0.0, 1.0)
    return Field.plane_wave(grid=grid, spectrum=spectrum).apply_amplitude(obj[None, :, :])


def _sampling_matched_focal_length_um(grid: Grid, wavelength_um: float, medium_index: float) -> float:
    return float(medium_index * grid.nx * (grid.dx_um**2) / wavelength_um)


def _kspace_stop_mask(grid: Grid, *, wavelength_um: float, na: float) -> jnp.ndarray:
    fx, fy = grid.frequency_grid()
    fr = jnp.sqrt(fx * fx + fy * fy)
    return (fr <= na / wavelength_um).astype(jnp.float32)


def _fourier_plane_spatial_stop_from_kcut(
    grid: Grid,
    *,
    wavelength_um: float,
    f_um: float,
    na: float,
    n_medium: float,
) -> jnp.ndarray:
    x_1d = (jnp.arange(grid.nx, dtype=jnp.float32) - grid.nx / 2.0) * grid.dx_um
    y_1d = (jnp.arange(grid.ny, dtype=jnp.float32) - grid.ny / 2.0) * grid.dy_um
    x, y = jnp.meshgrid(x_1d, y_1d, indexing="xy")
    fx_equiv = x * n_medium / (wavelength_um * f_um)
    fy_equiv = y * n_medium / (wavelength_um * f_um)
    fr_equiv = jnp.sqrt(fx_equiv * fx_equiv + fy_equiv * fy_equiv)
    return (fr_equiv <= na / wavelength_um).astype(jnp.float32)


def _spatial_circular_aperture(grid: Grid, diameter_um: float) -> jnp.ndarray:
    x, y = grid.spatial_grid()
    r2 = x * x + y * y
    radius = diameter_um / 2.0
    return (r2 <= radius * radius).astype(jnp.float32)


def _normalize_intensity(i: np.ndarray) -> np.ndarray:
    total = float(np.sum(i))
    if total <= 0:
        return i
    return i / total


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _raw_k_filter_intensity_jax(
    field_data: jnp.ndarray,
    *,
    input_aperture: jnp.ndarray,
    k_stop: jnp.ndarray,
) -> jnp.ndarray:
    data_in = field_data * input_aperture[None, :, :]
    data_k = jnp.fft.fftn(data_in, axes=(-2, -1))
    data_k_filtered = data_k * k_stop[None, :, :]
    data_out = jnp.fft.ifftn(data_k_filtered, axes=(-2, -1))
    return jnp.sum(jnp.abs(data_out) ** 2, axis=0)


def _save_heatmap(
    matrix: np.ndarray,
    *,
    xvals: list[float],
    yvals: list[float],
    title: str,
    path: Path,
    cbar_label: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.8))
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("lens_aperture_um")
    ax.set_ylabel("na_stop")
    ax.set_xticks(range(len(xvals)))
    ax.set_yticks(range(len(yvals)))
    ax.set_xticklabels([f"{x:g}" for x in xvals], rotation=45, ha="right")
    ax.set_yticklabels([f"{y:g}" for y in yvals])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    grid = Grid.from_extent(nx=GRID_N, ny=GRID_N, dx_um=GRID_DX_UM, dy_um=GRID_DX_UM)
    spectrum = Spectrum.from_scalar(WAVELENGTH_UM)
    field_in = _make_object_field(grid, spectrum)
    f_um = _sampling_matched_focal_length_um(grid, WAVELENGTH_UM, N_MEDIUM)

    rows: list[dict[str, float]] = []
    for na in NA_VALUES:
        for aperture in APERTURE_VALUES_UM:
            spatial_fourier_stop = _fourier_plane_spatial_stop_from_kcut(
                grid,
                wavelength_um=WAVELENGTH_UM,
                f_um=f_um,
                na=na,
                n_medium=N_MEDIUM,
            )
            k_stop = _kspace_stop_mask(grid, wavelength_um=WAVELENGTH_UM, na=na)
            input_aperture = _spatial_circular_aperture(grid, aperture)

            asm_4f = ASMPropagator(use_sampling_planner=True, medium_index=N_MEDIUM, na_limit=None)
            module_4f = OpticalModule(
                layers=(
                    ThinLensLayer(focal_length_um=f_um, aperture_diameter_um=aperture),
                    PropagationLayer(model=asm_4f, distance_um=f_um),
                    AmplitudeMaskLayer(amplitude_map=spatial_fourier_stop),
                    PropagationLayer(model=asm_4f, distance_um=f_um),
                    ThinLensLayer(focal_length_um=f_um, aperture_diameter_um=aperture),
                ),
                sensor=IntensitySensor(sum_wavelengths=True),
                auto_apply_na=False,
                medium_index=N_MEDIUM,
                na_fallback_to_effective=False,
            )
            module_k = OpticalModule(
                layers=(
                    AmplitudeMaskLayer(amplitude_map=input_aperture),
                    KAmplitudeMaskLayer(amplitude_map=k_stop, aperture_diameter_um=2.0 * f_um * na),
                    PhaseMaskLayer(phase_map_rad=0.0),
                ),
                sensor=IntensitySensor(sum_wavelengths=True),
                auto_apply_na=False,
                medium_index=N_MEDIUM,
                na_fallback_to_effective=False,
            )

            out_4f = np.asarray(module_4f.measure(field_in))
            out_k = np.asarray(module_k.measure(field_in))
            out_raw = np.asarray(
                _raw_k_filter_intensity_jax(
                    field_in.data,
                    input_aperture=input_aperture,
                    k_stop=k_stop,
                )
            )

            out_4f_n = _normalize_intensity(out_4f)
            out_raw_n = _normalize_intensity(out_raw)
            out_k_n = _normalize_intensity(out_k)
            rows.append(
                {
                    "na_stop": float(na),
                    "lens_aperture_um": float(aperture),
                    "f_um": float(f_um),
                    "mse_4f_vs_raw_abs": _mse(out_4f, out_raw),
                    "mse_k_vs_raw_abs": _mse(out_k, out_raw),
                    "mse_4f_vs_k_abs": _mse(out_4f, out_k),
                    "mse_4f_vs_raw_norm": _mse(out_4f_n, out_raw_n),
                    "mse_k_vs_raw_norm": _mse(out_k_n, out_raw_n),
                    "mse_4f_vs_k_norm": _mse(out_4f_n, out_k_n),
                    "throughput_4f": float(np.sum(out_4f)),
                    "throughput_k": float(np.sum(out_k)),
                    "throughput_raw": float(np.sum(out_raw)),
                    "k_stop_pixels": int(np.sum(np.asarray(k_stop))),
                    "spatial_stop_pixels": int(np.sum(np.asarray(spatial_fourier_stop))),
                }
            )
            print(
                f"[sweep] na={na:.4f}, aperture={aperture:.1f}, "
                f"mse4f_raw_norm={rows[-1]['mse_4f_vs_raw_norm']:.3e}"
            )

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SWEEP_DIR / "sweep_metrics.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved sweep CSV: {csv_path}")

    na_sorted = sorted(NA_VALUES)
    aperture_sorted = sorted(APERTURE_VALUES_UM)
    idx = {(r["na_stop"], r["lens_aperture_um"]): r for r in rows}

    mse_map = np.zeros((len(na_sorted), len(aperture_sorted)), dtype=np.float64)
    thr_map = np.zeros_like(mse_map)
    for iy, na in enumerate(na_sorted):
        for ix, ap in enumerate(aperture_sorted):
            row = idx[(na, ap)]
            mse_map[iy, ix] = float(row["mse_4f_vs_raw_norm"])
            thr_map[iy, ix] = float(row["throughput_4f"]) / max(
                float(row["throughput_raw"]), 1e-12
            )

    _save_heatmap(
        mse_map,
        xvals=aperture_sorted,
        yvals=na_sorted,
        title="Normalized MSE: 4f vs raw FFT",
        path=SWEEP_DIR / "heatmap_mse_4f_vs_raw_norm.png",
        cbar_label="MSE",
    )
    _save_heatmap(
        thr_map,
        xvals=aperture_sorted,
        yvals=na_sorted,
        title="Throughput Ratio: 4f/raw",
        path=SWEEP_DIR / "heatmap_throughput_ratio_4f_over_raw.png",
        cbar_label="ratio",
    )
    print(f"Saved sweep heatmaps in: {SWEEP_DIR}")


if __name__ == "__main__":
    main()
