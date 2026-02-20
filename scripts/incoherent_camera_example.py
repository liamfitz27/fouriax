#!/usr/bin/env python3
"""Incoherent camera imaging example using IncoherentImager."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from fouriax.optics import (
    Field,
    Grid,
    IncoherentImager,
    IntensitySensor,
    OpticalModule,
    RSPropagator,
    Spectrum,
    ThinLens,
)

ARTIFACTS_DIR = Path("artifacts")
FIG_PATH = ARTIFACTS_DIR / "incoherent_camera_example.png"
METRICS_PATH = ARTIFACTS_DIR / "incoherent_camera_metrics.json"


def _make_object_intensity(grid: Grid) -> jnp.ndarray:
    x, y = grid.spatial_grid()
    r = jnp.sqrt(x * x + y * y)
    disk = (r <= 20.0).astype(jnp.float32)
    ring = ((r >= 32.0) & (r <= 40.0)).astype(jnp.float32)
    bars = (jnp.cos(2.0 * jnp.pi * x / 9.0) > 0.0).astype(jnp.float32)
    envelope = jnp.exp(-(r / 48.0) ** 2)
    return jnp.clip(0.05 + 0.45 * disk + 0.25 * ring + 0.25 * bars * envelope, 0.0, 1.0)


def _center_crop(image: np.ndarray, size_px: int) -> np.ndarray:
    ny, nx = image.shape
    y0 = (ny - size_px) // 2
    x0 = (nx - size_px) // 2
    return image[y0 : y0 + size_px, x0 : x0 + size_px]


def main() -> None:
    grid = Grid.from_extent(nx=192, ny=192, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)

    focal_length_um = 900.0
    aperture_diameter_um = 90.0
    lens_na = aperture_diameter_um / (2.0 * focal_length_um)
    sensor_size_px = 96
    sensor_width_um = sensor_size_px * grid.dx_um
    hfov_deg = float(np.degrees(np.arctan((sensor_width_um * 0.5) / focal_length_um)))
    fov_deg = 2.0 * hfov_deg

    object_intensity = _make_object_intensity(grid)
    field_in = Field.plane_wave(grid=grid, spectrum=spectrum).apply_amplitude(
        jnp.sqrt(object_intensity)[None, :, :]
    )

    lens = ThinLens(
        focal_length_um=focal_length_um,
        aperture_diameter_um=aperture_diameter_um,
    )
    rs = RSPropagator(
        use_sampling_planner=True,
        warn_on_regime_mismatch=False,
        medium_index=1.0,
        na_limit=lens_na,
    )

    imager_psf = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=focal_length_um,
        psf_source="plane_wave_focus",
        normalize_psf=True,
        mode="psf",
    )
    imager_otf = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=focal_length_um,
        psf_source="plane_wave_focus",
        normalize_psf=True,
        mode="otf",
    )

    module_psf = OpticalModule(layers=(imager_psf,), sensor=IntensitySensor(sum_wavelengths=True))
    module_otf = OpticalModule(layers=(imager_otf,), sensor=IntensitySensor(sum_wavelengths=True))

    image_psf = np.asarray(module_psf.measure(field_in))
    image_otf = np.asarray(module_otf.measure(field_in))
    crop_psf = _center_crop(image_psf, size_px=sensor_size_px)
    crop_otf = _center_crop(image_otf, size_px=sensor_size_px)

    parity_mse_full = float(np.mean((image_psf - image_otf) ** 2))
    parity_mse_crop = float(np.mean((crop_psf - crop_otf) ** 2))
    print("=== Incoherent Camera Example ===")
    print(f"focal_length_um={focal_length_um:.1f}")
    print(f"aperture_diameter_um={aperture_diameter_um:.1f}")
    print(f"effective_na={lens_na:.4f}")
    print(f"sensor_size_px={sensor_size_px} (from full grid {grid.nx}x{grid.ny})")
    print(f"half_fov_deg={hfov_deg:.3f}, fov_deg={fov_deg:.3f}")
    print(f"PSF/OTF parity MSE (full)={parity_mse_full:.3e}")
    print(f"PSF/OTF parity MSE (crop)={parity_mse_crop:.3e}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "grid": {"nx": grid.nx, "ny": grid.ny, "dx_um": grid.dx_um, "dy_um": grid.dy_um},
        "wavelength_um": float(spectrum.wavelengths_um[0]),
        "focal_length_um": focal_length_um,
        "aperture_diameter_um": aperture_diameter_um,
        "effective_na": lens_na,
        "sensor_size_px": sensor_size_px,
        "sensor_width_um": sensor_width_um,
        "half_fov_deg": hfov_deg,
        "fov_deg": fov_deg,
        "parity_mse_full": parity_mse_full,
        "parity_mse_crop": parity_mse_crop,
    }
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved: {METRICS_PATH}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    psf = np.asarray(imager_psf.build_psf(field_in))[0]
    psf = psf / (np.max(psf) + 1e-12)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    ax = axes.ravel()
    ax[0].imshow(np.asarray(object_intensity), cmap="gray")
    ax[0].set_title("Object Intensity")
    ax[1].imshow(psf, cmap="magma")
    ax[1].set_title("Derived PSF (normalized)")
    ax[2].imshow(np.log10(psf + 1e-8), cmap="viridis")
    ax[2].set_title("log10(PSF + 1e-8)")
    ax[3].imshow(image_psf, cmap="magma")
    ax[3].set_title("Incoherent Image (PSF mode)")
    ax[4].imshow(image_otf, cmap="magma")
    ax[4].set_title("Incoherent Image (OTF mode)")
    ax[5].imshow(crop_psf, cmap="magma")
    ax[5].set_title("Sensor ROI Crop")
    for a in ax:
        a.set_xlabel("x pixel")
        a.set_ylabel("y pixel")
    fig.suptitle("Incoherent Camera Imaging (IncoherentImager)", fontsize=14, y=0.99)
    fig.text(
        0.5,
        0.01,
        (
            f"lambda={float(spectrum.wavelengths_um[0]):.3f} um | "
            f"f={focal_length_um:.1f} um | "
            f"D={aperture_diameter_um:.1f} um | "
            f"NA={lens_na:.4f} | "
            f"HFOV={hfov_deg:.3f} deg | FOV={fov_deg:.3f} deg | "
            f"sensor={sensor_size_px}px ({sensor_width_um:.1f} um)"
        ),
        ha="center",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )
    fig.savefig(FIG_PATH, dpi=150)
    print(f"saved: {FIG_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
