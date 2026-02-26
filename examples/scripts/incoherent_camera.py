"""Incoherent camera imaging example using IncoherentImager."""

#%% Imports
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
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

#%% Paths and Parameters
ARTIFACTS_DIR = Path("artifacts")
PLOT_PATH = ARTIFACTS_DIR / "incoherent_camera.png"

WAVELENGTH_UM = 0.532
GRID_N = 192
GRID_DX_UM = 1.0
FOCAL_LENGTH_UM = 900.0
APERTURE_DIAMETER_UM = 90.0
SENSOR_SIZE_PX = 96

#%% Helper Functions
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
    #%% Setup
    grid = Grid.from_extent(nx=GRID_N, ny=GRID_N, dx_um=GRID_DX_UM, dy_um=GRID_DX_UM)
    spectrum = Spectrum.from_scalar(WAVELENGTH_UM)

    lens_na = APERTURE_DIAMETER_UM / (2.0 * FOCAL_LENGTH_UM)
    sensor_width_um = SENSOR_SIZE_PX * grid.dx_um
    hfov_deg = float(np.degrees(np.arctan((sensor_width_um * 0.5) / FOCAL_LENGTH_UM)))
    fov_deg = 2.0 * hfov_deg

    object_intensity = _make_object_intensity(grid)
    field_in = Field.plane_wave(grid=grid, spectrum=spectrum).apply_amplitude(
        jnp.sqrt(object_intensity)[None, :, :]
    )

    lens = ThinLens(
        focal_length_um=FOCAL_LENGTH_UM,
        aperture_diameter_um=APERTURE_DIAMETER_UM,
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
        distance_um=FOCAL_LENGTH_UM,
        psf_source="plane_wave_focus",
        normalize_psf=True,
        mode="psf",
    )
    imager_otf = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=FOCAL_LENGTH_UM,
        psf_source="plane_wave_focus",
        normalize_psf=True,
        mode="otf",
    )

    module_psf = OpticalModule(layers=(imager_psf,), sensor=IntensitySensor(sum_wavelengths=True))
    module_otf = OpticalModule(layers=(imager_otf,), sensor=IntensitySensor(sum_wavelengths=True))

    #%% Evaluation
    image_psf = np.asarray(module_psf.measure(field_in))
    image_otf = np.asarray(module_otf.measure(field_in))
    crop_psf = _center_crop(image_psf, size_px=SENSOR_SIZE_PX)
    crop_otf = _center_crop(image_otf, size_px=SENSOR_SIZE_PX)

    parity_mse_full = float(np.mean((image_psf - image_otf) ** 2))
    parity_mse_crop = float(np.mean((crop_psf - crop_otf) ** 2))
    print("=== Incoherent Camera Example ===")
    print(f"focal_length_um={FOCAL_LENGTH_UM:.1f}")
    print(f"aperture_diameter_um={APERTURE_DIAMETER_UM:.1f}")
    print(f"effective_na={lens_na:.4f}")
    print(f"sensor_size_px={SENSOR_SIZE_PX} (from full grid {grid.nx}x{grid.ny})")
    print(f"half_fov_deg={hfov_deg:.3f}, fov_deg={fov_deg:.3f}")
    print(f"PSF/OTF parity MSE (full)={parity_mse_full:.3e}")
    print(f"PSF/OTF parity MSE (crop)={parity_mse_crop:.3e}")

    #%% Plot Results
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

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
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Incoherent Camera Imaging (IncoherentImager)", fontsize=14, y=0.99)
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
