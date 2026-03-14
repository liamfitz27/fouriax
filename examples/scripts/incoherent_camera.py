"""Incoherent camera imaging example using IncoherentImager."""

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
    parser = argparse.ArgumentParser(description="Incoherent Camera Example")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--wavelength-um", type=float, default=0.532)
    parser.add_argument("--grid-n", type=int, default=192)
    parser.add_argument("--grid-dx-um", type=float, default=1.0)
    parser.add_argument("--f1-um", type=float, default=2700.0)
    parser.add_argument("--f2-um", type=float, default=1350.0)
    parser.add_argument("--aperture-diameter-um", type=float, default=90.0)
    parser.add_argument("--sensor-size-px", type=int, default=96)
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    return parser.parse_args()


ARGS = parse_args()

ARTIFACTS_DIR = Path(ARGS.artifacts_dir)
PLOT_PATH = ARTIFACTS_DIR / "incoherent_camera.png"

WAVELENGTH_UM = ARGS.wavelength_um
GRID_N = ARGS.grid_n
GRID_DX_UM = ARGS.grid_dx_um
F1_UM = ARGS.f1_um
F2_UM = ARGS.f2_um
APERTURE_DIAMETER_UM = ARGS.aperture_diameter_um
SENSOR_SIZE_PX = ARGS.sensor_size_px
PLOT = not ARGS.no_plot
PARAXIAL_MAX_ANGLE_RAD = 0.05


# %% Helper Functions
def _make_object_intensity(grid: fx.Grid) -> jnp.ndarray:
    x, y = grid.spatial_grid()
    r = jnp.sqrt(x * x + y * y)
    disk = (r <= 20.0).astype(jnp.float32)
    ring = ((r >= 32.0) & (r <= 40.0)).astype(jnp.float32)
    bars = (jnp.cos(2.0 * jnp.pi * x / 9.0) > 0.0).astype(jnp.float32)
    envelope = jnp.exp(-((r / 48.0) ** 2))
    return jnp.clip(0.05 + 0.45 * disk + 0.25 * ring + 0.25 * bars * envelope, 0.0, 1.0)


def _grid_extent_um(grid: fx.Grid) -> tuple[float, float, float, float]:
    half_width_um = 0.5 * grid.nx * grid.dx_um
    half_height_um = 0.5 * grid.ny * grid.dy_um
    return (-half_width_um, half_width_um, -half_height_um, half_height_um)


def main() -> None:
    # %% Setup
    # Keep the example same-shape while allowing smoke tests to shrink workload.
    sensor_n = min(GRID_N, SENSOR_SIZE_PX)
    sensor_grid = fx.Grid.from_extent(nx=sensor_n, ny=sensor_n, dx_um=GRID_DX_UM, dy_um=GRID_DX_UM)
    spectrum = fx.Spectrum.from_scalar(WAVELENGTH_UM)
    if F1_UM <= 0 or F2_UM <= 0:
        raise ValueError("f1_um and f2_um must be strictly positive")

    object_distance_um = F1_UM
    image_distance_um = F2_UM
    focal_length_um = 1.0 / ((1.0 / object_distance_um) + (1.0 / image_distance_um))
    magnification_abs = image_distance_um / object_distance_um

    lens_na = APERTURE_DIAMETER_UM / (2.0 * image_distance_um)

    lens = fx.ThinLens(
        focal_length_um=focal_length_um,
        aperture_diameter_um=APERTURE_DIAMETER_UM,
    )
    rs = fx.RSPropagator(
        use_sampling_planner=True,
        warn_on_regime_mismatch=False,
        medium_index=1.0,
        na_limit=lens_na,
    )

    imager_psf = fx.IncoherentImager.for_finite_distance(
        optical_layer=lens,
        propagator=rs,
        object_distance_um=object_distance_um,
        image_distance_um=image_distance_um,
        normalize_psf=True,
        mode="psf",
    )
    imager_otf = fx.IncoherentImager.for_finite_distance(
        optical_layer=lens,
        propagator=rs,
        object_distance_um=object_distance_um,
        image_distance_um=image_distance_um,
        normalize_psf=True,
        mode="otf",
    )

    input_grid = imager_psf.infer_from_paraxial_limit(
        sensor_grid,
        paraxial_max_angle_rad=PARAXIAL_MAX_ANGLE_RAD,
    )
    object_intensity = fx.Intensity(
        data=_make_object_intensity(input_grid)[None, :, :],
        grid=input_grid,
        spectrum=spectrum,
    )
    sensor_width_um = sensor_grid.nx * sensor_grid.dx_um
    input_width_um = input_grid.nx * input_grid.dx_um
    half_object_fov_deg = float(np.degrees(np.arctan((input_width_um * 0.5) / object_distance_um)))
    object_fov_deg = 2.0 * half_object_fov_deg

    detector_array = fx.DetectorArray(
        detector_grid=sensor_grid,
        qe_curve=1.0,
    )
    field_template = fx.Field.zeros(grid=input_grid, spectrum=spectrum)

    # %% Evaluation
    image_psf = np.asarray(detector_array.measure(imager_psf.forward(object_intensity)))
    image_otf = np.asarray(detector_array.measure(imager_otf.forward(object_intensity)))

    parity_mse_sensor = float(np.mean((image_psf - image_otf) ** 2))
    print("=== Incoherent Camera Example ===")
    print(f"f1_um={object_distance_um:.1f}")
    print(f"f2_um={image_distance_um:.1f}")
    print(f"derived_focal_length_um={focal_length_um:.1f}")
    print(
        "thin_lens_residual="
        f"{(1.0 / focal_length_um) - (1.0 / object_distance_um) - (1.0 / image_distance_um):.3e}"
    )
    print(f"magnification_abs={magnification_abs:.3f}")
    print(f"aperture_diameter_um={APERTURE_DIAMETER_UM:.1f}")
    print(f"effective_na={lens_na:.4f}")
    print(f"paraxial_max_angle_rad={PARAXIAL_MAX_ANGLE_RAD:.3f}")
    print(f"input_grid_shape={input_grid.ny}x{input_grid.nx}, input_dx_um={input_grid.dx_um:.3f}")
    print(
        f"sensor_grid_shape={sensor_grid.ny}x{sensor_grid.nx}, "
        f"sensor_dx_um={sensor_grid.dx_um:.3f}"
    )
    print(f"input_width_um={input_width_um:.3f}, sensor_width_um={sensor_width_um:.3f}")
    print(f"half_object_fov_deg={half_object_fov_deg:.3f}, object_fov_deg={object_fov_deg:.3f}")
    print(f"PSF/OTF parity MSE (sensor)={parity_mse_sensor:.3e}")

    # %% Plot Results
    if PLOT:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        psf_sensor = np.asarray(detector_array.measure(imager_psf.build_psf(field_template)))
        psf_sensor = psf_sensor / (np.max(psf_sensor) + 1e-12)
        object_extent_um = _grid_extent_um(object_intensity.grid)
        sensor_extent_um = _grid_extent_um(sensor_grid)

        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        ax = axes.ravel()
        ax[0].imshow(
            np.asarray(object_intensity.data[0]),
            cmap="gray",
            extent=object_extent_um,
            origin="lower",
        )
        ax[0].set_title("Object Intensity (Input Plane)")
        ax[0].set_xlabel("x_obj [um]")
        ax[0].set_ylabel("y_obj [um]")
        ax[1].imshow(psf_sensor, cmap="magma", extent=sensor_extent_um, origin="lower")
        ax[1].set_title("Sensor-Sampled PSF")
        ax[1].set_xlabel("x_sensor [um]")
        ax[1].set_ylabel("y_sensor [um]")
        ax[2].imshow(
            np.log10(psf_sensor + 1e-8),
            cmap="viridis",
            extent=sensor_extent_um,
            origin="lower",
        )
        ax[2].set_title("log10(Sensor PSF + 1e-8)")
        ax[2].set_xlabel("x_sensor [um]")
        ax[2].set_ylabel("y_sensor [um]")
        ax[3].imshow(image_psf, cmap="magma", extent=sensor_extent_um, origin="lower")
        ax[3].set_title("Sensor Measurement (PSF mode)")
        ax[3].set_xlabel("x_sensor [um]")
        ax[3].set_ylabel("y_sensor [um]")
        ax[4].imshow(image_otf, cmap="magma", extent=sensor_extent_um, origin="lower")
        ax[4].set_title("Sensor Measurement (OTF mode)")
        ax[4].set_xlabel("x_sensor [um]")
        ax[4].set_ylabel("y_sensor [um]")
        ax[5].imshow(
            np.abs(image_psf - image_otf),
            cmap="magma",
            extent=sensor_extent_um,
            origin="lower",
        )
        ax[5].set_title("|PSF - OTF| on Sensor")
        ax[5].set_xlabel("x_sensor [um]")
        ax[5].set_ylabel("y_sensor [um]")
        fig.suptitle("Incoherent Camera Imaging (fx.IncoherentImager)", fontsize=14, y=0.99)
        fig.tight_layout()
        fig.savefig(PLOT_PATH, dpi=150)
        plt.close(fig)
        print(f"saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
