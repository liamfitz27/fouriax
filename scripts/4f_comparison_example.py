"""Compare physical 4f filtering vs k-space surrogate vs raw FFT baseline."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from fouriax.optics import (
    AmplitudeMask,
    Field,
    FourierTransform,
    Grid,
    IntensitySensor,
    InverseFourierTransform,
    KSpaceAmplitudeMask,
    OpticalModule,
    Spectrum,
    ThinLens,
    plan_propagation,
)

ARTIFACTS_DIR = Path("artifacts")
PLOT_PATH = ARTIFACTS_DIR / "4f_comparison_example.png"

# Fixed experiment settings.
WAVELENGTH_UM = 0.532
N_MEDIUM = 1.0
NA_STOP = 0.02
LENS_APERTURE_UM = 100.0
GRID_N = 256
GRID_DX_UM = 1.0


def _make_object_field(grid: Grid, spectrum: Spectrum) -> Field:
    x, y = grid.spatial_grid()
    r = jnp.sqrt(x * x + y * y)
    disk = (r <= 35.0).astype(jnp.float32)
    grating = (jnp.cos(2.0 * jnp.pi * x / 6.0) > 0.0).astype(jnp.float32)
    envelope = jnp.exp(-(r / 70.0) ** 2)
    obj = jnp.clip(0.25 + 0.55 * disk + 0.25 * grating * envelope, 0.0, 1.0)
    return Field.plane_wave(grid=grid, spectrum=spectrum).apply_amplitude(obj[None, :, :])


def _sampling_matched_focal_length_um(
    grid: Grid,
    wavelength_um: float,
    medium_index: float,
) -> float:
    # Sampling-matched 4f requires matching along x and y:
    # f = n * Nx * dx^2 / lambda = n * Ny * dy^2 / lambda.
    f_x = medium_index * grid.nx * (grid.dx_um**2) / wavelength_um
    f_y = medium_index * grid.ny * (grid.dy_um**2) / wavelength_um
    if not np.isclose(f_x, f_y, rtol=1e-6, atol=1e-9):
        raise ValueError(
            "sampling-matched focal length requires Nx*dx^2 ~= Ny*dy^2; "
            f"got f_x={f_x:.6f} um, f_y={f_y:.6f} um"
        )
    return float(f_x)


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


def main() -> None:
    grid = Grid.from_extent(nx=GRID_N, ny=GRID_N, dx_um=GRID_DX_UM, dy_um=GRID_DX_UM)
    spectrum = Spectrum.from_scalar(WAVELENGTH_UM)
    field_in = _make_object_field(grid, spectrum)
    f_um = _sampling_matched_focal_length_um(grid, WAVELENGTH_UM, N_MEDIUM)

    spatial_fourier_stop = _fourier_plane_spatial_stop_from_kcut(
        grid,
        wavelength_um=WAVELENGTH_UM,
        f_um=f_um,
        na=NA_STOP,
        n_medium=N_MEDIUM,
    )
    k_stop = _kspace_stop_mask(grid, wavelength_um=WAVELENGTH_UM, na=NA_STOP)
    input_aperture = _spatial_circular_aperture(grid, LENS_APERTURE_UM)

    # (1) Physical 4f path.
    propagator_4f = plan_propagation(
        mode="auto",
        grid=grid,
        spectrum=spectrum,
        distance_um=f_um,
    )
    module_4f = OpticalModule(
        layers=(
            ThinLens(focal_length_um=f_um, aperture_diameter_um=LENS_APERTURE_UM),
            propagator_4f,
            AmplitudeMask(amplitude_map=spatial_fourier_stop),
            propagator_4f,
            ThinLens(focal_length_um=f_um, aperture_diameter_um=LENS_APERTURE_UM),
        ),
        sensor=IntensitySensor(sum_wavelengths=True),
    )

    # (2) k-space surrogate.
    module_k = OpticalModule(
        layers=(
            AmplitudeMask(amplitude_map=input_aperture),
            FourierTransform(),
            KSpaceAmplitudeMask(amplitude_map=k_stop, aperture_diameter_um=2.0 * f_um * NA_STOP),
            InverseFourierTransform(),
        ),
        sensor=IntensitySensor(sum_wavelengths=True),
    )

    # (3) Raw FFT baseline (pure jax.numpy).
    out_raw = np.asarray(
        _raw_k_filter_intensity_jax(
            field_in.data,
            input_aperture=input_aperture,
            k_stop=k_stop,
        )
    )
    out_4f = np.asarray(module_4f.measure(field_in))
    out_k = np.asarray(module_k.measure(field_in))

    print("=== 4f Comparison ===")
    print(f"auto-selected propagator type={type(propagator_4f).__name__}")
    print(f"f_um={f_um:.6f}, NA={NA_STOP:.4f}, lens_aperture_um={LENS_APERTURE_UM:.1f}")
    print(
        "MSE: "
        f"4f/raw={_mse(out_4f, out_raw):.3e}, "
        f"k/raw={_mse(out_k, out_raw):.3e}, "
        f"4f/k={_mse(out_4f, out_k):.3e}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, img, title in zip(
        axes,
        [out_4f, out_k, out_raw],
        ["Physical 4f", "k-Surrogate", "Raw FFT/IFFT"],
        strict=True,
    ):
        im = ax.imshow(img, cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("x pixel")
        ax.set_ylabel("y pixel")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"Saved figure: {PLOT_PATH}")


if __name__ == "__main__":
    main()
