"""Port the cs_metasurface multispectral optimization task onto fouriax.

This script mirrors the task defined in
`cs_metasurface/metasurface_optimization/metasurface_optimization.py`:

- optimize a square-pillar metasurface from a 3-wavelength sweep library
- build PSFs from a plane wave at infinity
- downsample those PSFs onto a 64x64 sensor grid
- optimize the sensor-space sensing operator for a 64x64x3 datacube

The reference script uses a mutual-information objective on the singular
values of a DCT-parameterized sensing matrix. Here we keep the same sensing
task but score it through `fouriax.analysis.d_optimality(...)`. For the full
DCT basis used in the reference script, this is the same log-det objective
up to the prior scaling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.fft as jsp_fft
import matplotlib.pyplot as plt
import numpy as np
import optax

import fouriax as fx

EXPERIMENTS_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DATA_DIR = EXPERIMENTS_ROOT / "data"
EXPERIMENTS_ARTIFACTS_DIR = EXPERIMENTS_ROOT / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sensor-space multispectral metasurface optimization in fouriax.",
    )
    parser.add_argument(
        "--npz-path",
        type=str,
        default=str(
            EXPERIMENTS_DATA_DIR / "meta_atoms" / "square_pillar_0p7um_cell_sweep_results.npz"
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=str(EXPERIMENTS_ARTIFACTS_DIR / "rgb_incoherent_imager_d_opt"),
    )
    parser.add_argument("--speed-of-light", type=float, default=299_792_458.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sensor-size-px", type=int, default=64)
    parser.add_argument("--sensor-dx-um", type=float, default=3.5)
    parser.add_argument("--meta-dx-um", type=float, default=0.7)
    parser.add_argument("--wavelength-min-um", type=float, default=1.0)
    parser.add_argument("--wavelength-max-um", type=float, default=1.3)
    parser.add_argument("--num-wavelengths", type=int, default=3)
    parser.add_argument("--distance-um", type=float, default=200.0)
    parser.add_argument("--prior-variance", type=float, default=1.0)
    parser.add_argument("--rsvd-k", type=int, default=0)
    parser.add_argument("--rsvd-k-fraction", type=float, default=0.1)
    parser.add_argument("--rsvd-oversample", type=int, default=10)
    parser.add_argument("--rsvd-power-iters", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    return parser.parse_args()


ARGS = parse_args()

NPZ_PATH = Path(ARGS.npz_path)
ARTIFACTS_DIR = Path(ARGS.artifacts_dir)
PLOT_PATH = ARTIFACTS_DIR / "rgb_incoherent_imager_d_opt.png"
SUMMARY_PATH = ARTIFACTS_DIR / "rgb_incoherent_imager_d_opt_summary.json"

SPEED_OF_LIGHT_M_PER_S = ARGS.speed_of_light
SEED = ARGS.seed
SENSOR_SIZE_PX = ARGS.sensor_size_px
SENSOR_DX_UM = ARGS.sensor_dx_um
META_DX_UM = ARGS.meta_dx_um
WAVELENGTH_MIN_UM = ARGS.wavelength_min_um
WAVELENGTH_MAX_UM = ARGS.wavelength_max_um
NUM_WAVELENGTHS = ARGS.num_wavelengths
DISTANCE_UM = ARGS.distance_um
PRIOR_VARIANCE = ARGS.prior_variance
RSVD_K = ARGS.rsvd_k
RSVD_K_FRACTION = ARGS.rsvd_k_fraction
RSVD_OVERSAMPLE = ARGS.rsvd_oversample
RSVD_POWER_ITERS = ARGS.rsvd_power_iters
LR = ARGS.lr
STEPS = ARGS.steps
LOG_EVERY = ARGS.log_every
PLOT = not ARGS.no_plot


def load_square_pillar_library(npz_path: Path) -> fx.MetaAtomLibrary:
    """Load the same square-pillar LUT used by the fouriax metasurface example."""
    with np.load(npz_path) as data:
        freqs_hz = np.asarray(data["freqs"], dtype=np.float64).reshape(-1)
        side_lengths_m = np.asarray(data["side_lengths"], dtype=np.float64).reshape(-1)
        trans = np.asarray(data["trans"], dtype=np.float64)
        phase = np.asarray(data["phase"], dtype=np.float64)

    wavelengths_um = (SPEED_OF_LIGHT_M_PER_S / freqs_hz) * 1e6
    side_lengths_um = side_lengths_m * 1e6

    wav_order = np.argsort(wavelengths_um)
    side_order = np.argsort(side_lengths_um)

    wavelengths_um = wavelengths_um[wav_order]
    side_lengths_um = side_lengths_um[side_order]
    trans = trans[side_order, :][:, wav_order]
    phase = phase[side_order, :][:, wav_order]

    transmission_complex = trans.T * np.exp(1j * phase.T)
    return fx.MetaAtomLibrary.from_complex(
        wavelengths_um=jnp.asarray(wavelengths_um, dtype=jnp.float32),
        parameter_axes=(jnp.asarray(side_lengths_um, dtype=jnp.float32),),
        transmission_complex=jnp.asarray(transmission_complex),
    )


def reciprocal_wavelength_grid(
    wavelength_min_um: float,
    wavelength_max_um: float,
    num_wavelengths: int,
) -> jnp.ndarray:
    """Match the harmonic wavelength spacing used in the reference script."""
    return 1.0 / jnp.linspace(
        1.0 / wavelength_max_um,
        1.0 / wavelength_min_um,
        num_wavelengths,
        dtype=jnp.float32,
    )


def make_demo_sensor_object(sensor_grid: fx.Grid, num_wavelengths: int) -> jnp.ndarray:
    """Build a simple `64x64x3` sensor-space datacube for qualitative evaluation."""
    if num_wavelengths != 3:
        raise ValueError("demo object currently expects 3 wavelengths")

    x_um, y_um = sensor_grid.spatial_grid()
    r = jnp.sqrt(x_um * x_um + y_um * y_um)

    red = (r <= 18.0).astype(jnp.float32)
    green = ((jnp.abs(x_um) <= 10.0) | (jnp.abs(y_um) <= 10.0)).astype(jnp.float32)
    blue = ((r >= 28.0) & (r <= 40.0)).astype(jnp.float32)

    red = 0.05 + 0.95 * red
    green = 0.05 + 0.95 * green * jnp.exp(-((r / 70.0) ** 2))
    blue = 0.05 + 0.95 * blue
    return jnp.stack([red, green, blue], axis=0)


def render_rgb_cube(cube: np.ndarray) -> np.ndarray:
    """Map a `(3, ny, nx)` intensity cube to a displayable RGB image."""
    if cube.shape[0] != 3:
        raise ValueError(f"expected 3 channels, got shape {cube.shape}")
    cube = np.asarray(cube, dtype=np.float32)
    cube = cube / np.maximum(np.max(cube, axis=(1, 2), keepdims=True), 1e-12)
    return np.moveaxis(cube, 0, -1)

def dct_cube_from_coeffs(
    coeff_vec: jnp.ndarray,
    *,
    num_wavelengths: int,
    sensor_size_px: int,
) -> jnp.ndarray:
    """Map flattened DCT coefficients to a sensor-space datacube."""
    coeff_cube = coeff_vec.reshape((num_wavelengths, sensor_size_px, sensor_size_px))
    return jsp_fft.idctn(coeff_cube, axes=(-2, -1), norm="ortho").astype(jnp.float32)


def coeffs_from_dct_cube(scene_cube: jnp.ndarray) -> jnp.ndarray:
    """Adjoint DCT map from a sensor-space datacube back to flattened coefficients."""
    coeff_cube = jsp_fft.dctn(scene_cube, axes=(-2, -1), norm="ortho")
    return coeff_cube.reshape(-1).astype(jnp.float32)


def build_sensing_operator(
    sensor_conv: fx.LinearOperator,
    *,
    num_wavelengths: int,
    sensor_size_px: int,
) -> fx.LinearOperator:
    """Compose the DCT coefficient map with the cached sensor-space operator."""
    in_shape = (num_wavelengths * sensor_size_px * sensor_size_px,)
    out_shape = (sensor_size_px * sensor_size_px,)

    def matvec_fn(coeff_vec: jax.Array) -> jax.Array:
        scene_cube = dct_cube_from_coeffs(
            coeff_vec,
            num_wavelengths=num_wavelengths,
            sensor_size_px=sensor_size_px,
        )
        return (
            jnp.sum(sensor_conv.matvec(scene_cube), axis=0)
            .reshape(out_shape)
            .astype(jnp.float32)
        )

    def rmatvec_fn(sensor_vec: jax.Array) -> jax.Array:
        sensor_plane = sensor_vec.reshape((sensor_size_px, sensor_size_px))
        scene_adj = sensor_conv.rmatvec(jnp.broadcast_to(sensor_plane, sensor_conv.out_shape))
        return coeffs_from_dct_cube(scene_adj)

    return fx.LinearOperator(
        in_shape=in_shape,
        out_shape=out_shape,
        in_dtype=jnp.float32,
        out_dtype=jnp.float32,
        matvec_fn=matvec_fn,
        rmatvec_fn=rmatvec_fn,
    )


def randomized_svd_linear_operator(
    sensing_op: fx.LinearOperator,
    *,
    omega: jax.Array,
    k: int,
    q_iter: int = 0,
) -> jnp.ndarray:
    """Randomized SVD for a matrix-free sensing operator."""
    y = sensing_op.matmat(omega)
    for _ in range(q_iter):
        y = sensing_op.matmat(sensing_op.rmatmat(y))

    q, _ = jnp.linalg.qr(y)
    b = sensing_op.rmatmat(q).T
    _, singular_values, _ = jnp.linalg.svd(b, full_matrices=False)
    return singular_values[:k]


def main() -> None:
    if NUM_WAVELENGTHS != 3:
        raise ValueError("this port currently expects exactly 3 wavelengths")

    sensor_width_um = SENSOR_SIZE_PX * SENSOR_DX_UM
    meta_size_px = int(round(sensor_width_um / META_DX_UM))
    if meta_size_px <= 0:
        raise ValueError("meta grid size must be positive")
    if abs(meta_size_px * META_DX_UM - sensor_width_um) > 1e-6:
        raise ValueError(
            "sensor width must be divisible by the meta pitch so the sensor PSF "
            "can be binned exactly"
        )
    n_sensor = SENSOR_SIZE_PX * SENSOR_SIZE_PX
    n_sparse = n_sensor * NUM_WAVELENGTHS

    if RSVD_K > 0:
        rsvd_rank = RSVD_K
    else:
        rsvd_rank = int(RSVD_K_FRACTION * n_sparse)
    rsvd_rank = max(1, min(rsvd_rank, n_sensor - RSVD_OVERSAMPLE))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    library = load_square_pillar_library(NPZ_PATH)
    wavelengths_um = reciprocal_wavelength_grid(
        wavelength_min_um=WAVELENGTH_MIN_UM,
        wavelength_max_um=WAVELENGTH_MAX_UM,
        num_wavelengths=NUM_WAVELENGTHS,
    )
    spectrum = fx.Spectrum.from_array(wavelengths_um.astype(jnp.float32))
    optical_grid = fx.Grid.from_extent(
        nx=meta_size_px,
        ny=meta_size_px,
        dx_um=META_DX_UM,
        dy_um=META_DX_UM,
    )
    sensor_grid = fx.Grid.from_extent(
        nx=SENSOR_SIZE_PX,
        ny=SENSOR_SIZE_PX,
        dx_um=SENSOR_DX_UM,
        dy_um=SENSOR_DX_UM,
    )
    propagator = fx.plan_propagation(
        mode="asm",
        grid=optical_grid,
        spectrum=spectrum,
        distance_um=DISTANCE_UM,
    )

    if (
        float(jnp.min(wavelengths_um)) < float(jnp.min(library.wavelengths_um))
        or float(jnp.max(wavelengths_um)) > float(jnp.max(library.wavelengths_um))
    ):
        raise ValueError(
            "requested wavelengths fall outside the square-pillar library range: "
            f"requested=({float(jnp.min(wavelengths_um)):.3f}, "
            f"{float(jnp.max(wavelengths_um)):.3f}) um, "
            f"library=({float(jnp.min(library.wavelengths_um)):.3f}, "
            f"{float(jnp.max(library.wavelengths_um)):.3f}) um"
        )

    side_axis = library.parameter_axes[0]
    min_bounds = jnp.array([side_axis[0]], dtype=jnp.float32)
    max_bounds = jnp.array([side_axis[-1]], dtype=jnp.float32)
    psf_field_template = fx.Field.plane_wave(grid=optical_grid, spectrum=spectrum)
    sensor_array = fx.DetectorArray(
        detector_grid=sensor_grid,
        qe_curve=1.0,
        sum_wavelengths=False,
        resample_method="linear",
    )
    conv_template = fx.Intensity.zeros(grid=optical_grid, spectrum=spectrum)
    demo_object = make_demo_sensor_object(sensor_grid, NUM_WAVELENGTHS)
    rsvd_key = jax.random.PRNGKey(SEED + 1)
    rsvd_sketch = jax.random.normal(
        rsvd_key,
        (NUM_WAVELENGTHS * SENSOR_SIZE_PX * SENSOR_SIZE_PX, rsvd_rank + RSVD_OVERSAMPLE),
        dtype=jnp.float32,
    )

    def build_meta_layer(raw_params: jnp.ndarray) -> fx.MetaAtomInterpolationLayer:
        return fx.MetaAtomInterpolationLayer(
            library=library,
            raw_geometry_params=raw_params,
            min_geometry_params=min_bounds,
            max_geometry_params=max_bounds,
        )

    def build_optical_module(raw_params: jnp.ndarray) -> fx.OpticalModule:
        return fx.OpticalModule(layers=(build_meta_layer(raw_params), propagator))

    def build_incoherent_imager(raw_params: jnp.ndarray) -> fx.IncoherentImager:
        return fx.IncoherentImager(
            optical_layer=build_meta_layer(raw_params),
            propagator=propagator,
            distance_um=DISTANCE_UM,
            psf_source="plane_wave_focus",
            normalize_psf=True,
            normalization_reference="at_imaging_distance",
            mode="psf",
        )

    def d_opt_value(raw_params: jnp.ndarray) -> jnp.ndarray:
        measurement_op = build_incoherent_imager(raw_params).linear_operator(
            conv_template,
            cache="otf",
            flatten=False,
            conv_grid=sensor_grid,
        )
        sensing_op = build_sensing_operator(
            measurement_op,
            num_wavelengths=NUM_WAVELENGTHS,
            sensor_size_px=SENSOR_SIZE_PX,
        )
        singular_values = randomized_svd_linear_operator(
            sensing_op,
            omega=rsvd_sketch,
            k=rsvd_rank,
            q_iter=RSVD_POWER_ITERS,
        )
        fim_diag = jnp.square(singular_values).astype(jnp.float32)
        return fx.analysis.d_optimality(jnp.diag(fim_diag))

    def loss_fn(raw_params: jnp.ndarray) -> jnp.ndarray:
        d_opt = d_opt_value(raw_params)
        return jnp.where(jnp.isfinite(d_opt), -d_opt, 1e12)

    key = jax.random.PRNGKey(SEED)
    init_params = 0.05 * jax.random.normal(
        key,
        (optical_grid.ny, optical_grid.nx),
        dtype=jnp.float32,
    )
    optimizer = optax.adam(LR)

    print("=== RGB Incoherent Imager D-Optimality Experiment ===")
    print(f"library={NPZ_PATH}")
    print(
        "grids:"
        f" sensor={sensor_grid.nx}x{sensor_grid.ny} @ {sensor_grid.dx_um:.2f} um,"
        f" meta/field={optical_grid.nx}x{optical_grid.ny} @ {optical_grid.dx_um:.2f} um"
    )
    print(f"wavelengths_um={np.asarray(wavelengths_um)}")
    print(
        f"distance_um={DISTANCE_UM:.1f}, prior_variance={PRIOR_VARIANCE:.3f}, "
        f"rsvd_rank={rsvd_rank}"
    )
    print("optimizing...")

    result = fx.optim.optimize_optical_module(
        init_params=init_params,
        build_module=build_optical_module,
        loss_fn=loss_fn,
        optimizer=optimizer,
        steps=STEPS,
        log_every=LOG_EVERY,
    )

    reference_params = jnp.zeros_like(init_params)
    optimized_params = result.best_params

    reference_imager = build_incoherent_imager(reference_params)
    optimized_imager = build_incoherent_imager(optimized_params)

    reference_d_opt = d_opt_value(reference_params)
    optimized_d_opt = d_opt_value(optimized_params)
    optimized_measurement_op = optimized_imager.linear_operator(
        conv_template,
        cache="otf",
        flatten=False,
        conv_grid=sensor_grid,
    )
    optimized_singular_values = randomized_svd_linear_operator(
        build_sensing_operator(
            optimized_measurement_op,
            num_wavelengths=NUM_WAVELENGTHS,
            sensor_size_px=SENSOR_SIZE_PX,
        ),
        omega=rsvd_sketch,
        k=rsvd_rank,
        q_iter=RSVD_POWER_ITERS,
    )
    reference_measurement_op = reference_imager.linear_operator(
        conv_template,
        cache="otf",
        flatten=False,
        conv_grid=sensor_grid,
    )
    optimized_field_psf = optimized_imager.build_psf(psf_field_template).data.astype(jnp.float32)
    optimized_sensor_psf = sensor_array.expected(
        fx.Intensity(data=optimized_field_psf, grid=optical_grid, spectrum=spectrum)
    ).astype(jnp.float32)
    optimized_side_map = np.asarray(build_meta_layer(optimized_params).bounded_geometry_params()[0])

    plane_wave = fx.Field.plane_wave(grid=optical_grid, spectrum=spectrum)
    optimized_aperture = np.asarray(build_meta_layer(optimized_params).forward(plane_wave).data)

    reference_measurement = np.asarray(
        jnp.sum(reference_measurement_op.matvec(demo_object), axis=0)
    )
    optimized_measurement = np.asarray(
        jnp.sum(optimized_measurement_op.matvec(demo_object), axis=0)
    )

    summary = {
        "npz_path": str(NPZ_PATH),
        "sensor_shape": [sensor_grid.ny, sensor_grid.nx],
        "sensor_dx_um": float(sensor_grid.dx_um),
        "meta_shape": [optical_grid.ny, optical_grid.nx],
        "meta_dx_um": float(optical_grid.dx_um),
        "input_shape": [NUM_WAVELENGTHS, sensor_grid.ny, sensor_grid.nx],
        "wavelengths_um": np.asarray(wavelengths_um, dtype=np.float32).tolist(),
        "distance_um": float(DISTANCE_UM),
        "prior_variance": float(PRIOR_VARIANCE),
        "seed": SEED,
        "steps": STEPS,
        "lr": float(LR),
        "rsvd_rank": int(rsvd_rank),
        "rsvd_oversample": int(RSVD_OVERSAMPLE),
        "rsvd_power_iters": int(RSVD_POWER_ITERS),
        "best_step": result.best_step,
        "best_loss": float(result.best_loss),
        "final_loss": float(result.final_loss),
        "reference_d_opt": float(reference_d_opt),
        "optimized_d_opt": float(optimized_d_opt),
        "d_opt_gain": float(optimized_d_opt - reference_d_opt),
        "optimized_singular_values": np.asarray(optimized_singular_values).tolist(),
        "objective_note": (
            "Matches the reference sensor-space DCT sensing task. "
            "The retained sensing spectrum is scored with fouriax.analysis.d_optimality."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print(f"reference d_opt={float(reference_d_opt):.6f}")
    print(f"optimized d_opt={float(optimized_d_opt):.6f}")
    print(f"saved: {SUMMARY_PATH}")

    if PLOT:
        rgb_object = render_rgb_cube(np.asarray(demo_object))
        fig, axes = plt.subplots(4, 3, figsize=(14.0, 13.0))

        axes[0, 0].plot(result.history)
        axes[0, 0].set_title("Optimization Loss")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("-D-optimality")
        axes[0, 0].grid(alpha=0.3)

        side_im = axes[0, 1].imshow(optimized_side_map, cmap="viridis")
        axes[0, 1].set_title("Optimized Side Length (um)")
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])
        plt.colorbar(side_im, ax=axes[0, 1], fraction=0.046, pad=0.04)

        axes[0, 2].imshow(rgb_object)
        axes[0, 2].set_title("Demo 64x64x3 Datacube")
        axes[0, 2].set_xticks([])
        axes[0, 2].set_yticks([])

        ref_im = axes[1, 0].imshow(reference_measurement, cmap="magma")
        axes[1, 0].set_title(f"Reference Sensor Image\nD-opt={float(reference_d_opt):.3f}")
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
        plt.colorbar(ref_im, ax=axes[1, 0], fraction=0.046, pad=0.04)

        opt_im = axes[1, 1].imshow(optimized_measurement, cmap="magma")
        axes[1, 1].set_title(f"Optimized Sensor Image\nD-opt={float(optimized_d_opt):.3f}")
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        plt.colorbar(opt_im, ax=axes[1, 1], fraction=0.046, pad=0.04)

        axes[1, 2].plot(np.asarray(optimized_singular_values))
        axes[1, 2].set_title("Retained Singular Values")
        axes[1, 2].set_xlabel("Mode")
        axes[1, 2].grid(alpha=0.3)

        for idx, wavelength_um in enumerate(np.asarray(wavelengths_um)):
            phase_im = axes[2, idx].imshow(np.angle(optimized_aperture[idx]), cmap="twilight")
            axes[2, idx].set_title(f"Aperture Phase @ {float(wavelength_um):.3f} um")
            axes[2, idx].set_xticks([])
            axes[2, idx].set_yticks([])
            plt.colorbar(phase_im, ax=axes[2, idx], fraction=0.046, pad=0.04)

        for idx, wavelength_um in enumerate(np.asarray(wavelengths_um)):
            field_psf = np.asarray(optimized_field_psf[idx])
            field_psf = field_psf / np.maximum(np.max(field_psf), 1e-12)
            sensor_psf = np.asarray(optimized_sensor_psf[idx])
            sensor_psf = sensor_psf / np.maximum(np.max(sensor_psf), 1e-12)

            ax = axes[3, idx]
            ax.imshow(sensor_psf, cmap="inferno")
            ax.set_title(f"Sensor PSF @ {float(wavelength_um):.3f} um")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.03,
                0.97,
                f"field max={field_psf.max():.2f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                color="white",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.6},
            )

        fig.tight_layout()
        fig.savefig(PLOT_PATH, dpi=150)
        plt.close(fig)
        print(f"saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
