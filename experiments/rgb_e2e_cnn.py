"""End-to-end RGB metasurface + CNN reconstruction on cartoon-face data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn

import fouriax as fx

EXPERIMENTS_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DATA_DIR = EXPERIMENTS_ROOT / "data"
EXPERIMENTS_ARTIFACTS_DIR = EXPERIMENTS_ROOT / "artifacts"
REPO_ROOT = EXPERIMENTS_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fouriax RGB end-to-end metasurface + CNN experiment.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="gpu",
        help="JAX execution backend.",
    )
    parser.add_argument(
        "--train-npz",
        type=str,
        default="",
    )
    parser.add_argument(
        "--valid-npz",
        type=str,
        default="",
    )
    parser.add_argument("--data-root", type=str, default=str(EXPERIMENTS_DATA_DIR))
    parser.add_argument(
        "--meta-atom-npz",
        type=str,
        default=str(
            EXPERIMENTS_DATA_DIR / "meta_atoms" / "square_pillar_0p7um_cell_sweep_results.npz"
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=str(EXPERIMENTS_ARTIFACTS_DIR / "rgb_e2e_cnn"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--optical-lr", type=float, default=5e-2)
    parser.add_argument("--decoder-lr", type=float, default=1e-3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--noise-level", type=float, default=0.02)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--cnn-base-channels", type=int, default=64)
    parser.add_argument("--sensor-size-px", type=int, default=64)
    parser.add_argument("--sensor-dx-um", type=float, default=3.5)
    parser.add_argument("--meta-dx-um", type=float, default=0.7)
    parser.add_argument("--distance-um", type=float, default=500.0)
    parser.add_argument("--wavelength-min-um", type=float, default=1.0)
    parser.add_argument("--wavelength-max-um", type=float, default=1.3)
    parser.add_argument("--num-wavelengths", type=int, default=3)
    parser.add_argument("--speed-of-light", type=float, default=299_792_458.0)
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    return parser.parse_args()


ARGS = parse_args()

TRAIN_NPZ = Path(ARGS.train_npz) if ARGS.train_npz else None
VALID_NPZ = Path(ARGS.valid_npz) if ARGS.valid_npz else None
DATA_ROOT = Path(ARGS.data_root)
META_ATOM_NPZ = Path(ARGS.meta_atom_npz)
ARTIFACTS_DIR = Path(ARGS.artifacts_dir)
PLOT_PATH = ARTIFACTS_DIR / "rgb_e2e_cnn_overview.png"
SUMMARY_PATH = ARTIFACTS_DIR / "rgb_e2e_cnn_summary.json"
OPTIMIZED_PATH = ARTIFACTS_DIR / "rgb_e2e_cnn_optimized_artifacts.npz"

DEVICE = ARGS.device
SEED = ARGS.seed
TRAIN_SAMPLES = ARGS.train_samples
VAL_SAMPLES = ARGS.val_samples
BATCH_SIZE = ARGS.batch_size
EPOCHS = ARGS.epochs
OPTICAL_LR = ARGS.optical_lr
DECODER_LR = ARGS.decoder_lr
GRADIENT_CLIP = ARGS.gradient_clip
NOISE_LEVEL = ARGS.noise_level
PREVIEW_COUNT = ARGS.preview_count
CNN_BASE_CHANNELS = ARGS.cnn_base_channels
SENSOR_SIZE_PX = ARGS.sensor_size_px
SENSOR_DX_UM = ARGS.sensor_dx_um
META_DX_UM = ARGS.meta_dx_um
DISTANCE_UM = ARGS.distance_um
WAVELENGTH_MIN_UM = ARGS.wavelength_min_um
WAVELENGTH_MAX_UM = ARGS.wavelength_max_um
NUM_WAVELENGTHS = ARGS.num_wavelengths
SPEED_OF_LIGHT_M_PER_S = ARGS.speed_of_light
PLOT = not ARGS.no_plot


class RGBReconCNN(nn.Module):
    base_channels: int = 32

    @nn.compact
    def __call__(self, mono: jnp.ndarray) -> jnp.ndarray:
        if mono.ndim == 3:
            x = mono[..., None]
        else:
            x = mono

        x1 = nn.relu(nn.Conv(self.base_channels, (7, 7), padding="SAME")(x))
        x2 = nn.relu(nn.Conv(self.base_channels * 2, (5, 5), padding="SAME")(x1))
        x3 = nn.relu(nn.Conv(self.base_channels * 4, (3, 3), padding="SAME")(x2))
        x4 = nn.relu(nn.Conv(self.base_channels * 8, (3, 3), padding="SAME")(x3))

        y = nn.relu(nn.Conv(self.base_channels * 8, (3, 3), padding="SAME")(x4))
        y = nn.relu(nn.Conv(self.base_channels * 4, (3, 3), padding="SAME")(y))
        y = nn.relu(nn.Conv(self.base_channels * 2, (3, 3), padding="SAME")(y))
        y = nn.relu(nn.Conv(self.base_channels, (5, 5), padding="SAME")(y))

        rgb_logits = nn.Conv(3, (7, 7), padding="SAME")(y)
        rgb_logits = (
            rgb_logits
            + 0.1 * nn.Conv(3, (1, 1), padding="SAME")(x1)
            + 0.1 * nn.Conv(3, (1, 1), padding="SAME")(x2)
        )
        return rgb_logits.astype(jnp.float32)


def reciprocal_wavelength_grid(
    wavelength_min_um: float,
    wavelength_max_um: float,
    num_wavelengths: int,
) -> jnp.ndarray:
    return 1.0 / jnp.linspace(
        1.0 / wavelength_max_um,
        1.0 / wavelength_min_um,
        num_wavelengths,
        dtype=jnp.float32,
    )


def load_square_pillar_library(npz_path: Path) -> fx.MetaAtomLibrary:
    with np.load(npz_path) as data:
        freqs_hz = np.asarray(data["freqs"], dtype=np.float64).reshape(-1)
        side_lengths_m = np.asarray(data["side_lengths"], dtype=np.float64).reshape(-1)
        trans = np.asarray(data["trans"], dtype=np.float64)
        phase = np.asarray(data["phase"], dtype=np.float64)

    wavelengths_um = (SPEED_OF_LIGHT_M_PER_S / freqs_hz) * 1e6
    side_lengths_um = side_lengths_m * 1e6
    wav_order = np.argsort(wavelengths_um)
    side_order = np.argsort(side_lengths_um)

    transmission_complex = trans[side_order, :][:, wav_order].T * np.exp(
        1j * phase[side_order, :][:, wav_order].T
    )
    return fx.MetaAtomLibrary.from_complex(
        wavelengths_um=jnp.asarray(wavelengths_um[wav_order], dtype=jnp.float32),
        parameter_axes=(jnp.asarray(side_lengths_um[side_order], dtype=jnp.float32),),
        transmission_complex=jnp.asarray(transmission_complex),
    )


def discover_dataset_files(dataset_path: Path, *, subset_name: str) -> list[Path]:
    if dataset_path.is_file():
        return [dataset_path]
    if dataset_path.is_dir():
        shard_files = sorted(
            path
            for path in dataset_path.glob("*.npz")
            if f"_{subset_name}_" in path.name and "shard" in path.name
        )
        if shard_files:
            return shard_files
        all_npz = sorted(path for path in dataset_path.glob("*.npz") if path.is_file())
        if len(all_npz) == 1:
            return all_npz
        raise FileNotFoundError(
            f"could not find {subset_name!r} shard files under directory {dataset_path}"
        )
    raise FileNotFoundError(f"dataset path does not exist: {dataset_path}")


def load_rgb_subset(
    dataset_path: Path,
    *,
    subset_name: str,
    max_samples: int,
) -> np.ndarray:
    files = discover_dataset_files(dataset_path, subset_name=subset_name)
    arrays: list[np.ndarray] = []
    remaining = max_samples if max_samples > 0 else None

    for file_path in files:
        with np.load(file_path) as data:
            if "images" not in data:
                raise KeyError(f"expected key 'images' in {file_path}")
            images = np.asarray(data["images"], dtype=np.float32)
        if remaining is not None:
            if remaining <= 0:
                break
            images = images[:remaining]
            remaining -= images.shape[0]
        arrays.append(images)

    if not arrays:
        raise ValueError(f"no images loaded for subset={subset_name} from {dataset_path}")
    return np.concatenate(arrays, axis=0).astype(np.float32)


def psnr_from_mse(mse: jnp.ndarray) -> jnp.ndarray:
    return 20.0 * jnp.log10(1.0 / jnp.sqrt(jnp.maximum(mse, 1e-12)))


def render_rgb(image: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)


def main() -> None:
    if NUM_WAVELENGTHS != 3:
        raise ValueError("this experiment expects exactly 3 wavelengths")

    try:
        selected_device = jax.devices(DEVICE)[0]
    except (RuntimeError, IndexError) as exc:
        raise RuntimeError(f"requested JAX backend {DEVICE!r} is not available") from exc
    jax.config.update("jax_default_device", selected_device)
    print(
        "device="
        f"{selected_device.platform} kind={getattr(selected_device, 'device_kind', 'unknown')}"
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if bool(ARGS.train_npz) != bool(ARGS.valid_npz):
        raise ValueError("pass both --train-npz and --valid-npz, or neither")
    if ARGS.train_npz and ARGS.valid_npz:
        assert TRAIN_NPZ is not None and VALID_NPZ is not None
        train_source = TRAIN_NPZ
        valid_source = VALID_NPZ
    else:
        from experiments.data.process_cartoon_faces import ensure_processed_cartoon_faces_dataset

        dataset_dir = ensure_processed_cartoon_faces_dataset(
            data_root=DATA_ROOT,
            size=SENSOR_SIZE_PX,
            shard_size=5000,
            white_threshold=240,
            num_workers=0,
            seed=SEED,
            download_if_missing=True,
        )
        train_source = dataset_dir
        valid_source = dataset_dir

    print("loading datasets...")
    train = load_rgb_subset(train_source, subset_name="train", max_samples=TRAIN_SAMPLES)
    val = load_rgb_subset(valid_source, subset_name="valid", max_samples=VAL_SAMPLES)
    print(f"train={train.shape}, val={val.shape}")
    expected_image_shape = (SENSOR_SIZE_PX, SENSOR_SIZE_PX, NUM_WAVELENGTHS)
    if train.ndim != 4 or train.shape[1:] != expected_image_shape:
        raise ValueError(
            "training dataset image shape must match sensor-size-px and num-wavelengths: "
            f"expected (*, {expected_image_shape[0]}, {expected_image_shape[1]}, "
            f"{expected_image_shape[2]}), got {train.shape}"
        )
    if val.ndim != 4 or val.shape[1:] != expected_image_shape:
        raise ValueError(
            "validation dataset image shape must match sensor-size-px and num-wavelengths: "
            f"expected (*, {expected_image_shape[0]}, {expected_image_shape[1]}, "
            f"{expected_image_shape[2]}), got {val.shape}"
        )

    sensor_width_um = SENSOR_SIZE_PX * SENSOR_DX_UM
    meta_size_px = int(round(sensor_width_um / META_DX_UM))
    if meta_size_px <= 0:
        raise ValueError("meta grid size must be positive")
    if abs(meta_size_px * META_DX_UM - sensor_width_um) > 1e-6:
        raise ValueError("sensor width must be divisible by the meta pitch")

    library = load_square_pillar_library(META_ATOM_NPZ)
    wavelengths_um = reciprocal_wavelength_grid(
        wavelength_min_um=WAVELENGTH_MIN_UM,
        wavelength_max_um=WAVELENGTH_MAX_UM,
        num_wavelengths=NUM_WAVELENGTHS,
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
        mode="rs",
        grid=optical_grid,
        spectrum=spectrum,
        distance_um=DISTANCE_UM,
    )
    conv_template = fx.Intensity.zeros(grid=optical_grid, spectrum=spectrum)
    psf_field_template = fx.Field.plane_wave(grid=optical_grid, spectrum=spectrum)
    sensor_array = fx.DetectorArray(
        detector_grid=sensor_grid,
        qe_curve=1.0,
        sum_wavelengths=False,
        resample_method="linear",
    )

    side_axis = library.parameter_axes[0]
    min_bounds = jnp.array([side_axis[0]], dtype=jnp.float32)
    max_bounds = jnp.array([side_axis[-1]], dtype=jnp.float32)

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

    decoder = RGBReconCNN(base_channels=CNN_BASE_CHANNELS)
    key = jax.random.PRNGKey(SEED)
    key, k_init, k_train_noise, k_val_noise = jax.random.split(key, 4)
    init_optical_params = 0.05 * jax.random.normal(
        k_init,
        (optical_grid.ny, optical_grid.nx),
        dtype=jnp.float32,
    )
    init_decoder_params = decoder.init(
        {"params": k_init},
        jnp.zeros((1, SENSOR_SIZE_PX, SENSOR_SIZE_PX), dtype=jnp.float32),
    )

    train_noise_keys = np.asarray(jax.random.split(k_train_noise, train.shape[0]))
    val_noise_keys = np.asarray(jax.random.split(k_val_noise, val.shape[0]))

    def measure_batch(
        raw_params: jnp.ndarray,
        batch_rgb_nhwc: np.ndarray | jnp.ndarray,
        batch_noise_keys: np.ndarray | jnp.ndarray | None,
    ) -> jnp.ndarray:
        measurement_op = build_incoherent_imager(raw_params).linear_operator(
            conv_template,
            cache="otf",
            flatten=False,
            conv_grid=sensor_grid,
        )
        scene_cube = jnp.moveaxis(jnp.asarray(batch_rgb_nhwc, dtype=jnp.float32), -1, 1)
        spectral_sensor = jax.vmap(measurement_op.matvec)(scene_cube)
        mono = jnp.sum(spectral_sensor, axis=1)
        if batch_noise_keys is None or NOISE_LEVEL <= 0.0:
            return mono.astype(jnp.float32)
        batch_noise_keys = jnp.asarray(batch_noise_keys, dtype=jnp.uint32)
        noise_scale = NOISE_LEVEL * jnp.mean(jnp.abs(mono), axis=(-2, -1), keepdims=True)
        noise = jax.vmap(
            lambda sample_key, scale: scale * jax.random.normal(
                sample_key,
                (SENSOR_SIZE_PX, SENSOR_SIZE_PX),
                dtype=jnp.float32,
            )
        )(batch_noise_keys, noise_scale)
        return (mono + noise).astype(jnp.float32)

    def reconstruct_batch(
        raw_params: jnp.ndarray,
        decoder_params: dict[str, Any],
        batch_rgb_nhwc: np.ndarray | jnp.ndarray,
        batch_noise_keys: np.ndarray | jnp.ndarray | None,
    ) -> jnp.ndarray:
        mono = measure_batch(raw_params, batch_rgb_nhwc, batch_noise_keys)
        return decoder.apply(decoder_params, mono)

    def hybrid_batch_loss(
        optical_params: jnp.ndarray,
        decoder_params: dict[str, Any],
        batch: tuple[np.ndarray, np.ndarray] | tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        batch_images, batch_noise = batch
        recon = reconstruct_batch(optical_params, decoder_params, batch_images, batch_noise)
        target = jnp.asarray(batch_images, dtype=jnp.float32)
        return jnp.mean((recon - target) ** 2)

    optical_optimizer = optax.chain(
        optax.clip_by_global_norm(GRADIENT_CLIP),
        optax.adam(OPTICAL_LR),
    )
    decoder_optimizer = optax.chain(
        optax.clip_by_global_norm(GRADIENT_CLIP),
        optax.adam(DECODER_LR),
    )

    print("=== RGB E2E CNN Experiment ===")
    print(f"train_npz={train_source}")
    print(f"valid_npz={valid_source}")
    print(
        "grids:"
        f" sensor={sensor_grid.nx}x{sensor_grid.ny} @ {sensor_grid.dx_um:.2f} um,"
        f" meta/field={optical_grid.nx}x{optical_grid.ny} @ {optical_grid.dx_um:.2f} um"
    )
    print(f"wavelengths_um={np.asarray(wavelengths_um)}")
    print(
        f"epochs={EPOCHS}, batch_size={BATCH_SIZE}, "
        f"optical_lr={OPTICAL_LR:.2e}, decoder_lr={DECODER_LR:.2e}"
    )
    print("optimizing...")

    result = fx.optim.optimize_dataset_hybrid_module(
        init_optical_params=init_optical_params,
        init_decoder_params=init_decoder_params,
        build_module=build_optical_module,
        batch_loss_fn=hybrid_batch_loss,
        train_data=(train, train_noise_keys),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        val_data=(val, val_noise_keys),
        optical_optimizer=optical_optimizer,
        decoder_optimizer=decoder_optimizer,
        seed=SEED,
        jit=True,
    )

    history_steps = [record.step for record in result.params_result.val_history]
    history_val = [record.metrics["val_loss"] for record in result.params_result.val_history]
    history_train = [result.params_result.train_loss_history[step] for step in history_steps]
    best_val_loss = float(result.params_result.best_metric_value)

    preview_count = min(PREVIEW_COUNT, val.shape[0])
    preview_images = jnp.asarray(val[:preview_count], dtype=jnp.float32)
    preview_noise_keys = jnp.asarray(val_noise_keys[:preview_count], dtype=jnp.uint32)
    preview_recon = reconstruct_batch(
        result.best_optical_params,
        result.best_decoder_params,
        preview_images,
        preview_noise_keys,
    )
    preview_mono = measure_batch(result.best_optical_params, preview_images, preview_noise_keys)
    preview_mse = jnp.mean((preview_recon - preview_images) ** 2)
    preview_psnr = psnr_from_mse(preview_mse)

    optimized_imager = build_incoherent_imager(result.best_optical_params)
    optimized_field_psf = optimized_imager.build_psf(psf_field_template).data.astype(jnp.float32)
    optimized_sensor_psf = sensor_array.expected(
        fx.Intensity(data=optimized_field_psf, grid=optical_grid, spectrum=spectrum)
    ).astype(jnp.float32)
    optimized_side_map = np.asarray(
        build_meta_layer(result.best_optical_params).bounded_geometry_params()[0]
    )
    optimized_measurement_op = optimized_imager.linear_operator(
        conv_template,
        cache="otf",
        flatten=False,
        conv_grid=sensor_grid,
    )
    preview_spectral_sensor = jax.vmap(optimized_measurement_op.matvec)(
        jnp.moveaxis(preview_images, -1, 1)
    )

    summary = {
        "train_npz": str(train_source),
        "valid_npz": str(valid_source),
        "data_root": str(DATA_ROOT),
        "meta_atom_npz": str(META_ATOM_NPZ),
        "train_shape": list(train.shape),
        "val_shape": list(val.shape),
        "sensor_shape": [sensor_grid.ny, sensor_grid.nx],
        "meta_shape": [optical_grid.ny, optical_grid.nx],
        "sensor_dx_um": float(sensor_grid.dx_um),
        "meta_dx_um": float(optical_grid.dx_um),
        "wavelengths_um": np.asarray(wavelengths_um, dtype=np.float32).tolist(),
        "distance_um": float(DISTANCE_UM),
        "noise_level": float(NOISE_LEVEL),
        "batch_size": int(BATCH_SIZE),
        "epochs": int(EPOCHS),
        "optical_lr": float(OPTICAL_LR),
        "decoder_lr": float(DECODER_LR),
        "best_epoch": result.params_result.best_epoch,
        "best_val_loss": best_val_loss,
        "preview_mse": float(preview_mse),
        "preview_psnr_db": float(preview_psnr),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    np.savez_compressed(
        OPTIMIZED_PATH,
        optimized_side_lengths_um=optimized_side_map,
        optimized_field_psf=np.asarray(optimized_field_psf),
        optimized_sensor_psf=np.asarray(optimized_sensor_psf),
        preview_target=np.asarray(preview_images),
        preview_measurement=np.asarray(preview_mono),
        preview_reconstruction=np.asarray(preview_recon),
        preview_spectral_sensor=np.asarray(preview_spectral_sensor),
        wavelengths_um=np.asarray(wavelengths_um),
    )

    print(f"best_val_loss={best_val_loss:.6f}")
    print(f"preview_psnr_db={float(preview_psnr):.3f}")
    print(f"saved: {SUMMARY_PATH}")
    print(f"saved: {OPTIMIZED_PATH}")

    if PLOT:
        fig, axes = plt.subplots(4, 3, figsize=(13.5, 14.0))

        axes[0, 0].plot(result.params_result.train_loss_history)
        axes[0, 0].set_title("Train Loss")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(history_steps, history_train, label="train")
        axes[0, 1].plot(history_steps, history_val, label="val")
        axes[0, 1].set_title("Validation History")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        side_im = axes[0, 2].imshow(optimized_side_map, cmap="viridis")
        axes[0, 2].set_title("Optimized Side Length (um)")
        axes[0, 2].set_xticks([])
        axes[0, 2].set_yticks([])
        plt.colorbar(side_im, ax=axes[0, 2], fraction=0.046, pad=0.04)

        axes[1, 0].imshow(np.asarray(preview_mono[0]), cmap="magma")
        axes[1, 0].set_title("Mono Measurement")
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])

        axes[1, 1].imshow(render_rgb(np.asarray(preview_images[0])))
        axes[1, 1].set_title("Target RGB")
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])

        axes[1, 2].imshow(render_rgb(np.asarray(preview_recon[0])))
        axes[1, 2].set_title(f"Reconstruction\nPSNR={float(preview_psnr):.2f} dB")
        axes[1, 2].set_xticks([])
        axes[1, 2].set_yticks([])

        for idx, wavelength_um in enumerate(np.asarray(wavelengths_um)):
            ax = axes[2, idx]
            sensor_psf = np.asarray(optimized_sensor_psf[idx])
            sensor_psf = sensor_psf / np.maximum(sensor_psf.max(), 1e-12)
            ax.imshow(sensor_psf, cmap="inferno")
            ax.set_title(f"Sensor PSF @ {float(wavelength_um):.3f} um")
            ax.set_xticks([])
            ax.set_yticks([])

        for idx, wavelength_um in enumerate(np.asarray(wavelengths_um)):
            ax = axes[3, idx]
            sensor_slice = np.asarray(preview_spectral_sensor[0, idx])
            sensor_slice = sensor_slice / np.maximum(sensor_slice.max(), 1e-12)
            ax.imshow(sensor_slice, cmap="cividis")
            ax.set_title(f"Sensor Channel @ {float(wavelength_um):.3f} um")
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        fig.savefig(PLOT_PATH, dpi=150)
        plt.close(fig)
        print(f"saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
