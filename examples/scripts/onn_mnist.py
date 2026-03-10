"""Train a small optical neural network (ONN) on MNIST with OpticalModule."""

# %% Imports
from __future__ import annotations

# %% Paths and Parameters
import argparse
import urllib.request
from pathlib import Path

import jax
import jax.image
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from matplotlib.patches import Rectangle

from fouriax.optics import (
    DetectorArray,
    Field,
    Grid,
    IntensityMonitor,
    OpticalModule,
    PhaseMask,
    Spectrum,
    plan_propagation,
)
from fouriax.optim import optimize_dataset_optical_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ONN MNIST Example")
    parser.add_argument("--data-path", type=str, default="data/mnist.npz")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-phase-layers", type=int, default=4)
    parser.add_argument("--phase-mask-downsample", type=int, default=4)
    parser.add_argument("--nyquist-factor", type=float, default=1.0)
    parser.add_argument("--distance-um", type=float, default=50.0)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    return parser.parse_args()


ARGS = parse_args()

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
ARTIFACTS_DIR = Path(ARGS.artifacts_dir)
DATA_DIR = ARTIFACTS_DIR / "data"
MNIST_CACHE_PATH = DATA_DIR / "mnist.npz"
PLOT_PATH = ARTIFACTS_DIR / "onn_mnist_field_evolution.png"
SUMMARY_PATH = ARTIFACTS_DIR / "onn_mnist_summary.json"

DEVICE = ARGS.device
SEED = ARGS.seed
EPOCHS = ARGS.epochs
BATCH_SIZE = ARGS.batch_size
LEARNING_RATE = ARGS.learning_rate
NUM_PHASE_LAYERS = ARGS.num_phase_layers
PHASE_MASK_DOWNSAMPLE = ARGS.phase_mask_downsample
NYQUIST_FACTOR = ARGS.nyquist_factor
DISTANCE_UM = ARGS.distance_um
TRAIN_SAMPLES = ARGS.train_samples
TEST_SAMPLES = ARGS.test_samples
PLOT = not ARGS.no_plot


# %% Helper Functions
def load_mnist(cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        print(f"Downloading MNIST to {cache_path} ...")
        urllib.request.urlretrieve(MNIST_URL, cache_path)

    with np.load(cache_path) as data:
        x_train = data["x_train"].astype(np.float32) / 255.0
        y_train = data["y_train"].astype(np.int32)
        x_test = data["x_test"].astype(np.float32) / 255.0
        y_test = data["y_test"].astype(np.int32)
    return x_train, y_train, x_test, y_test


def resize_images_to_grid(images: np.ndarray, grid: Grid) -> np.ndarray:
    arr = jnp.asarray(images, dtype=jnp.float32)[..., None]
    resized = jax.image.resize(
        arr,
        shape=(arr.shape[0], grid.ny, grid.nx, 1),
        method="linear",
    )[..., 0]
    return np.asarray(resized, dtype=np.float32)


def main() -> None:
    # %% Setup
    jax.config.update("jax_platform_name", DEVICE)
    selected_device = jax.devices()[0]
    jax.config.update("jax_default_device", selected_device)
    print(
        "device="
        f"{selected_device.platform} kind={getattr(selected_device, 'device_kind', 'unknown')}"
    )

    input_grid = Grid.from_extent(nx=28, ny=28, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(1.55)
    propagator = plan_propagation(
        mode="auto",
        grid=input_grid,
        spectrum=spectrum,
        distance_um=DISTANCE_UM,
        nyquist_factor=NYQUIST_FACTOR,
        min_padding_factor=2.0,
    )
    work_grid = propagator.precomputed_grid or input_grid
    mask_nx = work_grid.nx // PHASE_MASK_DOWNSAMPLE
    mask_ny = work_grid.ny // PHASE_MASK_DOWNSAMPLE
    mask_grid = Grid.from_extent(
        nx=mask_nx,
        ny=mask_ny,
        dx_um=(work_grid.nx * work_grid.dx_um) / mask_nx,
        dy_um=(work_grid.ny * work_grid.dy_um) / mask_ny,
    )
    detector_grid = Grid.from_extent(
        nx=5,
        ny=2,
        dx_um=(work_grid.nx * work_grid.dx_um) / 5.0,
        dy_um=(work_grid.ny * work_grid.dy_um) / 2.0,
    )
    detector_array = DetectorArray(
        detector_grid=detector_grid,
    )

    def build_module(raw_params: jnp.ndarray) -> OpticalModule:
        layers = [IntensityMonitor(sum_wavelengths=True, output_domain="spatial")]
        for i in range(raw_params.shape[0]):
            upsampled_latent = jax.image.resize(
                raw_params[i],
                shape=(work_grid.ny, work_grid.nx),
                method="linear",
            )
            bounded_phase = 2.0 * jnp.pi * jax.nn.sigmoid(upsampled_latent)
            layers.append(PhaseMask(phase_map_rad=bounded_phase))
            layers.append(propagator)
            layers.append(IntensityMonitor(sum_wavelengths=True, output_domain="spatial"))
        return OpticalModule(layers=tuple(layers), sensor=detector_array)

    x_train, y_train, x_test, y_test = load_mnist(MNIST_CACHE_PATH)
    x_train = x_train[:TRAIN_SAMPLES]
    y_train = y_train[:TRAIN_SAMPLES]
    x_test = x_test[:TEST_SAMPLES]
    y_test = y_test[:TEST_SAMPLES]
    x_train = resize_images_to_grid(x_train, work_grid)
    x_test = resize_images_to_grid(x_test, work_grid)

    key = jax.random.PRNGKey(SEED)
    phase_params = 0.05 * jax.random.normal(
        key,
        (NUM_PHASE_LAYERS, mask_grid.ny, mask_grid.nx),
        dtype=jnp.float32,
    )

    # %% Loss Function and Optimization
    def logits_batch(raw_params: jnp.ndarray, images_3d: jnp.ndarray) -> jnp.ndarray:
        module = build_module(raw_params)
        field = Field(
            data=images_3d[:, None, :, :].astype(jnp.complex64),
            grid=work_grid,
            spectrum=spectrum,
        )
        return module.measure(field).reshape((images_3d.shape[0], -1))

    def batch_loss_fn(
        params: jnp.ndarray,
        batch: tuple[np.ndarray, np.ndarray] | tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        image_raw, label_raw = batch
        images = jnp.asarray(image_raw, dtype=jnp.float32)
        labels = jnp.asarray(label_raw, dtype=jnp.int32)
        logits = logits_batch(params, images)
        log_probs = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])

    def batch_accuracy(params: jnp.ndarray, images: np.ndarray, labels: np.ndarray) -> float:
        logits = np.asarray(logits_batch(params, jnp.asarray(images)))
        pred = np.argmax(logits, axis=1)
        return float(np.mean(pred == labels))

    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    train_data = (x_train, y_train)
    val_data = (x_test, y_test)

    result = optimize_dataset_optical_module(
        init_params=phase_params,
        build_module=build_module,
        batch_loss_fn=batch_loss_fn,
        optimizer=optimizer,
        train_data=train_data,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        val_data=val_data,
        seed=SEED + 1,
    )

    # %% Evaluation
    train_acc = batch_accuracy(result.params_result.best_params, x_train, y_train)
    test_acc = batch_accuracy(result.params_result.best_params, x_test, y_test)
    final_val_loss = (
        float(result.params_result.final_val_metrics["val_loss"])
        if result.params_result.final_val_metrics
        else float("nan")
    )
    print(
        f"final_train_acc={train_acc:.4f} final_test_acc={test_acc:.4f} "
        f"final_val_loss={final_val_loss:.4f}"
    )

    module = result.best_module
    sample_idx = 0
    test_image = jnp.asarray(x_test[sample_idx], dtype=jnp.float32)
    sample_field = Field(
        data=test_image[None, :, :].astype(jnp.complex64),
        grid=work_grid,
        spectrum=spectrum,
    )

    # %% Plot Results
    if PLOT:
        _, intensity_steps = module.observe(sample_field)
        phase_masks = [
            np.asarray(stage.phase_map_rad)
            for stage in module.layers
            if isinstance(stage, PhaseMask)
        ]
        titles = ["Input"] + [f"After Propagation {i + 1}" for i in range(len(intensity_steps) - 1)]
        n_cols = max(len(intensity_steps), len(phase_masks))
        fig_field, axes = plt.subplots(
            2,
            n_cols,
            figsize=(max(6.0, 2.8 * n_cols), 6.8),
            squeeze=False,
        )
        for col, ax in enumerate(axes[0]):
            if col >= len(intensity_steps):
                ax.axis("off")
                continue
            title = titles[col]
            image = intensity_steps[col]
            im = ax.imshow(np.asarray(image), cmap="inferno")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig_field.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            if col == len(intensity_steps) - 1:
                cell_w = work_grid.nx / detector_grid.nx
                cell_h = work_grid.ny / detector_grid.ny
                digit = 0
                for row in range(detector_grid.ny):
                    for det_col in range(detector_grid.nx):
                        ax.add_patch(
                            Rectangle(
                                (det_col * cell_w - 0.5, row * cell_h - 0.5),
                                cell_w,
                                cell_h,
                                fill=False,
                                edgecolor="red",
                                linewidth=1.2,
                            )
                        )
                        ax.text(
                            det_col * cell_w + 0.5 * cell_w - 0.5,
                            row * cell_h + 0.5 * cell_h - 0.5,
                            str(digit),
                            color="red",
                            fontsize=10,
                            fontweight="bold",
                            ha="center",
                            va="center",
                        )
                        digit += 1

        for col, ax in enumerate(axes[1]):
            if col >= len(phase_masks):
                ax.axis("off")
                continue
            phase = phase_masks[col]
            im = ax.imshow(phase, cmap="twilight", vmin=0.0, vmax=2.0 * np.pi)
            ax.set_title(f"Phase Mask {col + 1}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig_field.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

        fig_field.suptitle("ONN Intensity Checkpoints and Learned Phase Masks", y=0.98)
        fig_field.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        fig_field.savefig(PLOT_PATH, dpi=150)
        plt.close(fig_field)
        print(f"saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
