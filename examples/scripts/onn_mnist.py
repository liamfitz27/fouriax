"""Train a small optical neural network (ONN) on MNIST with OpticalModule."""

#%% Imports
from __future__ import annotations

import urllib.request
from pathlib import Path

import jax
import jax.image
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from fouriax.optics import (
    DetectorArray,
    Field,
    Grid,
    OpticalModule,
    PhaseMask,
    Spectrum,
    plan_propagation,
    plot_field_evolution,
)
from fouriax.optim import optimize_dataset_optical_module

#%% Paths and Parameters
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
ARTIFACTS_DIR = Path("artifacts")
DATA_DIR = ARTIFACTS_DIR / "data"
MNIST_CACHE_PATH = DATA_DIR / "mnist.npz"
PLOT_PATH = ARTIFACTS_DIR / "onn_mnist_field_evolution.png"
SUMMARY_PATH = ARTIFACTS_DIR / "onn_mnist_summary.json"

DEVICE = "cpu"
SEED = 0
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.2
NUM_PHASE_LAYERS = 4
PHASE_MASK_DOWNSAMPLE = 4
NYQUIST_FACTOR = 1.0
DISTANCE_UM = 50.0
TRAIN_SAMPLES = 1000
TEST_SAMPLES = 100

#%% Helper Functions
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
    #%% Setup
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
        layers = []
        for i in range(raw_params.shape[0]):
            upsampled_latent = jax.image.resize(
                raw_params[i],
                shape=(work_grid.ny, work_grid.nx),
                method="linear",
            )
            bounded_phase = 2.0 * jnp.pi * jax.nn.sigmoid(upsampled_latent)
            layers.append(PhaseMask(phase_map_rad=bounded_phase))
            layers.append(propagator)
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

    #%% Loss Function and Optimization
    def logits_single(raw_params: jnp.ndarray, image_2d: jnp.ndarray) -> jnp.ndarray:
        module = build_module(raw_params)
        field = Field(
            data=image_2d[None, :, :].astype(jnp.complex64),
            grid=work_grid,
            spectrum=spectrum,
        )
        return module.measure(field).reshape(-1)

    logits_batch = jax.vmap(logits_single, in_axes=(None, 0))

    def sample_loss_fn(
        params: jnp.ndarray,
        sample: tuple[np.ndarray, np.ndarray] | tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        image_raw, label_raw = sample
        image = jnp.asarray(image_raw, dtype=jnp.float32)
        label = jnp.asarray(label_raw, dtype=jnp.int32)
        logits = logits_single(params, image)
        log_probs = logits - jax.scipy.special.logsumexp(logits, axis=0)
        return -log_probs[label]

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
        sample_loss_fn=sample_loss_fn,
        optimizer=optimizer,
        train_data=train_data,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        val_data=val_data,
        seed=SEED + 1,
    )

    #%% Evaluation
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

    #%% Plot Results
    fig_field, _ = plot_field_evolution(
        module=module,
        field_in=sample_field,
        mode="intensity",
        wavelength_idx=0,
        log_scale=False,
    )
    fig_field.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig_field.savefig(PLOT_PATH, dpi=150)
    plt.close(fig_field)
    print(f"saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
