#!/usr/bin/env python3
"""Train a small optical neural network (ONN) on MNIST with OpticalModule."""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path
from typing import Sequence

import jax
import jax.image
import jax.numpy as jnp
import numpy as np
import optax

from fouriax.optics import (
    Field,
    Grid,
    IntensitySensor,
    OpticalLayer,
    OpticalModule,
    PhaseMask,
    Spectrum,
    plan_propagation,
    plot_field_evolution,
    recommend_nyquist_grid,
)

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


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


def build_detector_bank(grid: Grid) -> jnp.ndarray:
    """
    Build 10 non-overlapping detector regions (2 rows x 5 cols) over the sensor plane.
    Returns shape (10, ny, nx).
    """
    x_edges = np.linspace(0, grid.nx, 6, dtype=int)
    y_edges = np.linspace(0, grid.ny, 3, dtype=int)
    masks = np.zeros((10, grid.ny, grid.nx), dtype=np.float32)

    idx = 0
    for row in range(2):
        for col in range(5):
            x0, x1 = x_edges[col], x_edges[col + 1]
            y0, y1 = y_edges[row], y_edges[row + 1]
            masks[idx, y0:y1, x0:x1] = 1.0
            idx += 1
    return jnp.asarray(masks)


def resize_images_to_grid(images: np.ndarray, grid: Grid) -> np.ndarray:
    arr = jnp.asarray(images, dtype=jnp.float32)[..., None]
    resized = jax.image.resize(
        arr,
        shape=(arr.shape[0], grid.ny, grid.nx, 1),
        method="linear",
    )[..., 0]
    return np.asarray(resized, dtype=np.float32)


def build_onn_module(
    params_mask: jnp.ndarray,
    work_grid: Grid,
    propagator: OpticalLayer,
    detector_sensor: IntensitySensor,
) -> OpticalModule:
    layers = []
    for i in range(params_mask.shape[0]):
        upsampled_latent = jax.image.resize(
            params_mask[i],
            shape=(work_grid.ny, work_grid.nx),
            method="linear",
        )
        bounded_phase = 2.0 * jnp.pi * jax.nn.sigmoid(upsampled_latent)
        layers.append(PhaseMask(phase_map_rad=bounded_phase))
        layers.append(propagator)
    module = OpticalModule(layers=tuple(layers), sensor=detector_sensor)
    return module


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Execution device backend.",
    )
    args = parser.parse_args(argv)

    epochs = 10
    batch_size = 64
    learning_rate = 0.2
    num_phase_layers = 4
    phase_mask_downsample = 4
    nyquist_factor = 1.0
    distance_um = 50.0
    train_samples = 1000
    test_samples = 100
    seed = 0

    selected_device = jax.devices(args.device)[0]
    print(
        "device="
        f"{selected_device.platform} kind={getattr(selected_device, 'device_kind', 'unknown')}"
    )
    with jax.default_device(selected_device):
        input_grid = Grid.from_extent(nx=28, ny=28, dx_um=1.0, dy_um=1.0)
        spectrum = Spectrum.from_scalar(1.55)
        work_grid = recommend_nyquist_grid(
            grid=input_grid,
            spectrum=spectrum,
            nyquist_factor=nyquist_factor,
            min_padding_factor=2.0,
        )
        propagator = plan_propagation(
            mode="auto",
            grid=work_grid,
            spectrum=spectrum,
            distance_um=distance_um,
            precomputed_grid=work_grid,
            use_sampling_planner=False,
            nyquist_factor=nyquist_factor,
        )
        mask_nx = work_grid.nx // phase_mask_downsample
        mask_ny = work_grid.ny // phase_mask_downsample
        mask_grid = Grid.from_extent(
            nx=mask_nx,
            ny=mask_ny,
            dx_um=(work_grid.nx * work_grid.dx_um) / mask_nx,
            dy_um=(work_grid.ny * work_grid.dy_um) / mask_ny,
        )
        detector_masks = build_detector_bank(work_grid)
        detector_sensor = IntensitySensor(detector_masks=detector_masks, sum_wavelengths=True)

        x_train, y_train, x_test, y_test = load_mnist(Path("artifacts/data/mnist.npz"))
        x_train = x_train[:train_samples]
        y_train = y_train[:train_samples]
        x_test = x_test[:test_samples]
        y_test = y_test[:test_samples]
        x_train = resize_images_to_grid(x_train, work_grid)
        x_test = resize_images_to_grid(x_test, work_grid)

        key = jax.random.PRNGKey(seed)
        phase_params = 0.05 * jax.random.normal(
            key,
            (num_phase_layers, mask_grid.ny, mask_grid.nx),
            dtype=jnp.float32,
        )

        def logits_single(params_mask: jnp.ndarray, image_2d: jnp.ndarray) -> jnp.ndarray:
            module = build_onn_module(
                params_mask,
                work_grid,
                propagator,
                detector_sensor,
            )
            field = Field(
                data=image_2d[None, :, :].astype(jnp.complex64),
                grid=work_grid,
                spectrum=spectrum,
            )
            return module.measure(field)

        logits_batch = jax.vmap(logits_single, in_axes=(None, 0))

        def batch_loss(
            params: jnp.ndarray, images: jnp.ndarray, labels: jnp.ndarray
        ) -> jnp.ndarray:
            logits = logits_batch(params, images)
            log_probs = logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)
            nll = -log_probs[jnp.arange(labels.shape[0]), labels]
            return jnp.mean(nll)

        def batch_accuracy(params: jnp.ndarray, images: np.ndarray, labels: np.ndarray) -> float:
            logits = np.asarray(logits_batch(params, jnp.asarray(images)))
            pred = np.argmax(logits, axis=1)
            return float(np.mean(pred == labels))

        grad_fn = jax.grad(batch_loss)
        optimizer = optax.adam(learning_rate=learning_rate)
        opt_state = optimizer.init(phase_params)

        num_batches = int(np.ceil(len(x_train) / batch_size))
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            perm = np.random.default_rng(seed + epoch).permutation(len(x_train))
            x_train = x_train[perm]
            y_train = y_train[perm]

            epoch_losses: list[float] = []
            for bi in range(num_batches):
                lo = bi * batch_size
                hi = min((bi + 1) * batch_size, len(x_train))
                xb = jnp.asarray(x_train[lo:hi], dtype=jnp.float32)
                yb = jnp.asarray(y_train[lo:hi], dtype=jnp.int32)

                grads = grad_fn(phase_params, xb, yb)
                updates, opt_state = optimizer.update(grads, opt_state, phase_params)
                phase_params = optax.apply_updates(phase_params, updates)

                loss = batch_loss(phase_params, xb, yb)
                epoch_losses.append(float(loss))

            train_acc = batch_accuracy(phase_params, x_train, y_train)
            test_acc = batch_accuracy(phase_params, x_test, y_test)
            print(
                f"epoch={epoch:02d}/{epochs} "
                f"loss={np.mean(epoch_losses):.4f} "
                f"train_acc={train_acc:.4f} "
                f"test_acc={test_acc:.4f} "
                f"elapsed={time.time() - start_time:.2f}s "
            )

        module = build_onn_module(
            phase_params,
            work_grid,
            propagator,
            detector_sensor,
        )
        sample_idx = 0
        test_image = jnp.asarray(x_test[sample_idx], dtype=jnp.float32)
        sample_field = Field(
            data=test_image[None, :, :].astype(jnp.complex64),
            grid=work_grid,
            spectrum=spectrum,
        )

        # Plot field evolution
        fig_field, _ = plot_field_evolution(
            module=module,
            field_in=sample_field,
            mode="intensity",
            wavelength_idx=0,
            log_scale=False,
        )
        fig_field.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        out_dir = Path("artifacts")
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_field.savefig(out_dir / "onn_mnist_field_evolution.png", dpi=160)
        print("saved:", out_dir / "onn_mnist_field_evolution.png")


if __name__ == "__main__":
    main()
