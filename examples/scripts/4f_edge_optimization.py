"""Optimize a phase-only Fourier-plane filter for edge detection in a 4f system."""

#%% Imports
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from fouriax.example_utils import optimize_dataset_optical_module
from fouriax.optics import (
    ComplexMask,
    Field,
    Grid,
    IntensitySensor,
    OpticalModule,
    Spectrum,
    ThinLens,
)
from fouriax.optics.propagation import ASMPropagator

#%% Paths and Parameters
ARTIFACTS_DIR = Path("artifacts")
PLOT_PATH = ARTIFACTS_DIR / "4f_edge_optimization.png"
SUMMARY_PATH = ARTIFACTS_DIR / "4f_edge_optimization_summary.json"

SEED = 0
WAVELENGTH_UM = 0.532
N_MEDIUM = 1.0
GRID_N = 128
GRID_DX_UM = 2.0
EPOCHS = 25
LR = 5e-3
N_TRAIN_SCENES = 1000
N_TEST_SCENES = 100

#%% Helper Functions
def random_scene(key: jax.Array, grid: Grid) -> jnp.ndarray:
    noise = jax.random.normal(key, grid.shape)
    k = jnp.fft.fftn(noise, axes=(-2, -1))
    fx, fy = grid.frequency_grid()
    sigma_freq = 1.0 / (32.0 * grid.dx_um)
    lpf = jnp.exp(-(fx**2 + fy**2) / (2 * sigma_freq**2))
    smooth = jnp.real(jnp.fft.ifftn(k * lpf, axes=(-2, -1)))
    return (smooth > 0).astype(jnp.float32)


def edge_target(scene: jnp.ndarray) -> jnp.ndarray:
    padded = jnp.pad(scene, 1, mode="edge")
    gx = padded[1:-1, 2:] - padded[1:-1, :-2]
    gy = padded[2:, 1:-1] - padded[:-2, 1:-1]
    mag = jnp.sqrt(gx**2 + gy**2)
    return mag / jnp.maximum(jnp.max(mag), 1e-12)


def sampling_matched_f(grid: Grid) -> float:
    return N_MEDIUM * grid.nx * grid.dx_um**2 / WAVELENGTH_UM


def analytical_spiral_phase(grid: Grid) -> jnp.ndarray:
    x, y = grid.spatial_grid()
    return jnp.arctan2(y, x) + jnp.pi  # [0, 2π], centered on optical axis


def make_test_scene(grid: Grid) -> jnp.ndarray:
    x, y = grid.spatial_grid()
    half = grid.nx * grid.dx_um / 2.0
    scene = jnp.zeros(grid.shape, dtype=jnp.float32)
    scene = scene + ((jnp.abs(x - 0.2 * half) < 0.15 * half)
                      & (jnp.abs(y + 0.1 * half) < 0.15 * half)).astype(jnp.float32)
    scene = scene + ((jnp.abs(x + 0.3 * half) < 0.1 * half)
                      & (jnp.abs(y - 0.25 * half) < 0.1 * half)).astype(jnp.float32)
    r = jnp.sqrt((x + 0.1 * half) ** 2 + (y + 0.3 * half) ** 2)
    scene = scene + (r < 0.12 * half).astype(jnp.float32)
    return jnp.clip(scene, 0.0, 1.0)


def measure_scene(
    module: OpticalModule,
    scene: jnp.ndarray,
    grid: Grid,
    spectrum: Spectrum,
) -> jnp.ndarray:
    field_in = Field.plane_wave(grid=grid, spectrum=spectrum).apply_amplitude(
        scene[None, :, :],
    )
    return module.measure(field_in)[::-1, ::-1]


def main() -> None:
    #%% Setup
    grid = Grid.from_extent(nx=GRID_N, ny=GRID_N, dx_um=GRID_DX_UM, dy_um=GRID_DX_UM)
    spectrum = Spectrum.from_scalar(WAVELENGTH_UM)
    f_um = sampling_matched_f(grid)

    prop = ASMPropagator(
        distance_um=f_um, use_sampling_planner=False, warn_on_regime_mismatch=False,
    )
    lens = ThinLens(focal_length_um=f_um)

    def build_module(raw_phase: jnp.ndarray) -> OpticalModule:
        phase = 2.0 * jnp.pi * jax.nn.sigmoid(raw_phase)
        return OpticalModule(
            layers=(
                prop, lens, prop,
                ComplexMask(phase_map_rad=phase),
                prop, lens, prop,
            ),
            sensor=IntensitySensor(sum_wavelengths=True),
        )

    #%% Training Data
    key = jax.random.PRNGKey(SEED)
    key, *train_keys = jax.random.split(key, N_TRAIN_SCENES + 1)
    train_scenes = jnp.stack([random_scene(k, grid) for k in train_keys])
    train_targets = jnp.stack([edge_target(s) for s in train_scenes])
    key, *test_keys = jax.random.split(key, N_TEST_SCENES + 1)
    test_scenes = jnp.stack([random_scene(k, grid) for k in test_keys])
    test_targets = jnp.stack([edge_target(s) for s in test_scenes])

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for col in range(4):
        axes[0, col].imshow(np.asarray(train_scenes[col]), cmap="gray")
        axes[0, col].set_title(f"Scene {col}")
        axes[1, col].imshow(np.asarray(train_targets[col]), cmap="hot")
        axes[1, col].set_title(f"Edges {col}")
    for ax in axes.flat:
        ax.set_xlabel("x pixel")
        ax.set_ylabel("y pixel")
    fig.tight_layout()
    save_path = ARTIFACTS_DIR / "4f_edge_optimization_scenes.png"
    fig.savefig(save_path)
    plt.close(fig)

    #%% Loss Function and Optimization
    key, init_key = jax.random.split(key)
    raw_phase = 0.1 * jax.random.normal(init_key, (grid.ny, grid.nx))
    optimizer = optax.adam(LR)

    n_test_eval = min(10, N_TEST_SCENES)  # evaluate on a subset for speed
    val_data = (test_scenes[:n_test_eval], test_targets[:n_test_eval])
    
    def loss_fn(
        params: jnp.ndarray,
        batch: tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        scenes, targets = batch
        scene = scenes[0]
        target = targets[0]
        module = build_module(params)
        out = measure_scene(module, scene, grid, spectrum)
        out_n = out / jnp.maximum(jnp.max(out), 1e-12)
        return jnp.mean((out_n - target) ** 2)

    result = optimize_dataset_optical_module(
        init_params=raw_phase,
        build_module=build_module,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_data=(train_scenes, train_targets),
        batch_size=1,
        epochs=EPOCHS,
        val_data=val_data,
        seed=SEED,
    )
    train_history = result.params_result.train_loss_history
    test_history = [
        (record.step, record.metrics["val_loss"]) for record in result.params_result.val_history
    ]

    #%% Evaluation
    final_phase = np.asarray(2.0 * jnp.pi * jax.nn.sigmoid(result.params_result.best_params))
    test_scene = make_test_scene(grid)
    test_target = edge_target(test_scene)
    test_out = np.asarray(measure_scene(result.best_module, test_scene, grid, spectrum))
    test_out_n = test_out / np.max(test_out)
    cc = float(np.corrcoef(test_out_n.ravel(), np.asarray(test_target).ravel())[0, 1])
    print(f"Test-scene correlation: {cc:.4f}")

    spiral = np.asarray(analytical_spiral_phase(grid))

    #%% Plot Results
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].imshow(np.asarray(test_scene), cmap="gray")
    axes[0, 0].set_title("Test scene (held out)")

    axes[0, 1].imshow(np.asarray(test_target), cmap="hot")
    axes[0, 1].set_title("Target edges")

    im = axes[0, 2].imshow(test_out_n, cmap="hot")
    axes[0, 2].set_title(f"4f output (ρ = {cc:.3f})")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im_o = axes[1, 0].imshow(final_phase, cmap="twilight", vmin=0, vmax=2 * np.pi)
    axes[1, 0].set_title("Optimized phase")
    fig.colorbar(im_o, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_s = axes[1, 1].imshow(spiral, cmap="twilight", vmin=0, vmax=2 * np.pi)
    axes[1, 1].set_title("Analytical spiral phase")
    fig.colorbar(im_s, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].plot(train_history, alpha=0.5, label="Train (per-sample)")
    test_steps, test_vals = zip(*test_history, strict=True)
    axes[1, 2].plot(test_steps, test_vals, "o-", markersize=3, label="Test (mean)")
    axes[1, 2].set_title("Loss history")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("MSE")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(alpha=0.3)

    for ax in axes.flat:
        if ax.images:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
