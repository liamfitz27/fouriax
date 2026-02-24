"""Optimize a phase-only Fourier-plane filter for edge detection in a 4f system."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

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

ARTIFACTS_DIR = Path("artifacts")
PLOT_PATH = ARTIFACTS_DIR / "4f_edge_optimization_example.png"

WAVELENGTH_UM = 0.532
N_MEDIUM = 1.0
GRID_N = 128
GRID_DX_UM = 2.0
STEPS = 1000
LR = 1e-1
N_TRAIN_SCENES = 1000
N_TEST_SCENES = 100


def _sampling_matched_f(grid: Grid) -> float:
    return N_MEDIUM * grid.nx * grid.dx_um**2 / WAVELENGTH_UM


def _random_scene(key: jax.Array, grid: Grid) -> jnp.ndarray:
    noise = jax.random.normal(key, grid.shape)
    k = jnp.fft.fftn(noise, axes=(-2, -1))
    fx, fy = grid.frequency_grid()
    sigma_freq = 1.0 / (32.0 * grid.dx_um)
    lpf = jnp.exp(-(fx**2 + fy**2) / (2 * sigma_freq**2))
    smooth = jnp.real(jnp.fft.ifftn(k * lpf, axes=(-2, -1)))
    return (smooth > 0).astype(jnp.float32)


def _edge_target(scene: jnp.ndarray) -> jnp.ndarray:
    padded = jnp.pad(scene, 1, mode="edge")
    gx = padded[1:-1, 2:] - padded[1:-1, :-2]
    gy = padded[2:, 1:-1] - padded[:-2, 1:-1]
    mag = jnp.sqrt(gx**2 + gy**2)
    return mag / jnp.maximum(jnp.max(mag), 1e-12)


def _analytical_spiral_phase(grid: Grid) -> jnp.ndarray:
    x, y = grid.spatial_grid()
    return jnp.arctan2(y, x) + jnp.pi  # [0, 2π], centered on optical axis


def _make_test_scene(grid: Grid) -> jnp.ndarray:
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


def _apply_4f(
    phase: jnp.ndarray,
    scene: jnp.ndarray,
    grid: Grid,
    spectrum: Spectrum,
    prop: ASMPropagator,
    lens: ThinLens,
) -> jnp.ndarray:
    field_in = Field.plane_wave(grid=grid, spectrum=spectrum).apply_amplitude(
        scene[None, :, :],
    )
    module = OpticalModule(
        layers=(
            prop, lens, prop,
            ComplexMask(phase_map_rad=phase),
            prop, lens, prop,
        ),
        sensor=IntensitySensor(sum_wavelengths=True),
    )
    return module.measure(field_in)[::-1, ::-1]


def main() -> None:
    grid = Grid.from_extent(nx=GRID_N, ny=GRID_N, dx_um=GRID_DX_UM, dy_um=GRID_DX_UM)
    spectrum = Spectrum.from_scalar(WAVELENGTH_UM)
    f_um = _sampling_matched_f(grid)

    prop = ASMPropagator(
        distance_um=f_um, use_sampling_planner=False, warn_on_regime_mismatch=False,
    )
    lens = ThinLens(focal_length_um=f_um)

    # Pre-generate fixed training and test scenes.
    key = jax.random.PRNGKey(42)
    key, *train_keys = jax.random.split(key, N_TRAIN_SCENES + 1)
    train_scenes = jnp.stack([_random_scene(k, grid) for k in train_keys])
    train_targets = jnp.stack([_edge_target(s) for s in train_scenes])
    key, *test_keys = jax.random.split(key, N_TEST_SCENES + 1)
    test_scenes = jnp.stack([_random_scene(k, grid) for k in test_keys])
    test_targets = jnp.stack([_edge_target(s) for s in test_scenes])

    def loss_fn(raw_phase: jnp.ndarray, scene: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        phase = 2.0 * jnp.pi * jax.nn.sigmoid(raw_phase)
        out = _apply_4f(phase, scene, grid, spectrum, prop, lens)
        out_n = out / jnp.maximum(jnp.max(out), 1e-12)
        return jnp.mean((out_n - target) ** 2)

    @jax.jit
    def eval_loss(raw_phase: jnp.ndarray, scene: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        phase = 2.0 * jnp.pi * jax.nn.sigmoid(raw_phase)
        out = _apply_4f(phase, scene, grid, spectrum, prop, lens)
        out_n = out / jnp.maximum(jnp.max(out), 1e-12)
        return jnp.mean((out_n - target) ** 2)

    key, init_key = jax.random.split(key)
    raw_phase = 0.1 * jax.random.normal(init_key, (grid.ny, grid.nx))
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(raw_phase)
    vg = jax.jit(jax.value_and_grad(loss_fn))

    n_test_eval = min(10, N_TEST_SCENES)  # evaluate on a subset for speed
    train_history: list[float] = []
    test_history: list[tuple[int, float]] = []
    for step in range(STEPS):
        scene_idx = step % N_TRAIN_SCENES
        loss, grad = vg(raw_phase, train_scenes[scene_idx], train_targets[scene_idx])
        updates, opt_state = optimizer.update(grad, opt_state, raw_phase)
        raw_phase = optax.apply_updates(raw_phase, updates)
        train_history.append(float(loss))
        if step % 50 == 0 or step == STEPS - 1:
            test_losses = [float(eval_loss(raw_phase, test_scenes[i], test_targets[i]))
                           for i in range(n_test_eval)]
            test_loss = float(np.mean(test_losses))
            test_history.append((step, test_loss))
            print(f"step={step:03d}  train={float(loss):.6f}  test={test_loss:.6f}")

    # Evaluate on held-out structured test scene.
    final_phase = np.asarray(2.0 * jnp.pi * jax.nn.sigmoid(raw_phase))
    test_scene = _make_test_scene(grid)
    test_target = _edge_target(test_scene)
    test_out = np.asarray(_apply_4f(final_phase, test_scene, grid, spectrum, prop, lens))
    test_out_n = test_out / np.max(test_out)
    cc = float(np.corrcoef(test_out_n.ravel(), np.asarray(test_target).ravel())[0, 1])
    print(f"Test-scene correlation: {cc:.4f}")

    spiral = np.asarray(_analytical_spiral_phase(grid))

    # --- plot ---
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
    axes[1, 2].set_xlabel("Step")
    axes[1, 2].set_ylabel("MSE")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(alpha=0.3)

    for ax in axes.flat:
        if ax.images:
            ax.set_xlabel("x pixel")
            ax.set_ylabel("y pixel")

    fig.tight_layout()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"Saved: {PLOT_PATH}")

    summary = {
        "steps": STEPS,
        "learning_rate": LR,
        "n_train_scenes": N_TRAIN_SCENES,
        "n_test_scenes": N_TEST_SCENES,
        "initial_train_loss": train_history[0],
        "final_train_loss": train_history[-1],
        "final_test_loss": test_history[-1][1],
        "test_correlation": cc,
        "grid_n": GRID_N,
        "f_um": f_um,
    }
    with (ARTIFACTS_DIR / "4f_edge_optimization_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
