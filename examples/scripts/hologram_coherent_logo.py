"""Optimize a phase-only coherent hologram for a red-logo binary target."""

#%% Imports
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from PIL import Image

from fouriax.optics import (
    Field,
    Grid,
    OpticalModule,
    PhaseMask,
    Spectrum,
    plan_propagation,
)
from fouriax.optim import optimize_optical_module

#%% Paths and Parameters
IMAGE_PATH = Path("/Users/liam/Downloads/logo.jpg")
ARTIFACTS_DIR = Path("artifacts")
PLOT_PATH = ARTIFACTS_DIR / "hologram_coherent_logo_overview.png"

SEED = 0
NX = 128
NY = 128
DX_UM = 1.0
DY_UM = 1.0
WAVELENGTH_UM = 0.532
DISTANCE_UM = 1200.0
NYQUIST_FACTOR = 2.0
MIN_PADDING_FACTOR = 2.0
STEPS = 400
LR = 0.03

#%% Helper Functions
def load_logo_target(path: Path, grid: Grid) -> jnp.ndarray:
    """Load image and convert to binary target: white->0, red-logo->1."""
    img = Image.open(path).convert("RGB").resize((grid.nx, grid.ny), Image.Resampling.BILINEAR)
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    # Red logo: high red with suppressed green/blue channels.
    red_mask = (r >= 0.55) & (g <= 0.45) & (b <= 0.45)
    target = red_mask.astype(np.float32)
    return jnp.asarray(target, dtype=jnp.float32)


def main() -> None:
    #%% Setup
    grid = Grid.from_extent(nx=NX, ny=NY, dx_um=DX_UM, dy_um=DY_UM)
    spectrum = Spectrum.from_scalar(WAVELENGTH_UM)
    target = load_logo_target(IMAGE_PATH, grid=grid)

    field_in = Field.plane_wave(grid=grid, spectrum=spectrum)
    propagator = plan_propagation(
        mode="auto",
        grid=grid,
        spectrum=spectrum,
        distance_um=DISTANCE_UM,
        nyquist_factor=NYQUIST_FACTOR,
        min_padding_factor=MIN_PADDING_FACTOR,
    )

    def build_module(raw_phase: jnp.ndarray) -> OpticalModule:
        phase = 2.0 * jnp.pi * jax.nn.sigmoid(raw_phase)
        return OpticalModule(
            layers=(
                PhaseMask(phase_map_rad=phase[None, :, :]),
                propagator,
            )
        )

    #%% Loss Function and Optimization
    def loss_fn(raw_phase: jnp.ndarray) -> jnp.ndarray:
        module = build_module(raw_phase)
        intensity = module.forward(field_in).intensity()[0]
        intensity_norm = intensity / jnp.maximum(jnp.max(intensity), 1e-12)
        return jnp.mean((intensity_norm - target) ** 2)

    key = jax.random.PRNGKey(SEED)
    raw_phase = 0.1 * jax.random.normal(key, (grid.ny, grid.nx), dtype=jnp.float32)
    optimizer = optax.adam(LR)
    result = optimize_optical_module(
        init_params=raw_phase,
        build_module=build_module,
        loss_fn=loss_fn,
        optimizer=optimizer,
        steps=STEPS,
        log_every=50,
    )

    #%% Evaluation
    phase_opt = 2.0 * jnp.pi * jax.nn.sigmoid(result.best_params)
    recon = result.best_module.forward(field_in).intensity()[0]
    recon_norm = recon / jnp.maximum(jnp.max(recon), 1e-12)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    #%% Plot Results
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].imshow(np.asarray(target), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Target (red->1, white->0)")
    axes[0, 1].imshow(np.asarray(recon_norm), cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("Reconstruction (normalized)")
    phase_im = axes[1, 0].imshow(np.asarray(phase_opt), cmap="twilight")
    axes[1, 0].set_title("Optimized Phase (rad)")
    plt.colorbar(phase_im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 1].plot(result.history)
    axes[1, 1].set_title("Loss History")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("MSE")
    axes[1, 1].grid(alpha=0.3)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
