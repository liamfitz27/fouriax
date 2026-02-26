"""Optimize dual-polarization coherent holography with separate phase maps."""

#%% Imports
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from PIL import Image

from fouriax.example_utils.optim import optimize_optical_module
from fouriax.optics import (
    Field,
    Grid,
    JonesMatrixLayer,
    OpticalModule,
    Spectrum,
    plan_propagation,
)

#%% Paths and Parameters
IMAGE_PATH = Path("/Users/liam/Downloads/logo.jpg")
ARTIFACTS_DIR = Path("artifacts")
PLOT_PATH = ARTIFACTS_DIR / "holography_polarized_dual_overview.png"

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
    red_mask = (r >= 0.55) & (g <= 0.45) & (b <= 0.45)
    return jnp.asarray(red_mask.astype(np.float32), dtype=jnp.float32)


def main() -> None:
    #%% Setup
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"image not found: {IMAGE_PATH}")

    grid = Grid.from_extent(nx=NX, ny=NY, dx_um=DX_UM, dy_um=DY_UM)
    base_target = load_logo_target(IMAGE_PATH, grid=grid)
    target_x = base_target
    target_y = jnp.rot90(base_target, k=2, axes=(0, 1))

    spectrum = Spectrum.from_scalar(WAVELENGTH_UM)

    field_in = Field.plane_wave_jones(
        grid=grid,
        spectrum=spectrum,
        ex=1.0 + 0.0j,
        ey=1.0 + 0.0j,
    )
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
        jxx = jnp.exp(1j * phase[0]).astype(jnp.complex64)
        jyy = jnp.exp(1j * phase[1]).astype(jnp.complex64)
        zeros = jnp.zeros_like(jxx)
        jones = jnp.stack(
            [
                jnp.stack([jxx, zeros], axis=0),
                jnp.stack([zeros, jyy], axis=0),
            ],
            axis=0,
        )
        return OpticalModule(
            layers=(
                JonesMatrixLayer(jones_matrix=jones),
                propagator,
            )
        )

    #%% Loss Function and Optimization
    def loss_fn(raw_phase: jnp.ndarray) -> jnp.ndarray:
        module = build_module(raw_phase)
        out = module.forward(field_in)
        component_intensity = out.component_intensity()[0]
        ix = component_intensity[0]
        iy = component_intensity[1]
        ix_norm = ix / jnp.maximum(jnp.max(ix), 1e-12)
        iy_norm = iy / jnp.maximum(jnp.max(iy), 1e-12)
        return jnp.mean((ix_norm - target_x) ** 2) + jnp.mean((iy_norm - target_y) ** 2)

    key = jax.random.PRNGKey(SEED)
    raw_phase = 0.1 * jax.random.normal(key, (2, grid.ny, grid.nx), dtype=jnp.float32)
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
    out_opt = result.best_module.forward(field_in)
    component_intensity = out_opt.component_intensity()[0]
    recon_x = component_intensity[0]
    recon_y = component_intensity[1]
    recon_x_norm = recon_x / jnp.maximum(jnp.max(recon_x), 1e-12)
    recon_y_norm = recon_y / jnp.maximum(jnp.max(recon_y), 1e-12)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    #%% Plot Results
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes[0, 0].imshow(np.asarray(target_x), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Target X (original)")
    axes[0, 1].imshow(np.asarray(recon_x_norm), cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("Recon X")
    sx_im = axes[0, 2].imshow(np.asarray(phase_opt[0]), cmap="twilight")
    axes[0, 2].set_title("Optimized Phase X (rad)")
    plt.colorbar(sx_im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(np.asarray(target_y), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("Target Y (rotated 180 deg)")
    axes[1, 1].imshow(np.asarray(recon_y_norm), cmap="magma", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title("Recon Y")
    sy_im = axes[1, 2].imshow(np.asarray(phase_opt[1]), cmap="twilight")
    axes[1, 2].set_title("Optimized Phase Y (rad)")
    plt.colorbar(sy_im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
