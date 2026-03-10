"""Optimize a phase-only coherent hologram for a red-logo binary target."""

# %% Imports
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from PIL import Image

import fouriax as fx

# %% Paths and Parameters

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coherent Hologram Logo Example")
    parser.add_argument("--image-path", type=str, default="/Users/liam/Downloads/logo.jpg")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--dx-um", type=float, default=1.0)
    parser.add_argument("--dy-um", type=float, default=1.0)
    parser.add_argument("--wavelength-um", type=float, default=0.532)
    parser.add_argument("--distance-um", type=float, default=1200.0)
    parser.add_argument("--nyquist-factor", type=float, default=2.0)
    parser.add_argument("--min-padding-factor", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    return parser.parse_args()


ARGS = parse_args()

IMAGE_PATH = Path(ARGS.image_path)
ARTIFACTS_DIR = Path(ARGS.artifacts_dir)
PLOT_PATH = ARTIFACTS_DIR / "hologram_coherent_logo_overview.png"

SEED = ARGS.seed
NX = ARGS.nx
NY = ARGS.ny
DX_UM = ARGS.dx_um
DY_UM = ARGS.dy_um
WAVELENGTH_UM = ARGS.wavelength_um
DISTANCE_UM = ARGS.distance_um
NYQUIST_FACTOR = ARGS.nyquist_factor
MIN_PADDING_FACTOR = ARGS.min_padding_factor
STEPS = ARGS.steps
LR = ARGS.lr
PLOT = not ARGS.no_plot


# %% Helper Functions
def load_logo_target(path: Path, grid: fx.Grid) -> jnp.ndarray:
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
    # %% Setup
    grid = fx.Grid.from_extent(nx=NX, ny=NY, dx_um=DX_UM, dy_um=DY_UM)
    spectrum = fx.Spectrum.from_scalar(WAVELENGTH_UM)
    target = load_logo_target(IMAGE_PATH, grid=grid)

    field_in = fx.Field.plane_wave(grid=grid, spectrum=spectrum)
    propagator = fx.plan_propagation(
        mode="auto",
        grid=grid,
        spectrum=spectrum,
        distance_um=DISTANCE_UM,
        nyquist_factor=NYQUIST_FACTOR,
        min_padding_factor=MIN_PADDING_FACTOR,
    )

    def build_module(raw_phase: jnp.ndarray) -> fx.OpticalModule:
        phase = 2.0 * jnp.pi * jax.nn.sigmoid(raw_phase)
        return fx.OpticalModule(
            layers=(
                fx.PhaseMask(phase_map_rad=phase[None, :, :]),
                propagator,
            )
        )

    # %% Loss Function and Optimization
    def loss_fn(raw_phase: jnp.ndarray) -> jnp.ndarray:
        module = build_module(raw_phase)
        intensity = module.forward(field_in).intensity()[0]
        intensity_norm = intensity / jnp.maximum(jnp.max(intensity), 1e-12)
        return jnp.mean((intensity_norm - target) ** 2)

    key = jax.random.PRNGKey(SEED)
    raw_phase = 0.1 * jax.random.normal(key, (grid.ny, grid.nx), dtype=jnp.float32)
    optimizer = optax.adam(LR)
    result = fx.optim.optimize_optical_module(
        init_params=raw_phase,
        build_module=build_module,
        loss_fn=loss_fn,
        optimizer=optimizer,
        steps=STEPS,
        log_every=50,
    )

    # %% Evaluation
    phase_opt = 2.0 * jnp.pi * jax.nn.sigmoid(result.best_params)
    recon = result.best_module.forward(field_in).intensity()[0]
    recon_norm = recon / jnp.maximum(jnp.max(recon), 1e-12)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # %% Plot Results
    if PLOT:
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
