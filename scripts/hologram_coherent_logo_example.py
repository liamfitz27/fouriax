#!/usr/bin/env python3
"""Optimize a phase-only coherent hologram for a red-logo binary target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
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

IMAGE_PATH = Path("/Users/liam/Downloads/logo.jpg")


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--dx-um", type=float, default=1.0)
    parser.add_argument("--dy-um", type=float, default=1.0)
    parser.add_argument("--wavelength-um", type=float, default=0.532)
    parser.add_argument("--distance-um", type=float, default=1200.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"image not found: {IMAGE_PATH}")

    grid = Grid.from_extent(nx=args.nx, ny=args.ny, dx_um=args.dx_um, dy_um=args.dy_um)
    spectrum = Spectrum.from_scalar(args.wavelength_um)
    target = load_logo_target(IMAGE_PATH, grid=grid)

    field_in = Field.plane_wave(grid=grid, spectrum=spectrum)
    propagator = plan_propagation(
        mode="auto",
        grid=grid,
        spectrum=spectrum,
        distance_um=args.distance_um,
    )

    def loss_fn(raw_phase: jnp.ndarray) -> jnp.ndarray:
        phase = 2.0 * jnp.pi * jax.nn.sigmoid(raw_phase)
        module = OpticalModule(
            layers=(
                PhaseMask(phase_map_rad=phase[None, :, :]),
                propagator,
            )
        )
        intensity = module.forward(field_in).intensity()[0]
        intensity_norm = intensity / jnp.maximum(jnp.max(intensity), 1e-12)
        return jnp.mean((intensity_norm - target) ** 2)

    key = jax.random.PRNGKey(args.seed)
    raw_phase = 0.1 * jax.random.normal(key, (grid.ny, grid.nx), dtype=jnp.float32)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(raw_phase)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    history: list[float] = []
    best_loss = float("inf")
    best_raw_phase = raw_phase

    for step in range(args.steps):
        loss, grad = value_and_grad(raw_phase)
        updates, opt_state = optimizer.update(grad, opt_state, raw_phase)
        raw_phase = optax.apply_updates(raw_phase, updates)
        loss_val = float(loss)
        history.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val
            best_raw_phase = raw_phase
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step={step:04d} loss={loss_val:.6f}")

    phase_opt = 2.0 * jnp.pi * jax.nn.sigmoid(best_raw_phase)
    module_opt = OpticalModule(
        layers=(
            PhaseMask(phase_map_rad=phase_opt[None, :, :]),
            propagator,
        )
    )
    recon = module_opt.forward(field_in).intensity()[0]
    recon_norm = recon / jnp.maximum(jnp.max(recon), 1e-12)

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "image_path": str(IMAGE_PATH),
        "grid": {"nx": grid.nx, "ny": grid.ny, "dx_um": grid.dx_um, "dy_um": grid.dy_um},
        "wavelength_um": args.wavelength_um,
        "distance_um": args.distance_um,
        "steps": args.steps,
        "learning_rate": args.lr,
        "seed": args.seed,
        "initial_loss": history[0],
        "best_loss": best_loss,
        "final_loss": history[-1],
        "propagator_type": type(propagator).__name__,
    }
    summary_path = out_dir / "hologram_coherent_logo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    np.save(out_dir / "hologram_coherent_logo_phase.npy", np.asarray(phase_opt))
    np.save(out_dir / "hologram_coherent_logo_target.npy", np.asarray(target))
    np.save(out_dir / "hologram_coherent_logo_recon.npy", np.asarray(recon_norm))
    print(f"saved: {summary_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].imshow(np.asarray(target), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Target (red->1, white->0)")
    axes[0, 1].imshow(np.asarray(recon_norm), cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("Reconstruction (normalized)")
    phase_im = axes[1, 0].imshow(np.asarray(phase_opt), cmap="twilight")
    axes[1, 0].set_title("Optimized Phase (rad)")
    plt.colorbar(phase_im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 1].plot(history)
    axes[1, 1].set_title("Loss History")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("MSE")
    axes[1, 1].grid(alpha=0.3)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "hologram_coherent_logo_overview.png", dpi=160)
    plt.close(fig)
    print(f"saved: {out_dir / 'hologram_coherent_logo_overview.png'}")


if __name__ == "__main__":
    main()
