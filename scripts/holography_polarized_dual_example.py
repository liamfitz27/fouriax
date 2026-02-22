#!/usr/bin/env python3
"""Optimize dual-polarization coherent holography with separate phase maps."""

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
    JonesMatrixLayer,
    OpticalModule,
    Spectrum,
    plan_propagation,
    recommend_nyquist_grid,
)

IMAGE_PATH = Path("/Users/liam/Downloads/logo.jpg")


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
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"image not found: {IMAGE_PATH}")

    grid = Grid.from_extent(nx=args.nx, ny=args.ny, dx_um=args.dx_um, dy_um=args.dy_um)
    base_target = load_logo_target(IMAGE_PATH, grid=grid)
    target_x = base_target
    target_y = jnp.rot90(base_target, k=2, axes=(0, 1))

    wavelength_um = float(args.wavelength_um)
    spectrum = Spectrum.from_scalar(wavelength_um)

    field_in = Field.plane_wave_jones(
        grid=grid,
        spectrum=spectrum,
        ex=1.0 + 0.0j,
        ey=1.0 + 0.0j,
    )
    precomputed_grid = recommend_nyquist_grid(
        grid=grid,
        spectrum=spectrum,
        nyquist_factor=args.nyquist_factor,
        min_padding_factor=args.min_padding_factor,
    )
    propagator = plan_propagation(
        mode="auto",
        grid=grid,
        spectrum=spectrum,
        distance_um=args.distance_um,
        use_sampling_planner=False,
        precomputed_grid=precomputed_grid,
        warn_on_regime_mismatch=False,
    )

    def loss_fn(raw_phase: jnp.ndarray) -> jnp.ndarray:
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
        module = OpticalModule(
            layers=(
                JonesMatrixLayer(jones_matrix=jones),
                propagator,
            )
        )
        out = module.forward(field_in)
        component_intensity = out.component_intensity()[0]
        ix = component_intensity[0]
        iy = component_intensity[1]
        ix_norm = ix / jnp.maximum(jnp.max(ix), 1e-12)
        iy_norm = iy / jnp.maximum(jnp.max(iy), 1e-12)
        return jnp.mean((ix_norm - target_x) ** 2) + jnp.mean((iy_norm - target_y) ** 2)

    key = jax.random.PRNGKey(args.seed)
    raw_phase = 0.1 * jax.random.normal(key, (2, grid.ny, grid.nx), dtype=jnp.float32)
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
    jxx_opt = jnp.exp(1j * phase_opt[0]).astype(jnp.complex64)
    jyy_opt = jnp.exp(1j * phase_opt[1]).astype(jnp.complex64)
    zeros_opt = jnp.zeros_like(jxx_opt)
    jones_opt = jnp.stack(
        [
            jnp.stack([jxx_opt, zeros_opt], axis=0),
            jnp.stack([zeros_opt, jyy_opt], axis=0),
        ],
        axis=0,
    )
    module_opt = OpticalModule(layers=(JonesMatrixLayer(jones_matrix=jones_opt), propagator))
    out_opt = module_opt.forward(field_in)
    component_intensity = out_opt.component_intensity()[0]
    recon_x = component_intensity[0]
    recon_y = component_intensity[1]
    recon_x_norm = recon_x / jnp.maximum(jnp.max(recon_x), 1e-12)
    recon_y_norm = recon_y / jnp.maximum(jnp.max(recon_y), 1e-12)

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "image_path": str(IMAGE_PATH),
        "grid": {"nx": grid.nx, "ny": grid.ny, "dx_um": grid.dx_um, "dy_um": grid.dy_um},
        "wavelength_um": wavelength_um,
        "distance_um": args.distance_um,
        "nyquist_factor": args.nyquist_factor,
        "min_padding_factor": args.min_padding_factor,
        "steps": args.steps,
        "learning_rate": args.lr,
        "seed": args.seed,
        "initial_loss": history[0],
        "best_loss": best_loss,
        "final_loss": history[-1],
        "propagator_type": type(propagator).__name__,
        "cross_talk_x_to_y_mse": float(jnp.mean((recon_x_norm - target_y) ** 2)),
        "cross_talk_y_to_x_mse": float(jnp.mean((recon_y_norm - target_x) ** 2)),
    }
    summary_path = out_dir / "holography_polarized_dual_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved: {summary_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

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
    fig.savefig(out_dir / "holography_polarized_dual_overview.png", dpi=160)
    plt.close(fig)
    print(f"saved: {out_dir / 'holography_polarized_dual_overview.png'}")


if __name__ == "__main__":
    main()
