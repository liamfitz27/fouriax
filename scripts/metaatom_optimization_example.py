#!/usr/bin/env python3
"""Optimize shared square-pillar geometry in a propagated optical stack."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fouriax.optics import (
    Field,
    Grid,
    MetaAtomInterpolationLayer,
    MetaAtomLibrary,
    OpticalLayer,
    OpticalModule,
    PhaseMask,
    Spectrum,
    plan_propagation,
)

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


def load_square_pillar_library(npz_path: Path) -> MetaAtomLibrary:
    """Load LUT from NPZ keys: freqs [Hz], side_lengths [m], trans, phase."""
    with np.load(npz_path) as data:
        freqs_hz = np.asarray(data["freqs"], dtype=np.float64).reshape(-1)
        side_lengths_m = np.asarray(data["side_lengths"], dtype=np.float64).reshape(-1)
        trans = np.asarray(data["trans"], dtype=np.float64)
        phase = np.asarray(data["phase"], dtype=np.float64)

    wavelengths_um = (SPEED_OF_LIGHT_M_PER_S / freqs_hz) * 1e6
    side_lengths_um = side_lengths_m * 1e6

    wav_order = np.argsort(wavelengths_um)
    side_order = np.argsort(side_lengths_um)

    wavelengths_um = wavelengths_um[wav_order]
    side_lengths_um = side_lengths_um[side_order]

    trans = trans[side_order, :][:, wav_order]
    phase = phase[side_order, :][:, wav_order]

    transmission_complex = trans.T * np.exp(1j * phase.T)

    return MetaAtomLibrary.from_complex(
        wavelengths_um=jnp.asarray(wavelengths_um, dtype=jnp.float32),
        parameter_axes=(jnp.asarray(side_lengths_um, dtype=jnp.float32),),
        transmission_complex=jnp.asarray(transmission_complex),
    )


def build_module(
    library: MetaAtomLibrary,
    raw_params: jnp.ndarray,
    min_bounds: jnp.ndarray,
    max_bounds: jnp.ndarray,
    propagator: OpticalLayer,
) -> OpticalModule:
    return OpticalModule(
        layers=(
            MetaAtomInterpolationLayer(
                library=library,
                raw_geometry_params=raw_params,
                min_geometry_params=min_bounds,
                max_geometry_params=max_bounds,
            ),
            propagator,
        )
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    npz_path = repo_root / "data" / "meta_atoms" / "square_pillar_0p7um_cell_sweep_results.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing sweep file: {npz_path}")

    library = load_square_pillar_library(npz_path)

    selected_wavelength_um = 1.30
    nearest_idx = int(jnp.argmin(jnp.abs(library.wavelengths_um - selected_wavelength_um)))
    wavelength_um = float(library.wavelengths_um[nearest_idx])
    frequency_hz = SPEED_OF_LIGHT_M_PER_S / (wavelength_um * 1e-6)

    grid = Grid.from_extent(nx=64, ny=64, dx_um=0.7, dy_um=0.7)
    spectrum = Spectrum.from_scalar(wavelength_um)
    field_in = Field.plane_wave(grid=grid, spectrum=spectrum)

    distance_um = 100.0
    target_xy = (grid.nx // 2, grid.ny // 2)

    propagator = plan_propagation(
        mode="auto",
        grid=grid,
        spectrum=spectrum,
        distance_um=distance_um,
    )

    side_axis = library.parameter_axes[0]
    min_bounds = jnp.array([side_axis[0]], dtype=jnp.float32)
    max_bounds = jnp.array([side_axis[-1]], dtype=jnp.float32)

    def loss_fn(raw_params: jnp.ndarray) -> jnp.ndarray:
        module = build_module(
            library=library,
            raw_params=raw_params,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            propagator=propagator,
        )
        intensity = module.forward(field_in).intensity()
        center = intensity[0, target_xy[1], target_xy[0]]
        return -center

    key = jax.random.PRNGKey(0)
    raw_params = 0.1 * jax.random.normal(key, (grid.ny, grid.nx), dtype=jnp.float32)

    optimizer = optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(raw_params)
    value_and_grad = jax.value_and_grad(loss_fn)

    steps = 180
    history: list[float] = []
    for step in range(steps):
        loss, grads = value_and_grad(raw_params)
        updates, opt_state = optimizer.update(grads, opt_state, raw_params)
        raw_params = optax.apply_updates(raw_params, updates)
        history.append(float(loss))
        if step % 30 == 0 or step == steps - 1:
            print(f"step={step:03d} loss={float(loss):.6f}")

    final_module = build_module(
        library=library,
        raw_params=raw_params,
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        propagator=propagator,
    )

    final_intensity = np.asarray(final_module.forward(field_in).intensity())
    final_center_intensity = float(final_intensity[0, target_xy[1], target_xy[0]])
    optimized_profile = final_intensity[0, target_xy[1], :]

    # Reference: phase profile for ideal spherical wavefront convergence at distance_um.
    x_um, y_um = grid.spatial_grid()
    wavelength_um = float(spectrum.wavelengths_um[0])
    k = 2.0 * jnp.pi / wavelength_um
    hyperbolic_phase = -k * (
        jnp.sqrt(x_um * x_um + y_um * y_um + distance_um**2) - distance_um
    )
    reference_module = OpticalModule(
        layers=(
            PhaseMask(phase_map_rad=hyperbolic_phase[None, :, :]),
            propagator,
        )
    )

    reference_intensity = np.asarray(reference_module.forward(field_in).intensity())
    reference_center_intensity = float(reference_intensity[0, target_xy[1], target_xy[0]])
    reference_profile = reference_intensity[0, target_xy[1], :]

    final_layer = MetaAtomInterpolationLayer(
        library=library,
        raw_geometry_params=raw_params,
        min_geometry_params=min_bounds,
        max_geometry_params=max_bounds,
    )
    optimized_side_map = np.asarray(final_layer.bounded_geometry_params()[0])
    optimized_transmission = complex(
        np.asarray(
            library.interpolate_complex(
                final_layer.bounded_geometry_params()[:, target_xy[1], target_xy[0]][None, :],
                spectrum.wavelengths_um,
            )
        )[0, 0]
    )

    summary = {
        "sweep_file": str(npz_path.relative_to(repo_root)),
        "steps": steps,
        "learning_rate": 0.10,
        "selected_wavelength_um": wavelength_um,
        "selected_frequency_hz": frequency_hz,
        "side_length_range_um": [float(min_bounds[0]), float(max_bounds[0])],
        "optimized_side_length_um_mean": float(np.mean(optimized_side_map)),
        "optimized_side_length_um_min": float(np.min(optimized_side_map)),
        "optimized_side_length_um_max": float(np.max(optimized_side_map)),
        "initial_loss": history[0],
        "final_loss": history[-1],
        "optimized_transmission_abs": float(abs(optimized_transmission)),
        "optimized_transmission_phase_rad": float(np.angle(optimized_transmission)),
        "final_center_intensity": final_center_intensity,
        "reference_center_intensity": reference_center_intensity,
        "distance_um": distance_um,
    }

    out_dir = repo_root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "metaatom_opt_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved: {summary_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    axes[0, 0].plot(history)
    axes[0, 0].set_title("Loss History")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(alpha=0.3)

    side_im = axes[0, 1].imshow(optimized_side_map, cmap="viridis")
    axes[0, 1].set_title("Optimized Meta-Atom Side-Lengths")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    plt.colorbar(side_im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    focus_im = axes[1, 0].imshow(final_intensity[0], cmap="inferno")
    axes[1, 0].set_title("Optimized 2D Focal Spot")
    plt.colorbar(focus_im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].plot(optimized_profile, label="Optimized")
    axes[1, 1].plot(reference_profile, label="Hyperbolic-phase reference", linestyle="--")
    axes[1, 1].set_title("Center Row Profile")
    axes[1, 1].set_xlabel("x pixel")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    plot_path = out_dir / "metaatom_opt_overview.png"
    fig.savefig(plot_path, dpi=160)
    print(f"saved: {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
