"""Optimize shared square-pillar geometry in a propagated optical stack."""

#%% Imports
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from fouriax.optics import (
    Field,
    Grid,
    MetaAtomInterpolationLayer,
    MetaAtomLibrary,
    OpticalModule,
    PhaseMask,
    Spectrum,
    plan_propagation,
)
from fouriax.optim import optimize_optical_module

#%% Paths and Parameters
NPZ_PATH = Path("../../data/meta_atoms/square_pillar_0p7um_cell_sweep_results.npz")
ARTIFACTS_DIR = Path("artifacts")
PLOT_PATH = ARTIFACTS_DIR / "metaatom_opt_overview.png"
SUMMARY_PATH = ARTIFACTS_DIR / "metaatom_opt_summary.json"

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
SEED = 0
GRID_N = 64
GRID_DX_UM = 0.7
SELECTED_WAVELENGTH_UM = 1.30
DISTANCE_UM = 100.0
LR = 0.1
STEPS = 180


#%% Helper Functions
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


def main() -> None:
    #%% Setup
    library = load_square_pillar_library(NPZ_PATH)

    nearest_idx = int(jnp.argmin(jnp.abs(library.wavelengths_um - SELECTED_WAVELENGTH_UM)))
    wavelength_um = float(library.wavelengths_um[nearest_idx])
    grid = Grid.from_extent(nx=GRID_N, ny=GRID_N, dx_um=GRID_DX_UM, dy_um=GRID_DX_UM)
    spectrum = Spectrum.from_scalar(wavelength_um)
    field_in = Field.plane_wave(grid=grid, spectrum=spectrum)

    target_xy = (grid.nx // 2, grid.ny // 2)
    propagator = plan_propagation(
        mode="auto",
        grid=grid,
        spectrum=spectrum,
        distance_um=DISTANCE_UM,
    )

    side_axis = library.parameter_axes[0]
    min_bounds = jnp.array([side_axis[0]], dtype=jnp.float32)
    max_bounds = jnp.array([side_axis[-1]], dtype=jnp.float32)

    def build_module(raw_params: jnp.ndarray) -> OpticalModule:
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

    #%% Loss Function and Optimization
    def loss_fn(raw_params: jnp.ndarray) -> jnp.ndarray:
        module = build_module(raw_params)
        intensity = module.forward(field_in).intensity()
        center = intensity[0, target_xy[1], target_xy[0]]
        return -center

    key = jax.random.PRNGKey(SEED)
    raw_params = 0.1 * jax.random.normal(key, (grid.ny, grid.nx), dtype=jnp.float32)
    optimizer = optax.adam(learning_rate=LR)

    result = optimize_optical_module(
        init_params=raw_params,
        build_module=build_module,
        loss_fn=loss_fn,
        optimizer=optimizer,
        steps=STEPS,
        log_every=30,
    )

    #%% Evaluation
    final_intensity = np.asarray(result.best_module.forward(field_in).intensity())
    optimized_profile = final_intensity[0, target_xy[1], :]

    # Reference: phase profile for ideal spherical wavefront convergence at distance_um.
    x_um, y_um = grid.spatial_grid()
    k = 2.0 * jnp.pi / wavelength_um
    hyperbolic_phase = -k * (
        jnp.sqrt(x_um * x_um + y_um * y_um + DISTANCE_UM**2) - DISTANCE_UM
    )
    reference_module = OpticalModule(
        layers=(
            PhaseMask(phase_map_rad=hyperbolic_phase[None, :, :]),
            propagator,
        )
    )

    reference_intensity = np.asarray(reference_module.forward(field_in).intensity())
    reference_profile = reference_intensity[0, target_xy[1], :]

    final_layer = MetaAtomInterpolationLayer(
        library=library,
        raw_geometry_params=result.best_params,
        min_geometry_params=min_bounds,
        max_geometry_params=max_bounds,
    )
    optimized_side_map = np.asarray(final_layer.bounded_geometry_params()[0])
    #%% Plot Results
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    axes[0, 0].plot(result.history)
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
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    plt.colorbar(focus_im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].plot(optimized_profile, label="Optimized")
    axes[1, 1].plot(reference_profile, label="Hyperbolic-phase reference", linestyle="--")
    axes[1, 1].set_title("Center Row Profile")
    axes[1, 1].set_xlabel("x pixel")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
