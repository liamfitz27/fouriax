"""4f correlator — locate a small target pattern inside a larger scene.

Uses a proper 4f geometry (input at front focal plane) with a matched
filter H = F*{target} at the Fourier plane.  Compares the physical
output to a direct FFT cross-correlation as ground truth.
"""

# %% Imports
from __future__ import annotations

# %% Paths and Parameters
import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from fouriax.optics import (
    ComplexMask,
    DetectorArray,
    Field,
    Grid,
    OpticalModule,
    Spectrum,
    ThinLens,
    plan_propagation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="4f Correlator Example")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--wavelength-um", type=float, default=0.532)
    parser.add_argument("--n-medium", type=float, default=1.0)
    parser.add_argument("--grid-n", type=int, default=128)
    parser.add_argument("--grid-dx-um", type=float, default=2.0)
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    return parser.parse_args()


ARGS = parse_args()

ARTIFACTS_DIR = Path(ARGS.artifacts_dir)
PLOT_PATH = ARTIFACTS_DIR / "4f_correlator.png"

WAVELENGTH_UM = ARGS.wavelength_um
N_MEDIUM = ARGS.n_medium
GRID_N = ARGS.grid_n
GRID_DX_UM = ARGS.grid_dx_um
PLOT = not ARGS.no_plot


# %% Helper Functions
def _raw_correlate(scene: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Ground-truth cross-correlation intensity via direct FFT."""
    f_scene = jnp.fft.fftn(jnp.fft.ifftshift(scene), axes=(-2, -1))
    f_target = jnp.fft.fftn(jnp.fft.ifftshift(target), axes=(-2, -1))
    corr = jnp.fft.ifftn(f_scene * jnp.conj(f_target), axes=(-2, -1))
    return jnp.abs(jnp.fft.fftshift(corr)) ** 2


def _sampling_matched_focal_length(grid: Grid) -> float:
    """f = n·N·dx² / λ"""
    return N_MEDIUM * grid.nx * grid.dx_um**2 / WAVELENGTH_UM


def _matched_filter(target: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fourier-plane matched filter: (amplitude, phase_rad).

    Built in the centred (fftshift) convention to match the physical
    Fourier plane, where DC sits at the optical axis.
    """
    ft = jnp.fft.fftshift(
        jnp.fft.fftn(jnp.fft.ifftshift(target), axes=(-2, -1)),
        axes=(-2, -1),
    )
    amplitude = jnp.abs(ft) / jnp.max(jnp.abs(ft))
    phase_rad = -jnp.angle(ft)
    return amplitude, phase_rad


def _paraxial_validity_constraint_fom() -> float:
    """Dimensionless paraxial figure-of-merit (r_max / f)^2 for this grid pitch."""
    return (WAVELENGTH_UM / (2 * N_MEDIUM * GRID_DX_UM)) ** 2


def _make_rect(grid: Grid, cx_f: float, cy_f: float, w_f: float, h_f: float) -> jnp.ndarray:
    """Binary rectangle; positions/sizes are fractions of grid half-extent."""
    half = grid.nx * grid.dx_um / 2.0
    x, y = grid.spatial_grid()
    return (
        (jnp.abs(x - cx_f * half) <= w_f * half) & (jnp.abs(y - cy_f * half) <= h_f * half)
    ).astype(jnp.float32)


def _build_scene(grid: Grid) -> jnp.ndarray:
    """Three copies of the target square plus a distractor rectangle."""
    hw = 0.10
    scene = jnp.zeros(grid.shape, dtype=jnp.float32)
    for cx, cy in [(-0.35, -0.25), (0.20, 0.38), (0.40, -0.20)]:
        scene = scene + _make_rect(grid, cx, cy, hw, hw)
    scene = scene + 0.5 * _make_rect(grid, -0.18, 0.20, 0.18, 0.05)
    return jnp.clip(scene, 0.0, 1.0)


def _build_target(grid: Grid) -> jnp.ndarray:
    """Centred target square (same size as copies in the scene)."""
    return _make_rect(grid, 0.0, 0.0, 0.10, 0.10)


def main() -> None:
    # %% Setup
    grid = Grid.from_extent(nx=GRID_N, ny=GRID_N, dx_um=GRID_DX_UM, dy_um=GRID_DX_UM)
    spectrum = Spectrum.from_scalar(WAVELENGTH_UM)
    f_um = _sampling_matched_focal_length(grid)
    print(f"f = {f_um:.1f} µm  |  (r_max/f)² = {_paraxial_validity_constraint_fom():.4f}")

    scene = _build_scene(grid)
    target = _build_target(grid)
    field_in = Field.plane_wave(grid=grid, spectrum=spectrum).apply_amplitude(scene[None, :, :])

    amp, phase = _matched_filter(target)

    prop = plan_propagation(
        mode="auto",
        grid=grid,
        spectrum=spectrum,
        distance_um=f_um,
    )
    lens = ThinLens(focal_length_um=f_um)

    correlator = OpticalModule(
        layers=(
            prop,
            lens,
            prop,
            ComplexMask(amplitude_map=amp, phase_map_rad=phase),
            prop,
            lens,
            prop,
        ),
        sensor=DetectorArray(detector_grid=grid),
    )

    # %% Evaluation
    output_4f = np.asarray(correlator.measure(field_in))
    output_raw = np.asarray(_raw_correlate(scene, target))

    output_4f = output_4f[::-1, ::-1]
    norm = lambda x: x / np.max(x) if np.max(x) > 0 else x  # noqa: E731
    out_4f_n, out_raw_n = norm(output_4f), norm(output_raw)
    cc = float(np.corrcoef(out_4f_n.ravel(), out_raw_n.ravel())[0, 1])
    print(f"Correlation with ground truth: {cc:.4f}")

    # %% Plot Results
    if PLOT:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        axes[0].imshow(np.asarray(scene), cmap="gray")
        axes[0].set_title("Input scene")
        axes[1].imshow(np.asarray(target), cmap="gray")
        axes[1].set_title("Target pattern")

        for ax, img, title in zip(
            axes[2:],
            [out_4f_n, out_raw_n],
            [f"Physical 4f correlator (ρ = {cc:.4f})", "Raw FFT correlation"],
            strict=True,
        ):
            im = ax.imshow(img, cmap="hot")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOT_PATH, dpi=150)
        plt.close(fig)
        print(f"saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
