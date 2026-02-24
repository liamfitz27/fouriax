"""4f correlator — locate a small target pattern inside a larger scene.

Uses a proper 4f geometry (input at front focal plane) with a matched
filter H = F*{target} at the Fourier plane.  Compares the physical
output to a direct FFT cross-correlation as ground truth.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

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
PLOT_PATH = ARTIFACTS_DIR / "4f_correlator_example.png"

WAVELENGTH_UM = 0.532
N_MEDIUM = 1.0
GRID_N = 128
GRID_DX_UM = 2.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sampling_matched_focal_length(grid: Grid) -> float:
    """f = n·N·dx² / λ"""
    return N_MEDIUM * grid.nx * grid.dx_um**2 / WAVELENGTH_UM


def _make_rect(grid: Grid, cx_f: float, cy_f: float, w_f: float, h_f: float) -> jnp.ndarray:
    """Binary rectangle; positions/sizes are fractions of grid half-extent."""
    half = grid.nx * grid.dx_um / 2.0
    x, y = grid.spatial_grid()
    return ((jnp.abs(x - cx_f * half) <= w_f * half) &
            (jnp.abs(y - cy_f * half) <= h_f * half)).astype(jnp.float32)


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


def _raw_correlate(scene: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Ground-truth cross-correlation intensity via direct FFT."""
    f_scene = jnp.fft.fftn(jnp.fft.ifftshift(scene), axes=(-2, -1))
    f_target = jnp.fft.fftn(jnp.fft.ifftshift(target), axes=(-2, -1))
    corr = jnp.fft.ifftn(f_scene * jnp.conj(f_target), axes=(-2, -1))
    return jnp.abs(jnp.fft.fftshift(corr)) ** 2


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    grid = Grid.from_extent(nx=GRID_N, ny=GRID_N, dx_um=GRID_DX_UM, dy_um=GRID_DX_UM)
    spectrum = Spectrum.from_scalar(WAVELENGTH_UM)
    f_um = _sampling_matched_focal_length(grid)
    print(f"f = {f_um:.1f} µm  |  (r_max/f)² = "
          f"{(WAVELENGTH_UM / (2 * N_MEDIUM * GRID_DX_UM))**2:.4f}")

    scene = _build_scene(grid)
    target = _build_target(grid)
    field_in = Field.plane_wave(grid=grid, spectrum=spectrum).apply_amplitude(scene[None, :, :])

    # matched filter
    amp, phase = _matched_filter(target)

    # optics — no sampling planner so the grid stays matched to f
    prop = ASMPropagator(distance_um=f_um, use_sampling_planner=True,
                         warn_on_regime_mismatch=False)
    lens = ThinLens(focal_length_um=f_um)

    # proper 4f: prop(f) → Lens → prop(f) → filter → prop(f) → Lens → prop(f)
    correlator = OpticalModule(
        layers=(
            prop, lens, prop,
            ComplexMask(amplitude_map=amp, phase_map_rad=phase),
            prop, lens, prop,
        ),
        sensor=IntensitySensor(sum_wavelengths=True),
    )

    output_4f = np.asarray(correlator.measure(field_in))
    output_raw = np.asarray(_raw_correlate(scene, target))

    # double FT ≡ spatial inversion — flip so peaks align with ground truth
    output_4f = output_4f[::-1, ::-1]

    # normalise for comparison
    norm = lambda x: x / np.max(x) if np.max(x) > 0 else x  # noqa: E731
    out_4f_n, out_raw_n = norm(output_4f), norm(output_raw)
    cc = float(np.corrcoef(out_4f_n.ravel(), out_raw_n.ravel())[0, 1])
    print(f"Correlation with ground truth: {cc:.4f}")

    # --- plot ---
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
        ax.set_xlabel("x pixel")
        ax.set_ylabel("y pixel")
    fig.tight_layout()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"Saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
