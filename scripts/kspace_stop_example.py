"""Minimal mixed-domain stop example with NA schedule and analytic checks.

Pipeline:
1. Spatial layer (thin lens with finite aperture)
2. Propagation segment (ASM)
3. k-space stop (unity circular mask with small aperture metadata)
4. Spatial conversion layer (phase=0) and intensity sensor

This script compares:
- module-computed local NA schedule vs geometric expectation
- analytic k-cutoff vs measured out-of-band energy after the k-stop layer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from fouriax.optics import (
    ASMPropagator,
    Field,
    Grid,
    IntensitySensor,
    KAmplitudeMaskLayer,
    OpticalModule,
    PhaseMaskLayer,
    PropagationLayer,
    Spectrum,
    ThinLensLayer,
)


def _make_impulse_field(grid: Grid, spectrum: Spectrum) -> Field:
    data = jnp.zeros((spectrum.size, grid.ny, grid.nx), dtype=jnp.complex64)
    cy = grid.ny // 2
    cx = grid.nx // 2
    data = data.at[:, cy, cx].set(1.0 + 0.0j)
    return Field(data=data, grid=grid, spectrum=spectrum)


def _circular_k_stop_from_na(grid: Grid, wavelength_um: float, na: float) -> jnp.ndarray:
    fx, fy = grid.frequency_grid()
    fr = jnp.sqrt(fx * fx + fy * fy)
    f_cut = na / wavelength_um
    return (fr <= f_cut).astype(jnp.float32)


def _out_of_band_energy_ratio(k_field_2d: jnp.ndarray, mask_2d: jnp.ndarray) -> float:
    power = jnp.abs(k_field_2d) ** 2
    in_band = jnp.sum(power * mask_2d)
    out_band = jnp.sum(power * (1.0 - mask_2d))
    return float(out_band / (in_band + out_band + 1e-12))


def run_example(save_path: Path | None = None) -> None:
    # --- setup ---
    wavelength_um = 0.532
    medium_index = 1.0
    grid = Grid.from_extent(nx=256, ny=256, dx_um=0.8, dy_um=0.8)
    spectrum = Spectrum.from_scalar(wavelength_um)
    field_in = _make_impulse_field(grid=grid, spectrum=spectrum)

    # Geometry for NA scheduling check.
    lens_aperture_um = 120.0
    k_stop_aperture_um = 20.0
    segment_distance_um = 80.0

    # Geometric expected NA for the first propagation segment.
    r_lens = lens_aperture_um / 2.0
    r_kstop = k_stop_aperture_um / 2.0
    expected_na_geom = medium_index * np.sin(np.arctan(min(r_lens, r_kstop) / segment_distance_um))

    # Use the same NA for the actual unity circular k-stop transmission.
    k_stop_mask = _circular_k_stop_from_na(
        grid=grid,
        wavelength_um=wavelength_um,
        na=float(expected_na_geom),
    )

    module = OpticalModule(
        layers=(
            ThinLensLayer(focal_length_um=500.0, aperture_diameter_um=lens_aperture_um),
            PropagationLayer(
                model=ASMPropagator(use_sampling_planner=False, medium_index=medium_index),
                distance_um=segment_distance_um,
            ),
            KAmplitudeMaskLayer(
                amplitude_map=k_stop_mask,
                aperture_diameter_um=k_stop_aperture_um,
            ),
            # Domain conversion back to spatial for sensor interpretation.
            PhaseMaskLayer(phase_map_rad=0.0),
        ),
        sensor=IntensitySensor(sum_wavelengths=True),
        auto_apply_na=True,
        medium_index=medium_index,
        na_fallback_to_effective=False,
    )

    # --- NA schedule check ---
    schedule = module.na_schedule(field_in)
    # First propagation layer index is 1 in this module.
    na_local = schedule.get(1, np.nan)

    print("=== kspace_stop_example ===")
    print(f"Expected geometric NA (segment 1): {expected_na_geom:.6f}")
    print(f"Module local NA schedule[1]:      {na_local:.6f}")
    print(f"Absolute NA error:                {abs(na_local - expected_na_geom):.3e}")

    # --- k-stop analytic cutoff check on traced state ---
    states = module.trace(field_in, include_input=True)
    # states index: 0=input, 1=lens, 2=propagation, 3=k-stop, 4=phase(0) spatial
    after_k_stop = states[3]
    assert after_k_stop.domain == "kspace"

    ratio_oob = _out_of_band_energy_ratio(after_k_stop.data[0], k_stop_mask)
    print(f"Out-of-band energy ratio after k-stop: {ratio_oob:.3e}")
    print("Expected: near 0 for a hard unity circular stop.")

    # --- final sensor output ---
    measured = module.measure(field_in)
    intensity = np.asarray(measured)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    im = ax.imshow(intensity, cmap="magma")
    ax.set_title("Final Intensity (After k-stop + Back to Spatial)")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure: {save_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output path for the intensity figure.",
    )
    args = parser.parse_args()
    run_example(save_path=args.save)


if __name__ == "__main__":
    main()
