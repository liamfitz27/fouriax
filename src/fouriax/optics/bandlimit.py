from __future__ import annotations

import jax.numpy as jnp

from fouriax.optics.model import Grid


def build_na_mask(
    grid: Grid,
    *,
    wavelength_um: float,
    na_limit: float,
    medium_index: float = 1.0,
) -> jnp.ndarray:
    """
    Build a hard NA cutoff mask in spatial-frequency coordinates.

    The cutoff in cycles/um is f_cut = min(na_limit, medium_index) / wavelength_um.
    """
    if wavelength_um <= 0:
        raise ValueError("wavelength_um must be strictly positive")
    if na_limit <= 0:
        raise ValueError("na_limit must be strictly positive")
    if medium_index <= 0:
        raise ValueError("medium_index must be strictly positive")

    fx, fy = grid.frequency_grid()
    f_r = jnp.sqrt(fx * fx + fy * fy)
    f_cut = min(float(na_limit), float(medium_index)) / float(wavelength_um)
    return (f_r <= f_cut).astype(jnp.float32)
