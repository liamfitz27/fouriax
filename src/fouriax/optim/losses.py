"""Example-oriented loss helpers.

These are intentionally outside the core optics package.
"""

from __future__ import annotations

import jax.numpy as jnp


def focal_spot_loss(
    intensity: jnp.ndarray,
    target_xy: tuple[int, int],
    window_px: int = 2,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Maximize power concentration around a target focal spot."""
    if window_px < 0:
        raise ValueError("window_px must be >= 0")

    arr = jnp.asarray(intensity)
    if arr.ndim == 3:
        arr2d = jnp.sum(arr, axis=0)
    elif arr.ndim == 2:
        arr2d = arr
    else:
        raise ValueError("intensity must have shape (ny, nx) or (num_channels, ny, nx)")

    ny, nx = arr2d.shape
    tx, ty = int(target_xy[0]), int(target_xy[1])

    x = jnp.arange(nx)
    y = jnp.arange(ny)
    xmask = (x >= (tx - window_px)) & (x <= (tx + window_px))
    ymask = (y >= (ty - window_px)) & (y <= (ty + window_px))
    mask = ymask[:, None] & xmask[None, :]

    spot_power = jnp.sum(arr2d * mask.astype(arr2d.dtype))
    total_power = jnp.sum(arr2d)
    return 1.0 - (spot_power / (total_power + eps))
