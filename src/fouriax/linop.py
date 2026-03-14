from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

Array = jax.Array


@dataclass(frozen=True)
class LinearOperator:
    """Matrix-free linear operator with forward and adjoint actions."""

    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    in_dtype: jnp.dtype
    out_dtype: jnp.dtype
    matvec_fn: Callable[[Array], Array]
    rmatvec_fn: Callable[[Array], Array]

    def matvec(self, x: Array) -> Array:
        """Apply the forward linear map."""
        if x.shape != self.in_shape:
            raise ValueError(f"matvec shape mismatch: got {x.shape}, expected {self.in_shape}")
        return self.matvec_fn(x)

    def rmatvec(self, y: Array) -> Array:
        """Apply the Hermitian-adjoint linear map."""
        if y.shape != self.out_shape:
            raise ValueError(f"rmatvec shape mismatch: got {y.shape}, expected {self.out_shape}")
        return self.rmatvec_fn(y)

    def matmat(self, x: Array) -> Array:
        """Apply the forward map to a stack of vectors on the last axis."""
        if x.ndim < 1:
            raise ValueError("matmat input must have at least one axis")
        if x.shape[:-1] != self.in_shape:
            raise ValueError(
                f"matmat shape mismatch: got {x.shape[:-1]}, expected {self.in_shape} "
                "before the trailing stack axis"
            )
        return jax.vmap(self.matvec, in_axes=-1, out_axes=-1)(x)

    def rmatmat(self, y: Array) -> Array:
        """Apply the adjoint map to a stack of vectors on the last axis."""
        if y.ndim < 1:
            raise ValueError("rmatmat input must have at least one axis")
        if y.shape[:-1] != self.out_shape:
            raise ValueError(
                f"rmatmat shape mismatch: got {y.shape[:-1]}, expected {self.out_shape} "
                "before the trailing stack axis"
            )
        return jax.vmap(self.rmatvec, in_axes=-1, out_axes=-1)(y)

    def normal(self, x: Array) -> Array:
        """Apply the normal operator ``A* A``."""
        return self.rmatvec(self.matvec(x))
