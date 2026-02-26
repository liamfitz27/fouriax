"""Batching helpers for example scripts and notebooks."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, TypeVar, cast

import numpy as np


class _BatchableArray(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, key: Any) -> Any: ...


ArrayT = TypeVar("ArrayT", bound=_BatchableArray)


def batch_slices(
    n_items: int,
    batch_size: int,
    *,
    drop_last: bool = False,
) -> Iterator[tuple[int, int]]:
    """Yield `(lo, hi)` slices that cover `n_items` in minibatches."""
    if n_items < 0:
        raise ValueError("n_items must be >= 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for lo in range(0, n_items, batch_size):
        hi = min(lo + batch_size, n_items)
        if drop_last and (hi - lo) < batch_size:
            continue
        yield lo, hi


def random_batch_indices(
    rng: np.random.Generator,
    n_items: int,
    batch_size: int,
    *,
    replace: bool = True,
) -> np.ndarray:
    """Sample batch indices with replacement using a NumPy RNG."""
    if n_items <= 0:
        raise ValueError("n_items must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    size = batch_size if replace else min(batch_size, n_items)
    if replace:
        return rng.integers(0, n_items, size=(size,))
    return rng.choice(n_items, size=size, replace=False)


def shuffled_arrays(
    *arrays: ArrayT,
    rng: np.random.Generator,
) -> tuple[ArrayT, ...]:
    """Apply the same random permutation to multiple arrays."""
    if not arrays:
        return ()
    n_items = len(arrays[0])
    if any(len(arr) != n_items for arr in arrays):
        raise ValueError("all arrays must have the same length")
    perm = rng.permutation(n_items)
    return cast(tuple[ArrayT, ...], tuple(arr[perm] for arr in arrays))


def train_val_split(
    *arrays: ArrayT,
    val_fraction: float,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    shuffle: bool = True,
) -> tuple[tuple[ArrayT, ...], tuple[ArrayT, ...]]:
    """Split arrays into `(train_arrays, val_arrays)` with aligned indexing."""
    if not arrays:
        raise ValueError("at least one array is required")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")
    n_items = len(arrays[0])
    if any(len(arr) != n_items for arr in arrays):
        raise ValueError("all arrays must have the same length")
    if n_items < 2:
        raise ValueError("need at least 2 items for a train/val split")

    if rng is None:
        rng = np.random.default_rng(seed)
    elif seed is not None:
        raise ValueError("pass either rng or seed, not both")

    n_val = max(1, int(round(n_items * val_fraction)))
    n_val = min(n_val, n_items - 1)
    idx = np.arange(n_items)
    if shuffle:
        idx = rng.permutation(idx)

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    train_arrays = cast(tuple[ArrayT, ...], tuple(arr[train_idx] for arr in arrays))
    val_arrays = cast(tuple[ArrayT, ...], tuple(arr[val_idx] for arr in arrays))
    return train_arrays, val_arrays


def iter_minibatches(
    *arrays: ArrayT,
    batch_size: int,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    shuffle: bool = True,
    drop_last: bool = False,
) -> Iterator[tuple[ArrayT, ...]]:
    """Yield aligned minibatches from one or more arrays."""
    if not arrays:
        raise ValueError("at least one array is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    n_items = len(arrays[0])
    if any(len(arr) != n_items for arr in arrays):
        raise ValueError("all arrays must have the same length")
    if n_items == 0:
        raise ValueError("arrays must be non-empty")

    if rng is None:
        rng = np.random.default_rng(seed)
    elif seed is not None:
        raise ValueError("pass either rng or seed, not both")

    idx = np.arange(n_items)
    if shuffle:
        idx = rng.permutation(idx)

    for lo, hi in batch_slices(n_items, batch_size, drop_last=drop_last):
        batch_idx = idx[lo:hi]
        yield cast(tuple[ArrayT, ...], tuple(arr[batch_idx] for arr in arrays))
