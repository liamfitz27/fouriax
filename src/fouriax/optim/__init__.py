"""Lightweight helper utilities for optimization in example scripts and notebooks.

These are intentionally small wrappers around common NumPy/JAX/Optax patterns
to reduce duplication in examples. They are not part of the core optics API.
"""

from fouriax.optim.data import (
    batch_slices,
    iter_minibatches,
    random_batch_indices,
    shuffled_arrays,
    train_val_split,
)
from fouriax.optim.losses import focal_spot_loss
from fouriax.optim.optim import (
    BestValueTracker,
    DatasetOptResult,
    HybridModuleDatasetOptResult,
    ModuleDatasetOptResult,
    ModuleOptResult,
    ValidationRecord,
    apply_optax_updates,
    optimize_dataset_hybrid_module,
    optimize_dataset_optical_module,
    optimize_dataset_params,
    optimize_optical_module,
    should_log_step,
)

__all__ = [
    "batch_slices",
    "iter_minibatches",
    "random_batch_indices",
    "shuffled_arrays",
    "train_val_split",
    "focal_spot_loss",
    "BestValueTracker",
    "DatasetOptResult",
    "HybridModuleDatasetOptResult",
    "ModuleOptResult",
    "ModuleDatasetOptResult",
    "ValidationRecord",
    "apply_optax_updates",
    "optimize_dataset_hybrid_module",
    "optimize_dataset_optical_module",
    "optimize_dataset_params",
    "optimize_optical_module",
    "should_log_step",
]
