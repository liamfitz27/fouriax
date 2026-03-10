"""Optimisation helpers for gradient-based inverse design.

Provides training loops, dataset utilities, loss functions, and result
containers built on JAX and Optax.  These are convenience wrappers used
by the example scripts and notebooks; they are not part of the core
optics simulation API.
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
