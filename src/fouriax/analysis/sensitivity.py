"""Design sensitivity analysis and fabrication tolerance estimation.

Built on top of the Jacobian utilities in :mod:`fouriax.analysis.fisher`.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from fouriax.analysis.fisher import _flatten_params


def sensitivity_map(
    forward_fn: Callable[[Any], jnp.ndarray],
    params: Any,
    *,
    metric_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Any:
    """Per-parameter sensitivity of an output metric.

    For each element in the parameter pytree, computes the L2 norm of
    ``d(metric)/d(param_element)``.  High sensitivity means small perturbations
    in that parameter cause large changes in the metric.

    If ``metric_fn`` is ``None``, sensitivity is computed w.r.t. the raw
    forward output (using the full Jacobian row norm per parameter).

    Args:
        forward_fn: Maps params → output array.
        params: Parameter pytree.
        metric_fn: Optional scalar or vector function applied to the forward
            output before differentiating. If ``None``, uses identity.

    Returns:
        Pytree matching ``params`` structure, where each leaf contains
        per-element sensitivity values.
    """
    flat_params, unflatten = _flatten_params(params)

    def composed(flat_vec: jnp.ndarray) -> jnp.ndarray:
        output = forward_fn(unflatten(flat_vec))
        if metric_fn is not None:
            output = metric_fn(output)
        return output.ravel()

    jac = jax.jacobian(composed)(flat_params)  # (n_outputs, n_params)
    col_norms = jnp.sqrt(jnp.sum(jac * jac, axis=0))  # (n_params,)

    # Unflatten back to pytree structure
    leaves, treedef = jax.tree_util.tree_flatten(params)
    sizes = [leaf.size for leaf in leaves]
    shapes = [leaf.shape for leaf in leaves]
    splits = jnp.cumsum(jnp.array(sizes[:-1]))
    norm_leaves = jnp.split(col_norms, splits)
    reshaped = [leaf.reshape(shape) for leaf, shape in zip(norm_leaves, shapes, strict=True)]
    return jax.tree_util.tree_unflatten(treedef, reshaped)


def parameter_tolerance(
    forward_fn: Callable[[Any], jnp.ndarray],
    params: Any,
    *,
    target_change: float,
    metric_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Any:
    """Estimate allowable perturbation per parameter for a given output change.

    Uses a first-order (linear) approximation: for each parameter element,
    the tolerance is ``target_change / sensitivity``.  High sensitivity means
    tight tolerance.

    Args:
        forward_fn: Maps params → output array.
        params: Parameter pytree.
        target_change: Maximum acceptable change in the output metric.
        metric_fn: Optional metric function (same as :func:`sensitivity_map`).

    Returns:
        Pytree matching ``params`` structure with per-element tolerance values.
    """
    sens = sensitivity_map(forward_fn, params, metric_fn=metric_fn)

    def safe_reciprocal(leaf: jnp.ndarray) -> jnp.ndarray:
        return target_change / jnp.maximum(leaf, 1e-12)

    return jax.tree.map(safe_reciprocal, sens)
