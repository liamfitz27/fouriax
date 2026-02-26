"""Fisher information, Cramér–Rao bounds, and optimality criteria.

Provides two tiers of Fisher information computation:

1. **Closed-form FIM** for Gaussian or Poisson noise models, using a single
   Jacobian evaluation of the forward model.
2. **Score-based Monte Carlo FIM** for arbitrary distributions, using
   outer products of the score function averaged over samples.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, TypeVar, cast

import jax
import jax.numpy as jnp

ParamsT = TypeVar("ParamsT")


def _flatten_params(params: Any) -> tuple[jnp.ndarray, Callable[[jnp.ndarray], Any]]:
    """Flatten a pytree into a 1-D vector and return an unflatten function."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    shapes = [leaf.shape for leaf in leaves]
    sizes = [leaf.size for leaf in leaves]

    flat = jnp.concatenate([leaf.ravel() for leaf in leaves])

    def unflatten(flat_vec: jnp.ndarray) -> Any:
        splits = jnp.cumsum(jnp.array(sizes[:-1]))
        flat_leaves = jnp.split(flat_vec, splits)
        reshaped = [leaf.reshape(shape) for leaf, shape in zip(flat_leaves, shapes, strict=True)]
        return jax.tree_util.tree_unflatten(treedef, reshaped)

    return flat, unflatten


def jacobian_matrix(
    forward_fn: Callable[[Any], jnp.ndarray],
    params: Any,
    *,
    mode: Literal["forward", "reverse"] = "reverse",
) -> jnp.ndarray:
    """Compute the Jacobian d(output)/d(params).

    The parameter pytree is flattened internally, so the returned Jacobian has
    shape ``(n_outputs, n_params)`` regardless of pytree structure.

    Args:
        forward_fn: Maps params → 1-D output array.
        params: Parameter pytree to differentiate with respect to.
        mode: ``"reverse"`` (default) uses ``jax.jacobian``; ``"forward"``
            uses ``jax.jacfwd``.  Reverse is faster when ``n_outputs > n_params``.

    Returns:
        Jacobian array of shape ``(n_outputs, n_params)``.
    """
    flat_params, unflatten = _flatten_params(params)

    def flat_forward(flat_vec: jnp.ndarray) -> jnp.ndarray:
        return forward_fn(unflatten(flat_vec)).ravel()

    jac_fn = jax.jacobian if mode == "reverse" else jax.jacfwd
    return cast(jnp.ndarray, jac_fn(flat_forward)(flat_params))


def fisher_information(
    forward_fn: Callable[[Any], jnp.ndarray],
    params: Any,
    *,
    noise_model: Literal["gaussian", "poisson"] = "gaussian",
    noise_variance: jnp.ndarray | float | None = None,
) -> jnp.ndarray:
    """Closed-form Fisher information matrix for Gaussian or Poisson noise.

    For a forward model ``mu = forward_fn(params)`` observed with additive noise:

    * **Gaussian**: ``FIM = J^T diag(1/sigma^2) J`` where ``sigma^2`` is
      ``noise_variance`` (defaults to 1).
    * **Poisson**: ``FIM = J^T diag(1/mu) J`` where ``mu = forward_fn(params)``.

    Args:
        forward_fn: Maps params → 1-D predicted measurement vector.
        params: Parameter pytree.
        noise_model: ``"gaussian"`` or ``"poisson"``.
        noise_variance: Per-element variance for Gaussian model. Scalar or
            array matching output shape.  Ignored for Poisson.

    Returns:
        FIM of shape ``(n_params, n_params)``.
    """
    jac = jacobian_matrix(forward_fn, params, mode="reverse")  # (n_out, n_par)

    if noise_model == "gaussian":
        if noise_variance is None:
            inv_var = jnp.ones(jac.shape[0])
        else:
            inv_var = 1.0 / jnp.broadcast_to(jnp.asarray(noise_variance), (jac.shape[0],))
        return jac.T @ (inv_var[:, None] * jac)

    elif noise_model == "poisson":
        mu = forward_fn(params).ravel()
        inv_mu = 1.0 / jnp.maximum(mu, 1e-12)
        return jac.T @ (inv_mu[:, None] * jac)

    else:
        raise ValueError(f"Unknown noise_model: {noise_model!r}")


def score_fisher_information(
    log_prob_fn: Callable[[Any, jnp.ndarray], jnp.ndarray],
    params: Any,
    samples: jnp.ndarray,
) -> jnp.ndarray:
    """Fisher information via score function outer products (Monte Carlo).

    Estimates the FIM as::

        FIM ≈ (1/N) Σᵢ ∇_θ log p(yᵢ|θ) ∇_θ log p(yᵢ|θ)^T

    This is the general definition and applies to any distribution where the
    log-probability is differentiable w.r.t. ``params``.

    Args:
        log_prob_fn: ``(params, sample) → scalar`` log-probability.
        params: Parameter pytree to differentiate with respect to.
        samples: Array of shape ``(N, ...)`` drawn from ``p(y|params)``.

    Returns:
        FIM of shape ``(n_params, n_params)``.
    """
    flat_params, unflatten = _flatten_params(params)

    def flat_log_prob(flat_vec: jnp.ndarray, sample: jnp.ndarray) -> jnp.ndarray:
        return log_prob_fn(unflatten(flat_vec), sample)

    def score_outer(sample: jnp.ndarray) -> jnp.ndarray:
        score = jax.grad(flat_log_prob)(flat_params, sample)  # (n_params,)
        return jnp.outer(score, score)

    outer_products = jax.vmap(score_outer)(samples)  # (N, n_params, n_params)
    return jnp.mean(outer_products, axis=0)


def cramer_rao_bound(fim: jnp.ndarray, *, regularize: float = 1e-10) -> jnp.ndarray:
    """Cramér–Rao lower bound: diagonal of the inverse FIM.

    Each element gives the minimum achievable variance for the corresponding
    parameter under any unbiased estimator.

    Args:
        fim: Fisher information matrix of shape ``(n, n)``.
        regularize: Small constant added to the diagonal before inversion
            for numerical stability.

    Returns:
        Array of shape ``(n,)`` with per-parameter variance lower bounds.
    """
    n = fim.shape[0]
    fim_reg = fim + regularize * jnp.eye(n, dtype=fim.dtype)
    return jnp.diag(jnp.linalg.inv(fim_reg))


def d_optimality(fim: jnp.ndarray) -> jnp.ndarray:
    """D-optimality criterion: log-determinant of the FIM.

    Higher values indicate more information in the measurement. This is
    differentiable and commonly used as an optimization objective for
    experimental design.

    Args:
        fim: Fisher information matrix of shape ``(n, n)``.

    Returns:
        Scalar log-determinant.
    """
    sign, logdet = jnp.linalg.slogdet(fim)
    return cast(jnp.ndarray, jnp.where(sign > 0, logdet, -1e12))
