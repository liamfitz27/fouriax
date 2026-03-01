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

from fouriax.optics.noise import SensorNoiseModel

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
    noise_model: SensorNoiseModel | None = None,
) -> jnp.ndarray:
    """Closed-form Fisher information matrix for analytic noise models.

    For a forward model ``mu = forward_fn(params)``, this computes
    ``FIM = J^T Lambda J`` where ``J = dmu/dparams`` and ``Lambda`` is the
    analytic noise precision matrix provided by `noise_model`.

    If `noise_model` is omitted, unit-variance independent Gaussian noise is
    assumed, so ``Lambda = I`` and the result reduces to ``J^T J``.

    Args:
        forward_fn: Maps params → 1-D predicted measurement vector.
        params: Parameter pytree.
        noise_model: Optional analytic noise model. It must provide a
            `precision(expected)` method through `SensorNoiseModel`.

    Returns:
        FIM of shape ``(n_params, n_params)``.
    """
    jac = jacobian_matrix(forward_fn, params, mode="reverse")  # (n_out, n_par)
    if noise_model is None:
        precision = jnp.eye(jac.shape[0], dtype=jac.dtype)
    else:
        expected = forward_fn(params).ravel()
        precision = jnp.asarray(noise_model.precision(expected), dtype=jac.dtype)
        if precision.shape != (jac.shape[0], jac.shape[0]):
            raise ValueError(
                "noise_model.precision(expected) must return shape "
                f"{(jac.shape[0], jac.shape[0])}, got {precision.shape}"
            )
    return jac.T @ precision @ jac


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


def _as_square_matrix(
    value: jnp.ndarray | float,
    *,
    size: int,
    name: str,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    arr = jnp.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        return arr * jnp.eye(size, dtype=dtype)
    if arr.ndim == 1:
        if arr.shape[0] != size:
            raise ValueError(f"{name} length mismatch: got {arr.shape[0]}, expected {size}")
        return jnp.diag(arr)
    if arr.ndim == 2:
        if arr.shape != (size, size):
            raise ValueError(f"{name} shape mismatch: got {arr.shape}, expected {(size, size)}")
        return arr
    raise ValueError(f"{name} must be scalar, shape ({size},), or shape ({size}, {size})")


def d_optimality(
    fim: jnp.ndarray,
    *,
    prior_covariance: jnp.ndarray | float | None = None,
    prior_precision: jnp.ndarray | float | None = None,
    relative_to_prior: bool = True,
) -> jnp.ndarray:
    """D-optimality criterion, optionally with a Gaussian prior.

    Higher values indicate more information in the measurement. This is
    differentiable and commonly used as an optimization objective for
    experimental design.

    With a Gaussian prior on the parameter vector, this computes the
    posterior-precision log-determinant. If `relative_to_prior=True` (default),
    the prior baseline is subtracted:

    ``log det(FIM + Lambda_prior) - log det(Lambda_prior)``

    which is equivalent to ``log det(I + Sigma_prior FIM)`` and is proportional
    to the mutual information for a linear-Gaussian model.

    Args:
        fim: Fisher information matrix of shape ``(n, n)``.
        prior_covariance: Optional prior covariance ``Sigma_prior``. Scalar,
            diagonal vector, or full matrix.
        prior_precision: Optional prior precision ``Lambda_prior``. Scalar,
            diagonal vector, or full matrix.
        relative_to_prior: When a prior is provided, subtract the prior
            log-determinant baseline so the result measures information gain.

    Returns:
        Scalar log-determinant.
    """
    if prior_covariance is not None and prior_precision is not None:
        raise ValueError("provide at most one of prior_covariance or prior_precision")

    fim = jnp.asarray(fim)
    if fim.ndim != 2 or fim.shape[0] != fim.shape[1]:
        raise ValueError("fim must be a square matrix")

    if prior_covariance is None and prior_precision is None:
        sign, logdet = jnp.linalg.slogdet(fim)
        return cast(jnp.ndarray, jnp.where(sign > 0, logdet, -1e12))

    if prior_covariance is not None:
        sigma_prior = _as_square_matrix(
            prior_covariance,
            size=fim.shape[0],
            name="prior_covariance",
            dtype=fim.dtype,
        )
        lambda_prior = jnp.linalg.inv(sigma_prior)
    else:
        if prior_precision is None:
            raise ValueError(
                "prior_precision must be provided when prior_covariance is not set"
            )
        lambda_prior = _as_square_matrix(
            prior_precision,
            size=fim.shape[0],
            name="prior_precision",
            dtype=fim.dtype,
        )

    posterior_precision = fim + lambda_prior
    sign, logdet = jnp.linalg.slogdet(posterior_precision)
    if not relative_to_prior:
        return cast(jnp.ndarray, jnp.where(sign > 0, logdet, -1e12))

    prior_sign, prior_logdet = jnp.linalg.slogdet(lambda_prior)
    valid = (sign > 0) & (prior_sign > 0)
    return cast(jnp.ndarray, jnp.where(valid, logdet - prior_logdet, -1e12))
