"""Tests for fouriax.analysis (Fisher information, sensitivity, tolerance)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from fouriax.analysis import (
    cramer_rao_bound,
    d_optimality,
    fisher_information,
    jacobian_matrix,
    parameter_tolerance,
    score_fisher_information,
    sensitivity_map,
)
from fouriax.optics import GaussianNoise, PoissonNoise

# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------


def test_jacobian_linear():
    """Jacobian of a linear map y = Ax should be A."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def forward(params):
        return A @ params

    params = jnp.array([1.0, 1.0])
    jac = jacobian_matrix(forward, params)
    assert jac.shape == (3, 2)
    np.testing.assert_allclose(jac, A, atol=1e-5)


# ---------------------------------------------------------------------------
# Closed-form FIM
# ---------------------------------------------------------------------------


def test_fim_gaussian_linear():
    """For y = Ax + ε, Gaussian FIM = A^T Σ^{-1} A."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sigma2 = jnp.array([0.5, 1.0, 2.0])

    def forward(params):
        return A @ params

    params = jnp.zeros(2)
    fim = fisher_information(forward, params, noise_model=GaussianNoise(std=jnp.sqrt(sigma2)))
    expected = A.T @ (jnp.diag(1.0 / sigma2) @ A)
    np.testing.assert_allclose(fim, expected, atol=1e-5)


def test_fim_gaussian_unit_variance():
    """Gaussian FIM with default variance should equal J^T J."""
    A = jnp.array([[2.0, 0.0], [0.0, 3.0]])

    def forward(params):
        return A @ params

    params = jnp.zeros(2)
    fim = fisher_information(forward, params)
    expected = A.T @ A
    np.testing.assert_allclose(fim, expected, atol=1e-5)


def test_fim_poisson():
    """Poisson FIM = J^T diag(1/mu) J."""

    def forward(params):
        return jnp.exp(params)  # ensures positive mu

    params = jnp.array([1.0, 2.0, 0.5])
    fim = fisher_information(forward, params, noise_model=PoissonNoise(count_scale=1.0))

    mu = jnp.exp(params)
    # J = diag(mu), so J^T diag(1/mu) J = diag(mu) * diag(1/mu) * diag(mu) = diag(mu)
    expected = jnp.diag(mu)
    np.testing.assert_allclose(fim, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Score-based Monte Carlo FIM
# ---------------------------------------------------------------------------


def test_score_fim_gaussian_recovers_closed_form():
    """Score FIM with many Gaussian samples should converge to closed form."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    sigma2 = jnp.array([1.0, 1.0])
    params = jnp.array([0.5, -0.3])

    # Generate samples from y ~ N(A @ params, diag(sigma2))
    key = jax.random.PRNGKey(42)
    mu = A @ params
    samples = mu[None, :] + jax.random.normal(key, (10000, 2)) * jnp.sqrt(sigma2)[None, :]

    def log_prob(p, y):
        mu_p = A @ p
        return -0.5 * jnp.sum((y - mu_p) ** 2 / sigma2)

    score_fim = score_fisher_information(log_prob, params, samples)
    closed_fim = fisher_information(
        lambda p: A @ p,
        params,
        noise_model=GaussianNoise(std=jnp.sqrt(sigma2)),
    )

    # Monte Carlo estimate should be close with 10k samples
    np.testing.assert_allclose(score_fim, closed_fim, atol=0.15)


def test_score_fim_symmetry():
    """Score FIM should be symmetric positive semi-definite."""
    def log_prob(p, y):
        return -0.5 * jnp.sum((y - p) ** 2)

    params = jnp.array([1.0, 2.0])
    key = jax.random.PRNGKey(0)
    samples = params[None, :] + jax.random.normal(key, (500, 2))

    fim = score_fisher_information(log_prob, params, samples)
    np.testing.assert_allclose(fim, fim.T, atol=1e-6)
    eigenvalues = jnp.linalg.eigvalsh(fim)
    assert jnp.all(eigenvalues >= -1e-6)


# ---------------------------------------------------------------------------
# CRB and D-optimality
# ---------------------------------------------------------------------------


def test_crb_diagonal():
    """Diagonal FIM → CRB is element-wise reciprocal."""
    diag_vals = jnp.array([2.0, 5.0, 10.0])
    fim = jnp.diag(diag_vals)
    crb = cramer_rao_bound(fim, regularize=0.0)
    expected = 1.0 / diag_vals
    np.testing.assert_allclose(crb, expected, atol=1e-5)


def test_d_optimality_matches_slogdet():
    """D-optimality should match jnp.linalg.slogdet."""
    fim = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    d_opt = d_optimality(fim)
    _, expected = jnp.linalg.slogdet(fim)
    np.testing.assert_allclose(d_opt, expected, atol=1e-6)


def test_d_optimality_with_prior_covariance_matches_information_gain():
    """Prior-aware D-optimality should equal log det(I + Σ_prior FIM)."""
    fim = jnp.array([[2.0, 0.3], [0.3, 1.5]])
    sigma_prior = jnp.array([4.0, 9.0])
    d_opt = d_optimality(fim, prior_covariance=sigma_prior)
    _, expected = jnp.linalg.slogdet(jnp.eye(2) + jnp.diag(sigma_prior) @ fim)
    np.testing.assert_allclose(d_opt, expected, atol=1e-6)


def test_d_optimality_with_prior_precision_matches_posterior_precision():
    """Relative baseline can be disabled to return posterior precision log-det."""
    fim = jnp.array([[1.5, 0.2], [0.2, 0.9]])
    lambda_prior = jnp.array([2.0, 3.0])
    d_opt = d_optimality(fim, prior_precision=lambda_prior, relative_to_prior=False)
    _, expected = jnp.linalg.slogdet(fim + jnp.diag(lambda_prior))
    np.testing.assert_allclose(d_opt, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Sensitivity and tolerance
# ---------------------------------------------------------------------------


def test_sensitivity_map_finite_diff():
    """Sensitivity map should agree with finite-difference approximation."""
    A = jnp.array([[1.0, 3.0], [2.0, 1.0]])

    def forward(params):
        return A @ params

    params = jnp.array([1.0, 1.0])
    sens = sensitivity_map(forward, params)

    # For linear y = Ax, sensitivity of param j is ||A[:, j]||
    expected = jnp.sqrt(jnp.sum(A ** 2, axis=0))
    np.testing.assert_allclose(sens, expected, atol=1e-5)


def test_tolerance_inverse_of_sensitivity():
    """tolerance * sensitivity ≈ target_change."""
    A = jnp.array([[1.0, 3.0], [2.0, 1.0]])

    def forward(params):
        return A @ params

    params = jnp.array([1.0, 1.0])
    target = 0.01

    sens = sensitivity_map(forward, params)
    tol = parameter_tolerance(forward, params, target_change=target)

    product = sens * tol
    np.testing.assert_allclose(product, target, atol=1e-5)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_fim_differentiable():
    """Verify jax.grad flows through the FIM computation."""

    def objective(params):
        def forward(p):
            return jnp.array([p[0] * p[1], p[0] + p[1]])

        fim = fisher_information(forward, params)
        return d_optimality(fim)

    params = jnp.array([1.0, 2.0])
    grad = jax.grad(objective)(params)
    assert grad.shape == (2,)
    assert jnp.all(jnp.isfinite(grad))


def test_score_fim_differentiable():
    """Verify jax.grad flows through the score FIM computation."""

    def objective(params):
        def log_prob(p, y):
            return -0.5 * jnp.sum((y - p) ** 2)

        key = jax.random.PRNGKey(0)
        samples = params[None, :] + jax.random.normal(key, (100, 2))
        fim = score_fisher_information(log_prob, params, samples)
        return d_optimality(fim)

    params = jnp.array([1.0, 2.0])
    grad = jax.grad(objective)(params)
    assert grad.shape == (2,)
    assert jnp.all(jnp.isfinite(grad))
