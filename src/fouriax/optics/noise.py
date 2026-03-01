from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp


def _clip_output(
    values: jnp.ndarray,
    *,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> jnp.ndarray:
    if clip_min is None and clip_max is None:
        return values
    lo = -jnp.inf if clip_min is None else float(clip_min)
    hi = jnp.inf if clip_max is None else float(clip_max)
    return jnp.clip(values, lo, hi)


def _require_real_expected(expected: jnp.ndarray) -> jnp.ndarray:
    arr = jnp.asarray(expected)
    if jnp.iscomplexobj(arr):
        raise ValueError("noise models require real-valued expected measurements")
    return arr


class SensorNoiseModel(ABC):
    """Base interface for stochastic sensor noise applied to expected measurements."""

    @abstractmethod
    def sample(self, expected: jnp.ndarray, *, key: jax.Array) -> jnp.ndarray:
        """Draw a noisy sample from the measurement distribution."""

    @abstractmethod
    def expected_variance(self, expected: jnp.ndarray) -> jnp.ndarray:
        """Return per-element variance for the noise model at the expected signal."""

    def covariance(self, expected: jnp.ndarray) -> jnp.ndarray:
        """Return the analytic measurement covariance matrix.

        The default implementation assumes conditionally independent noise and
        constructs a diagonal covariance from `expected_variance(...)`.
        """
        variance = jnp.asarray(self.expected_variance(expected)).ravel()
        return jnp.diag(variance)

    def precision(self, expected: jnp.ndarray, *, regularize: float = 1e-12) -> jnp.ndarray:
        """Return the analytic inverse covariance (precision) matrix.

        The default implementation assumes conditionally independent noise and
        constructs a diagonal precision from `expected_variance(...)`.
        """
        variance = jnp.asarray(self.expected_variance(expected)).ravel()
        inv_var = 1.0 / jnp.maximum(variance, jnp.asarray(regularize, dtype=variance.dtype))
        return jnp.diag(inv_var)


@dataclass(frozen=True)
class PoissonNoise(SensorNoiseModel):
    """Shot noise model in normalized output units.

    `count_scale` maps expected intensity to expected counts before sampling.
    The returned noisy sample is divided back by `count_scale`, so the output
    remains in the same units as the clean sensor measurement.
    """

    count_scale: float = 1.0

    def _validate(self) -> None:
        if self.count_scale <= 0.0:
            raise ValueError("count_scale must be > 0")

    def sample(self, expected: jnp.ndarray, *, key: jax.Array) -> jnp.ndarray:
        self._validate()
        mu = jnp.maximum(_require_real_expected(expected), 0.0)
        lam = mu * jnp.asarray(self.count_scale, dtype=mu.dtype)
        counts = jax.random.poisson(key, lam=lam, shape=mu.shape)
        return counts.astype(mu.dtype) / jnp.asarray(self.count_scale, dtype=mu.dtype)

    def expected_variance(self, expected: jnp.ndarray) -> jnp.ndarray:
        self._validate()
        mu = jnp.maximum(_require_real_expected(expected), 0.0)
        return mu / jnp.asarray(self.count_scale, dtype=mu.dtype)


@dataclass(frozen=True)
class GaussianNoise(SensorNoiseModel):
    """Additive Gaussian sensor noise in measurement units."""

    std: float | jnp.ndarray
    clip_min: float | None = None
    clip_max: float | None = None

    def sample(self, expected: jnp.ndarray, *, key: jax.Array) -> jnp.ndarray:
        mu = _require_real_expected(expected)
        std = jnp.broadcast_to(jnp.asarray(self.std, dtype=mu.dtype), mu.shape)
        noise = std * jax.random.normal(key, shape=mu.shape, dtype=mu.dtype)
        return _clip_output(mu + noise, clip_min=self.clip_min, clip_max=self.clip_max)

    def expected_variance(self, expected: jnp.ndarray) -> jnp.ndarray:
        mu = _require_real_expected(expected)
        std = jnp.broadcast_to(jnp.asarray(self.std, dtype=mu.dtype), mu.shape)
        return std * std


@dataclass(frozen=True)
class PoissonGaussianNoise(SensorNoiseModel):
    """Shot noise plus additive Gaussian read noise in measurement units."""

    count_scale: float = 1.0
    read_noise_std: float = 0.0
    clip_min: float | None = 0.0
    clip_max: float | None = None

    def sample(self, expected: jnp.ndarray, *, key: jax.Array) -> jnp.ndarray:
        key_poisson, key_gauss = jax.random.split(key)
        shot = PoissonNoise(count_scale=self.count_scale).sample(expected, key=key_poisson)
        if self.read_noise_std <= 0.0:
            return _clip_output(shot, clip_min=self.clip_min, clip_max=self.clip_max)
        return GaussianNoise(
            std=self.read_noise_std,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
        ).sample(shot, key=key_gauss)

    def expected_variance(self, expected: jnp.ndarray) -> jnp.ndarray:
        mu = _require_real_expected(expected)
        shot_var = PoissonNoise(count_scale=self.count_scale).expected_variance(mu)
        read_var = jnp.asarray(self.read_noise_std, dtype=mu.dtype) ** 2
        return shot_var + read_var
