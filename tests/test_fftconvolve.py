import jax.numpy as jnp
import numpy as np
import pytest
from scipy import signal

from fouriax.core.fft import fftconvolve

# TODO(next tests):
# - Add broadcasting tests on non-convolution dimensions.
# - Add gradient checks with jax.grad vs finite-difference baselines.
# - Add dtype matrix tests: float32/float64/complex64/complex128.
# - Add explicit fft_shape override tests (including larger zero-padding).
# - Add JIT/static-arg behavior tests for mode/axes/fft_shape/sum_fft_axis.
# - Add edge-shape tests (odd/even kernels, singleton axes, higher-rank inputs).


def test_fftconvolve_full_matches_scipy_2d_real():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 7))
    k = rng.standard_normal((3, 5))

    got = np.asarray(fftconvolve(jnp.asarray(x), jnp.asarray(k), mode="full"))
    expected = signal.fftconvolve(x, k, mode="full")

    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=2e-6)


def test_fftconvolve_same_matches_scipy_2d_real():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((9, 6))
    k = rng.standard_normal((4, 3))

    got = np.asarray(fftconvolve(jnp.asarray(x), jnp.asarray(k), mode="same"))
    expected = signal.fftconvolve(x, k, mode="same")

    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=2e-6)


def test_fftconvolve_matches_scipy_complex():
    rng = np.random.default_rng(2)
    xr = rng.standard_normal((10,))
    xi = rng.standard_normal((10,))
    kr = rng.standard_normal((6,))
    ki = rng.standard_normal((6,))
    x = xr + 1j * xi
    k = kr + 1j * ki

    got = np.asarray(fftconvolve(jnp.asarray(x), jnp.asarray(k), mode="full"))
    expected = signal.fftconvolve(x, k, mode="full")

    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_fftconvolve_axes_matches_scipy():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 8, 7))
    k = rng.standard_normal((2, 3, 5))

    got = np.asarray(
        fftconvolve(jnp.asarray(x), jnp.asarray(k), mode="same", axes=(1, 2))
    )
    expected = signal.fftconvolve(x, k, mode="same", axes=(1, 2))

    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_fftconvolve_sum_fft_axis_matches_channel_sum():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((8, 7, 3))
    k = rng.standard_normal((3, 5, 3))

    got = np.asarray(
        fftconvolve(
            jnp.asarray(x),
            jnp.asarray(k),
            mode="full",
            axes=(0, 1),
            sum_fft_axis=2,
        )
    )

    expected_channels = [
        signal.fftconvolve(x[..., c], k[..., c], mode="full") for c in range(x.shape[-1])
    ]
    expected = np.sum(expected_channels, axis=0)

    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=2e-6)


def test_fftconvolve_invalid_mode_raises():
    x = jnp.ones((4,))
    k = jnp.ones((3,))
    with pytest.raises(ValueError, match="mode must be 'full' or 'same'"):
        fftconvolve(x, k, mode="valid")
