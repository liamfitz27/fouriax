import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fouriax.optics import AmplitudeMask, DetectorArray, Field, Grid, PoissonNoise, Spectrum


def _two_wavelength_field() -> Field:
    grid = Grid.from_extent(nx=4, ny=3, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_array(jnp.array([0.532, 0.633], dtype=jnp.float32))
    amp = jnp.stack(
        [
            jnp.ones(grid.shape, dtype=jnp.float32) * 2.0,
            jnp.ones(grid.shape, dtype=jnp.float32) * 3.0,
        ],
        axis=0,
    )
    data = amp.astype(jnp.complex64)
    return Field(data=data, grid=grid, spectrum=spectrum, domain="spatial")


def test_detector_array_expected_applies_qe_and_pixel_filter():
    field = _two_wavelength_field()
    qe = jnp.array([0.5, 1.0], dtype=jnp.float32)
    filter_mask = AmplitudeMask(
        amplitude_map=jnp.ones(field.grid.shape, dtype=jnp.float32) * 0.5,
    )
    sensor = DetectorArray(
        detector_grid=field.grid,
        qe_curve=qe,
        filter_mask=filter_mask,
    )

    measured = sensor.measure(field)
    expected_w0 = (2.0**2) * 0.5 * (0.5**2)
    expected_w1 = (3.0**2) * 1.0 * (0.5**2)
    assert measured.shape == field.grid.shape
    np.testing.assert_allclose(
        np.asarray(measured),
        np.full(field.grid.shape, expected_w0 + expected_w1, dtype=np.float32),
        atol=1e-6,
    )


def test_detector_array_filter_mask_must_match_detector_grid():
    field = _two_wavelength_field()
    bad_filter = AmplitudeMask(amplitude_map=jnp.ones((2, 2), dtype=jnp.float32))
    sensor = DetectorArray(
        detector_grid=field.grid,
        qe_curve=jnp.array([0.5, 1.0], dtype=jnp.float32),
        filter_mask=bad_filter,
    )

    with pytest.raises(ValueError, match="filter_mask must be defined on detector_grid"):
        _ = sensor.measure(field)


def test_detector_array_can_resample_to_readout_grid():
    field = _two_wavelength_field()
    readout_grid = Grid.from_extent(nx=2, ny=2, dx_um=2.0, dy_um=1.5)
    sensor = DetectorArray(
        detector_grid=readout_grid,
        qe_curve=1.0,
        resample_method="nearest",
    )

    measured = sensor.measure(field)
    assert measured.shape == (readout_grid.ny, readout_grid.nx)


def test_detector_array_noise_is_opt_in_via_key():
    field = _two_wavelength_field()
    sensor = DetectorArray(
        detector_grid=field.grid,
        qe_curve=1.0,
        noise_model=PoissonNoise(count_scale=100.0),
    )

    clean = sensor.measure(field)
    noisy = sensor.measure(field, key=jax.random.PRNGKey(0))
    assert clean.shape == noisy.shape
    # Sampling should perturb at least one element with overwhelming probability.
    assert not np.array_equal(np.asarray(clean), np.asarray(noisy))
