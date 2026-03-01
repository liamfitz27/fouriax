import jax
import jax.numpy as jnp
import numpy as np

from fouriax.optics import (
    Detector,
    DetectorArray,
    Field,
    GaussianNoise,
    Grid,
    OpticalModule,
    PoissonGaussianNoise,
    PoissonNoise,
    Spectrum,
)


def _plane_wave_field(*, amplitude: float = 2.0, n: int = 8) -> Field:
    grid = Grid.from_extent(nx=n, ny=n, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    return Field.plane_wave(grid=grid, spectrum=spectrum, amplitude=amplitude)


def test_detector_integrates_region_mask():
    field = _plane_wave_field(amplitude=2.0, n=4)
    region_mask = jnp.zeros(field.grid.shape, dtype=jnp.float32).at[:2, :2].set(1.0)
    detector = Detector(region_mask=region_mask, sum_wavelengths=True)

    measured = detector.measure(field)
    np.testing.assert_allclose(np.asarray(measured), 16.0, atol=1e-6)


def test_detector_array_noise_model_is_opt_in_via_key():
    field = _plane_wave_field(amplitude=2.0)
    sensor = DetectorArray(
        detector_grid=field.grid,
        sum_wavelengths=True,
        noise_model=PoissonNoise(count_scale=100.0),
    )

    clean = sensor.measure(field)
    expected = np.full(field.grid.shape, 4.0, dtype=np.float32)
    np.testing.assert_allclose(np.asarray(clean), expected, atol=1e-6)


def test_detector_array_poisson_noise_is_reproducible_for_same_key():
    field = _plane_wave_field(amplitude=2.0)
    sensor = DetectorArray(
        detector_grid=field.grid,
        sum_wavelengths=True,
        noise_model=PoissonNoise(count_scale=250.0),
    )
    key = jax.random.PRNGKey(0)

    sample_a = sensor.measure(field, key=key)
    sample_b = sensor.measure(field, key=key)
    np.testing.assert_array_equal(np.asarray(sample_a), np.asarray(sample_b))
    assert sample_a.shape == field.grid.shape
    assert sample_a.dtype == jnp.float32


def test_gaussian_noise_zero_std_matches_clean_measurement():
    field = _plane_wave_field(amplitude=1.5)
    sensor = DetectorArray(
        detector_grid=field.grid,
        sum_wavelengths=True,
        noise_model=GaussianNoise(std=0.0),
    )
    key = jax.random.PRNGKey(123)

    clean = sensor.measure(field)
    noisy = sensor.measure(field, key=key)
    np.testing.assert_allclose(np.asarray(noisy), np.asarray(clean), atol=1e-6)


def test_noise_model_expected_variance_matches_configuration():
    mu = jnp.array([0.5, 2.0, 4.0], dtype=jnp.float32)

    poisson_var = PoissonNoise(count_scale=20.0).expected_variance(mu)
    np.testing.assert_allclose(np.asarray(poisson_var), np.asarray(mu / 20.0), atol=1e-6)

    combo_var = PoissonGaussianNoise(
        count_scale=20.0,
        read_noise_std=0.3,
    ).expected_variance(mu)
    np.testing.assert_allclose(
        np.asarray(combo_var),
        np.asarray(mu / 20.0 + 0.3**2),
        atol=1e-6,
    )


def test_noise_model_analytic_covariance_and_precision_are_diagonal():
    mu = jnp.array([0.5, 2.0, 4.0], dtype=jnp.float32)
    noise = PoissonGaussianNoise(count_scale=20.0, read_noise_std=0.3)
    variance = noise.expected_variance(mu)
    covariance = noise.covariance(mu)
    precision = noise.precision(mu)

    np.testing.assert_allclose(np.asarray(covariance), np.diag(np.asarray(variance)), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(precision),
        np.diag(1.0 / np.asarray(variance)),
        atol=1e-6,
    )


def test_optical_module_measure_forwards_key_to_detector():
    field = _plane_wave_field(amplitude=1.0)
    detector = DetectorArray(
        detector_grid=field.grid,
        sum_wavelengths=True,
        noise_model=PoissonNoise(count_scale=50.0),
    )
    module = OpticalModule(layers=(), sensor=detector)
    key = jax.random.PRNGKey(7)

    direct = detector.measure(field, key=key)
    via_module = module.measure(field, key=key)
    np.testing.assert_array_equal(np.asarray(via_module), np.asarray(direct))
