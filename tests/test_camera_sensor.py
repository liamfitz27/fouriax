import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fouriax.optics import (
    AmplitudeMask,
    DetectorArray,
    Field,
    Grid,
    Intensity,
    PoissonNoise,
    Spectrum,
)


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


def test_detector_array_linear_resampling_distributes_coarse_pixels_over_finer_grid():
    src_grid = Grid.from_extent(nx=2, ny=2, dx_um=2.0, dy_um=2.0)
    spectrum = Spectrum.from_scalar(0.532)
    intensity = Intensity(
        data=jnp.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=jnp.float32),
        grid=src_grid,
        spectrum=spectrum,
    )
    detector_grid = Grid.from_extent(nx=4, ny=4, dx_um=1.0, dy_um=1.0)
    sensor = DetectorArray(
        detector_grid=detector_grid,
        qe_curve=1.0,
        resample_method="linear",
    )

    measured = sensor.measure(intensity)
    expected = np.asarray(
        [
            [0.25, 0.25, 0.50, 0.50],
            [0.25, 0.25, 0.50, 0.50],
            [0.75, 0.75, 1.00, 1.00],
            [0.75, 0.75, 1.00, 1.00],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(np.asarray(measured), expected, atol=1e-6)
    np.testing.assert_allclose(np.asarray(jnp.sum(measured)), 10.0, atol=1e-6)


def test_detector_array_integrates_intensity_without_complex_cancellation():
    grid = Grid.from_extent(nx=2, ny=1, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    data = jnp.asarray([[1.0 + 0.0j, -1.0 + 0.0j]], dtype=jnp.complex64).reshape((1, 1, 2))
    field = Field(data=data, grid=grid, spectrum=spectrum, domain="spatial")
    sensor = DetectorArray(
        detector_grid=Grid.from_extent(nx=1, ny=1, dx_um=2.0, dy_um=1.0),
        qe_curve=1.0,
        resample_method="linear",
    )

    measured = sensor.measure(field)
    np.testing.assert_allclose(
        np.asarray(measured),
        np.asarray([[2.0]], dtype=np.float32),
        atol=1e-6,
    )


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


def test_detector_array_accepts_explicit_intensity_input():
    field = _two_wavelength_field()
    intensity = field.to_intensity()
    sensor = DetectorArray(
        detector_grid=field.grid,
        qe_curve=1.0,
        sum_wavelengths=False,
    )

    from_field = sensor.measure(field)
    from_intensity = sensor.measure(intensity)
    np.testing.assert_allclose(np.asarray(from_intensity), np.asarray(from_field), atol=1e-6)


def test_detector_array_linear_operator_matches_expected():
    src_grid = Grid.from_extent(nx=2, ny=2, dx_um=2.0, dy_um=2.0)
    spectrum = Spectrum.from_array(jnp.array([0.532, 0.633], dtype=jnp.float32))
    intensity = Intensity(
        data=jnp.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[0.5, 1.5], [2.5, 3.5]],
            ],
            dtype=jnp.float32,
        ),
        grid=src_grid,
        spectrum=spectrum,
    )
    sensor = DetectorArray(
        detector_grid=Grid.from_extent(nx=4, ny=4, dx_um=1.0, dy_um=1.0),
        qe_curve=jnp.array([0.5, 1.0], dtype=jnp.float32),
        sum_wavelengths=False,
        resample_method="linear",
    )

    op = sensor.linear_operator(intensity, flatten=False)
    measured = sensor.expected(intensity)
    np.testing.assert_allclose(
        np.asarray(op.matvec(intensity.data)),
        np.asarray(measured),
        atol=1e-6,
    )


def test_detector_array_linear_operator_adjoint_and_flatten():
    src_grid = Grid.from_extent(nx=3, ny=2, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_array(jnp.array([0.532, 0.633], dtype=jnp.float32))
    template = Intensity(
        data=jnp.ones((spectrum.size, src_grid.ny, src_grid.nx), dtype=jnp.float32),
        grid=src_grid,
        spectrum=spectrum,
    )
    sensor = DetectorArray(
        detector_grid=Grid.from_extent(nx=2, ny=2, dx_um=1.5, dy_um=1.0),
        qe_curve=jnp.array([0.8, 0.6], dtype=jnp.float32),
        sum_wavelengths=True,
        resample_method="linear",
    )

    op = sensor.linear_operator(template, flatten=False)
    x = jnp.linspace(
        0.0,
        1.0,
        spectrum.size * src_grid.ny * src_grid.nx,
        dtype=jnp.float32,
    ).reshape((spectrum.size, src_grid.ny, src_grid.nx))
    y = jnp.linspace(
        1.0,
        2.0,
        sensor.detector_grid.ny * sensor.detector_grid.nx,
        dtype=jnp.float32,
    ).reshape((sensor.detector_grid.ny, sensor.detector_grid.nx))

    lhs = jnp.vdot(op.matvec(x), y)
    rhs = jnp.vdot(x, op.rmatvec(y))
    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), atol=1e-6, rtol=1e-6)

    op_flat = sensor.linear_operator(template, flatten=True)
    np.testing.assert_allclose(
        np.asarray(op_flat.matvec(x.reshape(-1)).reshape(op.out_shape)),
        np.asarray(op.matvec(x)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(op_flat.rmatvec(y.reshape(-1)).reshape(op.in_shape)),
        np.asarray(op.rmatvec(y)),
        atol=1e-6,
    )


def test_detector_array_rejects_complex_intensity_input():
    field = _two_wavelength_field()
    bad = Intensity(
        data=jnp.ones((field.spectrum.size, field.grid.ny, field.grid.nx), dtype=jnp.complex64),
        grid=field.grid,
        spectrum=field.spectrum,
    )
    sensor = DetectorArray(detector_grid=field.grid, qe_curve=1.0)

    with pytest.raises(ValueError, match="intensity data must be real-valued"):
        sensor.measure(bad)
