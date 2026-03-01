import jax
import jax.numpy as jnp
import numpy as np

from fouriax.optics import Field, FieldMonitor, Grid, IntensityMonitor, Spectrum


def _plane_wave_field(*, amplitude: float = 1.0) -> Field:
    grid = Grid.from_extent(nx=8, ny=6, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    return Field.plane_wave(grid=grid, spectrum=spectrum, amplitude=amplitude)


def test_field_monitor_defaults_to_input_domain_without_transform():
    field = _plane_wave_field(amplitude=2.0)
    field_k = field.to_kspace()

    measured = FieldMonitor(representation="complex").read(field_k)
    np.testing.assert_allclose(np.asarray(measured), np.asarray(field_k.data), atol=1e-6)


def test_field_monitor_can_convert_domain_on_read():
    field = _plane_wave_field(amplitude=2.0)
    field_k = field.to_kspace()

    measured = FieldMonitor(representation="complex", output_domain="spatial").read(field_k)
    np.testing.assert_allclose(np.asarray(measured), np.asarray(field.data), atol=1e-5)


def test_intensity_monitor_defaults_to_input_domain_without_transform():
    field = _plane_wave_field(amplitude=2.0)
    field_k = field.to_kspace()

    measured = IntensityMonitor().read(field_k)
    expected = np.abs(np.asarray(field_k.data)) ** 2
    np.testing.assert_allclose(np.asarray(measured), expected, atol=1e-6)


def test_intensity_monitor_can_convert_to_spatial_domain():
    field = _plane_wave_field(amplitude=2.0)
    field_k = field.to_kspace()

    measured = IntensityMonitor(output_domain="spatial").read(field_k)
    expected = np.abs(np.asarray(field.data)) ** 2
    np.testing.assert_allclose(np.asarray(measured), expected, atol=1e-5)


def test_monitor_domain_choice_preserves_gradients():
    grid = Grid.from_extent(nx=8, ny=8, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    monitor = IntensityMonitor(output_domain="kspace", sum_wavelengths=True)

    def loss_fn(amplitude: jnp.ndarray) -> jnp.ndarray:
        field = Field.plane_wave(grid=grid, spectrum=spectrum, amplitude=amplitude)
        intensity = monitor.read(field)
        return jnp.mean(intensity)

    grad = jax.grad(loss_fn)(jnp.asarray(1.5, dtype=jnp.float32))
    assert jnp.isfinite(grad)
    assert float(jnp.abs(grad)) > 0.0
