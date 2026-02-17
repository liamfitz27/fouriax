import jax.numpy as jnp
import numpy as np
import pytest

from fouriax.optics import (
    AmplitudeMaskLayer,
    ASMPropagator,
    ComplexMaskLayer,
    Field,
    FieldReadout,
    Grid,
    IntensitySensor,
    KPhaseMaskLayer,
    OpticalModule,
    PhaseMaskLayer,
    PropagationLayer,
    Spectrum,
    ThinLensLayer,
)


def test_optical_module_applies_layers_in_order():
    grid = Grid.from_extent(nx=8, ny=6, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    module = OpticalModule(
        layers=(
            PhaseMaskLayer(phase_map_rad=jnp.pi / 2.0),
            AmplitudeMaskLayer(amplitude_map=0.5),
        )
    )
    out = module.forward(field)

    expected = 0.5j * jnp.ones_like(out.data)
    np.testing.assert_allclose(np.asarray(out.data), np.asarray(expected), atol=1e-6)


def test_phase_mask_layer_rejects_bad_shape():
    grid = Grid.from_extent(nx=8, ny=6, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    layer = PhaseMaskLayer(phase_map_rad=jnp.zeros((3, 3), dtype=jnp.float32))
    with pytest.raises(ValueError, match="phase_map_rad shape mismatch"):
        layer.forward(field)


def test_complex_mask_matches_separate_amplitude_and_phase():
    grid = Grid.from_extent(nx=8, ny=6, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    amp = 0.25
    phase = jnp.pi / 3.0

    combined = ComplexMaskLayer(amplitude_map=amp, phase_map_rad=phase).forward(field)
    separate = field.apply_amplitude(amp).apply_phase(phase)

    np.testing.assert_allclose(
        np.asarray(combined.data),
        np.asarray(separate.data),
        atol=1e-6,
    )


def test_propagation_layer_matches_direct_propagator():
    grid = Grid.from_extent(nx=16, ny=16, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    model = ASMPropagator(use_sampling_planner=False)
    layer = PropagationLayer(model=model, distance_um=500.0)

    out_layer = layer.forward(field)
    out_direct = model.propagate(field, distance_um=500.0)
    np.testing.assert_allclose(
        np.asarray(out_layer.data),
        np.asarray(out_direct.data),
        atol=1e-6,
    )


def test_intensity_sensor_and_field_readout_representations():
    grid = Grid.from_extent(nx=4, ny=4, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_array(jnp.array([0.532, 0.633], dtype=jnp.float32))
    data = jnp.stack(
        [
            jnp.ones(grid.shape, dtype=jnp.complex64),
            2.0j * jnp.ones(grid.shape, dtype=jnp.complex64),
        ],
        axis=0,
    )
    field = Field(data=data, grid=grid, spectrum=spectrum)

    intensity = IntensitySensor(sum_wavelengths=False).measure(field)
    summed = IntensitySensor(sum_wavelengths=True).measure(field)
    np.testing.assert_allclose(np.asarray(intensity[0]), np.ones(grid.shape), atol=1e-6)
    np.testing.assert_allclose(np.asarray(intensity[1]), 4.0 * np.ones(grid.shape), atol=1e-6)
    np.testing.assert_allclose(np.asarray(summed), 5.0 * np.ones(grid.shape), atol=1e-6)

    real_imag = FieldReadout(representation="real_imag").measure(field)
    amp_phase = FieldReadout(representation="amplitude_phase").measure(field)
    assert real_imag.shape == (2, 4, 4, 2)
    assert amp_phase.shape == (2, 4, 4, 2)

    np.testing.assert_allclose(
        np.asarray(real_imag[..., 0]),
        np.asarray(field.data.real),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(real_imag[..., 1]),
        np.asarray(field.data.imag),
        atol=1e-6,
    )


def test_optical_module_na_schedule_uses_spatial_grid_and_k_layer_diameter():
    grid = Grid.from_extent(nx=40, ny=20, dx_um=1.0, dy_um=1.0)  # min diameter = 20 um
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    module = OpticalModule(
        layers=(
            PhaseMaskLayer(phase_map_rad=0.0),  # spatial stop from grid extent
            PropagationLayer(model=ASMPropagator(use_sampling_planner=False), distance_um=10.0),
            KPhaseMaskLayer(phase_map_rad=0.0, aperture_diameter_um=8.0),  # explicit k-stop
            PropagationLayer(model=ASMPropagator(use_sampling_planner=False), distance_um=10.0),
            ThinLensLayer(focal_length_um=20.0, aperture_diameter_um=6.0),
        ),
        auto_apply_na=True,
        medium_index=1.0,
    )

    schedule = module.na_schedule(field)
    assert 1 in schedule
    assert 3 in schedule
    assert schedule[1] > 0.0
    assert schedule[3] > 0.0
