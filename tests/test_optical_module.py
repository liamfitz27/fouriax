import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fouriax.optics import (
    AmplitudeMask,
    ASMPropagator,
    ComplexMask,
    Field,
    FieldReadout,
    FourierTransform,
    Grid,
    IntensitySensor,
    InverseFourierTransform,
    KSpacePhaseMask,
    OpticalModule,
    PhaseMask,
    Spectrum,
    ThinLens,
)
from fouriax.optics.na_planning import na_schedule


def test_optical_module_applies_layers_in_order():
    grid = Grid.from_extent(nx=8, ny=6, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    module = OpticalModule(
        layers=(
            PhaseMask(phase_map_rad=jnp.pi / 2.0),
            AmplitudeMask(amplitude_map=0.5),
        )
    )
    out = module.forward(field)

    expected = 0.5j * jnp.ones_like(out.data)
    np.testing.assert_allclose(np.asarray(out.data), np.asarray(expected), atol=1e-6)


def test_phase_mask_layer_rejects_bad_shape():
    grid = Grid.from_extent(nx=8, ny=6, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    layer = PhaseMask(phase_map_rad=jnp.zeros((3, 3), dtype=jnp.float32))
    with pytest.raises(ValueError, match="phase_map_rad shape mismatch"):
        layer.forward(field)


def test_complex_mask_matches_separate_amplitude_and_phase():
    grid = Grid.from_extent(nx=8, ny=6, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    amp = 0.25
    phase = jnp.pi / 3.0

    combined = ComplexMask(amplitude_map=amp, phase_map_rad=phase).forward(field)
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

    layer = ASMPropagator(use_sampling_planner=False, distance_um=500.0)

    out_layer = layer.forward(field)
    out_direct = ASMPropagator(use_sampling_planner=False, distance_um=500.0).forward(field)
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


def test_na_schedule_uses_spatial_grid_and_k_layer_diameter():
    grid = Grid.from_extent(nx=40, ny=20, dx_um=1.0, dy_um=1.0)  # min diameter = 20 um
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    module = OpticalModule(
        layers=(
            PhaseMask(phase_map_rad=0.0),  # spatial stop from grid extent
            ASMPropagator(use_sampling_planner=False, distance_um=10.0),
            KSpacePhaseMask(phase_map_rad=0.0, aperture_diameter_um=8.0),  # explicit k-stop
            ASMPropagator(use_sampling_planner=False, distance_um=10.0),
            ThinLens(focal_length_um=20.0, aperture_diameter_um=6.0),
        )
    )

    schedule = na_schedule(module.layers, field, medium_index=1.0)
    assert 1 in schedule
    assert 3 in schedule
    assert schedule[1] > 0.0
    assert schedule[3] > 0.0


def test_hybrid_spatial_k_stack_tracks_gradients():
    grid = Grid.from_extent(nx=16, ny=16, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)

    x, y = grid.spatial_grid()
    input_amp = jnp.exp(-((x + 1.5) ** 2 + (y - 0.5) ** 2) / (2.0 * 3.0**2))
    field_in = Field.plane_wave(grid=grid, spectrum=spectrum).apply_amplitude(input_amp[None, :, :])
    target = jnp.exp(-((x - 2.0) ** 2 + (y + 1.0) ** 2) / (2.0 * 2.4**2))

    phase_spatial_init = 0.08 * (x / jnp.max(jnp.abs(x))) + 0.03 * (y / jnp.max(jnp.abs(y)))
    phase_k_init = 0.05 * jnp.sin(0.4 * x) - 0.04 * jnp.cos(0.3 * y)
    params = {
        "phase_spatial": phase_spatial_init.astype(jnp.float32),
        "phase_k": phase_k_init.astype(jnp.float32),
    }

    def loss_fn(p: dict[str, jnp.ndarray]) -> jnp.ndarray:
        module = OpticalModule(
            layers=(
                PhaseMask(phase_map_rad=p["phase_spatial"]),
                ASMPropagator(
                    distance_um=12.0,
                    use_sampling_planner=False,
                    medium_index=1.0,
                    na_limit=None,
                ),
                FourierTransform(),
                KSpacePhaseMask(
                    phase_map_rad=p["phase_k"],
                    aperture_diameter_um=12.0,
                ),
                InverseFourierTransform(),
                ASMPropagator(
                    distance_um=8.0,
                    use_sampling_planner=False,
                    medium_index=1.0,
                    na_limit=None,
                ),
            ),
            sensor=IntensitySensor(sum_wavelengths=True),
        )
        measured = module.measure(field_in)
        return jnp.mean((measured - target) ** 2)

    grads = jax.grad(loss_fn)(params)
    assert grads["phase_spatial"].shape == (grid.ny, grid.nx)
    assert grads["phase_k"].shape == (grid.ny, grid.nx)
    assert bool(jnp.all(jnp.isfinite(grads["phase_spatial"])))
    assert bool(jnp.all(jnp.isfinite(grads["phase_k"])))
    assert float(jnp.linalg.norm(grads["phase_spatial"])) > 1e-10
    assert float(jnp.linalg.norm(grads["phase_k"])) > 1e-10


def test_optical_module_rejects_inline_propagator_without_distance():
    grid = Grid.from_extent(nx=16, ny=16, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    module = OpticalModule(layers=(ASMPropagator(),))
    with pytest.raises(ValueError, match="distance_um must be set for forward pass"):
        module.forward(field)


def test_optical_module_requires_explicit_domain_transforms():
    grid = Grid.from_extent(nx=16, ny=16, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    module_bad = OpticalModule(
        layers=(
            PhaseMask(phase_map_rad=0.0),
            KSpacePhaseMask(phase_map_rad=0.0),
        )
    )
    with pytest.raises(ValueError, match="requires kspace-domain input"):
        module_bad.forward(field)

    module_ok = OpticalModule(
        layers=(
            PhaseMask(phase_map_rad=0.0),
            FourierTransform(),
            KSpacePhaseMask(phase_map_rad=0.0),
            InverseFourierTransform(),
        )
    )
    out = module_ok.forward(field)
    assert out.domain == "spatial"
