import jax.numpy as jnp
import numpy as np
import pytest

from fouriax.optics import (
    ASMPropagator,
    DetectorArray,
    Field,
    FourierTransform,
    Grid,
    JonesMatrixLayer,
    KJonesMatrixLayer,
    KSpacePropagator,
    Spectrum,
)


def test_plane_wave_jones_has_expected_shape_and_intensity():
    grid = Grid.from_extent(nx=8, ny=6, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_array(jnp.array([0.532, 0.633], dtype=jnp.float32))
    field = Field.plane_wave_jones(grid=grid, spectrum=spectrum, ex=1.0 + 0.0j, ey=2.0 + 0.0j)

    assert field.polarization_mode == "jones"
    assert field.data.shape == (2, 2, 6, 8)
    np.testing.assert_allclose(np.asarray(field.intensity()), 5.0, atol=1e-6)


def test_jones_matrix_layer_matches_manual_component_mix():
    grid = Grid.from_extent(nx=10, ny=8, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave_jones(grid=grid, spectrum=spectrum, ex=1.0 + 0.0j, ey=0.0 + 0.0j)
    matrix = jnp.asarray([[1.0 + 0.0j, 0.5 + 0.0j], [0.25 + 0.0j, 2.0 + 0.0j]])

    out = JonesMatrixLayer(jones_matrix=matrix).forward(field)
    np.testing.assert_allclose(np.asarray(out.data[:, 0]), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(out.data[:, 1]), 0.25, atol=1e-6)


def test_k_jones_matrix_layer_requires_kspace():
    grid = Grid.from_extent(nx=10, ny=8, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave_jones(grid=grid, spectrum=spectrum)
    matrix = jnp.asarray([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]])

    with pytest.raises(ValueError, match="requires kspace-domain input"):
        KJonesMatrixLayer(jones_matrix=matrix).forward(field)

    out = KJonesMatrixLayer(jones_matrix=matrix).forward(FourierTransform().forward(field))
    assert out.domain == "kspace"


def test_jones_propagation_matches_scalar_per_component():
    grid = Grid.from_extent(nx=12, ny=10, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    jones = Field.plane_wave_jones(grid=grid, spectrum=spectrum, ex=1.2 + 0.3j, ey=-0.7 + 0.2j)
    scalar_ex = Field(data=jones.data[:, 0], grid=grid, spectrum=spectrum)
    scalar_ey = Field(data=jones.data[:, 1], grid=grid, spectrum=spectrum)

    rs_layer = ASMPropagator(distance_um=20.0, use_sampling_planner=False)
    out_jones = rs_layer.forward(jones)
    out_ex = rs_layer.forward(scalar_ex)
    out_ey = rs_layer.forward(scalar_ey)
    np.testing.assert_allclose(np.asarray(out_jones.data[:, 0]), np.asarray(out_ex.data), atol=1e-6)
    np.testing.assert_allclose(np.asarray(out_jones.data[:, 1]), np.asarray(out_ey.data), atol=1e-6)

    k_layer = KSpacePropagator(distance_um=20.0)
    out_jones_k = k_layer.forward(FourierTransform().forward(jones))
    out_ex_k = k_layer.forward(FourierTransform().forward(scalar_ex))
    out_ey_k = k_layer.forward(FourierTransform().forward(scalar_ey))
    np.testing.assert_allclose(
        np.asarray(out_jones_k.data[:, 0]), np.asarray(out_ex_k.data), atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(out_jones_k.data[:, 1]), np.asarray(out_ey_k.data), atol=1e-6
    )


def test_detector_array_channel_resolved_for_jones():
    grid = Grid.from_extent(nx=6, ny=4, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave_jones(grid=grid, spectrum=spectrum, ex=1.0 + 0.0j, ey=2.0j)

    total = DetectorArray(detector_grid=grid, sum_wavelengths=False).measure(field)
    by_channel = DetectorArray(
        detector_grid=grid,
        sum_wavelengths=False,
        channel_resolved=True,
    ).measure(field)

    assert total.shape == (1, 4, 6)
    assert by_channel.shape == (1, 2, 4, 6)
    np.testing.assert_allclose(np.asarray(total), 5.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(by_channel[:, 0]), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(by_channel[:, 1]), 4.0, atol=1e-6)
