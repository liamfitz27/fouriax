import jax.numpy as jnp
import numpy as np
import pytest

from fouriax.optics import (
    Field,
    FourierTransform,
    Grid,
    KSpaceAmplitudeMask,
    KSpaceComplexMask,
    KSpacePhaseMask,
    Spectrum,
)


def _manual_k_op(
    field: Field,
    *,
    amplitude: jnp.ndarray | float = 1.0,
    phase_rad: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    k = field.to_kspace().data
    mod = k * jnp.asarray(amplitude) * jnp.exp(1j * jnp.asarray(phase_rad))
    return mod


def test_k_phase_mask_matches_manual_kspace_multiplication():
    grid = Grid.from_extent(nx=14, ny=10, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum, phase=0.4)
    field_k = FourierTransform().forward(field)

    phase = jnp.linspace(0.0, jnp.pi, grid.nx * grid.ny, dtype=jnp.float32).reshape(
        1, grid.ny, grid.nx
    )
    out = KSpacePhaseMask(phase_map_rad=phase).forward(field_k)
    expected = _manual_k_op(field, phase_rad=phase)
    np.testing.assert_allclose(np.asarray(out.data), np.asarray(expected), atol=1e-6)
    assert out.domain == "kspace"


def test_k_amplitude_mask_matches_manual_kspace_multiplication():
    grid = Grid.from_extent(nx=14, ny=10, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum, phase=0.4)
    field_k = FourierTransform().forward(field)

    amp = jnp.linspace(0.2, 1.0, grid.nx * grid.ny, dtype=jnp.float32).reshape(
        1, grid.ny, grid.nx
    )
    out = KSpaceAmplitudeMask(amplitude_map=amp).forward(field_k)
    expected = _manual_k_op(field, amplitude=amp)
    np.testing.assert_allclose(np.asarray(out.data), np.asarray(expected), atol=1e-6)
    assert out.domain == "kspace"


def test_k_complex_mask_matches_manual_kspace_multiplication():
    grid = Grid.from_extent(nx=12, ny=12, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum, phase=0.1)
    field_k = FourierTransform().forward(field)

    amp = jnp.linspace(0.2, 1.0, grid.nx * grid.ny, dtype=jnp.float32).reshape(
        1, grid.ny, grid.nx
    )
    phase = jnp.linspace(-jnp.pi, jnp.pi, grid.nx * grid.ny, dtype=jnp.float32).reshape(
        1, grid.ny, grid.nx
    )
    out = KSpaceComplexMask(amplitude_map=amp, phase_map_rad=phase).forward(field_k)
    expected = _manual_k_op(field, amplitude=amp, phase_rad=phase)
    np.testing.assert_allclose(np.asarray(out.data), np.asarray(expected), atol=1e-6)
    assert out.domain == "kspace"


def test_k_phase_mask_rejects_bad_shape():
    grid = Grid.from_extent(nx=8, ny=6, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    with pytest.raises(ValueError, match="requires kspace-domain input"):
        KSpacePhaseMask(phase_map_rad=0.0).forward(field)

    field_k = FourierTransform().forward(field)
    with pytest.raises(ValueError, match="phase_map_rad shape mismatch"):
        KSpacePhaseMask(phase_map_rad=jnp.zeros((3, 3), dtype=jnp.float32)).forward(field_k)
