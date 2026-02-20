import jax.numpy as jnp
import numpy as np

from fouriax.optics import (
    ASMPropagator,
    Field,
    Grid,
    KSpacePhaseMask,
    KSpacePropagator,
    PhaseMask,
    Spectrum,
)


def test_field_spatial_kspace_roundtrip_matches_original():
    grid = Grid.from_extent(nx=12, ny=10, dx_um=0.8, dy_um=1.1)
    spectrum = Spectrum.from_array(jnp.array([0.532, 0.633], dtype=jnp.float32))
    base = Field.plane_wave(grid=grid, spectrum=spectrum, phase=0.3)
    phase_ramp = jnp.linspace(0.0, 1.0, grid.nx * grid.ny, dtype=jnp.float32).reshape(
        1, grid.ny, grid.nx
    )
    field = base.apply_phase(phase_ramp)

    rt = field.to_kspace().to_spatial()
    np.testing.assert_allclose(np.asarray(rt.data), np.asarray(field.data), atol=1e-6)
    assert rt.domain == "spatial"


def test_field_to_kspace_is_idempotent_and_preserves_sampling_metadata():
    grid = Grid.from_extent(nx=16, ny=8, dx_um=1.2, dy_um=0.7)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum).to_kspace()

    again = field.to_kspace()
    assert again is field
    assert again.domain == "kspace"
    assert again.kspace_pixel_size_cyc_per_um == grid.kspace_pixel_size_cyc_per_um()


def test_spatial_layer_auto_converts_kspace_input():
    grid = Grid.from_extent(nx=10, ny=10, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field_k = Field.plane_wave(grid=grid, spectrum=spectrum).to_kspace()

    out = PhaseMask(phase_map_rad=0.1).forward(field_k)
    assert out.domain == "spatial"


def test_k_layer_and_k_propagator_auto_convert_spatial_input():
    grid = Grid.from_extent(nx=10, ny=10, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    out_k = KSpacePhaseMask(phase_map_rad=0.2).forward(field)
    assert out_k.domain == "kspace"

    out_prop = KSpacePropagator(refractive_index=1.0).propagate(field, distance_um=5.0)
    assert out_prop.domain == "kspace"

    out_asm = ASMPropagator(use_sampling_planner=False).propagate(out_k, distance_um=5.0)
    assert out_asm.domain == "spatial"
