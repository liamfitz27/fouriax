import jax.numpy as jnp
import numpy as np

from fouriax.optics import (
    ASMPropagator,
    Field,
    Grid,
    KSpacePhaseMask,
    OpticalModule,
    PhaseMask,
    RSPropagator,
    Spectrum,
    ThinLens,
    build_na_mask,
)


def test_build_na_mask_respects_medium_cap():
    grid = Grid.from_extent(nx=64, ny=64, dx_um=1.0, dy_um=1.0)
    wl = 0.5
    mask_lo = build_na_mask(grid, wavelength_um=wl, na_limit=0.3, medium_index=1.0)
    mask_hi = build_na_mask(grid, wavelength_um=wl, na_limit=2.0, medium_index=0.2)

    assert float(jnp.sum(mask_lo)) > 0
    # hi NA should be capped by medium index, so still not full passband.
    assert float(jnp.sum(mask_hi)) < grid.nx * grid.ny


def test_na_schedule_local_segments_from_spatial_and_k_stops():
    grid = Grid.from_extent(nx=40, ny=20, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    module = OpticalModule(
        layers=(
            PhaseMask(phase_map_rad=0.0),  # spatial stop from grid extent
            ASMPropagator(use_sampling_planner=False, distance_um=10.0),
            KSpacePhaseMask(phase_map_rad=0.0, aperture_diameter_um=8.0),  # explicit k stop
            ASMPropagator(use_sampling_planner=False, distance_um=10.0),
            ThinLens(focal_length_um=20.0, aperture_diameter_um=6.0),
        ),
        auto_apply_na=True,
        medium_index=1.0,
    )

    schedule = module.na_schedule(field)
    assert 1 in schedule
    assert 3 in schedule
    assert schedule[1] > 0.0
    assert schedule[3] > 0.0
    assert schedule[3] < schedule[1]


def test_na_schedule_fallback_to_effective_for_unconstrained_module():
    grid = Grid.from_extent(nx=32, ny=32, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    module = OpticalModule(
        layers=(
            ASMPropagator(use_sampling_planner=False, distance_um=5.0),
            ASMPropagator(use_sampling_planner=False, distance_um=7.0),
        ),
        auto_apply_na=True,
        medium_index=1.2,
        na_fallback_to_effective=True,
    )

    schedule = module.na_schedule(field)
    assert set(schedule.keys()) == {0, 1}
    assert np.isclose(schedule[0], 1.2)
    assert np.isclose(schedule[1], 1.2)


def test_layer_na_injection_only_applies_to_models_with_na_limit():
    asm_layer = ASMPropagator(use_sampling_planner=False, na_limit=None, distance_um=5.0)
    rs_layer = RSPropagator(use_sampling_planner=False, na_limit=None, distance_um=5.0)

    asm_updated = OpticalModule._layer_with_na_if_supported(asm_layer, 0.25)
    rs_updated = OpticalModule._layer_with_na_if_supported(rs_layer, 0.25)

    assert isinstance(asm_updated, ASMPropagator)
    assert isinstance(rs_updated, RSPropagator)
    assert asm_updated.na_limit == 0.25
    assert rs_updated.na_limit == 0.25


def test_layer_na_injection_keeps_stricter_existing_limit():
    asm_layer = ASMPropagator(use_sampling_planner=False, na_limit=0.15, distance_um=5.0)
    rs_layer = RSPropagator(use_sampling_planner=False, na_limit=0.12, distance_um=5.0)

    asm_updated = OpticalModule._layer_with_na_if_supported(asm_layer, 0.25)
    rs_updated = OpticalModule._layer_with_na_if_supported(rs_layer, 0.25)

    assert isinstance(asm_updated, ASMPropagator)
    assert isinstance(rs_updated, RSPropagator)
    assert np.isclose(asm_updated.na_limit, 0.15)
    assert np.isclose(rs_updated.na_limit, 0.12)


def _out_of_band_energy_ratio(field: Field, wavelength_index: int, na_limit: float) -> float:
    k_field = field.to_kspace().data[wavelength_index]
    mask = build_na_mask(
        field.grid,
        wavelength_um=float(field.spectrum.wavelengths_um[wavelength_index]),
        na_limit=na_limit,
        medium_index=1.0,
    )
    power = jnp.abs(k_field) ** 2
    in_band = jnp.sum(power * mask)
    out_band = jnp.sum(power * (1.0 - mask))
    return float(out_band / (in_band + out_band + 1e-12))


def test_rs_propagator_na_limit_suppresses_out_of_band_spectrum():
    grid = Grid.from_extent(nx=64, ny=64, dx_um=0.6, dy_um=0.6)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)
    na_limit = 0.1

    out_unlimited = RSPropagator(
        use_sampling_planner=False,
        medium_index=1.0,
        na_limit=None,
        distance_um=20.0,
    ).forward(field)
    out_limited = RSPropagator(
        use_sampling_planner=False,
        medium_index=1.0,
        na_limit=na_limit,
        distance_um=20.0,
    ).forward(field)

    ratio_unlimited = _out_of_band_energy_ratio(
        out_unlimited,
        wavelength_index=0,
        na_limit=na_limit,
    )
    ratio_limited = _out_of_band_energy_ratio(out_limited, wavelength_index=0, na_limit=na_limit)

    assert ratio_limited < ratio_unlimited
    assert ratio_limited < 1e-3
