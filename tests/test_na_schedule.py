import jax.numpy as jnp
import numpy as np

from fouriax.optics import (
    ASMPropagator,
    Field,
    Grid,
    KPhaseMaskLayer,
    OpticalModule,
    PhaseMaskLayer,
    PropagationLayer,
    RSPropagator,
    Spectrum,
    ThinLensLayer,
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
            PhaseMaskLayer(phase_map_rad=0.0),  # spatial stop from grid extent
            PropagationLayer(model=ASMPropagator(use_sampling_planner=False), distance_um=10.0),
            KPhaseMaskLayer(phase_map_rad=0.0, aperture_diameter_um=8.0),  # explicit k stop
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
    assert schedule[3] < schedule[1]


def test_na_schedule_fallback_to_effective_for_unconstrained_module():
    grid = Grid.from_extent(nx=32, ny=32, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    module = OpticalModule(
        layers=(
            PropagationLayer(model=ASMPropagator(use_sampling_planner=False), distance_um=5.0),
            PropagationLayer(model=ASMPropagator(use_sampling_planner=False), distance_um=7.0),
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
    asm_layer = PropagationLayer(
        model=ASMPropagator(use_sampling_planner=False, na_limit=None), distance_um=5.0
    )
    rs_layer = PropagationLayer(model=RSPropagator(use_sampling_planner=False), distance_um=5.0)

    asm_updated = OpticalModule._layer_with_na_if_supported(asm_layer, 0.25)
    rs_updated = OpticalModule._layer_with_na_if_supported(rs_layer, 0.25)

    assert isinstance(asm_updated, PropagationLayer)
    assert isinstance(rs_updated, PropagationLayer)
    assert asm_updated.model.na_limit == 0.25
    assert not hasattr(rs_updated.model, "na_limit")
