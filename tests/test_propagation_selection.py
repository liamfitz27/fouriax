from dataclasses import replace

import numpy as np
import pytest
from scipy.special import j1

from fouriax.optics import (
    ASMPropagator,
    Field,
    FourierTransform,
    Grid,
    KSpacePropagator,
    RSPropagator,
    Spectrum,
)
from fouriax.optics.propagation import (
    critical_distance_um,
    plan_propagation,
    recommend_nyquist_grid,
    select_propagator_method,
)


def test_critical_distance_uses_expected_formula():
    grid = Grid.from_extent(nx=256, ny=128, dx_um=1.0, dy_um=0.5)
    spectrum = Spectrum.from_array([0.532, 0.633])
    z_crit = critical_distance_um(grid=grid, spectrum=spectrum)
    expected = min(grid.nx, grid.ny) * (max(grid.dx_um, grid.dy_um) ** 2) / 0.532
    assert abs(z_crit - expected) < 1e-5


def test_method_selection_matches_critical_regime():
    grid = Grid.from_extent(nx=128, ny=128, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.5)
    z_crit = critical_distance_um(grid=grid, spectrum=spectrum)
    assert select_propagator_method(grid, spectrum, 0.5 * z_crit) == "asm"
    assert select_propagator_method(grid, spectrum, z_crit) == "rs"
    assert select_propagator_method(grid, spectrum, 2.0 * z_crit) == "rs"


def test_recommend_nyquist_grid_respects_requested_density():
    grid = Grid.from_extent(nx=64, ny=64, dx_um=2.0, dy_um=2.0)
    spectrum = Spectrum.from_scalar(0.5)
    recommended = recommend_nyquist_grid(
        grid=grid,
        spectrum=spectrum,
        nyquist_factor=2.0,
        min_padding_factor=2.0,
    )
    assert recommended.nx >= grid.nx
    assert recommended.ny >= grid.ny
    assert recommended.dx_um <= grid.dx_um
    assert recommended.dy_um <= grid.dy_um


def test_specific_propagators_warn_when_used_outside_recommended_regime():
    grid = Grid.from_extent(nx=128, ny=128, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)
    z_crit = critical_distance_um(grid=grid, spectrum=spectrum)

    with pytest.warns(UserWarning, match="ASM selected outside recommended regime"):
        ASMPropagator(use_sampling_planner=False, distance_um=2.0 * z_crit).forward(field)

    with pytest.warns(UserWarning, match="RS selected outside recommended regime"):
        RSPropagator(use_sampling_planner=False, distance_um=0.5 * z_crit).forward(field)


def _airy_profile(
    r_um: np.ndarray,
    wavelength_um: float,
    focal_um: float,
    diameter_um: float,
) -> np.ndarray:
    alpha = np.pi * diameter_um * r_um / (wavelength_um * focal_um)
    profile = np.ones_like(alpha, dtype=np.float64)
    nonzero = alpha != 0.0
    profile[nonzero] = (2.0 * j1(alpha[nonzero]) / alpha[nonzero]) ** 2
    return profile


def _radial_row_profile(intensity_2d: np.ndarray, dx_um: float) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = intensity_2d.shape
    cy = ny // 2
    cx = nx // 2
    row = intensity_2d[cy, :]
    row = row / np.max(row)
    x_um = (np.arange(nx) - (nx - 1) / 2.0) * dx_um
    r_um = np.abs(x_um)
    return r_um[cx:], row[cx:]


def _mae_vs_airy(
    propagated_intensity_2d: np.ndarray,
    wavelength_um: float,
    focal_um: float,
    aperture_diameter_um: float,
    dx_um: float,
) -> float:
    r_um, profile = _radial_row_profile(propagated_intensity_2d, dx_um)
    expected_first_zero_um = 1.22 * wavelength_um * focal_um / aperture_diameter_um
    r_max = 0.9 * expected_first_zero_um
    n_compare = max(8, int(np.floor(r_max / dx_um)))
    r_compare = r_um[:n_compare]
    sim = profile[:n_compare]
    airy = _airy_profile(
        r_compare,
        wavelength_um=wavelength_um,
        focal_um=focal_um,
        diameter_um=aperture_diameter_um,
    )
    return float(np.mean(np.abs(sim - airy)))


def test_plan_propagation_selection_matches_best_method_in_near_and_far_regimes():
    wavelength_um = 0.532
    grid = Grid.from_extent(nx=256, ny=256, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(wavelength_um)
    aperture_diameter_um = 30.0
    z_crit = critical_distance_um(grid=grid, spectrum=spectrum)

    from fouriax.optics import ThinLens

    asm = ASMPropagator(use_sampling_planner=True, nyquist_factor=2.0, min_padding_factor=2.0)
    rs = RSPropagator(use_sampling_planner=True, nyquist_factor=2.0, min_padding_factor=2.0)
    field_in = Field.plane_wave(grid=grid, spectrum=spectrum)

    for distance_um, expected_method in ((0.5 * z_crit, "asm"), (2.0 * z_crit, "rs")):
        lens = ThinLens(
            focal_length_um=distance_um,
            aperture_diameter_um=aperture_diameter_um,
        )
        field_lens = lens.forward(field_in)

        out_asm = replace(asm, distance_um=distance_um).forward(field_lens)
        out_rs = replace(rs, distance_um=distance_um).forward(field_lens)
        mae_asm = _mae_vs_airy(
            propagated_intensity_2d=np.asarray(out_asm.intensity()[0]),
            wavelength_um=wavelength_um,
            focal_um=distance_um,
            aperture_diameter_um=aperture_diameter_um,
            dx_um=grid.dx_um,
        )
        mae_rs = _mae_vs_airy(
            propagated_intensity_2d=np.asarray(out_rs.intensity()[0]),
            wavelength_um=wavelength_um,
            focal_um=distance_um,
            aperture_diameter_um=aperture_diameter_um,
            dx_um=grid.dx_um,
        )

        planned = plan_propagation(
            mode="auto",
            grid=grid,
            spectrum=spectrum,
            distance_um=distance_um,
            input_domain="spatial",
            nyquist_factor=2.0,
        )
        assert isinstance(planned, ASMPropagator | RSPropagator)

        if expected_method == "asm":
            assert mae_asm < mae_rs
        else:
            assert mae_rs < mae_asm


def test_plan_propagation_uses_kspace_model_for_kspace_input():
    grid = Grid.from_extent(nx=64, ny=64, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field_k = Field.plane_wave(grid=grid, spectrum=spectrum).to_kspace()
    distance_um = 25.0

    planned = plan_propagation(
        mode="auto",
        grid=grid,
        spectrum=spectrum,
        distance_um=distance_um,
        input_domain="kspace",
        refractive_index=1.0,
    )

    assert isinstance(planned, KSpacePropagator)
    out_auto = planned.forward(field_k)
    out_k = KSpacePropagator(refractive_index=1.0, distance_um=distance_um).forward(field_k)

    np.testing.assert_allclose(np.asarray(out_auto.data), np.asarray(out_k.data), atol=1e-6)
    assert out_auto.domain == "kspace"


def test_plan_propagation_rejects_invalid_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        plan_propagation(  # type: ignore[arg-type]
            mode="invalid",
            grid=Grid.from_extent(nx=16, ny=16, dx_um=1.0, dy_um=1.0),
            spectrum=Spectrum.from_scalar(0.532),
            distance_um=10.0,
        )


def test_plan_propagation_mode_dispatch():
    grid = Grid.from_extent(nx=64, ny=64, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)
    distance_um = 25.0

    out_asm = plan_propagation(
        mode="asm",
        grid=grid,
        spectrum=spectrum,
        distance_um=distance_um,
        use_sampling_planner=False,
    ).forward(field)
    ref_asm = ASMPropagator(use_sampling_planner=False, distance_um=distance_um).forward(field)
    np.testing.assert_allclose(np.asarray(out_asm.data), np.asarray(ref_asm.data), atol=1e-6)

    out_rs = plan_propagation(
        mode="rs",
        grid=grid,
        spectrum=spectrum,
        distance_um=distance_um,
        use_sampling_planner=False,
    ).forward(field)
    ref_rs = RSPropagator(use_sampling_planner=False, distance_um=distance_um).forward(field)
    np.testing.assert_allclose(np.asarray(out_rs.data), np.asarray(ref_rs.data), atol=1e-6)

    field_k = FourierTransform().forward(field)
    out_k = plan_propagation(
        mode="kspace",
        grid=grid,
        spectrum=spectrum,
        distance_um=distance_um,
    ).forward(field_k)
    ref_k = KSpacePropagator(distance_um=distance_um).forward(field_k)
    np.testing.assert_allclose(np.asarray(out_k.data), np.asarray(ref_k.data), atol=1e-6)


def test_plan_propagation_auto_selects_rs_for_far_field():
    grid = Grid.from_extent(nx=64, ny=64, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    z_crit = critical_distance_um(grid=grid, spectrum=spectrum)
    p = plan_propagation(
        mode="auto",
        distance_um=2.0 * z_crit,
        grid=grid,
        spectrum=spectrum,
    )
    assert isinstance(p, RSPropagator)
