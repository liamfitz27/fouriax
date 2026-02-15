import numpy as np
from scipy.special import j1

from fouriax.optics import (
    ASMPropagator,
    Field,
    Grid,
    PropagationPolicy,
    RSPropagator,
    SamplingPlan,
    SamplingPlanner,
    Spectrum,
    ThinLensLayer,
)


def test_critical_distance_uses_expected_formula():
    grid = Grid.from_extent(nx=256, ny=128, dx_um=1.0, dy_um=0.5)
    spectrum = Spectrum.from_array([0.532, 0.633])
    policy = PropagationPolicy()

    z_crit = policy.critical_distance_um(grid=grid, spectrum=spectrum)
    expected = min(grid.nx, grid.ny) * (max(grid.dx_um, grid.dy_um) ** 2) / 0.532
    assert abs(z_crit - expected) < 1e-5


def test_policy_selects_asm_below_critical_distance():
    grid = Grid.from_extent(nx=256, ny=256, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.5)
    policy = PropagationPolicy()
    z_crit = policy.critical_distance_um(grid=grid, spectrum=spectrum)

    decision = policy.choose(
        grid=grid,
        spectrum=spectrum,
        distance_um=0.5 * z_crit,
    )
    assert decision.method == "asm"


def test_policy_selects_rs_above_critical_distance():
    grid = Grid.from_extent(nx=256, ny=256, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.5)
    policy = PropagationPolicy()
    z_crit = policy.critical_distance_um(grid=grid, spectrum=spectrum)

    decision = policy.choose(
        grid=grid,
        spectrum=spectrum,
        distance_um=2.0 * z_crit,
    )
    assert decision.method == "rs"


def test_policy_selects_rs_at_boundary():
    grid = Grid.from_extent(nx=128, ny=128, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.5)
    policy = PropagationPolicy(equality_tolerance=1e-6)
    z_crit = policy.critical_distance_um(grid=grid, spectrum=spectrum)

    decision = policy.choose(grid=grid, spectrum=spectrum, distance_um=z_crit)
    assert decision.method == "rs"


def test_policy_uses_sampling_plan_risk_override():
    grid = Grid.from_extent(nx=128, ny=128, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.5)
    planner = SamplingPlanner(safety_factor=2.0, min_padding_factor=2.0)
    plan = planner.recommend_grid(mask_grid=grid, spectrum=spectrum)
    forced_risky_plan = SamplingPlan(
        nx=plan.nx,
        ny=plan.ny,
        dx_um=plan.dx_um,
        dy_um=plan.dy_um,
        min_wavelength_um=plan.min_wavelength_um,
        safety_factor=plan.safety_factor,
        sampling_ratio=plan.sampling_ratio,
        is_sampling_safe=False,
        warning="forced for test",
    )
    policy = PropagationPolicy()
    z_crit = policy.critical_distance_um(grid=grid, spectrum=spectrum)

    decision = policy.choose(
        grid=grid,
        spectrum=spectrum,
        distance_um=0.5 * z_crit,
        plan=forced_risky_plan,
    )
    assert decision.method == "rs"


def test_policy_fast_mode_always_selects_asm():
    grid = Grid.from_extent(nx=128, ny=128, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    policy = PropagationPolicy(mode="fast")
    z_crit = policy.critical_distance_um(grid=grid, spectrum=spectrum)

    near = policy.choose(grid=grid, spectrum=spectrum, distance_um=0.5 * z_crit)
    far = policy.choose(grid=grid, spectrum=spectrum, distance_um=2.0 * z_crit)
    assert near.method == "asm"
    assert far.method == "asm"


def test_policy_accurate_mode_uses_strict_regime_selection():
    grid = Grid.from_extent(nx=128, ny=128, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    policy = PropagationPolicy(mode="accurate")
    z_crit = policy.critical_distance_um(grid=grid, spectrum=spectrum)

    near = policy.choose(grid=grid, spectrum=spectrum, distance_um=0.5 * z_crit)
    far = policy.choose(grid=grid, spectrum=spectrum, distance_um=2.0 * z_crit)
    assert near.method == "asm"
    assert far.method == "rs"


def test_policy_balanced_mode_cost_override_prefers_asm_when_rs_too_expensive():
    grid = Grid.from_extent(nx=128, ny=128, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    planner = SamplingPlanner(safety_factor=2.0, min_padding_factor=2.0)
    plan = planner.recommend_grid(mask_grid=grid, spectrum=spectrum)
    policy = PropagationPolicy(mode="balanced", rs_cost_ratio_threshold=2.0)
    z_crit = policy.critical_distance_um(grid=grid, spectrum=spectrum)

    decision = policy.choose(
        grid=grid,
        spectrum=spectrum,
        distance_um=2.0 * z_crit,
        plan=plan,
    )
    assert decision.regime_method == "rs"
    assert decision.method == "asm"


def test_policy_balanced_mode_accuracy_tolerance_override_prefers_asm():
    grid = Grid.from_extent(nx=128, ny=128, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    policy = PropagationPolicy(mode="balanced", rs_cost_ratio_threshold=1e6)
    z_crit = policy.critical_distance_um(grid=grid, spectrum=spectrum)

    decision = policy.choose(
        grid=grid,
        spectrum=spectrum,
        distance_um=2.0 * z_crit,
        mae_tolerance=0.01,
        estimated_mae={"asm": 0.005, "rs": 0.001},
    )
    assert decision.regime_method == "rs"
    assert decision.method == "asm"


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


def test_policy_choice_matches_lower_airy_error_in_near_and_far_regimes():
    wavelength_um = 0.532
    grid = Grid.from_extent(nx=256, ny=256, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(wavelength_um)
    aperture_diameter_um = 30.0
    policy = PropagationPolicy()

    # Use explicit planner instance so both propagators are tested identically.
    planner = SamplingPlanner(safety_factor=2.0, min_padding_factor=2.0)
    asm = ASMPropagator(sampling_planner=planner)
    rs = RSPropagator(sampling_planner=planner)

    z_crit = policy.critical_distance_um(grid=grid, spectrum=spectrum)
    regimes = [
        (0.5 * z_crit, "asm"),
        (2.0 * z_crit, "rs"),
    ]

    field_in = Field.plane_wave(grid=grid, spectrum=spectrum)

    for distance_um, expected_method in regimes:
        lens = ThinLensLayer(
            focal_length_um=distance_um,
            aperture_diameter_um=aperture_diameter_um,
        )
        field_lens = lens.forward(field_in)

        out_asm = asm.propagate(field_lens, distance_um=distance_um)
        out_rs = rs.propagate(field_lens, distance_um=distance_um)

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

        decision = policy.choose(grid=grid, spectrum=spectrum, distance_um=distance_um)
        assert decision.method == expected_method

        if expected_method == "asm":
            assert mae_asm < mae_rs
        else:
            assert mae_rs < mae_asm
