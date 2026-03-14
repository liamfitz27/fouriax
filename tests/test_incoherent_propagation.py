import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fouriax.optics import (
    Field,
    Grid,
    IncoherentImager,
    Intensity,
    PhaseMask,
    RSPropagator,
    Spectrum,
    ThinLens,
)


def _test_grid_and_spectrum() -> tuple[Grid, Spectrum]:
    grid = Grid.from_extent(nx=21, ny=21, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_array(jnp.array([0.532, 0.633], dtype=jnp.float32))
    return grid, spectrum


def _point_source_field(
    grid: Grid,
    spectrum: Spectrum,
    *,
    distance_um: float,
) -> Field:
    x, y = grid.spatial_grid()
    r = jnp.sqrt(x * x + y * y + distance_um**2)
    data = jnp.stack(
        [
            jnp.exp(1j * ((2.0 * jnp.pi) / wavelength_um) * r) / r
            for wavelength_um in spectrum.wavelengths_um
        ],
        axis=0,
    ).astype(jnp.complex64)
    return Field(data=data, grid=grid, spectrum=spectrum)


def test_psf_imager_build_psf_matches_manual_impulse_path():
    grid, spectrum = _test_grid_and_spectrum()
    lens = ThinLens(focal_length_um=60.0, aperture_diameter_um=14.0)
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False, na_limit=0.08)
    imager = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=60.0,
        psf_source="impulse",
        normalize_psf=False,
    )

    field_ref = Field.plane_wave(grid=grid, spectrum=spectrum)
    psf = imager.build_psf(field_ref)

    impulse = Field.zeros(grid=grid, spectrum=spectrum)
    cy = grid.ny // 2
    cx = grid.nx // 2
    impulse = Field(
        data=impulse.data.at[:, cy, cx].set(1.0 + 0.0j),
        grid=grid,
        spectrum=spectrum,
    )
    expected = (
        RSPropagator(
            use_sampling_planner=False,
            warn_on_regime_mismatch=False,
            na_limit=0.08,
            distance_um=60.0,
        )
        .forward(lens.forward(impulse))
        .to_spatial()
        .intensity()
    )
    np.testing.assert_allclose(np.asarray(psf.data), np.asarray(expected), atol=1e-6)


def test_psf_imager_build_psf_matches_manual_plane_wave_path():
    grid, spectrum = _test_grid_and_spectrum()
    lens = ThinLens(focal_length_um=70.0, aperture_diameter_um=16.0)
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False, na_limit=0.07)
    imager = IncoherentImager.for_far_field(
        optical_layer=lens,
        propagator=rs,
        image_distance_um=70.0,
        normalize_psf=False,
    )

    field_ref = Field.plane_wave(grid=grid, spectrum=spectrum)
    psf = imager.build_psf(field_ref)
    expected = (
        RSPropagator(
            use_sampling_planner=False,
            warn_on_regime_mismatch=False,
            na_limit=0.07,
            distance_um=70.0,
        )
        .forward(lens.forward(field_ref))
        .to_spatial()
        .intensity()
    )
    np.testing.assert_allclose(np.asarray(psf.data), np.asarray(expected), atol=1e-6)


def test_psf_imager_build_psf_matches_manual_finite_distance_point_source_path():
    grid, spectrum = _test_grid_and_spectrum()
    object_distance_um = 120.0
    image_distance_um = 60.0
    lens = ThinLens(focal_length_um=40.0, aperture_diameter_um=16.0)
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False, na_limit=0.07)
    imager = IncoherentImager.for_finite_distance(
        optical_layer=lens,
        propagator=rs,
        object_distance_um=object_distance_um,
        image_distance_um=image_distance_um,
        normalize_psf=False,
    )

    field_ref = Field.plane_wave(grid=grid, spectrum=spectrum)
    psf = imager.build_psf(field_ref)
    expected = (
        RSPropagator(
            use_sampling_planner=False,
            warn_on_regime_mismatch=False,
            na_limit=0.07,
            distance_um=image_distance_um,
        )
        .forward(lens.forward(_point_source_field(grid, spectrum, distance_um=object_distance_um)))
        .to_spatial()
        .intensity()
    )
    np.testing.assert_allclose(np.asarray(psf.data), np.asarray(expected), atol=1e-6)


def test_psf_imager_infer_from_paraxial_limit_far_field_preserves_sensor_pitch():
    grid, spectrum = _test_grid_and_spectrum()
    lens = ThinLens(focal_length_um=70.0, aperture_diameter_um=16.0)
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False, na_limit=0.07)
    imager = IncoherentImager.for_far_field(
        optical_layer=lens,
        propagator=rs,
        image_distance_um=70.0,
    )
    sensor_grid = Grid.from_extent(nx=15, ny=11, dx_um=0.7, dy_um=0.8)

    inferred = imager.infer_from_paraxial_limit(sensor_grid, paraxial_max_angle_rad=0.1)

    assert inferred == sensor_grid


def test_psf_imager_infer_from_paraxial_limit_finite_distance_scales_input_pitch():
    grid, spectrum = _test_grid_and_spectrum()
    lens = ThinLens(focal_length_um=40.0, aperture_diameter_um=16.0)
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False, na_limit=0.07)
    imager = IncoherentImager.for_finite_distance(
        optical_layer=lens,
        propagator=rs,
        object_distance_um=120.0,
        image_distance_um=60.0,
    )
    sensor_grid = Grid.from_extent(nx=15, ny=11, dx_um=1.0, dy_um=1.4)

    inferred = imager.infer_from_paraxial_limit(sensor_grid, paraxial_max_angle_rad=0.15)

    assert inferred.nx == sensor_grid.nx
    assert inferred.ny == sensor_grid.ny
    assert inferred.dx_um == pytest.approx(2.0)
    assert inferred.dy_um == pytest.approx(2.8)


def test_psf_imager_infer_from_paraxial_limit_rejects_sensor_outside_limit():
    grid, spectrum = _test_grid_and_spectrum()
    lens = ThinLens(focal_length_um=70.0, aperture_diameter_um=16.0)
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False, na_limit=0.07)
    imager = IncoherentImager.for_far_field(
        optical_layer=lens,
        propagator=rs,
        image_distance_um=70.0,
    )
    sensor_grid = Grid.from_extent(nx=21, ny=21, dx_um=1.0, dy_um=1.0)

    with pytest.raises(ValueError, match="sensor_grid exceeds the paraxial field of view"):
        imager.infer_from_paraxial_limit(sensor_grid, paraxial_max_angle_rad=0.1)


def test_psf_imager_infer_from_paraxial_limit_rejects_impulse_mode():
    grid, spectrum = _test_grid_and_spectrum()
    lens = ThinLens(focal_length_um=60.0, aperture_diameter_um=14.0)
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False, na_limit=0.08)
    imager = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=60.0,
        psf_source="impulse",
    )
    sensor_grid = Grid.from_extent(nx=9, ny=9, dx_um=1.0, dy_um=1.0)

    with pytest.raises(ValueError, match="infer_from_paraxial_limit is only supported"):
        imager.infer_from_paraxial_limit(sensor_grid, paraxial_max_angle_rad=0.2)


def test_psf_imager_psf_and_otf_modes_are_consistent():
    grid, spectrum = _test_grid_and_spectrum()
    lens = ThinLens(focal_length_um=65.0, aperture_diameter_um=15.0)
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False, na_limit=0.08)

    field = Field.plane_wave(grid=grid, spectrum=spectrum).apply_phase(0.3)
    intensity = field.to_intensity()
    psf_mode = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=65.0,
        psf_source="impulse",
        normalize_psf=True,
        mode="psf",
    )
    otf_mode = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=65.0,
        psf_source="impulse",
        normalize_psf=True,
        mode="otf",
    )

    out_psf = psf_mode.forward(intensity).data
    out_otf = otf_mode.forward(intensity).data
    np.testing.assert_allclose(np.asarray(out_psf), np.asarray(out_otf), atol=3e-5, rtol=1e-5)


def test_psf_imager_supports_gradients_through_optical_layer():
    grid, spectrum = _test_grid_and_spectrum()
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False, na_limit=0.08)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)
    intensity = field.to_intensity()
    x, y = grid.spatial_grid()
    target = jnp.exp(-((x**2 + y**2) / (2.0 * 3.5**2)))

    def loss_fn(scale: jnp.ndarray) -> jnp.ndarray:
        layer = PhaseMask(phase_map_rad=scale * (x + y))
        imager = IncoherentImager(
            optical_layer=layer,
            propagator=rs,
            distance_um=40.0,
            psf_source="impulse",
            normalize_psf=True,
            mode="otf",
        )
        out = imager.forward(intensity).data
        return jnp.mean((out[0] - target) ** 2)

    grad = jax.grad(loss_fn)(jnp.asarray(0.02, dtype=jnp.float32))
    assert bool(jnp.isfinite(grad))


def test_psf_imager_linear_operator_matches_forward():
    grid, spectrum = _test_grid_and_spectrum()
    lens = ThinLens(focal_length_um=65.0, aperture_diameter_um=15.0)
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False, na_limit=0.08)
    imager = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=65.0,
        psf_source="impulse",
        normalize_psf=True,
        mode="otf",
    )

    data = jnp.arange(
        spectrum.size * grid.ny * grid.nx,
        dtype=jnp.float32,
    ).reshape((spectrum.size, grid.ny, grid.nx))
    intensity = Intensity(data=data, grid=grid, spectrum=spectrum)

    op_psf = imager.linear_operator(intensity, cache="psf", flatten=False)
    op_otf = imager.linear_operator(intensity, cache="otf", flatten=False)

    out_forward = imager.forward(intensity).data
    out_psf = op_psf.matvec(data)
    out_otf = op_otf.matvec(data)

    np.testing.assert_allclose(np.asarray(out_psf), np.asarray(out_forward), atol=3e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(out_otf), np.asarray(out_forward), atol=3e-5, rtol=1e-5)


def test_psf_imager_linear_operator_adjoint_and_flatten():
    grid, spectrum = _test_grid_and_spectrum()
    lens = ThinLens(focal_length_um=65.0, aperture_diameter_um=15.0)
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False, na_limit=0.08)
    imager = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=65.0,
        psf_source="impulse",
        normalize_psf=True,
        mode="otf",
    )
    template = Field.plane_wave(grid=grid, spectrum=spectrum).to_intensity()

    op = imager.linear_operator(template, cache="otf", flatten=False)
    x = jnp.linspace(
        0.0,
        1.0,
        spectrum.size * grid.ny * grid.nx,
        dtype=jnp.float32,
    ).reshape((spectrum.size, grid.ny, grid.nx))
    y = jnp.linspace(
        1.0,
        2.0,
        spectrum.size * grid.ny * grid.nx,
        dtype=jnp.float32,
    ).reshape((spectrum.size, grid.ny, grid.nx))

    lhs = jnp.vdot(op.matvec(x), y)
    rhs = jnp.vdot(x, op.rmatvec(y))
    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), atol=3e-5, rtol=1e-5)

    op_flat = imager.linear_operator(template, cache="otf", flatten=True)
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    out_flat = op_flat.matvec(x_flat)
    adj_flat = op_flat.rmatvec(y_flat)

    assert op_flat.in_shape == (spectrum.size * grid.ny * grid.nx,)
    assert op_flat.out_shape == (spectrum.size * grid.ny * grid.nx,)
    np.testing.assert_allclose(
        np.asarray(out_flat.reshape((spectrum.size, grid.ny, grid.nx))),
        np.asarray(op.matvec(x)),
        atol=3e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(adj_flat.reshape((spectrum.size, grid.ny, grid.nx))),
        np.asarray(op.rmatvec(y)),
        atol=3e-5,
        rtol=1e-5,
    )


def test_psf_imager_linear_operator_requires_unbatched_template():
    grid, spectrum = _test_grid_and_spectrum()
    lens = ThinLens(focal_length_um=65.0, aperture_diameter_um=15.0)
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False, na_limit=0.08)
    imager = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=65.0,
        psf_source="impulse",
        normalize_psf=True,
        mode="otf",
    )

    base = Field.plane_wave(grid=grid, spectrum=spectrum).to_intensity()
    batched = Intensity(
        data=base.data[None, ...],
        grid=base.grid,
        spectrum=base.spectrum,
    )
    with pytest.raises(ValueError, match="unbatched template intensity"):
        imager.linear_operator(batched)


def test_psf_imager_auto_mode_matches_otf_on_large_grid():
    grid = Grid.from_extent(nx=96, ny=96, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    lens = ThinLens(focal_length_um=70.0, aperture_diameter_um=16.0)
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False, na_limit=0.07)
    field = Field.plane_wave(grid=grid, spectrum=spectrum).apply_phase(0.15)
    intensity = field.to_intensity()

    auto_mode = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=70.0,
        psf_source="impulse",
        normalize_psf=True,
        mode="auto",
    )
    otf_mode = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=70.0,
        psf_source="impulse",
        normalize_psf=True,
        mode="otf",
    )

    out_auto = auto_mode.forward(intensity).data
    out_otf = otf_mode.forward(intensity).data
    np.testing.assert_allclose(np.asarray(out_auto), np.asarray(out_otf), atol=3e-5, rtol=1e-5)


def test_psf_imager_normalization_distance_method():
    grid, spectrum = _test_grid_and_spectrum()
    lens = ThinLens(focal_length_um=50.0, aperture_diameter_um=12.0)
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)
    intensity = field.to_intensity()

    near_um = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=50.0,
        normalization_reference="near_1um",
    )
    assert near_um.normalization_distance_um(intensity) == 1.0

    near_wl = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=50.0,
        normalization_reference="near_wavelength",
    )
    assert np.isclose(
        near_wl.normalization_distance_um(intensity),
        float(jnp.min(spectrum.wavelengths_um)),
    )

    at_dist = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=50.0,
        normalization_reference="at_imaging_distance",
    )
    assert at_dist.normalization_distance_um(intensity) == 50.0


def test_psf_imager_near_reference_changes_psf_gain_vs_imaging_distance_reference():
    grid = Grid.from_extent(nx=31, ny=31, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    lens = ThinLens(focal_length_um=120.0, aperture_diameter_um=16.0)
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False, na_limit=0.06)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    near_ref = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=120.0,
        normalize_psf=True,
        normalization_reference="near_1um",
    )
    at_dist_ref = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=120.0,
        normalize_psf=True,
        normalization_reference="at_imaging_distance",
    )

    psf_near = near_ref.build_psf(field)
    psf_dist = at_dist_ref.build_psf(field)
    # At-imaging-distance normalization enforces unit-sum PSF on-grid.
    np.testing.assert_allclose(
        np.asarray(jnp.sum(psf_dist.data, axis=(-2, -1))),
        np.ones((1,)),
        atol=1e-5,
    )
    # Near-reference normalization is intentionally different when far-field spills beyond grid.
    assert float(jnp.sum(psf_near.data)) != pytest.approx(1.0, abs=1e-3)
