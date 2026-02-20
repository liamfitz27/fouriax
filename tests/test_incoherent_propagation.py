import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fouriax.optics import (
    Field,
    Grid,
    IncoherentImager,
    PhaseMask,
    RSPropagator,
    Spectrum,
    ThinLens,
)


def _test_grid_and_spectrum() -> tuple[Grid, Spectrum]:
    grid = Grid.from_extent(nx=21, ny=21, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_array(jnp.array([0.532, 0.633], dtype=jnp.float32))
    return grid, spectrum


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
    expected = rs.propagate(lens.forward(impulse), distance_um=60.0).to_spatial().intensity()
    np.testing.assert_allclose(np.asarray(psf), np.asarray(expected), atol=1e-6)


def test_psf_imager_build_psf_matches_manual_plane_wave_path():
    grid, spectrum = _test_grid_and_spectrum()
    lens = ThinLens(focal_length_um=70.0, aperture_diameter_um=16.0)
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False, na_limit=0.07)
    imager = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=70.0,
        psf_source="plane_wave_focus",
        normalize_psf=False,
    )

    field_ref = Field.plane_wave(grid=grid, spectrum=spectrum)
    psf = imager.build_psf(field_ref)
    expected = rs.propagate(lens.forward(field_ref), distance_um=70.0).to_spatial().intensity()
    np.testing.assert_allclose(np.asarray(psf), np.asarray(expected), atol=1e-6)


def test_psf_imager_psf_and_otf_modes_are_consistent():
    grid, spectrum = _test_grid_and_spectrum()
    lens = ThinLens(focal_length_um=65.0, aperture_diameter_um=15.0)
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False, na_limit=0.08)

    field = Field.plane_wave(grid=grid, spectrum=spectrum).apply_phase(0.3)
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

    out_psf = psf_mode.forward(field).intensity()
    out_otf = otf_mode.forward(field).intensity()
    np.testing.assert_allclose(np.asarray(out_psf), np.asarray(out_otf), atol=3e-5, rtol=1e-5)


def test_psf_imager_supports_gradients_through_optical_layer():
    grid, spectrum = _test_grid_and_spectrum()
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False, na_limit=0.08)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)
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
        out = imager.forward(field).intensity()
        return jnp.mean((out[0] - target) ** 2)

    grad = jax.grad(loss_fn)(jnp.asarray(0.02, dtype=jnp.float32))
    assert bool(jnp.isfinite(grad))


def test_psf_imager_auto_mode_matches_otf_on_large_grid():
    grid = Grid.from_extent(nx=96, ny=96, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_scalar(0.532)
    lens = ThinLens(focal_length_um=70.0, aperture_diameter_um=16.0)
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False, na_limit=0.07)
    field = Field.plane_wave(grid=grid, spectrum=spectrum).apply_phase(0.15)

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

    out_auto = auto_mode.forward(field).intensity()
    out_otf = otf_mode.forward(field).intensity()
    np.testing.assert_allclose(np.asarray(out_auto), np.asarray(out_otf), atol=3e-5, rtol=1e-5)


def test_psf_imager_normalization_distance_method():
    grid, spectrum = _test_grid_and_spectrum()
    lens = ThinLens(focal_length_um=50.0, aperture_diameter_um=12.0)
    rs = RSPropagator(use_sampling_planner=False, warn_on_regime_mismatch=False)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    near_um = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=50.0,
        normalization_reference="near_1um",
    )
    assert near_um.normalization_distance_um(field) == 1.0

    near_wl = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=50.0,
        normalization_reference="near_wavelength",
    )
    assert np.isclose(
        near_wl.normalization_distance_um(field),
        float(jnp.min(spectrum.wavelengths_um)),
    )

    at_dist = IncoherentImager(
        optical_layer=lens,
        propagator=rs,
        distance_um=50.0,
        normalization_reference="at_imaging_distance",
    )
    assert at_dist.normalization_distance_um(field) == 50.0


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
        np.asarray(jnp.sum(psf_dist, axis=(-2, -1))),
        np.ones((1,)),
        atol=1e-5,
    )
    # Near-reference normalization is intentionally different when far-field spills beyond grid.
    assert float(jnp.sum(psf_near)) != pytest.approx(1.0, abs=1e-3)
