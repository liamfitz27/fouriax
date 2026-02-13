import numpy as np
from scipy.special import j1

from fouriax.optics import Field, Grid, RSPropagator, Spectrum, ThinLensLayer


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


def test_thin_lens_focus_matches_airy_benchmark():
    wavelength_um = 0.532
    focal_um = 20_000.0
    aperture_diameter_um = 400.0

    grid = Grid.from_extent(nx=256, ny=256, dx_um=2.0, dy_um=2.0)
    spectrum = Spectrum.from_scalar(wavelength_um)
    field_in = Field.plane_wave(grid=grid, spectrum=spectrum)

    lens = ThinLensLayer(
        focal_length_um=focal_um,
        aperture_diameter_um=aperture_diameter_um,
    )
    propagator = RSPropagator()

    field_after_lens = lens.forward(field_in)
    field_focus = propagator.propagate(field_after_lens, distance_um=focal_um)

    intensity = np.asarray(field_focus.intensity()[0])
    center_y = grid.ny // 2
    center_x = grid.nx // 2
    row = intensity[center_y, :]
    row = row / np.max(row)

    x_um = (np.arange(grid.nx) - (grid.nx - 1) / 2.0) * grid.dx_um
    r_um = np.abs(x_um)

    expected_first_zero_um = 1.22 * wavelength_um * focal_um / aperture_diameter_um

    search_lo = int(max(1, np.floor(0.7 * expected_first_zero_um / grid.dx_um)))
    search_hi = int(np.ceil(1.3 * expected_first_zero_um / grid.dx_um))
    window = row[center_x + search_lo : center_x + search_hi]
    min_idx_local = int(np.argmin(window))
    first_zero_px = search_lo + min_idx_local
    measured_first_zero_um = first_zero_px * grid.dx_um

    rel_err_first_zero = (
        abs(measured_first_zero_um - expected_first_zero_um) / expected_first_zero_um
    )
    assert rel_err_first_zero < 0.20

    r_max = 0.9 * expected_first_zero_um
    n_compare = int(np.floor(r_max / grid.dx_um))
    sim_profile = row[center_x : center_x + n_compare]
    r_compare = r_um[center_x : center_x + n_compare]
    airy = _airy_profile(
        r_compare,
        wavelength_um=wavelength_um,
        focal_um=focal_um,
        diameter_um=aperture_diameter_um,
    )

    mean_abs_err = np.mean(np.abs(sim_profile - airy))
    assert mean_abs_err < 0.15
