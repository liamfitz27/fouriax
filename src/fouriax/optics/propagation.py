from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace

import jax.numpy as jnp
from jax.scipy import ndimage as jndimage

from fouriax.core.fft import fftconvolve
from fouriax.optics.interfaces import PropagationModel
from fouriax.optics.model import Field, Grid, Spectrum


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def critical_distance_um(grid: Grid, spectrum: Spectrum) -> float:
    """
    Compute conservative critical distance separating ASM and RS regimes.

    z_crit = N_eff * (dx_eff^2) / lambda_min
    where N_eff=min(nx, ny), dx_eff=max(dx_um, dy_um), lambda_min=min(wavelengths_um)
    """
    grid.validate()
    spectrum.validate()
    n_eff = min(grid.nx, grid.ny)
    dx_eff = max(grid.dx_um, grid.dy_um)
    lambda_min_um = float(jnp.min(spectrum.wavelengths_um))
    return float(n_eff * (dx_eff**2) / lambda_min_um)


def select_propagator_method(
    grid: Grid,
    spectrum: Spectrum,
    distance_um: float,
    equality_tolerance: float = 1e-6,
) -> str:
    """
    Select recommended method by critical-distance regime.

    - distance < z_crit * (1 - tol) -> ASM
    - otherwise -> RS
    """
    if distance_um <= 0:
        raise ValueError("distance_um must be strictly positive")
    if equality_tolerance < 0:
        raise ValueError("equality_tolerance must be >= 0")
    z_crit = critical_distance_um(grid=grid, spectrum=spectrum)
    if distance_um < z_crit * (1.0 - equality_tolerance):
        return "asm"
    return "rs"


def recommend_nyquist_grid(
    grid: Grid,
    spectrum: Spectrum,
    nyquist_factor: float = 2.0,
    min_padding_factor: float = 2.0,
) -> Grid:
    """
    Build a denser padded propagation grid from nyquist_factor and padding.
    """
    grid.validate()
    spectrum.validate()
    if nyquist_factor <= 0:
        raise ValueError("nyquist_factor must be strictly positive")
    if min_padding_factor < 1.0:
        raise ValueError("min_padding_factor must be >= 1.0")

    min_wavelength_um = float(jnp.min(spectrum.wavelengths_um))
    nyquist_limited_dx = min_wavelength_um / (2.0 * nyquist_factor)

    upsample_x = max(1, int(jnp.ceil(grid.dx_um / nyquist_limited_dx)))
    upsample_y = max(1, int(jnp.ceil(grid.dy_um / nyquist_limited_dx)))

    dx_um = grid.dx_um / upsample_x
    dy_um = grid.dy_um / upsample_y

    padded_nx = int(jnp.ceil(grid.nx * min_padding_factor)) * upsample_x
    padded_ny = int(jnp.ceil(grid.ny * min_padding_factor)) * upsample_y
    nx = _next_power_of_two(padded_nx)
    ny = _next_power_of_two(padded_ny)
    return Grid.from_extent(nx=nx, ny=ny, dx_um=dx_um, dy_um=dy_um)


def _resample_2d_to_grid(
    data_2d: jnp.ndarray,
    src_grid: Grid,
    dst_grid: Grid,
) -> jnp.ndarray:
    """
    Resample a 2D complex field from src_grid coordinates to dst_grid coordinates.

    Uses bilinear interpolation on real/imag components independently.
    Out-of-bounds points are zero-filled.
    """
    x_dst, y_dst = dst_grid.spatial_grid()
    src_x = x_dst / src_grid.dx_um + (src_grid.nx - 1) / 2.0
    src_y = y_dst / src_grid.dy_um + (src_grid.ny - 1) / 2.0
    coords = [src_y, src_x]

    real = jndimage.map_coordinates(
        data_2d.real,
        coords,
        order=1,
        mode="constant",
        cval=0.0,
    )
    imag = jndimage.map_coordinates(
        data_2d.imag,
        coords,
        order=1,
        mode="constant",
        cval=0.0,
    )
    return jnp.asarray(real + 1j * imag, dtype=data_2d.dtype)


def _prepare_field_with_grid(field: Field, target_grid: Grid | None) -> Field:
    """
    Apply Nyquist-driven resampling and padding for propagation stability.

    The resulting propagation grid may have both different shape and different
    spacing (dx_um, dy_um) compared to the input field grid.
    """
    if target_grid is None:
        return field
    if (
        target_grid.nx == field.grid.nx
        and target_grid.ny == field.grid.ny
        and target_grid.dx_um == field.grid.dx_um
        and target_grid.dy_um == field.grid.dy_um
    ):
        return field

    data = jnp.stack(
        [
            _resample_2d_to_grid(field.data[i], src_grid=field.grid, dst_grid=target_grid)
            for i in range(field.spectrum.size)
        ],
        axis=0,
    )
    return Field(data=data, grid=target_grid, spectrum=field.spectrum)


def _restore_to_original_grid(field: Field, original_grid: Grid) -> Field:
    if (
        field.grid.nx == original_grid.nx
        and field.grid.ny == original_grid.ny
        and field.grid.dx_um == original_grid.dx_um
        and field.grid.dy_um == original_grid.dy_um
    ):
        return field

    data = jnp.stack(
        [
            _resample_2d_to_grid(field.data[i], src_grid=field.grid, dst_grid=original_grid)
            for i in range(field.spectrum.size)
        ],
        axis=0,
    )
    return Field(data=data, grid=original_grid, spectrum=field.spectrum)


@dataclass(frozen=True)
class RSPropagator(PropagationModel):
    """
    Rayleigh-Sommerfeld propagator based on convolution with the RS impulse response.

    All length quantities use micrometers (um).
    """

    use_sampling_planner: bool = True
    nyquist_factor: float = 2.0
    min_padding_factor: float = 2.0
    precomputed_grid: Grid | None = None
    warn_on_regime_mismatch: bool = True
    equality_tolerance: float = 1e-6

    def impulse_response(
        self,
        field: Field,
        wavelength_um: float,
        distance_um: float,
    ) -> jnp.ndarray:
        if distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")

        x, y = field.grid.spatial_grid()
        z = jnp.asarray(distance_um, dtype=jnp.float32)
        wl = jnp.asarray(wavelength_um, dtype=jnp.float32)
        k = 2.0 * jnp.pi / wl

        r = jnp.sqrt(x * x + y * y + z * z)
        h = (z / (1j * wl * (r * r))) * jnp.exp(1j * k * r)
        return h.astype(jnp.complex64)

    def propagate(self, field: Field, distance_um: float) -> Field:
        self.validate_for(field)
        if distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")

        if self.warn_on_regime_mismatch:
            method = select_propagator_method(
                grid=field.grid,
                spectrum=field.spectrum,
                distance_um=distance_um,
                equality_tolerance=self.equality_tolerance,
            )
            if method != "rs":
                z_crit = critical_distance_um(grid=field.grid, spectrum=field.spectrum)
                warnings.warn(
                    "RS selected outside recommended regime "
                    f"(distance_um={distance_um:.3f}, z_crit_um={z_crit:.3f}, "
                    "recommended='asm'). Continuing with RS as requested.",
                    UserWarning,
                    stacklevel=2,
                )

        original_grid = field.grid
        target_grid = self.precomputed_grid
        if target_grid is None and self.use_sampling_planner:
            target_grid = recommend_nyquist_grid(
                grid=field.grid,
                spectrum=field.spectrum,
                nyquist_factor=self.nyquist_factor,
                min_padding_factor=self.min_padding_factor,
            )
        work_field = _prepare_field_with_grid(field, target_grid)

        area_um2 = work_field.grid.dx_um * work_field.grid.dy_um
        outputs = []
        for i, wavelength_um in enumerate(work_field.spectrum.wavelengths_um):
            kernel = self.impulse_response(work_field, wavelength_um, distance_um)
            propagated = fftconvolve(
                work_field.data[i],
                kernel,
                mode="same",
                axes=(-2, -1),
            )
            outputs.append(propagated * area_um2)

        data = jnp.stack(outputs, axis=0)
        propagated_field = Field(data=data, grid=work_field.grid, spectrum=work_field.spectrum)
        return _restore_to_original_grid(propagated_field, original_grid)


@dataclass(frozen=True)
class ASMPropagator(PropagationModel):
    """
    Angular Spectrum Method (ASM) propagator.

    All length quantities use micrometers (um).
    """

    use_sampling_planner: bool = True
    nyquist_factor: float = 2.0
    min_padding_factor: float = 2.0
    precomputed_grid: Grid | None = None
    warn_on_regime_mismatch: bool = True
    equality_tolerance: float = 1e-6

    def transfer_function(
        self,
        field: Field,
        wavelength_um: float,
        distance_um: float,
    ) -> jnp.ndarray:
        if distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")

        fx, fy = field.grid.frequency_grid()
        wl = jnp.asarray(wavelength_um, dtype=jnp.float32)
        z = jnp.asarray(distance_um, dtype=jnp.float32)
        k = 2.0 * jnp.pi / wl

        argument = 1.0 - (wl * fx) ** 2 - (wl * fy) ** 2
        kz = k * jnp.sqrt(argument.astype(jnp.complex64))
        return jnp.exp(1j * kz * z).astype(jnp.complex64)

    def propagate(self, field: Field, distance_um: float) -> Field:
        self.validate_for(field)
        if distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")

        if self.warn_on_regime_mismatch:
            method = select_propagator_method(
                grid=field.grid,
                spectrum=field.spectrum,
                distance_um=distance_um,
                equality_tolerance=self.equality_tolerance,
            )
            if method != "asm":
                z_crit = critical_distance_um(grid=field.grid, spectrum=field.spectrum)
                warnings.warn(
                    "ASM selected outside recommended regime "
                    f"(distance_um={distance_um:.3f}, z_crit_um={z_crit:.3f}, "
                    "recommended='rs'). Continuing with ASM as requested.",
                    UserWarning,
                    stacklevel=2,
                )

        original_grid = field.grid
        target_grid = self.precomputed_grid
        if target_grid is None and self.use_sampling_planner:
            target_grid = recommend_nyquist_grid(
                grid=field.grid,
                spectrum=field.spectrum,
                nyquist_factor=self.nyquist_factor,
                min_padding_factor=self.min_padding_factor,
            )
        work_field = _prepare_field_with_grid(field, target_grid)

        outputs = []
        for i, wavelength_um in enumerate(work_field.spectrum.wavelengths_um):
            transfer = self.transfer_function(work_field, wavelength_um, distance_um)
            spectrum = jnp.fft.fftn(work_field.data[i], axes=(-2, -1))
            propagated = jnp.fft.ifftn(spectrum * transfer, axes=(-2, -1))
            outputs.append(propagated)

        data = jnp.stack(outputs, axis=0)
        propagated_field = Field(data=data, grid=work_field.grid, spectrum=work_field.spectrum)
        return _restore_to_original_grid(propagated_field, original_grid)


@dataclass(frozen=True)
class AutoPropagator(PropagationModel):
    """
    Propagator wrapper that auto-selects ASM or RS.
    """

    asm: ASMPropagator = field(default_factory=ASMPropagator)
    rs: RSPropagator = field(default_factory=RSPropagator)
    equality_tolerance: float = 1e-6
    nyquist_factor: float = 2.0
    min_padding_factor: float = 2.0
    setup_grid: Grid | None = None
    setup_spectrum: Spectrum | None = None
    setup_distance_um: float | None = None
    precomputed_grid: Grid | None = None
    precomputed_method: str | None = None

    def __post_init__(self) -> None:
        if self.precomputed_grid is not None:
            asm = self.asm
            rs = self.rs
            if asm.precomputed_grid is None or asm.use_sampling_planner:
                asm = replace(
                    asm,
                    use_sampling_planner=False,
                    precomputed_grid=self.precomputed_grid,
                    warn_on_regime_mismatch=False,
                )
            if rs.precomputed_grid is None or rs.use_sampling_planner:
                rs = replace(
                    rs,
                    use_sampling_planner=False,
                    precomputed_grid=self.precomputed_grid,
                    warn_on_regime_mismatch=False,
                )
            object.__setattr__(self, "asm", asm)
            object.__setattr__(self, "rs", rs)

        if (
            self.setup_grid is not None
            and self.setup_spectrum is not None
            and self.setup_distance_um is not None
            and (self.precomputed_grid is None or self.precomputed_method is None)
        ):
            if self.setup_distance_um <= 0:
                raise ValueError("setup_distance_um must be strictly positive")
            self.setup_grid.validate()
            self.setup_spectrum.validate()

            precomputed_grid = self.precomputed_grid
            if precomputed_grid is None and (
                self.rs.use_sampling_planner or self.asm.use_sampling_planner
            ):
                precomputed_grid = recommend_nyquist_grid(
                    grid=self.setup_grid,
                    spectrum=self.setup_spectrum,
                    nyquist_factor=self.nyquist_factor,
                    min_padding_factor=self.min_padding_factor,
                )

            method = self.precomputed_method
            if method is None:
                method = select_propagator_method(
                    grid=self.setup_grid,
                    spectrum=self.setup_spectrum,
                    distance_um=self.setup_distance_um,
                    equality_tolerance=self.equality_tolerance,
                )
            object.__setattr__(self, "precomputed_grid", precomputed_grid)
            object.__setattr__(self, "precomputed_method", method)

            asm = self.asm
            rs = self.rs
            if precomputed_grid is not None:
                if asm.precomputed_grid is None or asm.use_sampling_planner:
                    asm = replace(
                        asm,
                        use_sampling_planner=False,
                        precomputed_grid=precomputed_grid,
                        warn_on_regime_mismatch=False,
                    )
                if rs.precomputed_grid is None or rs.use_sampling_planner:
                    rs = replace(
                        rs,
                        use_sampling_planner=False,
                        precomputed_grid=precomputed_grid,
                        warn_on_regime_mismatch=False,
                    )
            object.__setattr__(self, "asm", asm)
            object.__setattr__(self, "rs", rs)

    def propagate(self, field: Field, distance_um: float) -> Field:
        self.validate_for(field)
        if distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")

        method = self.precomputed_method
        if method is None:
            method = select_propagator_method(
                grid=field.grid,
                spectrum=field.spectrum,
                distance_um=distance_um,
                equality_tolerance=self.equality_tolerance,
            )

        if method == "asm":
            return self.asm.propagate(field, distance_um=distance_um)
        return self.rs.propagate(field, distance_um=distance_um)
