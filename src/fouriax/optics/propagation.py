from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Literal, cast

import jax.numpy as jnp
from jax.scipy import ndimage as jndimage

from fouriax.core.fft import fftconvolve
from fouriax.optics.bandlimit import build_na_mask
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
    return Field(
        data=data,
        grid=target_grid,
        spectrum=field.spectrum,
        domain=field.domain,
        kx_pixel_size_cyc_per_um=field.kx_pixel_size_cyc_per_um,
        ky_pixel_size_cyc_per_um=field.ky_pixel_size_cyc_per_um,
    )


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
    return Field(
        data=data,
        grid=original_grid,
        spectrum=field.spectrum,
        domain=field.domain,
        kx_pixel_size_cyc_per_um=field.kx_pixel_size_cyc_per_um,
        ky_pixel_size_cyc_per_um=field.ky_pixel_size_cyc_per_um,
    )


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
    medium_index: float = 1.0
    na_limit: float | None = None

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
        field = field.to_spatial()
        self.validate_for(field)
        if distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")
        if self.medium_index <= 0:
            raise ValueError("medium_index must be strictly positive")
        if self.na_limit is not None and self.na_limit <= 0:
            raise ValueError("na_limit must be strictly positive when provided")

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
            propagated = propagated * area_um2
            if self.na_limit is not None:
                na_mask = build_na_mask(
                    work_field.grid,
                    wavelength_um=float(wavelength_um),
                    na_limit=float(self.na_limit),
                    medium_index=float(self.medium_index),
                )
                propagated_spectrum = jnp.fft.fftn(propagated, axes=(-2, -1))
                propagated = jnp.fft.ifftn(
                    propagated_spectrum * na_mask.astype(jnp.complex64),
                    axes=(-2, -1),
                )
            outputs.append(propagated)

        data = jnp.stack(outputs, axis=0)
        propagated_field = Field(
            data=data,
            grid=work_field.grid,
            spectrum=work_field.spectrum,
            domain=work_field.domain,
            kx_pixel_size_cyc_per_um=work_field.kx_pixel_size_cyc_per_um,
            ky_pixel_size_cyc_per_um=work_field.ky_pixel_size_cyc_per_um,
        )
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
    medium_index: float = 1.0
    na_limit: float | None = None

    def transfer_function(
        self,
        field: Field,
        wavelength_um: float,
        distance_um: float,
    ) -> jnp.ndarray:
        if distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")
        if self.medium_index <= 0:
            raise ValueError("medium_index must be strictly positive")
        if self.na_limit is not None and self.na_limit <= 0:
            raise ValueError("na_limit must be strictly positive when provided")

        fx, fy = field.grid.frequency_grid()
        wl = jnp.asarray(wavelength_um, dtype=jnp.float32)
        z = jnp.asarray(distance_um, dtype=jnp.float32)
        n = jnp.asarray(self.medium_index, dtype=jnp.float32)
        k = 2.0 * jnp.pi * n / wl

        argument = 1.0 - (wl * fx / n) ** 2 - (wl * fy / n) ** 2
        kz = k * jnp.sqrt(argument.astype(jnp.complex64))
        transfer = jnp.exp(1j * kz * z).astype(jnp.complex64)
        if self.na_limit is None:
            return transfer
        na_mask = build_na_mask(
            field.grid,
            wavelength_um=float(wavelength_um),
            na_limit=float(self.na_limit),
            medium_index=float(self.medium_index),
        )
        return transfer * na_mask.astype(jnp.complex64)

    def propagate(self, field: Field, distance_um: float) -> Field:
        field = field.to_spatial()
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
        propagated_field = Field(
            data=data,
            grid=work_field.grid,
            spectrum=work_field.spectrum,
            domain=work_field.domain,
            kx_pixel_size_cyc_per_um=work_field.kx_pixel_size_cyc_per_um,
            ky_pixel_size_cyc_per_um=work_field.ky_pixel_size_cyc_per_um,
        )
        return _restore_to_original_grid(propagated_field, original_grid)


@dataclass(frozen=True)
class KSpacePropagator(PropagationModel):
    """
    k-space diagonal propagator using angular-spectrum phase advance.

    All length quantities use micrometers (um).
    """

    refractive_index: float = 1.0
    na_limit: float | None = None

    def transfer_function(
        self,
        field: Field,
        wavelength_um: float,
        distance_um: float,
    ) -> jnp.ndarray:
        if distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")
        if self.refractive_index <= 0:
            raise ValueError("refractive_index must be strictly positive")
        if self.na_limit is not None and self.na_limit <= 0:
            raise ValueError("na_limit must be strictly positive when provided")

        fx, fy = field.grid.frequency_grid()
        wl = jnp.asarray(wavelength_um, dtype=jnp.float32)
        z = jnp.asarray(distance_um, dtype=jnp.float32)
        n = jnp.asarray(self.refractive_index, dtype=jnp.float32)
        k = 2.0 * jnp.pi * n / wl
        argument = 1.0 - (wl * fx / n) ** 2 - (wl * fy / n) ** 2
        propagating = argument >= 0.0
        kz_real = k * jnp.sqrt(jnp.maximum(argument, 0.0))
        transfer = jnp.exp(1j * kz_real * z).astype(jnp.complex64)
        transfer = jnp.where(propagating, transfer, 0.0 + 0.0j)
        if self.na_limit is None:
            return transfer
        na_mask = build_na_mask(
            field.grid,
            wavelength_um=float(wavelength_um),
            na_limit=float(self.na_limit),
            medium_index=float(self.refractive_index),
        )
        return transfer * na_mask.astype(jnp.complex64)

    def propagate(self, field: Field, distance_um: float) -> Field:
        field = field.to_kspace()
        self.validate_for(field)
        if distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")

        outputs = []
        for i, wavelength_um in enumerate(field.spectrum.wavelengths_um):
            transfer = self.transfer_function(field, wavelength_um, distance_um)
            outputs.append(field.data[i] * transfer)

        data = jnp.stack(outputs, axis=0)
        return Field(
            data=data,
            grid=field.grid,
            spectrum=field.spectrum,
            domain="kspace",
            kx_pixel_size_cyc_per_um=field.kx_pixel_size_cyc_per_um,
            ky_pixel_size_cyc_per_um=field.ky_pixel_size_cyc_per_um,
        )


@dataclass(frozen=True)
class AutoPropagator(PropagationModel):
    """
    Propagator wrapper that auto-selects ASM, RS, or k-space propagation.
    """

    asm: ASMPropagator = field(default_factory=ASMPropagator)
    rs: RSPropagator = field(default_factory=RSPropagator)
    kspace: KSpacePropagator = field(default_factory=KSpacePropagator)
    equality_tolerance: float = 1e-6
    nyquist_factor: float = 2.0
    min_padding_factor: float = 2.0
    setup_grid: Grid | None = None
    setup_spectrum: Spectrum | None = None
    setup_distance_um: float | None = None
    distance_um: float | None = None
    precomputed_grid: Grid | None = None
    precomputed_method: Literal["asm", "rs", "kspace"] | None = None

    def __post_init__(self) -> None:
        if self.precomputed_method not in (None, "asm", "rs", "kspace"):
            raise ValueError("precomputed_method must be one of: asm, rs, kspace")
        if self.distance_um is not None and self.distance_um <= 0:
            raise ValueError("distance_um must be strictly positive when provided")

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

        planned_distance_um = (
            self.setup_distance_um if self.setup_distance_um is not None else self.distance_um
        )
        if (
            self.setup_grid is not None
            and self.setup_spectrum is not None
            and planned_distance_um is not None
            and (self.precomputed_grid is None or self.precomputed_method is None)
        ):
            if planned_distance_um <= 0:
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
                method = cast(
                    Literal["asm", "rs"],
                    select_propagator_method(
                        grid=self.setup_grid,
                        spectrum=self.setup_spectrum,
                        distance_um=planned_distance_um,
                        equality_tolerance=self.equality_tolerance,
                    ),
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

    def select_method(
        self,
        field: Field,
        distance_um: float,
    ) -> Literal["asm", "rs", "kspace"]:
        if distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")

        method = self.precomputed_method
        if method is not None:
            return method

        if field.domain == "kspace":
            return "kspace"
        return cast(
            Literal["asm", "rs"],
            select_propagator_method(
                grid=field.grid,
                spectrum=field.spectrum,
                distance_um=distance_um,
                equality_tolerance=self.equality_tolerance,
            ),
        )

    def resolved_model(self, field: Field, distance_um: float) -> PropagationModel:
        method = self.select_method(field=field, distance_um=distance_um)
        if method == "asm":
            return self.asm
        if method == "rs":
            return self.rs
        return self.kspace

    def propagate(self, field: Field, distance_um: float) -> Field:
        self.validate_for(field)
        if distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")

        model = self.resolved_model(field=field, distance_um=distance_um)
        return model.propagate(field, distance_um=distance_um)


@dataclass(frozen=True)
class Propagator(PropagationModel):
    """
    Public facade over ASM/RS/k-space/auto propagators.

    Use `mode` to select behavior:
    - "auto": choose ASM/RS by regime for spatial input, k-space for k-domain input
    - "asm": always ASM
    - "rs": always RS
    - "kspace": always k-space propagator
    """

    mode: Literal["auto", "asm", "rs", "kspace"] = "auto"
    distance_um: float | None = None
    use_sampling_planner: bool = True
    nyquist_factor: float = 2.0
    min_padding_factor: float = 2.0
    precomputed_grid: Grid | None = None
    warn_on_regime_mismatch: bool = True
    equality_tolerance: float = 1e-6
    medium_index: float = 1.0
    refractive_index: float = 1.0
    na_limit: float | None = None
    setup_grid: Grid | None = None
    setup_spectrum: Spectrum | None = None
    setup_distance_um: float | None = None
    auto_precomputed_method: Literal["asm", "rs", "kspace"] | None = None
    _resolved_precomputed_grid: Grid | None = field(default=None, init=False, repr=False)
    _resolved_precomputed_method: Literal["asm", "rs", "kspace"] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.mode not in ("auto", "asm", "rs", "kspace"):
            raise ValueError("mode must be one of: auto, asm, rs, kspace")
        if self.distance_um is not None and self.distance_um <= 0:
            raise ValueError("distance_um must be strictly positive when provided")
        if self.auto_precomputed_method is not None and self.auto_precomputed_method not in (
            "asm",
            "rs",
            "kspace",
        ):
            raise ValueError("auto_precomputed_method must be one of: asm, rs, kspace")

        if self.mode == "auto":
            auto = self._auto_model()
            object.__setattr__(self, "_resolved_precomputed_grid", auto.precomputed_grid)
            object.__setattr__(self, "_resolved_precomputed_method", auto.precomputed_method)
            return
        object.__setattr__(self, "_resolved_precomputed_grid", self.precomputed_grid)
        object.__setattr__(self, "_resolved_precomputed_method", self.mode)

    @property
    def precomputed_method(self) -> Literal["asm", "rs", "kspace"] | None:
        return self._resolved_precomputed_method

    @property
    def resolved_precomputed_grid(self) -> Grid | None:
        return self._resolved_precomputed_grid

    def _asm_model(self) -> ASMPropagator:
        return ASMPropagator(
            use_sampling_planner=self.use_sampling_planner,
            nyquist_factor=self.nyquist_factor,
            min_padding_factor=self.min_padding_factor,
            precomputed_grid=self.precomputed_grid,
            warn_on_regime_mismatch=self.warn_on_regime_mismatch,
            equality_tolerance=self.equality_tolerance,
            medium_index=self.medium_index,
            na_limit=self.na_limit,
        )

    def _rs_model(self) -> RSPropagator:
        return RSPropagator(
            use_sampling_planner=self.use_sampling_planner,
            nyquist_factor=self.nyquist_factor,
            min_padding_factor=self.min_padding_factor,
            precomputed_grid=self.precomputed_grid,
            warn_on_regime_mismatch=self.warn_on_regime_mismatch,
            equality_tolerance=self.equality_tolerance,
            medium_index=self.medium_index,
            na_limit=self.na_limit,
        )

    def _kspace_model(self) -> KSpacePropagator:
        return KSpacePropagator(
            refractive_index=self.refractive_index,
            na_limit=self.na_limit,
        )

    def _auto_model(self) -> AutoPropagator:
        return AutoPropagator(
            asm=self._asm_model(),
            rs=self._rs_model(),
            kspace=self._kspace_model(),
            equality_tolerance=self.equality_tolerance,
            nyquist_factor=self.nyquist_factor,
            min_padding_factor=self.min_padding_factor,
            setup_grid=self.setup_grid,
            setup_spectrum=self.setup_spectrum,
            setup_distance_um=self.setup_distance_um,
            distance_um=self.distance_um,
            precomputed_grid=self.precomputed_grid,
            precomputed_method=self.auto_precomputed_method,
        )

    def _effective_distance(self, distance_um: float | None) -> float:
        resolved = self.distance_um if distance_um is None else distance_um
        if resolved is None:
            raise ValueError(
                "distance_um must be provided either on Propagator or at propagate() call"
            )
        if resolved <= 0:
            raise ValueError("distance_um must be strictly positive")
        return float(resolved)

    def propagate(self, field: Field, distance_um: float | None = None) -> Field:
        self.validate_for(field)
        resolved_distance = self._effective_distance(distance_um)
        if self.mode == "auto":
            return self._auto_model().propagate(field, distance_um=resolved_distance)
        if self.mode == "asm":
            return self._asm_model().propagate(field, distance_um=resolved_distance)
        if self.mode == "rs":
            return self._rs_model().propagate(field, distance_um=resolved_distance)
        return self._kspace_model().propagate(field, distance_um=resolved_distance)
