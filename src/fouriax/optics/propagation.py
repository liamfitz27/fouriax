from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, cast

import jax.numpy as jnp
from jax.scipy import ndimage as jndimage

from fouriax.fft import fftconvolve
from fouriax.optics.interfaces import OpticalLayer
from fouriax.optics.model import Field, Grid, Spectrum
from fouriax.optics.na_planning import build_na_mask


def _require_domain(
    field: Field,
    *,
    expected: Literal["spatial", "kspace"],
    layer_name: str,
) -> None:
    if field.domain == expected:
        return
    if expected == "spatial":
        raise ValueError(
            f"{layer_name} requires spatial-domain input; "
            "insert InverseFourierTransform before this layer."
        )
    raise ValueError(
        f"{layer_name} requires kspace-domain input; "
        "insert FourierTransform before this layer."
    )


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

    if field.is_jones:
        data = jnp.stack(
            [
                jnp.stack(
                    [
                        _resample_2d_to_grid(
                            field.data[i, c],
                            src_grid=field.grid,
                            dst_grid=target_grid,
                        )
                        for c in range(field.num_polarization_channels)
                    ],
                    axis=0,
                )
                for i in range(field.spectrum.size)
            ],
            axis=0,
        )
    else:
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
        polarization_mode=field.polarization_mode,
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

    if field.is_jones:
        data = jnp.stack(
            [
                jnp.stack(
                    [
                        _resample_2d_to_grid(
                            field.data[i, c],
                            src_grid=field.grid,
                            dst_grid=original_grid,
                        )
                        for c in range(field.num_polarization_channels)
                    ],
                    axis=0,
                )
                for i in range(field.spectrum.size)
            ],
            axis=0,
        )
    else:
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
        polarization_mode=field.polarization_mode,
        domain=field.domain,
        kx_pixel_size_cyc_per_um=field.kx_pixel_size_cyc_per_um,
        ky_pixel_size_cyc_per_um=field.ky_pixel_size_cyc_per_um,
    )


@dataclass(frozen=True)
class RSPropagator(OpticalLayer):
    """
    Rayleigh-Sommerfeld propagator based on convolution with the RS delta response.

    All length quantities use micrometers (um).
    """

    distance_um: float | None = None
    use_sampling_planner: bool = True
    nyquist_factor: float = 2.0
    min_padding_factor: float = 2.0
    precomputed_grid: Grid | None = None
    warn_on_regime_mismatch: bool = True
    equality_tolerance: float = 1e-6
    medium_index: float = 1.0
    na_limit: float | None = None

    def delta_response(
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

    def parameters(self) -> dict[str, jnp.ndarray]:
        if self.distance_um is not None:
            return {"distance_um": jnp.asarray(self.distance_um, dtype=jnp.float32)}
        return {}

    def forward(self, field: Field) -> Field:
        distance_um = self.distance_um
        _require_domain(field, expected="spatial", layer_name="RSPropagator")
        self.validate_for(field)
        if distance_um is None:
            raise ValueError("distance_um must be set for forward pass")
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
            kernel = self.delta_response(work_field, wavelength_um, distance_um)
            if work_field.is_jones:
                propagated = fftconvolve(
                    work_field.data[i],
                    kernel[None, :, :],
                    mode="same",
                    axes=(-2, -1),
                )
            else:
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
            polarization_mode=work_field.polarization_mode,
            domain=work_field.domain,
            kx_pixel_size_cyc_per_um=work_field.kx_pixel_size_cyc_per_um,
            ky_pixel_size_cyc_per_um=work_field.ky_pixel_size_cyc_per_um,
        )
        return _restore_to_original_grid(propagated_field, original_grid)


@dataclass(frozen=True)
class ASMPropagator(OpticalLayer):
    """
    Angular Spectrum Method (ASM) propagator.

    All length quantities use micrometers (um).
    """

    distance_um: float | None = None
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

    def parameters(self) -> dict[str, jnp.ndarray]:
        if self.distance_um is not None:
            return {"distance_um": jnp.asarray(self.distance_um, dtype=jnp.float32)}
        return {}

    def forward(self, field: Field) -> Field:
        distance_um = self.distance_um
        _require_domain(field, expected="spatial", layer_name="ASMPropagator")
        self.validate_for(field)
        if distance_um is None:
            raise ValueError("distance_um must be set for forward pass")
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

        from fouriax.optics.layers import FourierTransform, InverseFourierTransform

        k_field = FourierTransform().forward(work_field)
        k_layer = KSpacePropagator(
            distance_um=distance_um,
            refractive_index=self.medium_index,
            na_limit=self.na_limit,
            include_evanescent=True,
        )
        propagated_k = k_layer.forward(k_field)
        propagated_spatial = InverseFourierTransform().forward(propagated_k)
        return _restore_to_original_grid(propagated_spatial, original_grid)


@dataclass(frozen=True)
class KSpacePropagator(OpticalLayer):
    """
    k-space diagonal propagator using angular-spectrum phase advance.

    All length quantities use micrometers (um).
    """

    distance_um: float | None = None
    refractive_index: float = 1.0
    na_limit: float | None = None
    include_evanescent: bool = False

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
        if self.include_evanescent:
            kz = k * jnp.sqrt(argument.astype(jnp.complex64))
            transfer = jnp.exp(1j * kz * z).astype(jnp.complex64)
        else:
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

    def parameters(self) -> dict[str, jnp.ndarray]:
        if self.distance_um is not None:
            return {"distance_um": jnp.asarray(self.distance_um, dtype=jnp.float32)}
        return {}

    def forward(self, field: Field) -> Field:
        distance_um = self.distance_um
        _require_domain(field, expected="kspace", layer_name="KSpacePropagator")
        self.validate_for(field)
        if distance_um is None:
            raise ValueError("distance_um must be set for forward pass")
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
            polarization_mode=field.polarization_mode,
            domain="kspace",
            kx_pixel_size_cyc_per_um=field.kx_pixel_size_cyc_per_um,
            ky_pixel_size_cyc_per_um=field.ky_pixel_size_cyc_per_um,
        )


def plan_propagation(
    *,
    mode: Literal["auto", "asm", "rs", "kspace"] = "auto",
    grid: Grid,
    spectrum: Spectrum,
    distance_um: float,
    input_domain: Literal["spatial", "kspace"] = "spatial",
    use_sampling_planner: bool = True,
    nyquist_factor: float = 2.0,
    min_padding_factor: float = 2.0,
    precomputed_grid: Grid | None = None,
    warn_on_regime_mismatch: bool = True,
    equality_tolerance: float = 1e-6,
    medium_index: float = 1.0,
    refractive_index: float = 1.0,
    na_limit: float | None = None,
) -> ASMPropagator | RSPropagator | KSpacePropagator:
    if distance_um <= 0:
        raise ValueError("distance_um must be strictly positive")
    if mode not in ("auto", "asm", "rs", "kspace"):
        raise ValueError("mode must be one of: auto, asm, rs, kspace")

    if mode == "kspace":
        return KSpacePropagator(
            distance_um=distance_um,
            refractive_index=refractive_index,
            na_limit=na_limit,
        )

    def _planned_grid() -> Grid | None:
        if input_domain == "kspace":
            return None
        if precomputed_grid is not None:
            return precomputed_grid
        if not use_sampling_planner:
            return None
        return recommend_nyquist_grid(
            grid=grid,
            spectrum=spectrum,
            nyquist_factor=nyquist_factor,
            min_padding_factor=min_padding_factor,
        )

    planned_grid = _planned_grid()
    selection_grid = planned_grid or grid

    def _warn_if_explicit_mode_mismatch(explicit_mode: Literal["asm", "rs"]) -> None:
        if not warn_on_regime_mismatch or input_domain == "kspace":
            return
        recommended = select_propagator_method(
            grid=selection_grid,
            spectrum=spectrum,
            distance_um=distance_um,
            equality_tolerance=equality_tolerance,
        )
        if recommended == explicit_mode:
            return
        z_crit = critical_distance_um(grid=selection_grid, spectrum=spectrum)
        warnings.warn(
            f"{explicit_mode.upper()} selected outside recommended regime "
            f"(distance_um={distance_um:.3f}, z_crit_um={z_crit:.3f}, "
            f"recommended='{recommended}'). Planning will continue with "
            f"{explicit_mode.upper()} as requested.",
            UserWarning,
            stacklevel=2,
        )

    if mode == "asm":
        _warn_if_explicit_mode_mismatch("asm")
        return ASMPropagator(
            distance_um=distance_um,
            use_sampling_planner=False,
            nyquist_factor=nyquist_factor,
            min_padding_factor=min_padding_factor,
            precomputed_grid=planned_grid,
            warn_on_regime_mismatch=False,
            equality_tolerance=equality_tolerance,
            medium_index=medium_index,
            na_limit=na_limit,
        )
    if mode == "rs":
        _warn_if_explicit_mode_mismatch("rs")
        return RSPropagator(
            distance_um=distance_um,
            use_sampling_planner=False,
            nyquist_factor=nyquist_factor,
            min_padding_factor=min_padding_factor,
            precomputed_grid=planned_grid,
            warn_on_regime_mismatch=False,
            equality_tolerance=equality_tolerance,
            medium_index=medium_index,
            na_limit=na_limit,
        )

    if input_domain == "kspace":
        return KSpacePropagator(
            distance_um=distance_um,
            refractive_index=refractive_index,
            na_limit=na_limit,
        )

    method = cast(
        Literal["asm", "rs"],
        select_propagator_method(
            grid=selection_grid,
            spectrum=spectrum,
            distance_um=distance_um,
            equality_tolerance=equality_tolerance,
        ),
    )
    if method == "asm":
        return ASMPropagator(
            distance_um=distance_um,
            use_sampling_planner=False,
            nyquist_factor=nyquist_factor,
            min_padding_factor=min_padding_factor,
            precomputed_grid=planned_grid,
            warn_on_regime_mismatch=False,
            equality_tolerance=equality_tolerance,
            medium_index=medium_index,
            na_limit=na_limit,
        )
    return RSPropagator(
        distance_um=distance_um,
        use_sampling_planner=False,
        nyquist_factor=nyquist_factor,
        min_padding_factor=min_padding_factor,
        precomputed_grid=planned_grid,
        warn_on_regime_mismatch=False,
        equality_tolerance=equality_tolerance,
        medium_index=medium_index,
        na_limit=na_limit,
    )
