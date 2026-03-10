from __future__ import annotations

import math
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
    """Compute the critical propagation distance separating the ASM and RS regimes.

    The boundary is estimated as::

        z_crit = N_eff * dx_eff**2 / lambda_min

    where ``N_eff = min(nx, ny)``, ``dx_eff = max(dx_um, dy_um)``, and
    ``lambda_min`` is the shortest wavelength in *spectrum*.  Distances below
    ``z_crit`` favour ASM; distances at or above favour RS.

    Args:
        grid: Spatial sampling grid.
        spectrum: Wavelength set.

    Returns:
        Critical distance in micrometers.
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
    """Select the recommended propagation method for a given distance.

    Compares *distance_um* against the critical distance (see
    :func:`critical_distance_um`) and returns:

    - ``"asm"`` when ``distance_um < z_crit * (1 - equality_tolerance)``
    - ``"rs"`` otherwise

    Args:
        grid: Spatial sampling grid.
        spectrum: Wavelength set.
        distance_um: Propagation distance in micrometers (must be positive).
        equality_tolerance: Relative tolerance around the boundary.

    Returns:
        ``"asm"`` or ``"rs"``.
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
    """Build a padded, Nyquist-safe propagation grid.

    The returned grid has finer pixel spacing (if needed to satisfy the
    Nyquist criterion for the shortest wavelength) and is zero-padded to
    at least ``min_padding_factor`` times the original extent, rounded up
    to the next power of two for FFT efficiency.

    Args:
        grid: Original spatial sampling grid.
        spectrum: Wavelength set used to determine Nyquist limits.
        nyquist_factor: Sampling density relative to the Nyquist limit.
            A value of 2.0 means two samples per shortest half-wavelength.
        min_padding_factor: Minimum spatial-extent multiplier for
            zero-padding (must be ≥ 1.0).

    Returns:
        A new ``Grid`` suitable for numerically stable propagation.
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

    leading_shape = field.data.shape[:-2]
    flat_data = field.data.reshape((-1, field.grid.ny, field.grid.nx))
    data = jnp.stack(
        [
            _resample_2d_to_grid(flat_data[i], src_grid=field.grid, dst_grid=target_grid)
            for i in range(flat_data.shape[0])
        ],
        axis=0,
    ).reshape((*leading_shape, target_grid.ny, target_grid.nx))
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

    leading_shape = field.data.shape[:-2]
    flat_data = field.data.reshape((-1, field.grid.ny, field.grid.nx))
    data = jnp.stack(
        [
            _resample_2d_to_grid(flat_data[i], src_grid=field.grid, dst_grid=original_grid)
            for i in range(flat_data.shape[0])
        ],
        axis=0,
    ).reshape((*leading_shape, original_grid.ny, original_grid.nx))
    return Field(
        data=data,
        grid=original_grid,
        spectrum=field.spectrum,
        polarization_mode=field.polarization_mode,
        domain=field.domain,
        kx_pixel_size_cyc_per_um=field.kx_pixel_size_cyc_per_um,
        ky_pixel_size_cyc_per_um=field.ky_pixel_size_cyc_per_um,
    )


def _flatten_batch_axes(field: Field) -> tuple[jnp.ndarray, tuple[int, ...]]:
    batch_shape = field.batch_shape
    flat_batch = math.prod(batch_shape) if batch_shape else 1
    if field.is_jones:
        flat = field.data.reshape(
            (
                flat_batch,
                field.spectrum.size,
                field.num_polarization_channels,
                field.grid.ny,
                field.grid.nx,
            )
        )
    else:
        flat = field.data.reshape((flat_batch, field.spectrum.size, field.grid.ny, field.grid.nx))
    return flat, batch_shape


def _restore_batch_axes(
    field: Field,
    flat_data: jnp.ndarray,
    batch_shape: tuple[int, ...],
) -> jnp.ndarray:
    if field.is_jones:
        return flat_data.reshape(
            (
                *batch_shape,
                field.spectrum.size,
                field.num_polarization_channels,
                field.grid.ny,
                field.grid.nx,
            )
        )
    return flat_data.reshape((*batch_shape, field.spectrum.size, field.grid.ny, field.grid.nx))


@dataclass(frozen=True)
class RSPropagator(OpticalLayer):
    """Rayleigh–Sommerfeld propagator via convolution with the RS impulse response.

    Requires **spatial-domain** input.  Supports both scalar and Jones fields.
    When ``use_sampling_planner`` is ``True`` (default) the field is
    automatically resampled / padded to a Nyquist-safe grid before
    propagation and restored to the original grid afterwards.

    All length quantities are in micrometers.

    Args:
        distance_um: Propagation distance in micrometers.
        use_sampling_planner: Whether to apply automatic Nyquist resampling.
        nyquist_factor: Oversampling relative to the Nyquist limit.
        min_padding_factor: Minimum padding multiplier for the propagation
            grid.
        precomputed_grid: Optional pre-built propagation grid, bypassing
            the automatic planner.
        warn_on_regime_mismatch: Emit a warning when the propagation
            distance falls outside the recommended RS regime.
        equality_tolerance: Tolerance for the regime boundary.
        medium_index: Refractive index of the propagation medium.
        na_limit: Optional numerical aperture cutoff applied in k-space
            after propagation.
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
        """Return the Rayleigh-Sommerfeld impulse response on ``field.grid``.

        Args:
            field: Spatial-domain field whose grid defines the sampling lattice.
            wavelength_um: Wavelength to propagate in micrometers.
            distance_um: Propagation distance in micrometers.

        Returns:
            Complex array with shape ``(ny, nx)`` sampled on ``field.grid``.

        Raises:
            ValueError: If ``distance_um`` is not strictly positive.
        """
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
        """Return trainable parameters exposed by this layer."""
        if self.distance_um is not None:
            return {"distance_um": jnp.asarray(self.distance_um, dtype=jnp.float32)}
        return {}

    def forward(self, field: Field) -> Field:
        """Propagate a spatial-domain field with the RS convolution model.

        Args:
            field: Spatial-domain input field with scalar or Jones data layout.

        Returns:
            Spatial-domain field sampled on the original input grid. If the
            sampling planner is enabled, resampling and padding are applied
            internally and then undone before returning.

        Raises:
            ValueError: If the input domain is wrong or propagation parameters
                are invalid.
        """
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
        flat_data, batch_shape = _flatten_batch_axes(work_field)

        area_um2 = work_field.grid.dx_um * work_field.grid.dy_um
        outputs = []
        for i, wavelength_um in enumerate(work_field.spectrum.wavelengths_um):
            kernel = self.delta_response(work_field, wavelength_um, distance_um)
            propagated = fftconvolve(
                flat_data[:, i],
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

        data = _restore_batch_axes(work_field, jnp.stack(outputs, axis=1), batch_shape)
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
    """Angular Spectrum Method (ASM) propagator.

    Requires **spatial-domain** input.  Internally transforms the field
    to k-space, applies the ASM transfer function via
    :class:`KSpacePropagator`, and transforms back.  Supports both
    scalar and Jones fields.

    When ``use_sampling_planner`` is ``True`` the field is automatically
    resampled / padded for numerical stability.

    All length quantities are in micrometers.

    Args:
        distance_um: Propagation distance in micrometers.
        use_sampling_planner: Whether to apply automatic Nyquist resampling.
        nyquist_factor: Oversampling relative to the Nyquist limit.
        min_padding_factor: Minimum padding multiplier.
        precomputed_grid: Optional pre-built propagation grid.
        warn_on_regime_mismatch: Emit a warning when the distance falls
            outside the recommended ASM regime.
        equality_tolerance: Tolerance for the regime boundary.
        medium_index: Refractive index of the propagation medium.
        na_limit: Optional numerical aperture cutoff.
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
        """Build the ASM transfer function for one wavelength on ``field.grid``.

        Args:
            field: Spatial-domain field whose sampling grid defines the
                frequency lattice.
            wavelength_um: Wavelength to propagate in micrometers.
            distance_um: Propagation distance in micrometers.

        Returns:
            Complex array with shape ``(ny, nx)`` in reciprocal-space sampling
            order.

        Raises:
            ValueError: If the distance, refractive index, or NA limit is
                invalid.
        """
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
        """Return trainable parameters exposed by this layer."""
        if self.distance_um is not None:
            return {"distance_um": jnp.asarray(self.distance_um, dtype=jnp.float32)}
        return {}

    def forward(self, field: Field) -> Field:
        """Propagate a spatial-domain field with the Angular Spectrum Method.

        Args:
            field: Spatial-domain input field with scalar or Jones data layout.

        Returns:
            Spatial-domain field sampled on the original input grid. Internal
            resampling and padding are restored away before the result is
            returned.

        Raises:
            ValueError: If the input domain is wrong or propagation parameters
                are invalid.
        """
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
    """Diagonal k-space propagator using the angular-spectrum phase advance.

    Requires **k-space-domain** input.  Multiplies each frequency
    component by ``exp(j * kz * distance_um)`` where *kz* is derived
    from the dispersion relation.  Supports scalar and Jones fields.

    All length quantities are in micrometers.

    Args:
        distance_um: Propagation distance in micrometers.
        refractive_index: Refractive index of the medium.
        na_limit: Optional numerical aperture cutoff applied as a hard
            k-space mask.
        include_evanescent: If ``True``, evanescent modes are included
            (complex *kz*). If ``False`` (default), evanescent modes are
            zeroed out.
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
        """Build the diagonal k-space transfer function for one wavelength.

        Args:
            field: K-space field whose grid defines the sampled frequency axes.
            wavelength_um: Wavelength to propagate in micrometers.
            distance_um: Propagation distance in micrometers.

        Returns:
            Complex array with shape ``(ny, nx)`` matching the sampled k-space
            grid.

        Raises:
            ValueError: If the distance, refractive index, or NA limit is
                invalid.
        """
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
        """Return trainable parameters exposed by this layer."""
        if self.distance_um is not None:
            return {"distance_um": jnp.asarray(self.distance_um, dtype=jnp.float32)}
        return {}

    def forward(self, field: Field) -> Field:
        """Propagate a k-space field by diagonal transfer multiplication.

        Args:
            field: K-space input field with scalar or Jones data layout.

        Returns:
            K-space field with the same shape, batch axes, and sampling metadata
            as the input.

        Raises:
            ValueError: If the input domain is wrong or propagation parameters
                are invalid.
        """
        distance_um = self.distance_um
        _require_domain(field, expected="kspace", layer_name="KSpacePropagator")
        self.validate_for(field)
        if distance_um is None:
            raise ValueError("distance_um must be set for forward pass")
        if distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")

        flat_data, batch_shape = _flatten_batch_axes(field)
        outputs = []
        for i, wavelength_um in enumerate(field.spectrum.wavelengths_um):
            transfer = self.transfer_function(field, wavelength_um, distance_um)
            outputs.append(flat_data[:, i] * transfer)

        data = _restore_batch_axes(field, jnp.stack(outputs, axis=1), batch_shape)
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
    """Plan a propagation layer for a given field sampling regime.

    When **mode** is ``"auto"`` the function selects a propagator based on
    the critical-distance heuristic (ASM for short distances, RS for long
    distances).  If *input_domain* is ``"kspace"`` the returned layer is
    always a :class:`KSpacePropagator`.

    Explicit modes ``"asm"``, ``"rs"``, or ``"kspace"`` force the
    corresponding propagator and emit a warning (if enabled) when the
    choice falls outside the recommended regime.

    All length parameters are in micrometers.

    Args:
        mode: Propagator selection strategy.
        grid: Spatial sampling grid.
        spectrum: Wavelength set.
        distance_um: Propagation distance in micrometers (must be positive).
        input_domain: Expected domain of the input field.
        use_sampling_planner: Apply automatic Nyquist grid planning.
        nyquist_factor: Oversampling relative to the Nyquist limit.
        min_padding_factor: Minimum padding multiplier.
        precomputed_grid: Optional pre-built propagation grid.
        warn_on_regime_mismatch: Emit a warning on sub-optimal mode
            selection.
        equality_tolerance: Tolerance for the ASM/RS boundary.
        medium_index: Refractive index used by ASM / RS propagators.
        refractive_index: Refractive index used by the k-space propagator.
        na_limit: Optional numerical aperture cutoff.

    Returns:
        A configured propagator layer ready for ``forward()``.

    Raises:
        ValueError: If *distance_um* is non-positive or *mode* is invalid.
    """
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
