from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Grid:
    """Regular 2D sampling lattice with pixel spacing in micrometers.

    A ``Grid`` defines the discrete spatial (or frequency) extent of an optical
    field.  Pixel centres are placed symmetrically about the origin so that the
    field is centred at ``(0, 0)``.

    Args:
        nx: Number of grid points along the x-axis.
        ny: Number of grid points along the y-axis.
        dx_um: Pixel pitch along x in micrometers.
        dy_um: Pixel pitch along y in micrometers.
    """

    nx: int
    ny: int
    dx_um: float
    dy_um: float

    @classmethod
    def from_extent(cls, nx: int, ny: int, dx_um: float, dy_um: float) -> "Grid":
        """Create a validated grid.

        Args:
            nx: Number of grid points along x.
            ny: Number of grid points along y.
            dx_um: Pixel pitch along x in micrometers.
            dy_um: Pixel pitch along y in micrometers.

        Returns:
            A validated ``Grid`` instance.

        Raises:
            ValueError: If sizes or spacings are non-positive.
        """
        grid = cls(nx=nx, ny=ny, dx_um=dx_um, dy_um=dy_um)
        grid.validate()
        return grid

    @property
    def shape(self) -> tuple[int, int]:
        """Spatial array shape as ``(ny, nx)``."""
        return (self.ny, self.nx)

    def frequency_grid(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return 2D spatial-frequency coordinate arrays.

        Frequencies are in cycles per micrometer and follow the FFT ordering
        produced by ``jnp.fft.fftfreq``.

        Returns:
            ``(fx, fy)`` each with shape ``(ny, nx)``.
        """
        self.validate()
        fx_1d = jnp.fft.fftfreq(self.nx, d=self.dx_um)
        fy_1d = jnp.fft.fftfreq(self.ny, d=self.dy_um)
        fx, fy = jnp.meshgrid(fx_1d, fy_1d, indexing="xy")
        return fx, fy

    def spatial_grid(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return 2D spatial coordinate arrays centred at the origin.

        Coordinates are in micrometers.

        Returns:
            ``(x, y)`` each with shape ``(ny, nx)``.
        """
        self.validate()
        x_1d = (jnp.arange(self.nx) - (self.nx - 1) / 2.0) * self.dx_um
        y_1d = (jnp.arange(self.ny) - (self.ny - 1) / 2.0) * self.dy_um
        x, y = jnp.meshgrid(x_1d, y_1d, indexing="xy")
        return x, y

    def kspace_pixel_size_cyc_per_um(self) -> tuple[float, float]:
        """Return k-grid sampling intervals ``(dfx, dfy)`` in cycles per micrometer.

        These are the reciprocal-space pixel pitches implied by the spatial
        grid dimensions.
        """
        self.validate()
        return (1.0 / (self.nx * self.dx_um), 1.0 / (self.ny * self.dy_um))

    def validate(self) -> None:
        """Validate grid dimensions and pixel pitches.

        Raises:
            ValueError: If any grid size or pixel pitch is non-positive.
        """
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("grid sizes nx and ny must be positive integers")
        if self.dx_um <= 0 or self.dy_um <= 0:
            raise ValueError("grid spacings dx_um and dy_um must be positive")


@dataclass(frozen=True)
class Spectrum:
    """Immutable collection of wavelengths in micrometers.

    All wavelengths must be strictly positive.  Use :meth:`from_scalar` for a
    single wavelength or :meth:`from_array` for broadband simulations.

    Args:
        wavelengths_um: 1-D array of wavelengths in micrometers.  Must contain
            at least one strictly positive value.
    """

    wavelengths_um: jnp.ndarray

    @classmethod
    def from_scalar(cls, wavelength_um: float) -> "Spectrum":
        """Create a single-wavelength spectrum.

        Args:
            wavelength_um: Wavelength in micrometers (must be positive).
        """
        spectrum = cls(wavelengths_um=jnp.asarray([wavelength_um], dtype=jnp.float32))
        spectrum.validate()
        return spectrum

    @classmethod
    def from_array(cls, wavelengths_um: jnp.ndarray) -> "Spectrum":
        """Create a multi-wavelength spectrum from an array.

        Args:
            wavelengths_um: 1-D array-like of wavelengths in micrometers.
                All values must be strictly positive.
        """
        spectrum = cls(wavelengths_um=jnp.asarray(wavelengths_um, dtype=jnp.float32))
        spectrum.validate()
        return spectrum

    @property
    def size(self) -> int:
        """Number of wavelength samples stored in the spectrum."""
        return int(self.wavelengths_um.shape[0])

    def validate(self) -> None:
        """Validate wavelength storage and positivity constraints.

        Raises:
            ValueError: If the wavelength array is not 1-D, is empty, or
                contains non-positive values during eager execution.
        """
        if self.wavelengths_um.ndim != 1:
            raise ValueError("spectrum wavelengths_um must be a 1D array")
        if self.wavelengths_um.size == 0:
            raise ValueError("spectrum must contain at least one wavelength")
        # During JAX tracing, Python bool conversion on traced arrays is invalid.
        # Keep strict positivity checks for eager execution and skip in traced mode.
        if isinstance(self.wavelengths_um, jax.core.Tracer):
            return
        if np.any(np.asarray(self.wavelengths_um) <= 0):
            raise ValueError("all wavelengths_um must be strictly positive")


@dataclass(frozen=True)
class Intensity:
    """Real-valued spatial intensity over a 2D grid for one or more wavelengths.

    The expected data layout is ``(*batch, num_wavelengths, ny, nx)``.

    Args:
        data: Real-valued intensity samples.
        grid: Spatial sampling grid.
        spectrum: Wavelength set.
    """

    data: jnp.ndarray
    grid: Grid
    spectrum: Spectrum

    @classmethod
    def zeros(
        cls,
        grid: Grid,
        spectrum: Spectrum,
        dtype=jnp.float32,
    ) -> "Intensity":
        """Create an intensity image initialised to zero."""
        intensity = cls(
            data=jnp.zeros((spectrum.size, grid.ny, grid.nx), dtype=dtype),
            grid=grid,
            spectrum=spectrum,
        )
        intensity.validate()
        return intensity

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading batch dimensions before wavelength and spatial axes."""
        return self.data.shape[:-3]

    @property
    def has_batch(self) -> bool:
        """Whether the intensity includes one or more leading batch axes."""
        return bool(self.batch_shape)

    def power(self) -> jnp.ndarray:
        """Integrated power per wavelength (spatial sum of intensity)."""
        return jnp.sum(self.data, axis=(-2, -1))

    def sum_wavelengths(self) -> jnp.ndarray:
        """Sum the wavelength axis while preserving batch and spatial axes."""
        return jnp.sum(self.data, axis=-3)

    def validate(self) -> None:
        """Validate intensity metadata and the trailing shape convention."""
        self.grid.validate()
        self.spectrum.validate()
        if self.data.ndim < 3:
            raise ValueError(
                "intensity data must have shape (*batch, num_wavelengths, ny, nx)"
            )
        expected_shape = (self.spectrum.size, self.grid.ny, self.grid.nx)
        if self.data.shape[-3:] != expected_shape:
            raise ValueError(
                "intensity data shape mismatch: got "
                f"{self.data.shape}, expected trailing shape {expected_shape}"
            )
        if jnp.issubdtype(self.data.dtype, jnp.complexfloating):
            raise ValueError("intensity data must be real-valued")


@dataclass(frozen=True)
class Field:
    """Complex optical field over a 2D grid for one or more wavelengths.

    Data layout depends on the polarization mode:

    - **scalar**: ``(*batch, num_wavelengths, ny, nx)``
    - **jones**: ``(*batch, num_wavelengths, 2, ny, nx)`` with channel
      order ``(Ex, Ey)``

    Leading batch dimensions are optional. ``domain`` records whether the
    data is currently in the spatial or k-space representation.

    Args:
        data: Complex array with the shape described above.
        grid: Spatial sampling grid.
        spectrum: Wavelength set.
        polarization_mode: ``"scalar"`` or ``"jones"``.
        domain: ``"spatial"`` or ``"kspace"``.
        kx_pixel_size_cyc_per_um: k-space pixel pitch along x
            (cycles/µm).  Defaults to the value implied by ``grid``.
        ky_pixel_size_cyc_per_um: k-space pixel pitch along y
            (cycles/µm).  Defaults to the value implied by ``grid``.
    """

    data: jnp.ndarray
    grid: Grid
    spectrum: Spectrum
    polarization_mode: Literal["scalar", "jones"] = "scalar"
    domain: Literal["spatial", "kspace"] = "spatial"
    kx_pixel_size_cyc_per_um: float | None = None
    ky_pixel_size_cyc_per_um: float | None = None

    def __post_init__(self) -> None:
        if self.polarization_mode not in ("scalar", "jones"):
            raise ValueError("polarization_mode must be one of: scalar, jones")
        if self.domain not in ("spatial", "kspace"):
            raise ValueError("domain must be one of: spatial, kspace")

        default_kx, default_ky = self.grid.kspace_pixel_size_cyc_per_um()
        kx = (
            default_kx
            if self.kx_pixel_size_cyc_per_um is None
            else self.kx_pixel_size_cyc_per_um
        )
        ky = (
            default_ky
            if self.ky_pixel_size_cyc_per_um is None
            else self.ky_pixel_size_cyc_per_um
        )

        if kx <= 0 or ky <= 0:
            raise ValueError("kx/ky pixel sizes must be strictly positive")

        object.__setattr__(self, "kx_pixel_size_cyc_per_um", float(kx))
        object.__setattr__(self, "ky_pixel_size_cyc_per_um", float(ky))

    @classmethod
    def zeros(
        cls,
        grid: Grid,
        spectrum: Spectrum,
        dtype=jnp.complex64,
    ) -> "Field":
        """Create a scalar field initialised to zero.

        Returns:
            A spatial-domain scalar field with shape
            ``(num_wavelengths, ny, nx)``.
        """
        data = jnp.zeros((spectrum.size, grid.ny, grid.nx), dtype=dtype)
        field = cls(data=data, grid=grid, spectrum=spectrum)
        field.validate()
        return field

    @classmethod
    def plane_wave(
        cls,
        grid: Grid,
        spectrum: Spectrum,
        amplitude: float | jnp.ndarray = 1.0,
        phase: float | jnp.ndarray = 0.0,
        dtype=jnp.complex64,
    ) -> "Field":
        """Create a uniform scalar plane wave.

        Args:
            grid: Spatial sampling grid.
            spectrum: Wavelength set.
            amplitude: Scalar or broadcastable amplitude.
            phase: Scalar or broadcastable phase in radians.
            dtype: Complex dtype for the field data.

        Returns:
            A spatial-domain scalar field with shape
            ``(num_wavelengths, ny, nx)``.
        """
        amp = jnp.asarray(amplitude, dtype=jnp.float32)
        phs = jnp.asarray(phase, dtype=jnp.float32)
        base = jnp.ones((spectrum.size, grid.ny, grid.nx), dtype=dtype)
        data = base * amp * jnp.exp(1j * phs)
        field = cls(data=data, grid=grid, spectrum=spectrum)
        field.validate()
        return field

    @classmethod
    def plane_wave_jones(
        cls,
        grid: Grid,
        spectrum: Spectrum,
        ex: complex | jnp.ndarray = 1.0 + 0.0j,
        ey: complex | jnp.ndarray = 0.0 + 0.0j,
        dtype=jnp.complex64,
    ) -> "Field":
        """Create a uniform Jones-vector plane wave.

        Args:
            grid: Spatial sampling grid.
            spectrum: Wavelength set.
            ex: Complex amplitude of the x-polarisation component.
            ey: Complex amplitude of the y-polarisation component.
            dtype: Complex dtype for the field data.

        Returns:
            A spatial-domain Jones field with shape
            ``(num_wavelengths, 2, ny, nx)``.
        """
        ex_arr = jnp.asarray(ex, dtype=dtype)
        ey_arr = jnp.asarray(ey, dtype=dtype)
        base = jnp.ones((spectrum.size, grid.ny, grid.nx), dtype=dtype)
        ex_data = base * ex_arr
        ey_data = base * ey_arr
        data = jnp.stack([ex_data, ey_data], axis=1)
        field = cls(
            data=data,
            grid=grid,
            spectrum=spectrum,
            polarization_mode="jones",
        )
        field.validate()
        return field

    @property
    def is_jones(self) -> bool:
        """Whether the field stores Jones-vector polarization channels."""
        return self.polarization_mode == "jones"

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading batch dimensions before wavelength and spatial axes."""
        return self.data.shape[:-4] if self.is_jones else self.data.shape[:-3]

    @property
    def has_batch(self) -> bool:
        """Whether the field includes one or more leading batch axes."""
        return bool(self.batch_shape)

    @property
    def num_polarization_channels(self) -> int:
        """Number of polarization channels represented in ``data``."""
        return 2 if self.is_jones else 1

    def component_intensity(self) -> jnp.ndarray:
        """Per-polarisation-channel intensity.

        Returns:
            Array with shape ``(*batch, num_wavelengths, num_channels, ny, nx)``.
            For scalar fields ``num_channels`` is 1; for Jones fields it is 2.
        """
        if self.is_jones:
            return jnp.abs(self.data) ** 2
        return (jnp.abs(self.data) ** 2)[..., None, :, :]

    def intensity(self) -> jnp.ndarray:
        """Total intensity summed over polarisation channels.

        Returns:
            Array with shape ``(*batch, num_wavelengths, ny, nx)``.
        """
        return jnp.sum(self.component_intensity(), axis=-3)

    def to_intensity(self) -> Intensity:
        """Return the spatial irradiance represented by this field.

        K-space fields are converted to spatial domain before intensity is
        computed, so the returned object is always suitable for detector and
        incoherent-imaging paths.
        """
        field_spatial = self.to_spatial()
        intensity = Intensity(
            data=field_spatial.intensity().astype(field_spatial.data.real.dtype),
            grid=field_spatial.grid,
            spectrum=field_spatial.spectrum,
        )
        intensity.validate()
        return intensity

    def phase(self) -> jnp.ndarray:
        """Element-wise phase angle in radians.

        Returns:
            Real array with the same shape as ``data``.
        """
        return jnp.angle(self.data)

    def power(self) -> jnp.ndarray:
        """Integrated power per wavelength (spatial sum of intensity).

        Returns:
            Array with shape ``(*batch, num_wavelengths)``.
        """
        return jnp.sum(self.intensity(), axis=(-2, -1))

    def normalize_power(self, target: float = 1.0, eps: float = 1e-12) -> "Field":
        """Return a copy of this field scaled so that each wavelength channel
        has total power equal to *target*.

        Args:
            target: Desired total power per wavelength channel.
            eps: Minimum power threshold; raises if any channel is below this.

        Raises:
            ValueError: If any wavelength channel has near-zero power.
        """
        current = self.power()
        if not bool(jnp.all(current > eps)):
            raise ValueError("cannot normalize field with near-zero power")
        scale = jnp.sqrt(jnp.asarray(target, dtype=current.dtype) / current)
        if self.is_jones:
            scale = scale[..., None, None, None]
        else:
            scale = scale[..., None, None]
        return Field(
            data=self.data * scale,
            grid=self.grid,
            spectrum=self.spectrum,
            polarization_mode=self.polarization_mode,
            domain=self.domain,
            kx_pixel_size_cyc_per_um=self.kx_pixel_size_cyc_per_um,
            ky_pixel_size_cyc_per_um=self.ky_pixel_size_cyc_per_um,
        )

    def apply_phase(self, phase_map: jnp.ndarray | float) -> "Field":
        """Multiply the field by ``exp(j * phase_map)``.

        Args:
            phase_map: Phase in radians, broadcastable to the field data.
                Accepted shapes include scalar, ``(ny, nx)``, or
                ``(num_wavelengths, ny, nx)``.
        """
        phs = jnp.asarray(phase_map, dtype=jnp.float32)
        if self.is_jones and phs.ndim in (2, 3):
            phs = phs[:, None, :, :] if phs.ndim == 3 else phs[None, None, :, :]
        return Field(
            data=self.data * jnp.exp(1j * phs),
            grid=self.grid,
            spectrum=self.spectrum,
            polarization_mode=self.polarization_mode,
            domain=self.domain,
            kx_pixel_size_cyc_per_um=self.kx_pixel_size_cyc_per_um,
            ky_pixel_size_cyc_per_um=self.ky_pixel_size_cyc_per_um,
        )

    def apply_amplitude(self, mask: jnp.ndarray | float) -> "Field":
        """Multiply the field amplitude by a real-valued mask.

        Args:
            mask: Amplitude transmission, broadcastable to the field data.
                Accepted shapes include scalar, ``(ny, nx)``, or
                ``(num_wavelengths, ny, nx)``.
        """
        amp = jnp.asarray(mask, dtype=self.data.real.dtype)
        if self.is_jones and amp.ndim in (2, 3):
            amp = amp[:, None, :, :] if amp.ndim == 3 else amp[None, None, :, :]
        return Field(
            data=self.data * amp,
            grid=self.grid,
            spectrum=self.spectrum,
            polarization_mode=self.polarization_mode,
            domain=self.domain,
            kx_pixel_size_cyc_per_um=self.kx_pixel_size_cyc_per_um,
            ky_pixel_size_cyc_per_um=self.ky_pixel_size_cyc_per_um,
        )

    def to_kspace(self) -> "Field":
        """Return this field in k-space domain.

        Applies a 2-D FFT over the spatial axes. If the field is already in
        k-space the same object is returned unchanged. Grid, spectrum, and
        polarisation metadata are preserved.
        """
        if self.domain == "kspace":
            return self
        data = jnp.fft.fftn(self.data, axes=(-2, -1))
        kx, ky = self.grid.kspace_pixel_size_cyc_per_um()
        return Field(
            data=data,
            grid=self.grid,
            spectrum=self.spectrum,
            polarization_mode=self.polarization_mode,
            domain="kspace",
            kx_pixel_size_cyc_per_um=kx,
            ky_pixel_size_cyc_per_um=ky,
        )

    def to_spatial(self) -> "Field":
        """Return this field in spatial domain.

        Applies a 2-D inverse FFT over the spatial axes. If the field is
        already spatial the same object is returned unchanged. Grid,
        spectrum, and polarisation metadata are preserved.
        """
        if self.domain == "spatial":
            return self
        data = jnp.fft.ifftn(self.data, axes=(-2, -1))
        kx, ky = self.grid.kspace_pixel_size_cyc_per_um()
        return Field(
            data=data,
            grid=self.grid,
            spectrum=self.spectrum,
            polarization_mode=self.polarization_mode,
            domain="spatial",
            kx_pixel_size_cyc_per_um=kx,
            ky_pixel_size_cyc_per_um=ky,
        )

    @property
    def spatial_pixel_size_um(self) -> tuple[float, float]:
        """Spatial sampling pitch ``(dx_um, dy_um)`` in micrometers."""
        return (self.grid.dx_um, self.grid.dy_um)

    @property
    def kspace_pixel_size_cyc_per_um(self) -> tuple[float, float]:
        """Reciprocal-space sampling pitch ``(dkx, dky)`` in cycles per micrometer."""
        if self.kx_pixel_size_cyc_per_um is None or self.ky_pixel_size_cyc_per_um is None:
            raise ValueError("k-space pixel size metadata is not initialized")
        return (
            float(cast(float, self.kx_pixel_size_cyc_per_um)),
            float(cast(float, self.ky_pixel_size_cyc_per_um)),
        )

    def validate(self) -> None:
        """Validate field metadata and the trailing shape convention.

        Scalar fields must have trailing shape
        ``(num_wavelengths, ny, nx)``. Jones fields must have trailing shape
        ``(num_wavelengths, 2, ny, nx)`` with the channel axis ordered as
        ``(Ex, Ey)``.

        Raises:
            ValueError: If the grid or spectrum is invalid, or if ``data`` does
                not match the expected trailing shape for the selected
                polarization mode.
        """
        self.grid.validate()
        self.spectrum.validate()
        expected_shape: tuple[int, ...]
        if self.polarization_mode == "scalar":
            if self.data.ndim < 3:
                raise ValueError(
                    "scalar field data must have shape (*batch, num_wavelengths, ny, nx)"
                )
            expected_shape = (self.spectrum.size, self.grid.ny, self.grid.nx)
        else:
            if self.data.ndim < 4:
                raise ValueError(
                    "jones field data must have shape (*batch, num_wavelengths, 2, ny, nx)"
                )
            expected_shape = (self.spectrum.size, 2, self.grid.ny, self.grid.nx)
        if self.data.shape[-len(expected_shape) :] != expected_shape:
            raise ValueError(
                "field data shape mismatch: got "
                f"{self.data.shape}, expected trailing shape {expected_shape}"
            )
