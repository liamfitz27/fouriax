from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Grid:
    """2D spatial grid with spacing in micrometers (um)."""

    nx: int
    ny: int
    dx_um: float
    dy_um: float

    @classmethod
    def from_extent(cls, nx: int, ny: int, dx_um: float, dy_um: float) -> "Grid":
        grid = cls(nx=nx, ny=ny, dx_um=dx_um, dy_um=dy_um)
        grid.validate()
        return grid

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    def frequency_grid(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return spatial frequency grids in cycles per micrometer."""
        self.validate()
        fx_1d = jnp.fft.fftfreq(self.nx, d=self.dx_um)
        fy_1d = jnp.fft.fftfreq(self.ny, d=self.dy_um)
        fx, fy = jnp.meshgrid(fx_1d, fy_1d, indexing="xy")
        return fx, fy

    def spatial_grid(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return centered spatial grids in micrometers."""
        self.validate()
        x_1d = (jnp.arange(self.nx) - (self.nx - 1) / 2.0) * self.dx_um
        y_1d = (jnp.arange(self.ny) - (self.ny - 1) / 2.0) * self.dy_um
        x, y = jnp.meshgrid(x_1d, y_1d, indexing="xy")
        return x, y

    def kspace_pixel_size_cyc_per_um(self) -> tuple[float, float]:
        """Return k-grid sampling intervals `(dfx, dfy)` in cycles per micrometer."""
        self.validate()
        return (1.0 / (self.nx * self.dx_um), 1.0 / (self.ny * self.dy_um))

    def validate(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("grid sizes nx and ny must be positive integers")
        if self.dx_um <= 0 or self.dy_um <= 0:
            raise ValueError("grid spacings dx_um and dy_um must be positive")


@dataclass(frozen=True)
class Spectrum:
    """Collection of wavelengths in micrometers (um)."""

    wavelengths_um: jnp.ndarray

    @classmethod
    def from_scalar(cls, wavelength_um: float) -> "Spectrum":
        spectrum = cls(wavelengths_um=jnp.asarray([wavelength_um], dtype=jnp.float32))
        spectrum.validate()
        return spectrum

    @classmethod
    def from_array(cls, wavelengths_um: jnp.ndarray) -> "Spectrum":
        spectrum = cls(wavelengths_um=jnp.asarray(wavelengths_um, dtype=jnp.float32))
        spectrum.validate()
        return spectrum

    @property
    def size(self) -> int:
        return int(self.wavelengths_um.shape[0])

    def validate(self) -> None:
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
class Field:
    """
    Complex optical field over a 2D grid for one or more wavelengths.

    Data shape is:
    - scalar mode: `(num_wavelengths, ny, nx)`
    - jones mode: `(num_wavelengths, 2, ny, nx)` with channel order `(Ex, Ey)`
    `domain` indicates whether `data` is represented in spatial or k-space.
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
        return self.polarization_mode == "jones"

    @property
    def num_polarization_channels(self) -> int:
        return 2 if self.is_jones else 1

    def component_intensity(self) -> jnp.ndarray:
        if self.is_jones:
            return jnp.abs(self.data) ** 2
        return (jnp.abs(self.data) ** 2)[:, None, :, :]

    def intensity(self) -> jnp.ndarray:
        return jnp.sum(self.component_intensity(), axis=1)

    def phase(self) -> jnp.ndarray:
        return jnp.angle(self.data)

    def power(self) -> jnp.ndarray:
        return jnp.sum(self.intensity(), axis=(-2, -1))

    def normalize_power(self, target: float = 1.0, eps: float = 1e-12) -> "Field":
        current = self.power()
        if not bool(jnp.all(current > eps)):
            raise ValueError("cannot normalize field with near-zero power")
        scale = jnp.sqrt(jnp.asarray(target, dtype=current.dtype) / current)
        if self.is_jones:
            scale = scale[:, None, None, None]
        else:
            scale = scale[:, None, None]
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
        """Return this field in k-space domain."""
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
        """Return this field in spatial domain."""
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
        return (self.grid.dx_um, self.grid.dy_um)

    @property
    def kspace_pixel_size_cyc_per_um(self) -> tuple[float, float]:
        if self.kx_pixel_size_cyc_per_um is None or self.ky_pixel_size_cyc_per_um is None:
            raise ValueError("k-space pixel size metadata is not initialized")
        return (
            float(cast(float, self.kx_pixel_size_cyc_per_um)),
            float(cast(float, self.ky_pixel_size_cyc_per_um)),
        )

    def validate(self) -> None:
        self.grid.validate()
        self.spectrum.validate()
        expected_shape: tuple[int, ...]
        if self.polarization_mode == "scalar":
            if self.data.ndim != 3:
                raise ValueError(
                    "scalar field data must have shape (num_wavelengths, ny, nx)"
                )
            expected_shape = (self.spectrum.size, self.grid.ny, self.grid.nx)
        else:
            if self.data.ndim != 4:
                raise ValueError(
                    "jones field data must have shape (num_wavelengths, 2, ny, nx)"
                )
            expected_shape = (self.spectrum.size, 2, self.grid.ny, self.grid.nx)
        if self.data.shape != expected_shape:
            raise ValueError(
                f"field data shape mismatch: got {self.data.shape}, expected {expected_shape}"
            )
