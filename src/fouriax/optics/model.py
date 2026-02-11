from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


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
        if not bool(jnp.all(self.wavelengths_um > 0)):
            raise ValueError("all wavelengths_um must be strictly positive")


@dataclass(frozen=True)
class Field:
    """
    Complex optical field over a 2D grid for one or more wavelengths.

    Data shape is `(num_wavelengths, ny, nx)`.
    """

    data: jnp.ndarray
    grid: Grid
    spectrum: Spectrum

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

    def intensity(self) -> jnp.ndarray:
        return jnp.abs(self.data) ** 2

    def phase(self) -> jnp.ndarray:
        return jnp.angle(self.data)

    def power(self) -> jnp.ndarray:
        return jnp.sum(self.intensity(), axis=(-2, -1))

    def normalize_power(self, target: float = 1.0, eps: float = 1e-12) -> "Field":
        current = self.power()
        if not bool(jnp.all(current > eps)):
            raise ValueError("cannot normalize field with near-zero power")
        scale = jnp.sqrt(jnp.asarray(target, dtype=current.dtype) / current)
        scale = scale[:, None, None]
        return Field(data=self.data * scale, grid=self.grid, spectrum=self.spectrum)

    def apply_phase(self, phase_map: jnp.ndarray | float) -> "Field":
        phs = jnp.asarray(phase_map, dtype=jnp.float32)
        return Field(
            data=self.data * jnp.exp(1j * phs),
            grid=self.grid,
            spectrum=self.spectrum,
        )

    def apply_amplitude(self, mask: jnp.ndarray | float) -> "Field":
        amp = jnp.asarray(mask, dtype=self.data.real.dtype)
        return Field(data=self.data * amp, grid=self.grid, spectrum=self.spectrum)

    def validate(self) -> None:
        self.grid.validate()
        self.spectrum.validate()
        if self.data.ndim != 3:
            raise ValueError("field data must have shape (num_wavelengths, ny, nx)")
        expected_shape = (self.spectrum.size, self.grid.ny, self.grid.nx)
        if self.data.shape != expected_shape:
            raise ValueError(
                f"field data shape mismatch: got {self.data.shape}, expected {expected_shape}"
            )
