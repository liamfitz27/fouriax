from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np

from fouriax.optics.interfaces import OpticalLayer
from fouriax.optics.model import Field


@dataclass(frozen=True)
class MetaAtomLibrary:
    """Regular-grid meta-atom transmission library over wavelength and geometry axes."""

    wavelengths_um: jnp.ndarray
    parameter_axes: tuple[jnp.ndarray, ...]
    transmission_real: jnp.ndarray
    transmission_imag: jnp.ndarray

    @classmethod
    def from_complex(
        cls,
        wavelengths_um: jnp.ndarray,
        parameter_axes: tuple[jnp.ndarray, ...],
        transmission_complex: jnp.ndarray,
    ) -> "MetaAtomLibrary":
        transmission = jnp.asarray(transmission_complex)
        return cls(
            wavelengths_um=jnp.asarray(wavelengths_um, dtype=jnp.float32),
            parameter_axes=tuple(jnp.asarray(axis, dtype=jnp.float32) for axis in parameter_axes),
            transmission_real=jnp.asarray(transmission.real, dtype=jnp.float32),
            transmission_imag=jnp.asarray(transmission.imag, dtype=jnp.float32),
        )

    @property
    def num_parameters(self) -> int:
        return len(self.parameter_axes)

    @property
    def parameter_shape(self) -> tuple[int, ...]:
        return tuple(int(axis.shape[0]) for axis in self.parameter_axes)

    def validate(self) -> None:
        if self.wavelengths_um.ndim != 1:
            raise ValueError("wavelengths_um must be a 1D array")
        if self.wavelengths_um.size == 0:
            raise ValueError("wavelengths_um must contain at least one value")

        if not self.parameter_axes:
            raise ValueError("parameter_axes must include at least one geometry axis")
        for i, axis in enumerate(self.parameter_axes):
            if axis.ndim != 1:
                raise ValueError(f"parameter axis {i} must be 1D")
            if axis.shape[0] < 2:
                raise ValueError(f"parameter axis {i} must have at least 2 points")
            axis_np = np.asarray(axis)
            if np.any(np.diff(axis_np) <= 0):
                raise ValueError(f"parameter axis {i} must be strictly increasing")

        expected_shape = (self.wavelengths_um.shape[0], *self.parameter_shape)
        if self.transmission_real.shape != expected_shape:
            raise ValueError(
                "transmission_real shape mismatch: "
                f"got {self.transmission_real.shape}, expected {expected_shape}"
            )
        if self.transmission_imag.shape != expected_shape:
            raise ValueError(
                "transmission_imag shape mismatch: "
                f"got {self.transmission_imag.shape}, expected {expected_shape}"
            )

    def transmission_complex(self) -> jnp.ndarray:
        return self.transmission_real + 1j * self.transmission_imag

    def _multilinear_interpolate_parameter_grid(
        self,
        values: jnp.ndarray,
        x_flat: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Interpolate `values` over geometry axes for multiple geometry points.

        Args:
            values: shape (num_wavelengths, p1, p2, ..., pk)
            x_flat: shape (n_points, k) geometry params in physical units.

        Returns:
            shape (n_points, num_wavelengths)
        """
        if x_flat.ndim != 2 or x_flat.shape[1] != self.num_parameters:
            raise ValueError(
                f"geometry parameter shape mismatch: got {x_flat.shape}, "
                f"expected (n_points, {self.num_parameters})"
            )

        n_points = x_flat.shape[0]
        lower_indices: list[jnp.ndarray] = []
        upper_indices: list[jnp.ndarray] = []
        fractions: list[jnp.ndarray] = []

        for dim, axis in enumerate(self.parameter_axes):
            xi = x_flat[:, dim]
            idx_hi = jnp.searchsorted(axis, xi, side="right")
            idx_lo = jnp.clip(idx_hi - 1, 0, axis.shape[0] - 2)
            idx_hi = idx_lo + 1

            x_lo = axis[idx_lo]
            x_hi = axis[idx_hi]
            frac = (xi - x_lo) / (x_hi - x_lo)

            lower_indices.append(idx_lo)
            upper_indices.append(idx_hi)
            fractions.append(frac)

        parameter_shape = self.parameter_shape
        flat_values = values.reshape((values.shape[0], int(np.prod(parameter_shape))))

        strides: list[int] = []
        running = 1
        for size in reversed(parameter_shape[1:]):
            running *= size
            strides.insert(0, running)
        strides.append(1)

        out = jnp.zeros((values.shape[0], n_points), dtype=values.dtype)
        for bits in product((0, 1), repeat=self.num_parameters):
            weight = jnp.ones((n_points,), dtype=values.dtype)
            flat_index = jnp.zeros((n_points,), dtype=jnp.int32)
            for dim, bit in enumerate(bits):
                idx = upper_indices[dim] if bit else lower_indices[dim]
                frac = fractions[dim]
                weight = weight * (frac if bit else (1.0 - frac))
                flat_index = flat_index + idx.astype(jnp.int32) * jnp.int32(strides[dim])
            sampled = jnp.take(flat_values, flat_index, axis=1)
            out = out + sampled * weight[None, :]

        return out.T

    def interpolate_complex(
        self,
        geometry_params: jnp.ndarray,
        wavelengths_um: jnp.ndarray,
    ) -> jnp.ndarray:
        """Interpolate complex transmission at geometry parameters and requested wavelengths."""
        self.validate()

        geom = jnp.asarray(geometry_params, dtype=jnp.float32)
        wav = jnp.asarray(wavelengths_um, dtype=jnp.float32)
        if wav.ndim != 1:
            raise ValueError("wavelengths_um must be 1D")
        if geom.ndim < 1 or geom.shape[-1] != self.num_parameters:
            raise ValueError(
                "geometry_params must have trailing dimension equal to library.num_parameters"
            )

        orig_shape = geom.shape[:-1]
        x_flat = geom.reshape((-1, self.num_parameters))

        real_at_library_wavelengths = self._multilinear_interpolate_parameter_grid(
            self.transmission_real,
            x_flat,
        )
        imag_at_library_wavelengths = self._multilinear_interpolate_parameter_grid(
            self.transmission_imag,
            x_flat,
        )

        def interp_one(fp: jnp.ndarray) -> jnp.ndarray:
            return jnp.interp(wav, self.wavelengths_um, fp)

        real_interp = jax.vmap(interp_one)(real_at_library_wavelengths)
        imag_interp = jax.vmap(interp_one)(imag_at_library_wavelengths)

        out = real_interp + 1j * imag_interp
        return out.reshape((*orig_shape, wav.shape[0]))


@dataclass(frozen=True)
class MetaAtomInterpolationLayer(OpticalLayer):
    """Layer that applies a geometry-parameterized meta-atom transmission model."""

    library: MetaAtomLibrary
    raw_geometry_params: jnp.ndarray
    min_geometry_params: jnp.ndarray
    max_geometry_params: jnp.ndarray

    def bounded_geometry_params(self) -> jnp.ndarray:
        """Map unconstrained parameters into physical bounds via sigmoid."""
        raw = jnp.asarray(self.raw_geometry_params, dtype=jnp.float32)
        min_v = jnp.asarray(self.min_geometry_params, dtype=jnp.float32)
        max_v = jnp.asarray(self.max_geometry_params, dtype=jnp.float32)

        if min_v.ndim != 1 or max_v.ndim != 1:
            raise ValueError("min/max geometry bounds must be 1D arrays")
        if min_v.shape != max_v.shape:
            raise ValueError("min/max geometry bounds must have the same shape")
        if min_v.shape[0] != self.library.num_parameters:
            raise ValueError("min/max geometry bounds length must match library parameter count")

        if raw.ndim == 1:
            if raw.shape[0] != self.library.num_parameters:
                raise ValueError(
                    "raw_geometry_params length must match library parameter axes count"
                )
            return min_v + (max_v - min_v) * jax.nn.sigmoid(raw)

        if raw.ndim >= 2:
            if raw.shape[0] == self.library.num_parameters:
                raw_param_first = raw
            elif self.library.num_parameters == 1:
                raw_param_first = raw[None, ...]
            else:
                raise ValueError(
                    "raw_geometry_params map must have leading parameter dimension"
                )
            min_map = min_v[:, None, None]
            max_map = max_v[:, None, None]
            return min_map + (max_map - min_map) * jax.nn.sigmoid(raw_param_first)

        raise ValueError("raw_geometry_params must be 1D or parameter map")

    def forward(self, field: Field) -> Field:
        self.validate_for(field)

        geometry = self.bounded_geometry_params()

        if geometry.ndim == 1:
            transmission = self.library.interpolate_complex(
                geometry_params=geometry[None, :],
                wavelengths_um=field.spectrum.wavelengths_um,
            )[0]
            modulation = transmission[:, None, None].astype(field.data.dtype)
            return Field(
                data=field.data * modulation,
                grid=field.grid,
                spectrum=field.spectrum,
                domain=field.domain,
                kx_pixel_size_cyc_per_um=field.kx_pixel_size_cyc_per_um,
                ky_pixel_size_cyc_per_um=field.ky_pixel_size_cyc_per_um,
            )

        if geometry.shape[1:] != (field.grid.ny, field.grid.nx):
            raise ValueError(
                "geometry map shape mismatch: "
                f"got {geometry.shape[1:]}, expected {(field.grid.ny, field.grid.nx)}"
            )

        geometry_points = jnp.moveaxis(geometry, 0, -1)
        transmission = self.library.interpolate_complex(
            geometry_params=geometry_points,
            wavelengths_um=field.spectrum.wavelengths_um,
        )
        modulation = jnp.moveaxis(transmission, -1, 0).astype(field.data.dtype)
        return Field(
            data=field.data * modulation,
            grid=field.grid,
            spectrum=field.spectrum,
            domain=field.domain,
            kx_pixel_size_cyc_per_um=field.kx_pixel_size_cyc_per_um,
            ky_pixel_size_cyc_per_um=field.ky_pixel_size_cyc_per_um,
        )

    def validate_for(self, field: Field) -> None:
        super().validate_for(field)
        self.library.validate()

        wav = np.asarray(field.spectrum.wavelengths_um)
        lib_wav = np.asarray(self.library.wavelengths_um)
        if wav.min() < lib_wav.min() or wav.max() > lib_wav.max():
            raise ValueError(
                "field wavelengths fall outside library wavelength range; "
                f"field range=({wav.min()}, {wav.max()}), "
                f"library range=({lib_wav.min()}, {lib_wav.max()})"
            )

    def parameters(self) -> dict[str, jnp.ndarray]:
        return {
            "raw_geometry_params": jnp.asarray(self.raw_geometry_params, dtype=jnp.float32),
            "bounded_geometry_params": self.bounded_geometry_params(),
            "min_geometry_params": jnp.asarray(self.min_geometry_params, dtype=jnp.float32),
            "max_geometry_params": jnp.asarray(self.max_geometry_params, dtype=jnp.float32),
        }
