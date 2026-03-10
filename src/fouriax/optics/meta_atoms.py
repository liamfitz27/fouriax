from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from fouriax.optics.interfaces import OpticalLayer
from fouriax.optics.model import Field


@dataclass(frozen=True)
class MetaAtomLibrary:
    """Regular-grid meta-atom transmission library over wavelength and geometry axes.

    Stores complex transmission coefficients on a structured grid of
    wavelengths and one or more geometry parameter axes (e.g. pillar
    width, height).  Transmission at arbitrary geometry/wavelength
    points is obtained by multilinear interpolation.

    Args:
        wavelengths_um: 1-D array of library wavelengths in micrometers,
            used as the spectral interpolation axis.
        parameter_axes: Tuple of 1-D arrays, each defining a strictly
            increasing geometry axis in physical units.
        transmission_real: Real part of the complex transmission with
            shape ``(num_wavelengths, p1, p2, ..., pk)``.
        transmission_imag: Imaginary part, same shape as
            ``transmission_real``.
    """

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
        """Build a library from a complex-valued transmission array.

        Args:
            wavelengths_um: 1-D wavelength array in micrometers.
            parameter_axes: Tuple of 1-D geometry axes.
            transmission_complex: Complex array with shape
                ``(num_wavelengths, p1, p2, ..., pk)``.

        Returns:
            Validated library contents split into real and imaginary tensors.
        """
        transmission = jnp.asarray(transmission_complex)
        return cls(
            wavelengths_um=jnp.asarray(wavelengths_um, dtype=jnp.float32),
            parameter_axes=tuple(jnp.asarray(axis, dtype=jnp.float32) for axis in parameter_axes),
            transmission_real=jnp.asarray(transmission.real, dtype=jnp.float32),
            transmission_imag=jnp.asarray(transmission.imag, dtype=jnp.float32),
        )

    @property
    def num_parameters(self) -> int:
        """Number of geometry parameters spanned by the library."""
        return len(self.parameter_axes)

    @property
    def parameter_shape(self) -> tuple[int, ...]:
        """Grid size along each geometry-parameter axis."""
        return tuple(int(axis.shape[0]) for axis in self.parameter_axes)

    def validate(self) -> None:
        """Validate wavelength axes, geometry axes, and transmission tensor shapes.

        Raises:
            ValueError: If axis dimensionality, monotonicity, or transmission
                shapes do not match the library definition.
        """
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
        """Reconstruct the full complex transmission array.

        Returns:
            Complex array with shape ``(num_wavelengths, p1, p2, ..., pk)``.
        """
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
        """Interpolate complex transmission at arbitrary geometry and wavelength points.

        Geometry interpolation is multilinear over the library's parameter
        grid; spectral interpolation is linear between library wavelengths.

        Args:
            geometry_params: Geometry coordinates with trailing dimension
                equal to ``num_parameters``.  Spatial maps of shape
                ``(ny, nx, num_parameters)`` are supported.
            wavelengths_um: 1-D array of query wavelengths in micrometers.

        Returns:
            Complex transmission with shape ``(*spatial_dims, num_query_wavelengths)``.
        """
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
    """Optical layer applying a geometry-parameterised meta-atom transmission.

    Raw (unconstrained) geometry parameters are mapped into physical
    bounds via a sigmoid transform, then looked up in the attached
    :class:`MetaAtomLibrary` to obtain per-pixel complex transmission.

    Supports two polarisation modes:

    - ``"scalar"``: a single set of geometry parameters modulates both
      polarisation channels identically.
    - ``"jones_diagonal"``: independent geometry parameters for the
      ``Ex`` and ``Ey`` channels, producing a diagonal Jones
      transmission.

    Args:
        library: Pre-built meta-atom transmission library.
        raw_geometry_params: Unconstrained optimisation variables.
            Shape depends on mode and whether parameters are global or
            spatially varying.
        min_geometry_params: 1-D lower bounds for each geometry axis.
        max_geometry_params: 1-D upper bounds for each geometry axis.
        polarization_mode: ``"scalar"`` or ``"jones_diagonal"``.
    """

    library: MetaAtomLibrary
    raw_geometry_params: jnp.ndarray
    min_geometry_params: jnp.ndarray
    max_geometry_params: jnp.ndarray
    polarization_mode: Literal["scalar", "jones_diagonal"] = "scalar"

    def _bounded_geometry_params_scalar(self, raw: jnp.ndarray) -> jnp.ndarray:
        """Map scalar-mode raw parameters into bounded physical geometry values."""
        min_v = jnp.asarray(self.min_geometry_params, dtype=jnp.float32)
        max_v = jnp.asarray(self.max_geometry_params, dtype=jnp.float32)
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

    def _bounded_geometry_params_jones_diagonal(self, raw: jnp.ndarray) -> jnp.ndarray:
        """Map Jones-diagonal raw parameters into bounded physical geometry values."""
        min_v = jnp.asarray(self.min_geometry_params, dtype=jnp.float32)
        max_v = jnp.asarray(self.max_geometry_params, dtype=jnp.float32)
        num_p = self.library.num_parameters
        if raw.ndim == 2:
            if raw.shape != (2, num_p):
                raise ValueError(
                    "raw_geometry_params must have shape (2, num_parameters) for "
                    "jones_diagonal global mode"
                )
            return min_v[None, :] + (max_v - min_v)[None, :] * jax.nn.sigmoid(raw)
        if raw.ndim == 3 and num_p == 1:
            raw_param = raw[:, None, :, :]
        elif raw.ndim == 4:
            raw_param = raw
        else:
            raise ValueError(
                "raw_geometry_params must have shape (2, num_parameters, ny, nx) for "
                "jones_diagonal map mode"
            )
        if raw_param.shape[:2] != (2, num_p):
            raise ValueError(
                "raw_geometry_params must have leading axes (2, num_parameters) in "
                "jones_diagonal mode"
            )
        min_map = min_v[None, :, None, None]
        max_map = max_v[None, :, None, None]
        return min_map + (max_map - min_map) * jax.nn.sigmoid(raw_param)

    def bounded_geometry_params(self) -> jnp.ndarray:
        """Map unconstrained raw parameters into physical bounds via sigmoid.

        Returns:
            Bounded geometry parameters in the same layout as
            ``raw_geometry_params``.
        """
        raw = jnp.asarray(self.raw_geometry_params, dtype=jnp.float32)
        min_v = jnp.asarray(self.min_geometry_params, dtype=jnp.float32)
        max_v = jnp.asarray(self.max_geometry_params, dtype=jnp.float32)
        if min_v.ndim != 1 or max_v.ndim != 1:
            raise ValueError("min/max geometry bounds must be 1D arrays")
        if min_v.shape != max_v.shape:
            raise ValueError("min/max geometry bounds must have the same shape")
        if min_v.shape[0] != self.library.num_parameters:
            raise ValueError("min/max geometry bounds length must match library parameter count")
        if self.polarization_mode == "scalar":
            return self._bounded_geometry_params_scalar(raw)
        if self.polarization_mode == "jones_diagonal":
            return self._bounded_geometry_params_jones_diagonal(raw)
        raise ValueError("polarization_mode must be one of: scalar, jones_diagonal")

    def forward(self, field: Field) -> Field:
        """Apply interpolated meta-atom transmission to the input field.

        Args:
            field: Scalar field for ``polarization_mode="scalar"`` or Jones
                field for ``polarization_mode="jones_diagonal"``.

        Returns:
            Field with the same grid, spectrum, domain, and batch axes as the
            input, modulated by the interpolated complex transmission.
        """
        self.validate_for(field)
        geometry = self.bounded_geometry_params()
        if self.polarization_mode == "jones_diagonal":
            if not field.is_jones:
                raise ValueError(
                    "MetaAtomInterpolationLayer with polarization_mode='jones_diagonal' "
                    "requires a jones field"
                )
            if geometry.ndim == 2:
                tx = self.library.interpolate_complex(
                    geometry_params=geometry[0][None, :],
                    wavelengths_um=field.spectrum.wavelengths_um,
                )[0]
                ty = self.library.interpolate_complex(
                    geometry_params=geometry[1][None, :],
                    wavelengths_um=field.spectrum.wavelengths_um,
                )[0]
                out_x = field.data[..., 0, :, :] * tx[:, None, None].astype(field.data.dtype)
                out_y = field.data[..., 1, :, :] * ty[:, None, None].astype(field.data.dtype)
            else:
                if geometry.ndim != 4 or geometry.shape[0] != 2:
                    raise ValueError(
                        "geometry shape mismatch for jones_diagonal map mode"
                    )
                if geometry.shape[2:] != (field.grid.ny, field.grid.nx):
                    raise ValueError(
                        "geometry map shape mismatch: got "
                        f"{geometry.shape[2:]}, expected {(field.grid.ny, field.grid.nx)}"
                    )
                geom_x = jnp.moveaxis(geometry[0], 0, -1)
                geom_y = jnp.moveaxis(geometry[1], 0, -1)
                tx = self.library.interpolate_complex(
                    geometry_params=geom_x,
                    wavelengths_um=field.spectrum.wavelengths_um,
                )
                ty = self.library.interpolate_complex(
                    geometry_params=geom_y,
                    wavelengths_um=field.spectrum.wavelengths_um,
                )
                tx_mod = jnp.moveaxis(tx, -1, 0).astype(field.data.dtype)
                ty_mod = jnp.moveaxis(ty, -1, 0).astype(field.data.dtype)
                out_x = field.data[..., 0, :, :] * tx_mod
                out_y = field.data[..., 1, :, :] * ty_mod
            return Field(
                data=jnp.stack([out_x, out_y], axis=-3),
                grid=field.grid,
                spectrum=field.spectrum,
                polarization_mode=field.polarization_mode,
                domain=field.domain,
                kx_pixel_size_cyc_per_um=field.kx_pixel_size_cyc_per_um,
                ky_pixel_size_cyc_per_um=field.ky_pixel_size_cyc_per_um,
            )

        if geometry.ndim == 1:
            transmission = self.library.interpolate_complex(
                geometry_params=geometry[None, :],
                wavelengths_um=field.spectrum.wavelengths_um,
            )[0]
            modulation = transmission[:, None, None].astype(field.data.dtype)
            if field.is_jones:
                modulation = modulation[:, None, :, :]
            return Field(
                data=field.data * modulation,
                grid=field.grid,
                spectrum=field.spectrum,
                polarization_mode=field.polarization_mode,
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
        if field.is_jones:
            modulation = modulation[:, None, :, :]
        return Field(
            data=field.data * modulation,
            grid=field.grid,
            spectrum=field.spectrum,
            polarization_mode=field.polarization_mode,
            domain=field.domain,
            kx_pixel_size_cyc_per_um=field.kx_pixel_size_cyc_per_um,
            ky_pixel_size_cyc_per_um=field.ky_pixel_size_cyc_per_um,
        )

    def validate_for(self, field: Field) -> None:
        """Validate field compatibility and library wavelength coverage."""
        super().validate_for(field)
        self.library.validate()
        if self.polarization_mode not in ("scalar", "jones_diagonal"):
            raise ValueError("polarization_mode must be one of: scalar, jones_diagonal")
        if self.polarization_mode == "jones_diagonal" and not field.is_jones:
            raise ValueError(
                "MetaAtomInterpolationLayer with polarization_mode='jones_diagonal' "
                "requires a jones field"
            )

        wav = np.asarray(field.spectrum.wavelengths_um)
        lib_wav = np.asarray(self.library.wavelengths_um)
        if wav.min() < lib_wav.min() or wav.max() > lib_wav.max():
            raise ValueError(
                "field wavelengths fall outside library wavelength range; "
                f"field range=({wav.min()}, {wav.max()}), "
                f"library range=({lib_wav.min()}, {lib_wav.max()})"
            )

    def parameters(self) -> dict[str, jnp.ndarray]:
        """Return raw and bounded geometry parameters plus optimization bounds."""
        return {
            "raw_geometry_params": jnp.asarray(self.raw_geometry_params, dtype=jnp.float32),
            "bounded_geometry_params": self.bounded_geometry_params(),
            "min_geometry_params": jnp.asarray(self.min_geometry_params, dtype=jnp.float32),
            "max_geometry_params": jnp.asarray(self.max_geometry_params, dtype=jnp.float32),
            "polarization_mode": jnp.asarray(
                0 if self.polarization_mode == "scalar" else 1, dtype=jnp.int32
            ),
        }
