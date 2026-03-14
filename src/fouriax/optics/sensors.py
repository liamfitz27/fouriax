from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from fouriax.linop import LinearOperator
from fouriax.optics.interfaces import Sensor
from fouriax.optics.layers import AmplitudeMask
from fouriax.optics.model import Field, Grid, Intensity
from fouriax.optics.noise import SensorNoiseModel


def _apply_optional_noise(
    measured: jnp.ndarray,
    *,
    noise_model: SensorNoiseModel | None,
    key: jax.Array | None,
) -> jnp.ndarray:
    if noise_model is None or key is None:
        return measured
    return noise_model.sample(measured, key=key)


def _pixel_edges(grid: Grid, *, axis: Literal["x", "y"]) -> jnp.ndarray:
    n = grid.nx if axis == "x" else grid.ny
    d = grid.dx_um if axis == "x" else grid.dy_um
    return (jnp.arange(n + 1, dtype=jnp.float32) - (n / 2.0)) * d


def _overlap_fraction_matrix(src_edges: jnp.ndarray, dst_edges: jnp.ndarray) -> jnp.ndarray:
    src_left = src_edges[:-1][None, :]
    src_right = src_edges[1:][None, :]
    dst_left = dst_edges[:-1][:, None]
    dst_right = dst_edges[1:][:, None]
    overlap = jnp.maximum(
        0.0,
        jnp.minimum(dst_right, src_right) - jnp.maximum(dst_left, src_left),
    )
    src_width = jnp.maximum(src_right - src_left, 1e-12)
    return overlap / src_width


@dataclass(frozen=True)
class Detector(Sensor):
    """Intensity detector integrated over a single region of the field grid.

    Computes total intensity within an optional spatial mask, optionally
    summing over wavelengths and resolving polarisation channels.

    The default output shape is a scalar per batch element (wavelengths
    summed, channels merged).  Set ``sum_wavelengths=False`` to retain
    the wavelength axis.  Set ``channel_resolved=True`` with Jones input
    to preserve the ``(Ex, Ey)`` split.

    Args:
        region_mask: Spatial weighting mask with shape ``(ny, nx)``, or
            ``None`` for a uniform mask covering the whole grid.
        sum_wavelengths: Sum over the wavelength axis in the output.
        channel_resolved: Preserve per-channel intensities for Jones
            fields.
        noise_model: Optional stochastic noise model applied during
            sampling.
    """

    region_mask: jnp.ndarray | None = None
    sum_wavelengths: bool = True
    channel_resolved: bool = False
    noise_model: SensorNoiseModel | None = None

    def measure(self, field: Field | Intensity, *, key: jax.Array | None = None) -> jnp.ndarray:
        """Return the detector output, optionally with sampled noise."""
        measured = self.expected(field)
        return _apply_optional_noise(measured, noise_model=self.noise_model, key=key)

    def expected(self, field: Field | Intensity) -> jnp.ndarray:
        """Return the deterministic detector signal before noise sampling.

        Args:
            field: Input field or intensity. Field input is converted to spatial
                intensity before integration.

        Returns:
            Array with shape ``(*batch,)``, ``(*batch, num_wavelengths)``,
            ``(*batch, 2)``, or ``(*batch, num_wavelengths, 2)`` depending on
            ``sum_wavelengths`` and ``channel_resolved``.
        """
        self.validate_for(field)
        mask = self._resolved_region_mask(field.grid)
        if isinstance(field, Field):
            field_spatial = field.to_spatial()
            if self.channel_resolved and field_spatial.is_jones:
                measured = jnp.einsum(
                    "...wcxy,xy->...wc",
                    field_spatial.component_intensity(),
                    mask,
                )
                return jnp.sum(measured, axis=-2) if self.sum_wavelengths else measured
            intensity = field_spatial.to_intensity()
        else:
            intensity = field

        measured = jnp.einsum("...wxy,xy->...w", intensity.data, mask)
        return jnp.sum(measured, axis=-1) if self.sum_wavelengths else measured

    def sample(self, field: Field | Intensity, *, key: jax.Array) -> jnp.ndarray:
        """Sample the detector output with the configured stochastic noise model."""
        return _apply_optional_noise(
            self.expected(field),
            noise_model=self.noise_model,
            key=key,
        )

    def _resolved_region_mask(self, grid: Grid) -> jnp.ndarray:
        if self.region_mask is None:
            return jnp.ones(grid.shape, dtype=jnp.float32)

        mask = jnp.asarray(self.region_mask, dtype=jnp.float32)
        if mask.shape != (grid.ny, grid.nx):
            raise ValueError(
                "region_mask must have shape "
                f"{(grid.ny, grid.nx)}, got {mask.shape}"
            )
        return mask


@dataclass(frozen=True)
class DetectorArray(Sensor):
    """Grid-based detector array with optional QE weighting and noise.

    Integrates field intensity onto a coarser ``detector_grid`` by
    binning field pixels into detector super-pixels.  Supports per-
    wavelength quantum-efficiency curves, an optional pre-detection
    amplitude filter, and stochastic noise models.

    Output shape is ``(*batch, [num_wavelengths,] [2,] det_ny, det_nx)``
    depending on ``sum_wavelengths`` and ``channel_resolved``.

    Args:
        detector_grid: Detector pixel grid (typically coarser than the
            field grid).
        qe_curve: Quantum efficiency — scalar, 1-D array of shape
            ``(num_wavelengths,)``, or ``None``.
        filter_mask: Optional ``AmplitudeMask`` applied to the intensity
            before integration.
        sum_wavelengths: Sum over the wavelength axis.
        channel_resolved: Preserve per-polarisation channel intensities.
        resample_method: Interpolation method for detector binning.
        noise_model: Optional stochastic noise model.
    """

    detector_grid: Grid
    qe_curve: jnp.ndarray | float | None = 1.0
    filter_mask: AmplitudeMask | None = None
    sum_wavelengths: bool = True
    channel_resolved: bool = False
    resample_method: Literal["nearest", "linear"] = "linear"
    noise_model: SensorNoiseModel | None = None

    def measure(self, field: Field | Intensity, *, key: jax.Array | None = None) -> jnp.ndarray:
        """Return the detector-array readout, optionally with sampled noise."""
        measured = self.expected(field)
        return _apply_optional_noise(measured, noise_model=self.noise_model, key=key)

    def expected(self, field: Field | Intensity) -> jnp.ndarray:
        """Return the deterministic detector-array signal before noise.

        Args:
            field: Input field or intensity. Field input is converted to spatial
                intensity before integration.

        Returns:
            Array with shape ``(*batch, det_ny, det_nx)``,
            ``(*batch, num_wavelengths, det_ny, det_nx)``,
            ``(*batch, 2, det_ny, det_nx)``, or
            ``(*batch, num_wavelengths, 2, det_ny, det_nx)`` depending on
            ``sum_wavelengths`` and ``channel_resolved``.
        """
        self.validate_for(field)
        if isinstance(field, Field):
            field_spatial = field.to_spatial()
            channel_resolved = self.channel_resolved and field_spatial.is_jones
            if channel_resolved:
                intensity = field_spatial.component_intensity()
            else:
                intensity = field_spatial.to_intensity().data
            grid = field_spatial.grid
            spectrum_size = field_spatial.spectrum.size
            dtype = field_spatial.data.real.dtype
        else:
            channel_resolved = False
            intensity = field.data
            grid = field.grid
            spectrum_size = field.spectrum.size
            dtype = field.data.dtype

        return self._apply_readout(
            intensity,
            field_grid=grid,
            spectrum_size=spectrum_size,
            dtype=dtype,
            channel_resolved=channel_resolved,
        )

    def sample(self, field: Field | Intensity, *, key: jax.Array) -> jnp.ndarray:
        """Sample the detector-array output with the configured noise model."""
        return _apply_optional_noise(
            self.expected(field),
            noise_model=self.noise_model,
            key=key,
        )

    def _apply_readout(
        self,
        intensity: jnp.ndarray,
        *,
        field_grid: Grid,
        spectrum_size: int,
        dtype: jnp.dtype,
        channel_resolved: bool,
    ) -> jnp.ndarray:
        """Apply deterministic detector readout to intensity data."""
        intensity_det = self._integrate_intensity(intensity, field_grid)
        weight = self._detector_intensity_weight(
            spectrum_size=spectrum_size,
            dtype=dtype,
            detector_grid=self.detector_grid,
        )
        if channel_resolved:
            intensity_det = intensity_det * weight[:, None, :, :]
        else:
            intensity_det = intensity_det * weight

        return (
            jnp.sum(intensity_det, axis=-4 if channel_resolved else -3)
            if self.sum_wavelengths
            else intensity_det
        )

    def linear_operator(
        self,
        template: Intensity,
        *,
        flatten: bool = False,
    ) -> LinearOperator:
        """Return the deterministic intensity-readout operator for this detector.

        The operator corresponds to :meth:`expected` on ``Intensity`` input and
        does not include stochastic noise sampling.
        """
        self.validate_for(template)
        if template.data.ndim != 3:
            raise ValueError(
                "linear_operator currently requires an unbatched template intensity with shape "
                "(num_wavelengths, ny, nx)"
            )

        in_tensor_shape: tuple[int, int, int] = (
            template.spectrum.size,
            template.grid.ny,
            template.grid.nx,
        )
        out_tensor_shape: tuple[int, ...]
        if self.sum_wavelengths:
            out_tensor_shape = (self.detector_grid.ny, self.detector_grid.nx)
        else:
            out_tensor_shape = (
                template.spectrum.size,
                self.detector_grid.ny,
                self.detector_grid.nx,
            )

        def _forward_tensor(x: jax.Array) -> jax.Array:
            return self._apply_readout(
                x,
                field_grid=template.grid,
                spectrum_size=template.spectrum.size,
                dtype=template.data.dtype,
                channel_resolved=False,
            )

        in_shape: tuple[int, ...]
        out_shape: tuple[int, ...]
        if flatten:
            in_shape = (math.prod(in_tensor_shape),)
            out_shape = (math.prod(out_tensor_shape),)

            def matvec_fn(x: jax.Array) -> jax.Array:
                return _forward_tensor(x.reshape(in_tensor_shape)).reshape(out_shape)

            def rmatvec_fn(y: jax.Array) -> jax.Array:
                (out,) = jax.linear_transpose(
                    _forward_tensor,
                    jnp.zeros(in_tensor_shape, dtype=template.data.dtype),
                )(y.reshape(out_tensor_shape))
                return jnp.asarray(out, dtype=template.data.dtype).reshape(in_shape)
        else:
            in_shape = in_tensor_shape
            out_shape = out_tensor_shape

            def matvec_fn(x: jax.Array) -> jax.Array:
                return _forward_tensor(x)

            def rmatvec_fn(y: jax.Array) -> jax.Array:
                (out,) = jax.linear_transpose(
                    _forward_tensor,
                    jnp.zeros(in_tensor_shape, dtype=template.data.dtype),
                )(y)
                return jnp.asarray(out, dtype=template.data.dtype)

        return LinearOperator(
            in_shape=in_shape,
            out_shape=out_shape,
            in_dtype=template.data.dtype,
            out_dtype=template.data.dtype,
            matvec_fn=matvec_fn,
            rmatvec_fn=rmatvec_fn,
        )

    def _integrate_intensity(self, intensity: jnp.ndarray, field_grid: Grid) -> jnp.ndarray:
        if self.resample_method == "linear":
            return self._resample_intensity_linear(intensity, field_grid)
        if self.resample_method != "nearest":
            raise ValueError("resample_method must be one of: nearest, linear")
        x_idx, y_idx, valid = self._detector_bin_indices(field_grid)
        n_det = self.detector_grid.nx * self.detector_grid.ny
        flat_idx = y_idx * self.detector_grid.nx + x_idx
        flat_idx = jnp.where(valid, flat_idx, 0)
        leading_shape = intensity.shape[:-2]
        values = intensity.reshape((-1, field_grid.ny * field_grid.nx))
        values = jnp.where(valid[None, :], values, 0.0)
        integrated = jnp.zeros((values.shape[0], n_det), dtype=intensity.dtype)
        integrated = integrated.at[:, flat_idx].add(values)
        return integrated.reshape((*leading_shape, self.detector_grid.ny, self.detector_grid.nx))

    def _resample_intensity_linear(self, intensity: jnp.ndarray, field_grid: Grid) -> jnp.ndarray:
        # Redistribute each source pixel over overlapping detector pixels so
        # total signal is preserved when the detector covers the same extent.
        wx = _overlap_fraction_matrix(
            _pixel_edges(field_grid, axis="x"),
            _pixel_edges(self.detector_grid, axis="x"),
        )
        wy = _overlap_fraction_matrix(
            _pixel_edges(field_grid, axis="y"),
            _pixel_edges(self.detector_grid, axis="y"),
        )
        return jnp.einsum("...yx,ay,bx->...ab", intensity, wy, wx)

    def _detector_bin_indices(
        self,
        field_grid: Grid,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x_centers = (jnp.arange(field_grid.nx) - (field_grid.nx - 1) / 2.0) * field_grid.dx_um
        y_centers = (jnp.arange(field_grid.ny) - (field_grid.ny - 1) / 2.0) * field_grid.dy_um
        x, y = jnp.meshgrid(x_centers, y_centers, indexing="xy")

        det_x_min = -0.5 * self.detector_grid.nx * self.detector_grid.dx_um
        det_y_min = -0.5 * self.detector_grid.ny * self.detector_grid.dy_um
        x_idx = jnp.floor((x - det_x_min) / self.detector_grid.dx_um).astype(jnp.int32)
        y_idx = jnp.floor((y - det_y_min) / self.detector_grid.dy_um).astype(jnp.int32)
        valid = (
            (x_idx >= 0)
            & (x_idx < self.detector_grid.nx)
            & (y_idx >= 0)
            & (y_idx < self.detector_grid.ny)
        )
        return x_idx.reshape(-1), y_idx.reshape(-1), valid.reshape(-1)

    def _detector_intensity_weight(
        self,
        *,
        spectrum_size: int,
        dtype: jnp.dtype,
        detector_grid: Grid,
    ) -> jnp.ndarray:
        qe = self._qe_intensity_map(
            spectrum_size=spectrum_size,
            dtype=dtype,
            detector_grid=detector_grid,
        )
        if self.filter_mask is None:
            return qe

        self._validate_filter_mask(detector_grid)
        amp = jnp.asarray(self.filter_mask.amplitude_map, dtype=dtype)
        if amp.ndim == 2:
            filt = (amp * amp)[None, :, :]
        else:
            if amp.shape[0] not in (1, spectrum_size):
                raise ValueError(
                    "filter_mask wavelength axis mismatch: "
                    f"got {amp.shape[0]}, expected 1 or {spectrum_size}"
                )
            filt = amp * amp
            if filt.shape[0] == 1:
                filt = jnp.broadcast_to(filt, qe.shape)
        return qe * filt

    def _qe_intensity_map(
        self,
        *,
        spectrum_size: int,
        dtype: jnp.dtype,
        detector_grid: Grid,
    ) -> jnp.ndarray:
        num_wavelengths = spectrum_size
        ny, nx = detector_grid.ny, detector_grid.nx
        if self.qe_curve is None:
            qe_vec = jnp.ones((num_wavelengths,), dtype=dtype)
        else:
            qe = jnp.asarray(self.qe_curve, dtype=dtype)
            if qe.ndim == 0:
                qe_vec = jnp.ones((num_wavelengths,), dtype=dtype) * qe
            else:
                if qe.ndim != 1:
                    raise ValueError("qe_curve must be scalar or shape (num_wavelengths,)")
                if qe.shape[0] not in (1, num_wavelengths):
                    raise ValueError(
                        "qe_curve length mismatch: "
                        f"got {qe.shape[0]}, expected 1 or {num_wavelengths}"
                    )
                qe_vec = (
                    qe
                    if qe.shape[0] == num_wavelengths
                    else jnp.ones((num_wavelengths,), dtype=dtype) * qe[0]
                )
        return qe_vec[:, None, None] * jnp.ones((num_wavelengths, ny, nx), dtype=dtype)

    def _validate_filter_mask(self, detector_grid: Grid) -> None:
        if self.filter_mask is None:
            return

        amp = jnp.asarray(self.filter_mask.amplitude_map)
        expected_shape = (detector_grid.ny, detector_grid.nx)
        if amp.ndim == 2 and amp.shape == expected_shape:
            return
        if amp.ndim == 3 and amp.shape[1:] == expected_shape:
            return
        raise ValueError(
            "filter_mask must be defined on detector_grid with shape "
            "(ny, nx) or (num_wavelengths|1, ny, nx)"
        )
