from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from fouriax.optics.interfaces import Sensor
from fouriax.optics.layers import AmplitudeMask
from fouriax.optics.model import Field, Grid
from fouriax.optics.monitors import IntensityMonitor
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


@dataclass(frozen=True)
class Detector(Sensor):
    """Intensity detector integrated over one region on the field grid."""

    region_mask: jnp.ndarray | None = None
    sum_wavelengths: bool = True
    channel_resolved: bool = False
    noise_model: SensorNoiseModel | None = None

    def measure(self, field: Field, *, key: jax.Array | None = None) -> jnp.ndarray:
        measured = self.expected(field)
        return _apply_optional_noise(measured, noise_model=self.noise_model, key=key)

    def expected(self, field: Field) -> jnp.ndarray:
        field_spatial = field.to_spatial()
        mask = self._resolved_region_mask(field_spatial)
        measured = IntensityMonitor(
            sum_wavelengths=self.sum_wavelengths,
            detector_masks=mask[None, :, :],
            channel_resolved=self.channel_resolved,
            output_domain=None,
        ).read(field_spatial)
        return jnp.squeeze(measured, axis=-1)

    def sample(self, field: Field, *, key: jax.Array) -> jnp.ndarray:
        return _apply_optional_noise(
            self.expected(field),
            noise_model=self.noise_model,
            key=key,
        )

    def _resolved_region_mask(self, field: Field) -> jnp.ndarray:
        if self.region_mask is None:
            return jnp.ones(field.grid.shape, dtype=jnp.float32)

        mask = jnp.asarray(self.region_mask, dtype=jnp.float32)
        if mask.shape != (field.grid.ny, field.grid.nx):
            raise ValueError(
                "region_mask must have shape "
                f"{(field.grid.ny, field.grid.nx)}, got {mask.shape}"
            )
        return mask


@dataclass(frozen=True)
class DetectorArray(Sensor):
    """Grid-based detector array with optional QE weighting, filters, and noise."""

    detector_grid: Grid
    qe_curve: jnp.ndarray | float | None = 1.0
    filter_mask: AmplitudeMask | None = None
    sum_wavelengths: bool = True
    channel_resolved: bool = False
    resample_method: Literal["nearest", "linear"] = "linear"
    noise_model: SensorNoiseModel | None = None

    def measure(self, field: Field, *, key: jax.Array | None = None) -> jnp.ndarray:
        measured = self.expected(field)
        return _apply_optional_noise(measured, noise_model=self.noise_model, key=key)

    def expected(self, field: Field) -> jnp.ndarray:
        field_spatial = field.to_spatial()
        channel_resolved = self.channel_resolved and field_spatial.is_jones
        if channel_resolved:
            intensity = field_spatial.component_intensity()
        else:
            intensity = field_spatial.intensity()

        intensity_det = self._integrate_intensity(intensity, field_spatial.grid)
        weight = self._detector_intensity_weight(field_spatial)
        if channel_resolved:
            intensity_det = intensity_det * weight[:, None, :, :]
        else:
            intensity_det = intensity_det * weight

        return (
            jnp.sum(intensity_det, axis=-4 if channel_resolved else -3)
            if self.sum_wavelengths
            else intensity_det
        )

    def sample(self, field: Field, *, key: jax.Array) -> jnp.ndarray:
        return _apply_optional_noise(
            self.expected(field),
            noise_model=self.noise_model,
            key=key,
        )

    def _integrate_intensity(self, intensity: jnp.ndarray, field_grid: Grid) -> jnp.ndarray:
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

    def _detector_intensity_weight(self, field: Field) -> jnp.ndarray:
        qe = self._qe_intensity_map(field)
        if self.filter_mask is None:
            return qe

        self._validate_filter_mask(field)
        amp = jnp.asarray(self.filter_mask.amplitude_map, dtype=field.data.real.dtype)
        if amp.ndim == 2:
            filt = (amp * amp)[None, :, :]
        else:
            filt = amp * amp
            if filt.shape[0] == 1:
                filt = jnp.broadcast_to(filt, qe.shape)
        return qe * filt

    def _qe_intensity_map(self, field: Field) -> jnp.ndarray:
        num_wavelengths = field.spectrum.size
        dtype = field.data.real.dtype
        ny, nx = self.detector_grid.ny, self.detector_grid.nx
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

    def _validate_filter_mask(self, field: Field) -> None:
        if self.filter_mask is None:
            return

        amp = jnp.asarray(self.filter_mask.amplitude_map)
        expected_shape = (self.detector_grid.ny, self.detector_grid.nx)
        if amp.ndim == 2 and amp.shape == expected_shape:
            return
        if amp.ndim == 3 and amp.shape[1:] == expected_shape:
            if amp.shape[0] in (1, field.spectrum.size):
                return
        raise ValueError(
            "filter_mask must be defined on detector_grid with shape "
            "(ny, nx) or (num_wavelengths|1, ny, nx)"
        )
