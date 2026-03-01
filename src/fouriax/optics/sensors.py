from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import jax
import jax.image
import jax.numpy as jnp

from fouriax.optics.interfaces import Sensor
from fouriax.optics.layers import AmplitudeMask
from fouriax.optics.model import Field, Grid
from fouriax.optics.monitors import FieldMonitor, IntensityMonitor
from fouriax.optics.noise import SensorNoiseModel


@dataclass(frozen=True)
class CameraSensor(Sensor):
    """Camera-like sensor with explicit pixel-grid readout.

    The expected camera response is computed from field intensity as:
    `I_cam(λ, y, x) = |F(λ, y, x) * sqrt(QE(λ)) * E_resampled(λ, y, x)|²`,
    where `F` is an optional externally provided per-pixel `AmplitudeMask`
    (`filter_mask`) and the QE weighting is applied via an internal
    `AmplitudeMask` on the same pixel grid.
    """

    pixel_grid: Grid
    qe_curve: jnp.ndarray | float | None = 1.0
    filter_mask: AmplitudeMask | None = None
    pixel_resample_method: Literal["nearest", "linear"] = "linear"
    noise_model: SensorNoiseModel | None = None

    def measure(self, field: Field, *, key: jax.Array | None = None) -> jnp.ndarray:
        measured = self.expected(field)
        return self._apply_noise(measured, key=key)

    def expected(self, field: Field) -> jnp.ndarray:
        field_px = self._field_on_pixel_grid(field)
        qe_mask = self._build_qe_mask(field_px)
        field_weighted = qe_mask.forward(field_px)
        if self.filter_mask is not None:
            self._validate_filter_mask(field_weighted)
            field_weighted = self.filter_mask.forward(field_weighted)
        return IntensityMonitor(
            sum_wavelengths=True,
            output_domain=None,
        ).read(field_weighted)

    def sample(self, field: Field, *, key: jax.Array) -> jnp.ndarray:
        return self._apply_noise(self.expected(field), key=key)

    def _build_qe_mask(self, field: Field) -> AmplitudeMask:
        num_wavelengths = field.spectrum.size
        dtype = field.data.real.dtype
        ny, nx = field.grid.ny, field.grid.nx
        if self.qe_curve is None:
            qe = jnp.ones((num_wavelengths,), dtype=dtype)
            qe_amp = jnp.sqrt(jnp.maximum(qe, 0.0))
            qe_map = qe_amp[:, None, None] * jnp.ones((num_wavelengths, ny, nx), dtype=dtype)
            return AmplitudeMask(amplitude_map=qe_map)
        qe = jnp.asarray(self.qe_curve, dtype=dtype)
        if qe.ndim == 0:
            return AmplitudeMask(amplitude_map=jnp.sqrt(jnp.maximum(qe, 0.0)))
        if qe.ndim != 1:
            raise ValueError("qe_curve must be scalar or shape (num_wavelengths,)")
        if qe.shape[0] not in (1, num_wavelengths):
            raise ValueError(
                f"qe_curve length mismatch: got {qe.shape[0]}, expected 1 or {num_wavelengths}"
            )
        qe_vec = (
            qe
            if qe.shape[0] == num_wavelengths
            else jnp.ones((num_wavelengths,), dtype=dtype) * qe[0]
        )
        qe_amp = jnp.sqrt(jnp.maximum(qe_vec, 0.0))
        qe_map = qe_amp[:, None, None] * jnp.ones((num_wavelengths, ny, nx), dtype=dtype)
        return AmplitudeMask(amplitude_map=qe_map)

    def _field_on_pixel_grid(self, field: Field) -> Field:
        field_spatial = field.to_spatial()
        if (
            field_spatial.grid.nx == self.pixel_grid.nx
            and field_spatial.grid.ny == self.pixel_grid.ny
            and field_spatial.grid.dx_um == self.pixel_grid.dx_um
            and field_spatial.grid.dy_um == self.pixel_grid.dy_um
        ):
            return field_spatial

        target_shape: tuple[int, ...] = (
            field_spatial.spectrum.size,
            self.pixel_grid.ny,
            self.pixel_grid.nx,
        )
        if field_spatial.is_jones:
            target_shape = (
                field_spatial.spectrum.size,
                2,
                self.pixel_grid.ny,
                self.pixel_grid.nx,
            )

        resized_real = jax.image.resize(
            field_spatial.data.real,
            shape=target_shape,
            method=self.pixel_resample_method,
        )
        resized_imag = jax.image.resize(
            field_spatial.data.imag,
            shape=target_shape,
            method=self.pixel_resample_method,
        )
        resized_data = cast(jnp.ndarray, resized_real + 1j * resized_imag)
        return Field(
            data=resized_data,
            grid=self.pixel_grid,
            spectrum=field_spatial.spectrum,
            polarization_mode=field_spatial.polarization_mode,
            domain="spatial",
        )

    def _validate_filter_mask(self, field: Field) -> None:
        if self.filter_mask is None:
            return
        amp = jnp.asarray(self.filter_mask.amplitude_map)
        expected_shape = (self.pixel_grid.ny, self.pixel_grid.nx)
        if amp.ndim == 2 and amp.shape == expected_shape:
            return
        if amp.ndim == 3 and amp.shape[1:] == expected_shape:
            if amp.shape[0] in (1, field.spectrum.size):
                return
        raise ValueError(
            "filter_mask must be defined on pixel_grid with shape "
            "(ny, nx) or (num_wavelengths|1, ny, nx)"
        )

    def _apply_noise(self, measured: jnp.ndarray, *, key: jax.Array | None) -> jnp.ndarray:
        if self.noise_model is None or key is None:
            return measured
        return self.noise_model.sample(measured, key=key)


@dataclass(frozen=True)
class IntensitySensor(Sensor):
    """Compatibility wrapper around `IntensityMonitor` plus optional noise."""

    sum_wavelengths: bool = False
    detector_masks: jnp.ndarray | None = None
    channel_resolved: bool = False
    noise_model: SensorNoiseModel | None = None

    def measure(self, field: Field, *, key: jax.Array | None = None) -> jnp.ndarray:
        measured = IntensityMonitor(
            sum_wavelengths=self.sum_wavelengths,
            detector_masks=self.detector_masks,
            channel_resolved=self.channel_resolved,
            output_domain="spatial",
        ).read(field)
        if self.noise_model is None or key is None:
            return measured
        return self.noise_model.sample(measured, key=key)


@dataclass(frozen=True)
class FieldReadout(Sensor):
    """Compatibility wrapper around `FieldMonitor`."""

    representation: str = "complex"

    def measure(self, field: Field, *, key: jax.Array | None = None) -> jnp.ndarray:
        del key
        return FieldMonitor(
            representation=self.representation,
            output_domain="spatial",
        ).read(field)
