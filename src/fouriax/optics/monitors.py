from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp

from fouriax.optics.interfaces import Monitor
from fouriax.optics.model import Field


def _field_in_domain(
    field: Field,
    output_domain: Literal["spatial", "kspace"] | None,
) -> Field:
    if output_domain is None:
        return field
    if output_domain == "spatial":
        return field.to_spatial()
    if output_domain == "kspace":
        return field.to_kspace()
    raise ValueError("output_domain must be one of: spatial, kspace, or None")


@dataclass(frozen=True)
class IntensityMonitor(Monitor):
    """Deterministic field-intensity monitor.

    If `detector_masks` is provided with shape `(num_detectors, ny, nx)`, the
    monitor integrates intensity within each detector region instead of
    returning a full image.
    """

    sum_wavelengths: bool = False
    detector_masks: jnp.ndarray | None = None
    channel_resolved: bool = False
    output_domain: Literal["spatial", "kspace"] | None = None

    def read(self, field: Field) -> jnp.ndarray:
        field = _field_in_domain(field, self.output_domain)
        self.validate_for(field)
        channel_resolved = self.channel_resolved and field.is_jones
        if channel_resolved:
            intensity = field.component_intensity()
        else:
            intensity = field.intensity()
        if self.detector_masks is not None:
            masks = jnp.asarray(self.detector_masks, dtype=jnp.float32)
            if masks.ndim != 3:
                raise ValueError("detector_masks must have shape (num_detectors, ny, nx)")
            if masks.shape[1:] != (field.grid.ny, field.grid.nx):
                raise ValueError(
                    f"detector_masks spatial shape mismatch: got {masks.shape[1:]}, "
                    f"expected {(field.grid.ny, field.grid.nx)}"
                )
            if channel_resolved:
                per_wavelength = jnp.einsum("...wcxy,dxy->...wcd", intensity, masks)
            else:
                per_wavelength = jnp.einsum("...wxy,dxy->...wd", intensity, masks)
            return (
                jnp.sum(per_wavelength, axis=-3 if channel_resolved else -2)
                if self.sum_wavelengths
                else per_wavelength
            )
        if self.sum_wavelengths:
            return jnp.sum(intensity, axis=-4 if channel_resolved else -3)
        return intensity


@dataclass(frozen=True)
class FieldMonitor(Monitor):
    """Deterministic field readout monitor.

    Supported values for `representation`:
    - "complex": complex array:
      - scalar `(*batch, wavelengths, ny, nx)`
      - jones `(*batch, wavelengths, 2, ny, nx)`
    - "real_imag": stacked real/imag in trailing axis
    - "amplitude_phase": stacked amplitude/phase in trailing axis
    """

    representation: str = "complex"
    output_domain: Literal["spatial", "kspace"] | None = None

    def read(self, field: Field) -> jnp.ndarray:
        field = _field_in_domain(field, self.output_domain)
        self.validate_for(field)
        if self.representation == "complex":
            return field.data
        if self.representation == "real_imag":
            return jnp.stack([field.data.real, field.data.imag], axis=-1)
        if self.representation == "amplitude_phase":
            return jnp.stack([jnp.abs(field.data), jnp.angle(field.data)], axis=-1)
        raise ValueError(
            "representation must be one of: complex, real_imag, amplitude_phase"
        )
