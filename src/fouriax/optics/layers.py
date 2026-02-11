from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from fouriax.optics.interfaces import OpticalLayer
from fouriax.optics.model import Field


@dataclass(frozen=True)
class ThinLensLayer(OpticalLayer):
    """
    Ideal thin lens phase transform.

    All length quantities use micrometers (um).
    """

    focal_length_um: float
    aperture_diameter_um: float | None = None

    def forward(self, field: Field) -> Field:
        self.validate_for(field)
        if self.focal_length_um <= 0:
            raise ValueError("focal_length_um must be strictly positive")

        x, y = field.grid.spatial_grid()
        r2 = x * x + y * y
        phase_stack = []
        for wavelength_um in field.spectrum.wavelengths_um:
            phi = -jnp.pi * r2 / (wavelength_um * self.focal_length_um)
            phase_stack.append(phi)
        phase = jnp.stack(phase_stack, axis=0)

        transformed = field.apply_phase(phase)

        if self.aperture_diameter_um is not None:
            if self.aperture_diameter_um <= 0:
                raise ValueError("aperture_diameter_um must be strictly positive")
            radius = self.aperture_diameter_um / 2.0
            aperture = (r2 <= radius * radius).astype(transformed.data.real.dtype)
            transformed = transformed.apply_amplitude(aperture[None, :, :])

        return transformed

    def parameters(self) -> dict[str, jnp.ndarray]:
        params: dict[str, jnp.ndarray] = {
            "focal_length_um": jnp.asarray(self.focal_length_um, dtype=jnp.float32)
        }
        if self.aperture_diameter_um is not None:
            params["aperture_diameter_um"] = jnp.asarray(
                self.aperture_diameter_um, dtype=jnp.float32
            )
        return params
