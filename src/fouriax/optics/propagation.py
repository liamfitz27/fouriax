from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from fouriax.core.fft import fftconvolve
from fouriax.optics.interfaces import PropagationModel
from fouriax.optics.model import Field


@dataclass(frozen=True)
class RSPropagator(PropagationModel):
    """
    Rayleigh-Sommerfeld propagator based on convolution with the RS impulse response.

    All length quantities use micrometers (um).
    """

    def impulse_response(
        self,
        field: Field,
        wavelength_um: float,
        distance_um: float,
    ) -> jnp.ndarray:
        if distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")

        x, y = field.grid.spatial_grid()
        z = jnp.asarray(distance_um, dtype=jnp.float32)
        wl = jnp.asarray(wavelength_um, dtype=jnp.float32)
        k = 2.0 * jnp.pi / wl

        r = jnp.sqrt(x * x + y * y + z * z)
        h = (z / (1j * wl * (r * r))) * jnp.exp(1j * k * r)
        return h.astype(jnp.complex64)

    def propagate(self, field: Field, distance_um: float) -> Field:
        self.validate_for(field)
        if distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")

        area_um2 = field.grid.dx_um * field.grid.dy_um
        outputs = []
        for i, wavelength_um in enumerate(field.spectrum.wavelengths_um):
            kernel = self.impulse_response(field, float(wavelength_um), distance_um)
            propagated = fftconvolve(
                field.data[i],
                kernel,
                mode="same",
                axes=(-2, -1),
            )
            outputs.append(propagated * area_um2)

        data = jnp.stack(outputs, axis=0)
        return Field(data=data, grid=field.grid, spectrum=field.spectrum)
