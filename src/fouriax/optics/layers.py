from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from fouriax.optics.interfaces import OpticalLayer, PropagationModel, Sensor
from fouriax.optics.model import Field


def _expand_map(
    value: jnp.ndarray | float,
    field: Field,
    *,
    dtype: jnp.dtype,
    name: str,
) -> jnp.ndarray:
    arr = jnp.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        return arr
    if arr.ndim == 2:
        if arr.shape != (field.grid.ny, field.grid.nx):
            raise ValueError(
                f"{name} shape mismatch: got {arr.shape}, expected {(field.grid.ny, field.grid.nx)}"
            )
        return arr[None, :, :]
    if arr.ndim == 3:
        if arr.shape[1:] != (field.grid.ny, field.grid.nx):
            raise ValueError(
                f"{name} shape mismatch: got {arr.shape[1:]}, "
                f"expected {(field.grid.ny, field.grid.nx)}"
            )
        if arr.shape[0] not in (1, field.spectrum.size):
            raise ValueError(
                f"{name} wavelength axis mismatch: got {arr.shape[0]}, expected 1 or "
                f"{field.spectrum.size}"
            )
        return arr
    raise ValueError(f"{name} must be scalar, (ny, nx), or (num_wavelengths, ny, nx)")


@dataclass(frozen=True)
class PropagationLayer(OpticalLayer):
    """Optical layer wrapper around a propagation model and fixed distance."""

    model: PropagationModel
    distance_um: float

    def forward(self, field: Field) -> Field:
        self.validate_for(field)
        if self.distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")
        return self.model.propagate(field, distance_um=self.distance_um)

    def parameters(self) -> dict[str, jnp.ndarray]:
        return {"distance_um": jnp.asarray(self.distance_um, dtype=jnp.float32)}


@dataclass(frozen=True)
class OpticalModule(OpticalLayer):
    """Composable sequence of optical layers."""

    layers: tuple[OpticalLayer, ...]
    sensor: Sensor | None = None

    def forward(self, field: Field) -> Field:
        self.validate_for(field)
        output = field
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def measure(self, field: Field) -> jnp.ndarray:
        """Forward through layers and apply the configured sensor."""
        if self.sensor is None:
            raise ValueError("OpticalModule has no sensor configured")
        output = self.forward(field)
        return self.sensor.measure(output)

    def trace(self, field: Field, include_input: bool = True) -> list[Field]:
        """Return intermediate fields through the module."""
        self.validate_for(field)
        output = field
        states: list[Field] = [output] if include_input else []
        for layer in self.layers:
            output = layer.forward(output)
            states.append(output)
        return states

    def parameters(self) -> dict[str, jnp.ndarray]:
        params: dict[str, jnp.ndarray] = {}
        for i, layer in enumerate(self.layers):
            for key, value in layer.parameters().items():
                params[f"layer_{i}.{key}"] = value
        return params


@dataclass(frozen=True)
class PhaseMaskLayer(OpticalLayer):
    """Generic phase-only modulation layer."""

    phase_map_rad: jnp.ndarray | float

    def forward(self, field: Field) -> Field:
        self.validate_for(field)
        phase = _expand_map(
            self.phase_map_rad,
            field,
            dtype=jnp.float32,
            name="phase_map_rad",
        )
        return field.apply_phase(phase)

    def parameters(self) -> dict[str, jnp.ndarray]:
        return {"phase_map_rad": jnp.asarray(self.phase_map_rad, dtype=jnp.float32)}


@dataclass(frozen=True)
class AmplitudeMaskLayer(OpticalLayer):
    """Generic amplitude modulation layer."""

    amplitude_map: jnp.ndarray | float

    def forward(self, field: Field) -> Field:
        self.validate_for(field)
        amplitude = _expand_map(
            self.amplitude_map,
            field,
            dtype=field.data.real.dtype,
            name="amplitude_map",
        )
        return field.apply_amplitude(amplitude)

    def parameters(self) -> dict[str, jnp.ndarray]:
        return {"amplitude_map": jnp.asarray(self.amplitude_map, dtype=jnp.float32)}


@dataclass(frozen=True)
class ComplexMaskLayer(OpticalLayer):
    """Generic complex-valued modulation layer (amplitude and phase)."""

    amplitude_map: jnp.ndarray | float = 1.0
    phase_map_rad: jnp.ndarray | float = 0.0

    def forward(self, field: Field) -> Field:
        self.validate_for(field)
        amplitude = _expand_map(
            self.amplitude_map,
            field,
            dtype=field.data.real.dtype,
            name="amplitude_map",
        )
        phase = _expand_map(
            self.phase_map_rad,
            field,
            dtype=jnp.float32,
            name="phase_map_rad",
        )
        return field.apply_amplitude(amplitude).apply_phase(phase)

    def parameters(self) -> dict[str, jnp.ndarray]:
        return {
            "amplitude_map": jnp.asarray(self.amplitude_map, dtype=jnp.float32),
            "phase_map_rad": jnp.asarray(self.phase_map_rad, dtype=jnp.float32),
        }


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
