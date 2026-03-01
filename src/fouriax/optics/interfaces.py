from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from fouriax.optics.model import Field


class OpticalLayer(ABC):
    """Base interface for field-to-field optical transformations."""

    @abstractmethod
    def forward(self, field: Field) -> Field:
        """Transform an input optical field."""

    def validate_for(self, field: Field) -> None:
        """Validate layer compatibility with the input field."""
        field.validate()

    def parameters(self) -> dict[str, jnp.ndarray]:
        """Return trainable or configurable layer parameters."""
        return {}


class Sensor(ABC):
    """Base interface for converting fields into measurements."""

    @abstractmethod
    def measure(self, field: Field, *, key: jax.Array | None = None) -> jnp.ndarray:
        """Produce a sensor measurement from a field."""

    def validate_for(self, field: Field) -> None:
        """Validate sensor compatibility with the input field."""
        field.validate()


class Monitor(ABC):
    """Base interface for deterministic, non-physical field readouts."""

    @abstractmethod
    def read(self, field: Field) -> jnp.ndarray:
        """Produce a deterministic monitoring readout from a field."""

    def validate_for(self, field: Field) -> None:
        """Validate monitor compatibility with the input field."""
        field.validate()
