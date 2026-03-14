from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from fouriax.optics.model import Field, Intensity


class OpticalLayer(ABC):
    """Base interface for field-to-field optical transformations.

    Subclasses must implement :meth:`forward`, which receives a
    :class:`~fouriax.Field` and returns a transformed ``Field`` with
    the same grid and spectrum (unless the layer explicitly changes them).

    Override :meth:`validate_for` to add layer-specific compatibility
    checks (e.g. domain or polarisation requirements).  Override
    :meth:`parameters` to expose trainable or configurable values used
    during gradient-based optimisation.
    """

    @abstractmethod
    def forward(self, field: Field) -> Field:
        """Apply this layer's optical transformation to ``field``.

        Args:
            field: Input optical field to transform.

        Returns:
            Transformed field. Subclasses should preserve batch axes,
            wavelength axes, and metadata unless the layer explicitly changes
            them.
        """

    def validate_for(self, field: Field) -> None:
        """Check that *field* is compatible with this layer.

        The default implementation validates the field itself.  Subclasses
        should call ``super().validate_for(field)`` and add any
        layer-specific checks such as domain, polarization, or shape
        requirements.
        """
        field.validate()

    def parameters(self) -> dict[str, jnp.ndarray]:
        """Return trainable or configurable layer parameters.

        Returns an empty dict by default.  Subclasses override this to
        expose arrays that should be updated during optimisation.
        """
        return {}


class Sensor(ABC):
    """Base interface for converting optical fields into detector measurements.

    Subclasses implement :meth:`measure`, which maps a ``Field`` or
    ``Intensity`` to a real-valued measurement array.  An optional *key*
    argument supports stochastic sensor noise models.
    """

    @abstractmethod
    def measure(
        self,
        field: Field | Intensity,
        *,
        key: jax.Array | None = None,
    ) -> jnp.ndarray:
        """Produce a measurement from *field*.

        Args:
            field: Input optical field or spatial intensity.
            key: Optional JAX PRNG key for stochastic noise sampling.

        Returns:
            Measurement array derived from ``field``. Concrete sensors define
            the exact shape and reduction over wavelength or polarization axes.
        """

    def validate_for(self, field: Field | Intensity) -> None:
        """Check that ``field`` is compatible with this sensor."""
        field.validate()


class IncoherentLayer(ABC):
    """Base interface for incoherent intensity-to-intensity transforms."""

    @abstractmethod
    def forward(self, intensity: Intensity) -> Intensity:
        """Apply this layer's transformation to ``intensity``."""

    def validate_for(self, intensity: Intensity) -> None:
        """Check that ``intensity`` is compatible with this layer."""
        intensity.validate()

    def parameters(self) -> dict[str, jnp.ndarray]:
        """Return trainable or configurable layer parameters."""
        return {}


class Monitor(ABC):
    """Base interface for deterministic, non-physical field readouts.

    Monitors are placed inline in an :class:`~fouriax.OpticalModule`
    layer stack to record intermediate quantities (e.g. intensity
    distributions) without altering the field.
    """

    @abstractmethod
    def read(self, field: Field) -> jnp.ndarray:
        """Produce a deterministic readout from ``field``.

        Args:
            field: Input optical field.

        Returns:
            Diagnostic array derived from ``field`` without stochastic noise or
            physical detector effects.
        """

    def validate_for(self, field: Field) -> None:
        """Check that ``field`` is compatible with this monitor."""
        field.validate()
