"""Core optics data models."""

from fouriax.optics.interfaces import OpticalLayer, PropagationModel, Sensor
from fouriax.optics.layers import (
    AmplitudeMaskLayer,
    ComplexMaskLayer,
    OpticalModule,
    PhaseMaskLayer,
    PropagationLayer,
    ThinLensLayer,
)
from fouriax.optics.losses import focal_spot_loss
from fouriax.optics.model import Field, Grid, Spectrum
from fouriax.optics.plotting import plot_field_evolution
from fouriax.optics.propagation import (
    ASMPropagator,
    AutoPropagator,
    RSPropagator,
    critical_distance_um,
    recommend_nyquist_grid,
    select_propagator_method,
)
from fouriax.optics.sensors import FieldReadout, IntensitySensor

__all__ = [
    "Field",
    "Grid",
    "Spectrum",
    "OpticalLayer",
    "Sensor",
    "PropagationModel",
    "PropagationLayer",
    "OpticalModule",
    "PhaseMaskLayer",
    "AmplitudeMaskLayer",
    "ComplexMaskLayer",
    "ThinLensLayer",
    "IntensitySensor",
    "FieldReadout",
    "focal_spot_loss",
    "ASMPropagator",
    "RSPropagator",
    "AutoPropagator",
    "critical_distance_um",
    "select_propagator_method",
    "recommend_nyquist_grid",
    "plot_field_evolution",
]
