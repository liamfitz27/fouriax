"""fouriax package."""

from fouriax.core.fft import fftconvolve
from fouriax.optics import (
    ASMPropagator,
    AutoPropagator,
    Field,
    Grid,
    OpticalLayer,
    PropagationDecision,
    PropagationModel,
    PropagationPolicy,
    RSPropagator,
    SamplingPlan,
    SamplingPlanner,
    Sensor,
    Spectrum,
    ThinLensLayer,
    focal_spot_loss,
)

__all__ = [
    "fftconvolve",
    "Grid",
    "Spectrum",
    "Field",
    "OpticalLayer",
    "Sensor",
    "PropagationModel",
    "ThinLensLayer",
    "focal_spot_loss",
    "ASMPropagator",
    "RSPropagator",
    "AutoPropagator",
    "SamplingPlan",
    "SamplingPlanner",
    "PropagationDecision",
    "PropagationPolicy",
]
