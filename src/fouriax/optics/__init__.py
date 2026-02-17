"""Core optics data models."""

from fouriax.optics.bandlimit import build_na_mask
from fouriax.optics.interfaces import OpticalLayer, PropagationModel, Sensor
from fouriax.optics.layers import (
    AmplitudeMaskLayer,
    ComplexMaskLayer,
    KAmplitudeMaskLayer,
    KComplexMaskLayer,
    KPhaseMaskLayer,
    OpticalModule,
    PhaseMaskLayer,
    PropagationLayer,
    ThinLensLayer,
)
from fouriax.optics.losses import (
    build_full_sensing_matrix_dct,
    dct_basis_synthesis_matrix_2d,
    focal_spot_loss,
    mutual_information_loss,
    mutual_information_loss_from_matrix,
    randomized_svd,
    total_coherence_loss,
    total_coherence_loss_from_matrix,
)
from fouriax.optics.meta_atoms import MetaAtomInterpolationLayer, MetaAtomLibrary
from fouriax.optics.model import Field, Grid, Spectrum
from fouriax.optics.plotting import plot_field_evolution
from fouriax.optics.propagation import (
    ASMPropagator,
    AutoPropagator,
    KSpacePropagator,
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
    "build_na_mask",
    "OpticalLayer",
    "Sensor",
    "PropagationModel",
    "PropagationLayer",
    "OpticalModule",
    "PhaseMaskLayer",
    "AmplitudeMaskLayer",
    "ComplexMaskLayer",
    "KPhaseMaskLayer",
    "KAmplitudeMaskLayer",
    "KComplexMaskLayer",
    "ThinLensLayer",
    "IntensitySensor",
    "FieldReadout",
    "focal_spot_loss",
    "dct_basis_synthesis_matrix_2d",
    "build_full_sensing_matrix_dct",
    "randomized_svd",
    "total_coherence_loss",
    "total_coherence_loss_from_matrix",
    "mutual_information_loss",
    "mutual_information_loss_from_matrix",
    "MetaAtomLibrary",
    "MetaAtomInterpolationLayer",
    "ASMPropagator",
    "RSPropagator",
    "KSpacePropagator",
    "AutoPropagator",
    "critical_distance_um",
    "select_propagator_method",
    "recommend_nyquist_grid",
    "plot_field_evolution",
]
