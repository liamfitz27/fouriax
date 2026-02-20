"""Core optics data models."""

from fouriax.optics.bandlimit import build_na_mask
from fouriax.optics.interfaces import OpticalLayer, Sensor
from fouriax.optics.layers import (
    AmplitudeMask,
    ComplexMask,
    FourierTransform,
    IncoherentImager,
    InverseFourierTransform,
    KSpaceAmplitudeMask,
    KSpaceComplexMask,
    KSpacePhaseMask,
    OpticalModule,
    PhaseMask,
    ThinLens,
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
from fouriax.optics.na_planning import (
    apply_na_limits,
    collect_propagation_segments,
    collect_stops,
    effective_na,
    layer_with_na_if_supported,
    na_schedule,
    propagation_layer_view,
)
from fouriax.optics.plotting import plot_field_evolution
from fouriax.optics.propagation import (
    ASMPropagator,
    KSpacePropagator,
    RSPropagator,
    critical_distance_um,
    plan_propagation,
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
    "OpticalModule",
    "PhaseMask",
    "AmplitudeMask",
    "ComplexMask",
    "FourierTransform",
    "InverseFourierTransform",
    "KSpacePhaseMask",
    "KSpaceAmplitudeMask",
    "KSpaceComplexMask",
    "ThinLens",
    "IncoherentImager",
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
    "plan_propagation",
    "critical_distance_um",
    "select_propagator_method",
    "recommend_nyquist_grid",
    "plot_field_evolution",
    "propagation_layer_view",
    "collect_stops",
    "collect_propagation_segments",
    "effective_na",
    "na_schedule",
    "layer_with_na_if_supported",
    "apply_na_limits",
]
