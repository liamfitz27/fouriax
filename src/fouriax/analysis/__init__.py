"""Fisher information, sensitivity analysis, and design optimality utilities.

Compute Jacobians, Fisher information matrices, Cramér–Rao bounds,
D-optimality scores, sensitivity maps, and parameter tolerance
estimates for optical measurement systems.
"""

from fouriax.analysis.fisher import (
    cramer_rao_bound,
    d_optimality,
    fisher_information,
    jacobian_matrix,
    score_fisher_information,
)
from fouriax.analysis.sensitivity import (
    parameter_tolerance,
    sensitivity_map,
)

__all__ = [
    "jacobian_matrix",
    "fisher_information",
    "score_fisher_information",
    "cramer_rao_bound",
    "d_optimality",
    "sensitivity_map",
    "parameter_tolerance",
]
