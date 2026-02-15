from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp

from fouriax.optics.model import Grid, Spectrum


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@dataclass(frozen=True)
class SamplingPlan:
    """Recommended propagation grid and safety diagnostics."""

    nx: int
    ny: int
    dx_um: float
    dy_um: float
    min_wavelength_um: float
    safety_factor: float
    sampling_ratio: float
    is_sampling_safe: bool
    warning: str | None = None

    @property
    def grid(self) -> Grid:
        return Grid.from_extent(self.nx, self.ny, self.dx_um, self.dy_um)


@dataclass(frozen=True)
class PropagationDecision:
    """Decision output for automatic propagator selection."""

    method: str
    regime_method: str
    critical_distance_um: float
    distance_to_critical_ratio: float
    estimated_rs_to_asm_cost_ratio: float
    reason: str


@dataclass(frozen=True)
class SamplingPlanner:
    """
    Recommend a propagation grid from the mask grid and spectrum.

    All length units are micrometers (um).
    """

    safety_factor: float = 2.0
    min_padding_factor: float = 2.0

    def recommend_grid(self, mask_grid: Grid, spectrum: Spectrum) -> SamplingPlan:
        mask_grid.validate()
        spectrum.validate()
        if self.safety_factor <= 0:
            raise ValueError("safety_factor must be strictly positive")
        if self.min_padding_factor < 1.0:
            raise ValueError("min_padding_factor must be >= 1.0")

        min_wavelength_um = float(jnp.min(spectrum.wavelengths_um))
        nyquist_limited_dx = min_wavelength_um / (2.0 * self.safety_factor)

        upsample_x = max(1, int(jnp.ceil(mask_grid.dx_um / nyquist_limited_dx)))
        upsample_y = max(1, int(jnp.ceil(mask_grid.dy_um / nyquist_limited_dx)))

        dx_um = mask_grid.dx_um / upsample_x
        dy_um = mask_grid.dy_um / upsample_y

        padded_nx = int(jnp.ceil(mask_grid.nx * self.min_padding_factor)) * upsample_x
        padded_ny = int(jnp.ceil(mask_grid.ny * self.min_padding_factor)) * upsample_y
        nx = _next_power_of_two(padded_nx)
        ny = _next_power_of_two(padded_ny)

        sampling_ratio = min_wavelength_um / (2.0 * max(dx_um, dy_um))
        is_sampling_safe = sampling_ratio >= self.safety_factor
        warning = None
        if not is_sampling_safe:
            warning = "Sampling ratio below requested safety factor; aliasing risk may be high."

        return SamplingPlan(
            nx=nx,
            ny=ny,
            dx_um=dx_um,
            dy_um=dy_um,
            min_wavelength_um=min_wavelength_um,
            safety_factor=self.safety_factor,
            sampling_ratio=float(sampling_ratio),
            is_sampling_safe=is_sampling_safe,
            warning=warning,
        )

    def estimate_aliasing_margin(self, plan: SamplingPlan) -> float:
        return plan.sampling_ratio / plan.safety_factor

    def validate_grid(self, grid: Grid, spectrum: Spectrum) -> SamplingPlan:
        return self.recommend_grid(mask_grid=grid, spectrum=spectrum)


@dataclass(frozen=True)
class PropagationPolicy:
    """
    Choose between RS and ASM using the critical crossover distance:

    z_crit = N * (dx^2) / lambda

    Modes:
    - fast: always choose ASM.
    - balanced: regime rule with optional cost and accuracy overrides.
    - accurate: strict regime rule.
    """

    mode: Literal["fast", "balanced", "accurate"] = "balanced"
    equality_tolerance: float = 1e-6
    rs_cost_ratio_threshold: float = 8.0

    def critical_distance_um(self, grid: Grid, spectrum: Spectrum) -> float:
        """
        Compute conservative z_crit for non-square sampling.

        Uses:
        - N_eff = min(nx, ny)
        - dx_eff = max(dx_um, dy_um)
        - lambda_min for conservative multi-wavelength behavior.
        """
        grid.validate()
        spectrum.validate()
        n_eff = min(grid.nx, grid.ny)
        dx_eff = max(grid.dx_um, grid.dy_um)
        lambda_min_um = float(jnp.min(spectrum.wavelengths_um))
        return float(n_eff * (dx_eff**2) / lambda_min_um)

    def estimate_cost_ratio_rs_to_asm(self, grid: Grid) -> float:
        """
        Estimate RS/ASM relative cost with an FFT complexity proxy.

        ASM proxy: P log2(P), where P=nx*ny.
        RS proxy:  P log2(P), where P=(2*nx-1)*(2*ny-1).
        """
        grid.validate()
        n_asm = float(grid.nx * grid.ny)
        n_rs = float((2 * grid.nx - 1) * (2 * grid.ny - 1))
        asm_cost = n_asm * jnp.log2(n_asm + 1.0)
        rs_cost = n_rs * jnp.log2(n_rs + 1.0)
        return float(rs_cost / asm_cost)

    def _regime_method(self, critical_distance_um: float, distance_um: float) -> str:
        if distance_um < critical_distance_um * (1.0 - self.equality_tolerance):
            return "asm"
        return "rs"

    def choose(
        self,
        grid: Grid,
        spectrum: Spectrum,
        distance_um: float,
        plan: SamplingPlan | None = None,
        mae_tolerance: float | None = None,
        estimated_mae: dict[str, float] | None = None,
    ) -> PropagationDecision:
        grid.validate()
        spectrum.validate()
        if distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")
        if self.equality_tolerance < 0:
            raise ValueError("equality_tolerance must be >= 0")
        if self.rs_cost_ratio_threshold <= 0:
            raise ValueError("rs_cost_ratio_threshold must be > 0")
        if mae_tolerance is not None and mae_tolerance <= 0:
            raise ValueError("mae_tolerance must be > 0 when provided")

        critical = self.critical_distance_um(grid=grid, spectrum=spectrum)
        ratio = distance_um / critical
        regime_method = self._regime_method(critical_distance_um=critical, distance_um=distance_um)
        cost_grid = plan.grid if plan is not None else grid
        cost_ratio = self.estimate_cost_ratio_rs_to_asm(cost_grid)

        if plan is not None and not plan.is_sampling_safe:
            return PropagationDecision(
                method="rs",
                regime_method=regime_method,
                critical_distance_um=critical,
                distance_to_critical_ratio=float(ratio),
                estimated_rs_to_asm_cost_ratio=cost_ratio,
                reason="Selected RS because sampling plan flagged aliasing risk.",
            )

        if self.mode == "fast":
            return PropagationDecision(
                method="asm",
                regime_method=regime_method,
                critical_distance_um=critical,
                distance_to_critical_ratio=float(ratio),
                estimated_rs_to_asm_cost_ratio=cost_ratio,
                reason="Selected ASM because policy mode is fast.",
            )

        if self.mode == "accurate":
            return PropagationDecision(
                method=regime_method,
                regime_method=regime_method,
                critical_distance_um=critical,
                distance_to_critical_ratio=float(ratio),
                estimated_rs_to_asm_cost_ratio=cost_ratio,
                reason="Selected strict regime method because policy mode is accurate.",
            )

        method = regime_method
        reason = "Selected regime method in balanced mode."

        if regime_method == "rs" and cost_ratio > self.rs_cost_ratio_threshold:
            method = "asm"
            reason = (
                "Selected ASM in balanced mode because projected RS/ASM cost ratio "
                "exceeded threshold."
            )

        if (
            mae_tolerance is not None
            and estimated_mae is not None
            and "asm" in estimated_mae
            and estimated_mae["asm"] <= mae_tolerance
        ):
            method = "asm"
            reason = "Selected ASM in balanced mode because estimated ASM error met tolerance."

        return PropagationDecision(
            method=method,
            regime_method=regime_method,
            critical_distance_um=critical,
            distance_to_critical_ratio=float(ratio),
            estimated_rs_to_asm_cost_ratio=cost_ratio,
            reason=reason,
        )
