from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import jax.numpy as jnp
from jax.scipy import ndimage as jndimage

from fouriax.core.fft import fftconvolve
from fouriax.optics.interfaces import PropagationModel
from fouriax.optics.model import Field, Grid, Spectrum
from fouriax.optics.planning import PropagationPolicy, SamplingPlan, SamplingPlanner


def _resample_2d_to_grid(
    data_2d: jnp.ndarray,
    src_grid: Grid,
    dst_grid: Grid,
) -> jnp.ndarray:
    """
    Resample a 2D complex field from src_grid coordinates to dst_grid coordinates.

    Uses bilinear interpolation on real/imag components independently.
    Out-of-bounds points are zero-filled.
    """
    x_dst, y_dst = dst_grid.spatial_grid()
    src_x = x_dst / src_grid.dx_um + (src_grid.nx - 1) / 2.0
    src_y = y_dst / src_grid.dy_um + (src_grid.ny - 1) / 2.0
    coords = [src_y, src_x]

    real = jndimage.map_coordinates(
        data_2d.real,
        coords,
        order=1,
        mode="constant",
        cval=0.0,
    )
    imag = jndimage.map_coordinates(
        data_2d.imag,
        coords,
        order=1,
        mode="constant",
        cval=0.0,
    )
    return jnp.asarray(real + 1j * imag, dtype=data_2d.dtype)


def _prepare_field_with_plan(field: Field, plan: SamplingPlan) -> Field:
    """
    Apply planner-driven resampling and padding for propagation stability.

    The resulting propagation grid may have both different shape and different
    spacing (dx_um, dy_um) compared to the input field grid.
    """
    target_grid = plan.grid
    if (
        target_grid.nx == field.grid.nx
        and target_grid.ny == field.grid.ny
        and target_grid.dx_um == field.grid.dx_um
        and target_grid.dy_um == field.grid.dy_um
    ):
        return field

    data = jnp.stack(
        [
            _resample_2d_to_grid(field.data[i], src_grid=field.grid, dst_grid=target_grid)
            for i in range(field.spectrum.size)
        ],
        axis=0,
    )
    return Field(data=data, grid=target_grid, spectrum=field.spectrum)


def _restore_to_original_grid(field: Field, original_grid: Grid) -> Field:
    if (
        field.grid.nx == original_grid.nx
        and field.grid.ny == original_grid.ny
        and field.grid.dx_um == original_grid.dx_um
        and field.grid.dy_um == original_grid.dy_um
    ):
        return field

    data = jnp.stack(
        [
            _resample_2d_to_grid(field.data[i], src_grid=field.grid, dst_grid=original_grid)
            for i in range(field.spectrum.size)
        ],
        axis=0,
    )
    return Field(data=data, grid=original_grid, spectrum=field.spectrum)


@dataclass(frozen=True)
class RSPropagator(PropagationModel):
    """
    Rayleigh-Sommerfeld propagator based on convolution with the RS impulse response.

    All length quantities use micrometers (um).
    """

    use_sampling_planner: bool = True
    sampling_planner: SamplingPlanner = field(default_factory=SamplingPlanner)
    precomputed_plan: SamplingPlan | None = None

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

        original_grid = field.grid
        plan = self.precomputed_plan
        if plan is None and self.use_sampling_planner:
            plan = self.sampling_planner.recommend_grid(
                mask_grid=field.grid,
                spectrum=field.spectrum,
            )
        work_field = _prepare_field_with_plan(field, plan) if plan is not None else field

        area_um2 = work_field.grid.dx_um * work_field.grid.dy_um
        outputs = []
        for i, wavelength_um in enumerate(work_field.spectrum.wavelengths_um):
            kernel = self.impulse_response(work_field, wavelength_um, distance_um)
            propagated = fftconvolve(
                work_field.data[i],
                kernel,
                mode="same",
                axes=(-2, -1),
            )
            outputs.append(propagated * area_um2)

        data = jnp.stack(outputs, axis=0)
        propagated_field = Field(data=data, grid=work_field.grid, spectrum=work_field.spectrum)
        return _restore_to_original_grid(propagated_field, original_grid)


@dataclass(frozen=True)
class ASMPropagator(PropagationModel):
    """
    Angular Spectrum Method (ASM) propagator.

    All length quantities use micrometers (um).
    """

    use_sampling_planner: bool = True
    sampling_planner: SamplingPlanner = field(default_factory=SamplingPlanner)
    precomputed_plan: SamplingPlan | None = None

    def transfer_function(
        self,
        field: Field,
        wavelength_um: float,
        distance_um: float,
    ) -> jnp.ndarray:
        if distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")

        fx, fy = field.grid.frequency_grid()
        wl = jnp.asarray(wavelength_um, dtype=jnp.float32)
        z = jnp.asarray(distance_um, dtype=jnp.float32)
        k = 2.0 * jnp.pi / wl

        argument = 1.0 - (wl * fx) ** 2 - (wl * fy) ** 2
        kz = k * jnp.sqrt(argument.astype(jnp.complex64))
        return jnp.exp(1j * kz * z).astype(jnp.complex64)

    def propagate(self, field: Field, distance_um: float) -> Field:
        self.validate_for(field)
        if distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")

        original_grid = field.grid
        plan = self.precomputed_plan
        if plan is None and self.use_sampling_planner:
            plan = self.sampling_planner.recommend_grid(
                mask_grid=field.grid,
                spectrum=field.spectrum,
            )
        work_field = _prepare_field_with_plan(field, plan) if plan is not None else field

        outputs = []
        for i, wavelength_um in enumerate(work_field.spectrum.wavelengths_um):
            transfer = self.transfer_function(work_field, wavelength_um, distance_um)
            spectrum = jnp.fft.fftn(work_field.data[i], axes=(-2, -1))
            propagated = jnp.fft.ifftn(spectrum * transfer, axes=(-2, -1))
            outputs.append(propagated)

        data = jnp.stack(outputs, axis=0)
        propagated_field = Field(data=data, grid=work_field.grid, spectrum=work_field.spectrum)
        return _restore_to_original_grid(propagated_field, original_grid)


@dataclass(frozen=True)
class AutoPropagator(PropagationModel):
    """
    Propagator wrapper that auto-selects ASM or RS.
    """

    asm: ASMPropagator = field(default_factory=ASMPropagator)
    rs: RSPropagator = field(default_factory=RSPropagator)
    policy_mode: Literal["fast", "balanced", "accurate"] = "balanced"
    equality_tolerance: float = 1e-6
    rs_cost_ratio_threshold: float = 8.0
    nyquist_fraction: float = 2.0
    min_padding_factor: float = 2.0
    setup_grid: Grid | None = None
    setup_spectrum: Spectrum | None = None
    setup_distance_um: float | None = None
    precomputed_plan: SamplingPlan | None = None
    precomputed_method: str | None = None

    def __post_init__(self) -> None:
        if self.precomputed_plan is not None:
            asm = self.asm
            rs = self.rs
            if asm.precomputed_plan is None or asm.use_sampling_planner:
                asm = replace(
                    asm,
                    use_sampling_planner=False,
                    precomputed_plan=self.precomputed_plan,
                )
            if rs.precomputed_plan is None or rs.use_sampling_planner:
                rs = replace(
                    rs,
                    use_sampling_planner=False,
                    precomputed_plan=self.precomputed_plan,
                )
            object.__setattr__(self, "asm", asm)
            object.__setattr__(self, "rs", rs)

        if (
            self.setup_grid is not None
            and self.setup_spectrum is not None
            and self.setup_distance_um is not None
            and (self.precomputed_plan is None or self.precomputed_method is None)
        ):
            if self.setup_distance_um <= 0:
                raise ValueError("setup_distance_um must be strictly positive")
            self.setup_grid.validate()
            self.setup_spectrum.validate()
            policy = PropagationPolicy(
                mode=self.policy_mode,
                equality_tolerance=self.equality_tolerance,
                rs_cost_ratio_threshold=self.rs_cost_ratio_threshold,
            )
            plan = self.precomputed_plan
            if plan is None and (self.rs.use_sampling_planner or self.asm.use_sampling_planner):
                planner = SamplingPlanner(
                    nyquist_fraction=self.nyquist_fraction,
                    min_padding_factor=self.min_padding_factor,
                )
                plan = planner.recommend_grid(
                    mask_grid=self.setup_grid,
                    spectrum=self.setup_spectrum,
                )
            method = self.precomputed_method
            if method is None:
                method = policy.choose(
                    grid=self.setup_grid,
                    spectrum=self.setup_spectrum,
                    distance_um=self.setup_distance_um,
                    plan=plan,
                ).method
            object.__setattr__(self, "precomputed_plan", plan)
            object.__setattr__(self, "precomputed_method", method)

            asm = self.asm
            rs = self.rs
            if plan is not None:
                if asm.precomputed_plan is None or asm.use_sampling_planner:
                    asm = replace(asm, use_sampling_planner=False, precomputed_plan=plan)
                if rs.precomputed_plan is None or rs.use_sampling_planner:
                    rs = replace(rs, use_sampling_planner=False, precomputed_plan=plan)
            object.__setattr__(self, "asm", asm)
            object.__setattr__(self, "rs", rs)

    def propagate(self, field: Field, distance_um: float) -> Field:
        self.validate_for(field)
        if distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")

        plan = self.precomputed_plan
        if plan is None and (self.rs.use_sampling_planner or self.asm.use_sampling_planner):
            planner = SamplingPlanner(
                nyquist_fraction=self.nyquist_fraction,
                min_padding_factor=self.min_padding_factor,
            )
            plan = planner.recommend_grid(mask_grid=field.grid, spectrum=field.spectrum)

        method = self.precomputed_method
        if method is None:
            decision = PropagationPolicy(
                mode=self.policy_mode,
                equality_tolerance=self.equality_tolerance,
                rs_cost_ratio_threshold=self.rs_cost_ratio_threshold,
            ).choose(
                grid=field.grid,
                spectrum=field.spectrum,
                distance_um=distance_um,
                plan=plan,
            )
            method = decision.method

        if method == "asm":
            return self.asm.propagate(field, distance_um=distance_um)
        return self.rs.propagate(field, distance_um=distance_um)
