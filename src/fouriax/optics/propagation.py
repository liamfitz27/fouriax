from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

import jax.numpy as jnp
from jax.scipy import ndimage as jndimage

from fouriax.core.fft import fftconvolve
from fouriax.optics.interfaces import PropagationModel
from fouriax.optics.model import Field, Grid
from fouriax.optics.planning import PropagationPolicy, SamplingPlanner


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


def _prepare_field_with_planning(field: Field, planner: SamplingPlanner) -> Field:
    """
    Apply planner-driven resampling and padding for propagation stability.

    The resulting propagation grid may have both different shape and different
    spacing (dx_um, dy_um) compared to the input field grid.
    """
    plan = planner.recommend_grid(mask_grid=field.grid, spectrum=field.spectrum)
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
    sampling_planner: SamplingPlanner = dataclass_field(default_factory=SamplingPlanner)

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
        work_field = (
            _prepare_field_with_planning(field, self.sampling_planner)
            if self.use_sampling_planner
            else field
        )

        area_um2 = work_field.grid.dx_um * work_field.grid.dy_um
        outputs = []
        for i, wavelength_um in enumerate(work_field.spectrum.wavelengths_um):
            kernel = self.impulse_response(work_field, float(wavelength_um), distance_um)
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
    sampling_planner: SamplingPlanner = dataclass_field(default_factory=SamplingPlanner)

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
        work_field = (
            _prepare_field_with_planning(field, self.sampling_planner)
            if self.use_sampling_planner
            else field
        )

        outputs = []
        for i, wavelength_um in enumerate(work_field.spectrum.wavelengths_um):
            transfer = self.transfer_function(work_field, float(wavelength_um), distance_um)
            spectrum = jnp.fft.fftn(work_field.data[i], axes=(-2, -1))
            propagated = jnp.fft.ifftn(spectrum * transfer, axes=(-2, -1))
            outputs.append(propagated)

        data = jnp.stack(outputs, axis=0)
        propagated_field = Field(data=data, grid=work_field.grid, spectrum=work_field.spectrum)
        return _restore_to_original_grid(propagated_field, original_grid)


@dataclass(frozen=True)
class AutoPropagator(PropagationModel):
    """
    Propagator wrapper that selects ASM or RS using PropagationPolicy.
    """

    asm: ASMPropagator = dataclass_field(default_factory=ASMPropagator)
    rs: RSPropagator = dataclass_field(default_factory=RSPropagator)
    policy: PropagationPolicy = dataclass_field(default_factory=PropagationPolicy)

    def propagate(self, field: Field, distance_um: float) -> Field:
        self.validate_for(field)
        if distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")

        planner = self.rs.sampling_planner if self.rs.use_sampling_planner else SamplingPlanner()
        plan = planner.recommend_grid(mask_grid=field.grid, spectrum=field.spectrum)
        decision = self.policy.choose(
            grid=field.grid,
            spectrum=field.spectrum,
            distance_um=distance_um,
            plan=plan,
        )
        if decision.method == "asm":
            return self.asm.propagate(field, distance_um=distance_um)
        return self.rs.propagate(field, distance_um=distance_um)
