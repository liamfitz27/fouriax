from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, cast

import jax.numpy as jnp

from fouriax.core.fft import fftconvolve
from fouriax.optics.interfaces import OpticalLayer, Sensor
from fouriax.optics.model import Field


def _expand_map(
    value: jnp.ndarray | float,
    field: Field,
    *,
    dtype: jnp.dtype,
    name: str,
) -> jnp.ndarray:
    arr = jnp.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        return arr
    if arr.ndim == 2:
        if arr.shape != (field.grid.ny, field.grid.nx):
            raise ValueError(
                f"{name} shape mismatch: got {arr.shape}, expected {(field.grid.ny, field.grid.nx)}"
            )
        return arr[None, :, :]
    if arr.ndim == 3:
        if arr.shape[1:] != (field.grid.ny, field.grid.nx):
            raise ValueError(
                f"{name} shape mismatch: got {arr.shape[1:]}, "
                f"expected {(field.grid.ny, field.grid.nx)}"
            )
        if arr.shape[0] not in (1, field.spectrum.size):
            raise ValueError(
                f"{name} wavelength axis mismatch: got {arr.shape[0]}, expected 1 or "
                f"{field.spectrum.size}"
            )
        return arr
    raise ValueError(f"{name} must be scalar, (ny, nx), or (num_wavelengths, ny, nx)")


def _with_field_metadata(
    data: jnp.ndarray,
    field: Field,
    *,
    domain: Literal["spatial", "kspace"] | None = None,
) -> Field:
    return Field(
        data=data,
        grid=field.grid,
        spectrum=field.spectrum,
        domain=field.domain if domain is None else domain,
        kx_pixel_size_cyc_per_um=field.kx_pixel_size_cyc_per_um,
        ky_pixel_size_cyc_per_um=field.ky_pixel_size_cyc_per_um,
    )


def _expand_psf_for_field(psf: jnp.ndarray, field: Field) -> jnp.ndarray:
    arr = jnp.asarray(psf)
    if arr.ndim == 2:
        if arr.shape != (field.grid.ny, field.grid.nx):
            raise ValueError(
                f"psf shape mismatch: got {arr.shape}, expected {(field.grid.ny, field.grid.nx)}"
            )
        return arr[None, :, :]
    if arr.ndim == 3:
        if arr.shape[1:] != (field.grid.ny, field.grid.nx):
            raise ValueError(
                "psf shape mismatch: got "
                f"{arr.shape[1:]}, expected {(field.grid.ny, field.grid.nx)}"
            )
        if arr.shape[0] not in (1, field.spectrum.size):
            raise ValueError(
                f"psf wavelength axis mismatch: got {arr.shape[0]}, expected 1 or "
                f"{field.spectrum.size}"
            )
        return arr
    raise ValueError("psf must have shape (ny, nx) or (num_wavelengths, ny, nx)")


def _centered_delta_field(field: Field) -> Field:
    data = jnp.zeros_like(field.data)
    cy = field.grid.ny // 2
    cx = field.grid.nx // 2
    data = data.at[:, cy, cx].set(1.0 + 0.0j)
    return Field(
        data=data,
        grid=field.grid,
        spectrum=field.spectrum,
        domain="spatial",
        kx_pixel_size_cyc_per_um=field.kx_pixel_size_cyc_per_um,
        ky_pixel_size_cyc_per_um=field.ky_pixel_size_cyc_per_um,
    )


def _linear_convolve_same_with_otf(
    image_2d: jnp.ndarray,
    otf_2d: jnp.ndarray,
    *,
    kernel_shape: tuple[int, int],
) -> jnp.ndarray:
    ky, kx = kernel_shape
    ny, nx = image_2d.shape
    full_shape = (ny + ky - 1, nx + kx - 1)
    if otf_2d.shape != full_shape:
        raise ValueError(f"otf shape mismatch: got {otf_2d.shape}, expected {full_shape}")
    image_fft = jnp.fft.fftn(image_2d, s=full_shape, axes=(-2, -1))
    full = jnp.fft.ifftn(image_fft * otf_2d, s=full_shape, axes=(-2, -1))
    y0 = (ky - 1) // 2
    x0 = (kx - 1) // 2
    return full[y0 : y0 + ny, x0 : x0 + nx].real


@dataclass(frozen=True)
class OpticalModule(OpticalLayer):
    """Composable sequence of optical layers."""

    layers: tuple[OpticalLayer, ...]
    sensor: Sensor | None = None
    auto_apply_na: bool = False
    medium_index: float = 1.0
    na_fallback_to_effective: bool = False

    @classmethod
    def _propagation_layer_view(cls, layer: OpticalLayer) -> OpticalLayer | None:
        from fouriax.optics.propagation import (
            ASMPropagator,
            AutoPropagator,
            CoherentPropagator,
            KSpacePropagator,
            RSPropagator,
        )

        if isinstance(
            layer,
            (
                ASMPropagator,
                AutoPropagator,
                CoherentPropagator,
                KSpacePropagator,
                RSPropagator,
            ),
        ):
            distance_um = getattr(layer, "distance_um", None)
            if distance_um is None:
                raise ValueError(
                    " layer in OpticalModule.layers must define "
                    "distance_um"
                )
            if float(distance_um) <= 0:
                raise ValueError("distance_um must be strictly positive")
            return layer
        return None

    @staticmethod
    def _spatial_grid_diameter_um(field: Field) -> float:
        return float(min(field.grid.nx * field.grid.dx_um, field.grid.ny * field.grid.dy_um))

    @staticmethod
    def _is_k_layer(layer: OpticalLayer) -> bool:
        return isinstance(layer, (KSpacePhaseMask, KSpaceAmplitudeMask, KSpaceComplexMask))

    def _layer_stop_diameter_um(self, layer: OpticalLayer, field: Field) -> float | None:
        grid_diameter_um = self._spatial_grid_diameter_um(field)
        aperture_diameter_um = getattr(layer, "aperture_diameter_um", None)

        if self._is_k_layer(layer):
            if aperture_diameter_um is None:
                return None
            if aperture_diameter_um <= 0:
                raise ValueError("k-space layer aperture_diameter_um must be strictly positive")
            return float(aperture_diameter_um)

        if aperture_diameter_um is None:
            return grid_diameter_um
        if aperture_diameter_um <= 0:
            raise ValueError("aperture_diameter_um must be strictly positive")
        return float(min(aperture_diameter_um, grid_diameter_um))

    def _collect_stops(self, field: Field) -> list[tuple[float, float, float]]:
        z_um = 0.0
        stops: list[tuple[float, float, float]] = []
        for layer in self.layers:
            propagation_layer = self._propagation_layer_view(layer)
            if propagation_layer is not None:
                z_um += float(getattr(propagation_layer, "distance_um"))  # noqa: B009
                continue
            stop_diameter_um = self._layer_stop_diameter_um(cast(OpticalLayer, layer), field)
            if stop_diameter_um is None:
                continue
            radius_um = float(stop_diameter_um) / 2.0
            stops.append((z_um, radius_um, radius_um))
        return stops

    def _collect_propagation_segments(self) -> dict[int, tuple[float, float]]:
        z_um = 0.0
        segments: dict[int, tuple[float, float]] = {}
        for i, layer in enumerate(self.layers):
            propagation_layer = self._propagation_layer_view(layer)
            if propagation_layer is not None:
                distance_um = float(getattr(propagation_layer, "distance_um"))  # noqa: B009
                if distance_um <= 0:
                    raise ValueError("distance_um must be strictly positive")
                z0 = z_um
                z1 = z_um + distance_um
                segments[i] = (z0, z1)
                z_um = z1
        return segments

    @staticmethod
    def _local_na_for_segment(
        z0_um: float,
        z1_um: float,
        stops: list[tuple[float, float, float]],
        *,
        medium_index: float,
    ) -> float | None:
        if len(stops) < 2:
            return None
        z_mid = 0.5 * (z0_um + z1_um)
        prev_stop = None
        next_stop = None
        for stop in stops:
            if stop[0] <= z_mid:
                prev_stop = stop
            if stop[0] >= z_mid and next_stop is None:
                next_stop = stop
        if prev_stop is None or next_stop is None:
            return None
        dz_um = next_stop[0] - prev_stop[0]
        if dz_um <= 0:
            return None
        theta_x = jnp.arctan(min(prev_stop[1], next_stop[1]) / dz_um)
        theta_y = jnp.arctan(min(prev_stop[2], next_stop[2]) / dz_um)
        na_x = float(medium_index * jnp.sin(theta_x))
        na_y = float(medium_index * jnp.sin(theta_y))
        return float(min(na_x, na_y, medium_index))

    def na_schedule(self, field: Field) -> dict[int, float]:
        """
        Compute per-propagation-layer local NA limits from adjacent stop geometry.

        Keys are layer indices in `self.layers`.
        """
        if self.medium_index <= 0:
            raise ValueError("medium_index must be strictly positive")
        stops = self._collect_stops(field)
        segments = self._collect_propagation_segments()
        schedule: dict[int, float] = {}
        for i, (z0, z1) in segments.items():
            local_na = self._local_na_for_segment(
                z0,
                z1,
                stops,
                medium_index=self.medium_index,
            )
            if local_na is not None:
                schedule[i] = local_na
        if schedule:
            return schedule
        if not self.na_fallback_to_effective:
            return {}
        effective = self.effective_na(field=field, medium_index=self.medium_index)
        if effective <= 0:
            return {}
        return {i: effective for i in segments}

    @classmethod
    def _layer_with_na_if_supported(
        cls,
        layer: OpticalLayer,
        na_limit: float | None,
    ) -> OpticalLayer:
        propagation_layer = cls._propagation_layer_view(layer)
        if propagation_layer is None:
            return layer
        if na_limit is None:
            return propagation_layer
        if not hasattr(propagation_layer, "na_limit"):
            return propagation_layer
        existing_na = getattr(propagation_layer, "na_limit", None)  # noqa: B009
        merged_na = na_limit if existing_na is None else min(float(existing_na), float(na_limit))
        return replace(propagation_layer, na_limit=merged_na)  # type: ignore[type-var]

    @staticmethod
    def _field_with_domain(field: Field, domain: Literal["spatial", "kspace"]) -> Field:
        if field.domain == domain:
            return field
        return Field(
            data=field.data,
            grid=field.grid,
            spectrum=field.spectrum,
            domain=domain,
            kx_pixel_size_cyc_per_um=field.kx_pixel_size_cyc_per_um,
            ky_pixel_size_cyc_per_um=field.ky_pixel_size_cyc_per_um,
        )

    @classmethod
    def _layer_output_domain(
        cls,
        layer: OpticalLayer,
        current_domain: Literal["spatial", "kspace"],
    ) -> Literal["spatial", "kspace"]:
        propagation_layer = cls._propagation_layer_view(layer)
        if propagation_layer is not None:
            from fouriax.optics.propagation import ASMPropagator, KSpacePropagator, RSPropagator

            if isinstance(propagation_layer, (ASMPropagator, RSPropagator)):
                return "spatial"
            if isinstance(propagation_layer, KSpacePropagator):
                return "kspace"
            return current_domain
        if isinstance(
            layer,
            (PhaseMask, AmplitudeMask, ComplexMask, ThinLens, IncoherentImager),
        ):
            return "spatial"
        if isinstance(layer, (KSpacePhaseMask, KSpaceAmplitudeMask, KSpaceComplexMask)):
            return "kspace"
        return current_domain

    @classmethod
    def _resolve_auto_propagator_layer(
        cls,
        layer: OpticalLayer,
        field: Field,
    ) -> OpticalLayer:
        propagation_layer = cls._propagation_layer_view(layer)
        if propagation_layer is None:
            return layer
        from fouriax.optics.propagation import AutoPropagator

        if not isinstance(propagation_layer, AutoPropagator):
            return propagation_layer
        resolved_model = propagation_layer.resolved_model(
            field=field,
            distance_um=float(getattr(propagation_layer, "distance_um")),  # noqa: B009
        )
        return replace(resolved_model, distance_um=getattr(propagation_layer, "distance_um"))  # noqa: B009

    def planned_layers(self, field: Field) -> tuple[OpticalLayer, ...]:
        """
        Build a concrete per-layer execution plan for this input field.

        This resolves AutoPropagator layers to concrete propagation models up front
        and injects per-segment NA limits before executing data-path operations.
        """
        self.validate_for(field)
        schedule = self.na_schedule(field) if self.auto_apply_na else {}
        plan: list[OpticalLayer] = []
        current_domain: Literal["spatial", "kspace"] = field.domain
        for i, layer in enumerate(self.layers):
            planning_field = self._field_with_domain(field, current_domain)
            resolved = self._resolve_auto_propagator_layer(layer, planning_field)
            resolved = self._layer_with_na_if_supported(resolved, schedule.get(i))
            propagation_layer = self._propagation_layer_view(resolved)
            plan.append(
                propagation_layer
                if propagation_layer is not None
                else cast(OpticalLayer, resolved)
            )
            current_domain = self._layer_output_domain(resolved, current_domain)
        return tuple(plan)

    def forward(self, field: Field) -> Field:
        plan = self.planned_layers(field)
        output = field
        for layer in plan:
            output = layer.forward(output)
        return output

    def measure(self, field: Field) -> jnp.ndarray:
        """Forward through layers and apply the configured sensor."""
        if self.sensor is None:
            raise ValueError("OpticalModule has no sensor configured")
        output = self.forward(field)
        return self.sensor.measure(output)

    def trace(self, field: Field, include_input: bool = True) -> list[Field]:
        """Return intermediate fields through the module."""
        plan = self.planned_layers(field)
        output = field
        states: list[Field] = [output] if include_input else []
        for layer in plan:
            output = layer.forward(output)
            states.append(output)
        return states

    def parameters(self) -> dict[str, jnp.ndarray]:
        params: dict[str, jnp.ndarray] = {}
        for i, layer in enumerate(self.layers):
            propagation_layer = self._propagation_layer_view(layer)
            layer_obj = (
                propagation_layer if propagation_layer is not None else cast(OpticalLayer, layer)
            )
            for key, value in layer_obj.parameters().items():
                params[f"layer_{i}.{key}"] = value
        return params

    def effective_na(self, field: Field | None = None, medium_index: float = 1.0) -> float:
        """
        Estimate effective system NA from aperture stops and propagation distances.

        This computes a geometric, consecutive-stop estimate:
        - Collect aperture stops from layers that expose `aperture_diameter_um`.
        - Track axial position using `.distance_um`.
        - Compute limiting angle between consecutive stops.
        """
        if medium_index <= 0:
            raise ValueError("medium_index must be strictly positive")

        if field is None:
            return float(medium_index)
        stops = self._collect_stops(field)

        if len(stops) < 2:
            return float(medium_index)

        na_limits: list[float] = []
        for i in range(len(stops) - 1):
            z0, ax0, ay0 = stops[i]
            z1, ax1, ay1 = stops[i + 1]
            dz_um = z1 - z0
            if dz_um <= 0:
                continue

            theta_x = jnp.arctan(min(ax0, ax1) / dz_um)
            theta_y = jnp.arctan(min(ay0, ay1) / dz_um)
            na_x = float(medium_index * jnp.sin(theta_x))
            na_y = float(medium_index * jnp.sin(theta_y))
            na_limits.extend([na_x, na_y])

        if not na_limits:
            return float(medium_index)
        return float(min(min(na_limits), medium_index))


@dataclass(frozen=True)
class PhaseMask(OpticalLayer):
    """Generic phase-only modulation layer."""

    phase_map_rad: jnp.ndarray | float

    def forward(self, field: Field) -> Field:
        field = field.to_spatial()
        self.validate_for(field)
        phase = _expand_map(
            self.phase_map_rad,
            field,
            dtype=jnp.float32,
            name="phase_map_rad",
        )
        return field.apply_phase(phase)

    def parameters(self) -> dict[str, jnp.ndarray]:
        return {"phase_map_rad": jnp.asarray(self.phase_map_rad, dtype=jnp.float32)}


@dataclass(frozen=True)
class AmplitudeMask(OpticalLayer):
    """Generic amplitude modulation layer."""

    amplitude_map: jnp.ndarray | float

    def forward(self, field: Field) -> Field:
        field = field.to_spatial()
        self.validate_for(field)
        amplitude = _expand_map(
            self.amplitude_map,
            field,
            dtype=field.data.real.dtype,
            name="amplitude_map",
        )
        return field.apply_amplitude(amplitude)

    def parameters(self) -> dict[str, jnp.ndarray]:
        return {"amplitude_map": jnp.asarray(self.amplitude_map, dtype=jnp.float32)}


@dataclass(frozen=True)
class ComplexMask(OpticalLayer):
    """Generic complex-valued modulation layer (amplitude and phase)."""

    amplitude_map: jnp.ndarray | float = 1.0
    phase_map_rad: jnp.ndarray | float = 0.0

    def forward(self, field: Field) -> Field:
        field = field.to_spatial()
        self.validate_for(field)
        amplitude = _expand_map(
            self.amplitude_map,
            field,
            dtype=field.data.real.dtype,
            name="amplitude_map",
        )
        phase = _expand_map(
            self.phase_map_rad,
            field,
            dtype=jnp.float32,
            name="phase_map_rad",
        )
        return field.apply_amplitude(amplitude).apply_phase(phase)

    def parameters(self) -> dict[str, jnp.ndarray]:
        return {
            "amplitude_map": jnp.asarray(self.amplitude_map, dtype=jnp.float32),
            "phase_map_rad": jnp.asarray(self.phase_map_rad, dtype=jnp.float32),
        }


@dataclass(frozen=True)
class KSpacePhaseMask(OpticalLayer):
    """k-domain phase-only modulation layer."""

    phase_map_rad: jnp.ndarray | float
    aperture_diameter_um: float | None = None

    def forward(self, field: Field) -> Field:
        field = field.to_kspace()
        self.validate_for(field)
        phase = _expand_map(
            self.phase_map_rad,
            field,
            dtype=jnp.float32,
            name="phase_map_rad",
        )
        data = field.data * jnp.exp(1j * phase)
        return _with_field_metadata(data, field)

    def parameters(self) -> dict[str, jnp.ndarray]:
        params = {"phase_map_rad": jnp.asarray(self.phase_map_rad, dtype=jnp.float32)}
        if self.aperture_diameter_um is not None:
            params["aperture_diameter_um"] = jnp.asarray(
                self.aperture_diameter_um, dtype=jnp.float32
            )
        return params


@dataclass(frozen=True)
class KSpaceAmplitudeMask(OpticalLayer):
    """k-domain amplitude modulation layer."""

    amplitude_map: jnp.ndarray | float
    aperture_diameter_um: float | None = None

    def forward(self, field: Field) -> Field:
        field = field.to_kspace()
        self.validate_for(field)
        amplitude = _expand_map(
            self.amplitude_map,
            field,
            dtype=field.data.real.dtype,
            name="amplitude_map",
        )
        data = field.data * amplitude
        return _with_field_metadata(data, field)

    def parameters(self) -> dict[str, jnp.ndarray]:
        params = {"amplitude_map": jnp.asarray(self.amplitude_map, dtype=jnp.float32)}
        if self.aperture_diameter_um is not None:
            params["aperture_diameter_um"] = jnp.asarray(
                self.aperture_diameter_um, dtype=jnp.float32
            )
        return params


@dataclass(frozen=True)
class KSpaceComplexMask(OpticalLayer):
    """k-domain complex-valued modulation layer (amplitude and phase)."""

    amplitude_map: jnp.ndarray | float = 1.0
    phase_map_rad: jnp.ndarray | float = 0.0
    aperture_diameter_um: float | None = None

    def forward(self, field: Field) -> Field:
        field = field.to_kspace()
        self.validate_for(field)
        amplitude = _expand_map(
            self.amplitude_map,
            field,
            dtype=field.data.real.dtype,
            name="amplitude_map",
        )
        phase = _expand_map(
            self.phase_map_rad,
            field,
            dtype=jnp.float32,
            name="phase_map_rad",
        )
        data = field.data * amplitude * jnp.exp(1j * phase)
        return _with_field_metadata(data, field)

    def parameters(self) -> dict[str, jnp.ndarray]:
        params = {
            "amplitude_map": jnp.asarray(self.amplitude_map, dtype=jnp.float32),
            "phase_map_rad": jnp.asarray(self.phase_map_rad, dtype=jnp.float32),
        }
        if self.aperture_diameter_um is not None:
            params["aperture_diameter_um"] = jnp.asarray(
                self.aperture_diameter_um, dtype=jnp.float32
            )
        return params


@dataclass(frozen=True)
class ThinLens(OpticalLayer):
    """
    Ideal thin lens phase transform.

    All length quantities use micrometers (um).
    """

    focal_length_um: float
    aperture_diameter_um: float | None = None

    def forward(self, field: Field) -> Field:
        field = field.to_spatial()
        self.validate_for(field)
        if self.focal_length_um <= 0:
            raise ValueError("focal_length_um must be strictly positive")

        x, y = field.grid.spatial_grid()
        r2 = x * x + y * y
        phase_stack = []
        for wavelength_um in field.spectrum.wavelengths_um:
            phi = -jnp.pi * r2 / (wavelength_um * self.focal_length_um)
            phase_stack.append(phi)
        phase = jnp.stack(phase_stack, axis=0)

        transformed = field.apply_phase(phase)

        if self.aperture_diameter_um is not None:
            if self.aperture_diameter_um <= 0:
                raise ValueError("aperture_diameter_um must be strictly positive")
            radius = self.aperture_diameter_um / 2.0
            aperture = (r2 <= radius * radius).astype(transformed.data.real.dtype)
            transformed = transformed.apply_amplitude(aperture[None, :, :])

        return transformed

    def parameters(self) -> dict[str, jnp.ndarray]:
        params: dict[str, jnp.ndarray] = {
            "focal_length_um": jnp.asarray(self.focal_length_um, dtype=jnp.float32)
        }
        if self.aperture_diameter_um is not None:
            params["aperture_diameter_um"] = jnp.asarray(
                self.aperture_diameter_um, dtype=jnp.float32
            )
        return params


@dataclass(frozen=True)
class IncoherentImager(OpticalLayer):
    """
    Incoherent shift-invariant imager built from coherent optics + propagation.

    The PSF is constructed internally by propagating a calibration source through:
    `optical_layer -> propagator(distance_um)` and taking output intensity.
    """

    optical_layer: OpticalLayer
    propagator: OpticalLayer
    distance_um: float
    psf_source: Literal["impulse", "plane_wave_focus"] = "impulse"
    normalize_psf: bool = True
    enforce_nonnegative_psf: bool = True
    mode: Literal["psf", "otf", "auto"] = "auto"
    normalization_reference: Literal["near_1um", "near_wavelength", "at_imaging_distance"] = (
        "near_wavelength"
    )
    normalization_reference_distance_um: float | None = None

    def normalization_distance_um(self, field: Field) -> float:
        """
        Return the propagation distance used to compute PSF normalization sums.

        This can be kept near-field (e.g. 1 um or wavelength-scale) so most power
        remains inside the simulated grid, reducing normalization bias from far-field
        truncation.
        """
        if self.normalization_reference_distance_um is not None:
            if self.normalization_reference_distance_um <= 0:
                raise ValueError("normalization_reference_distance_um must be strictly positive")
            return float(self.normalization_reference_distance_um)
        if self.normalization_reference == "near_1um":
            return 1.0
        if self.normalization_reference == "near_wavelength":
            return max(1e-3, float(jnp.min(field.spectrum.wavelengths_um)))
        if self.normalization_reference == "at_imaging_distance":
            return float(self.distance_um)
        raise ValueError(
            "normalization_reference must be one of: "
            "near_1um, near_wavelength, at_imaging_distance"
        )

    def build_psf(self, field: Field) -> jnp.ndarray:
        spatial_field = field.to_spatial()
        self.validate_for(spatial_field)
        if self.distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")
        if self.psf_source not in ("impulse", "plane_wave_focus"):
            raise ValueError("psf_source must be one of: impulse, plane_wave_focus")

        if self.psf_source == "impulse":
            source = _centered_delta_field(spatial_field)
        else:
            source = Field.plane_wave(grid=spatial_field.grid, spectrum=spatial_field.spectrum)

        coherent = self.optical_layer.forward(source)
        propagator = replace(self.propagator, distance_um=self.distance_um)  # type: ignore[type-var]
        response = propagator.forward(coherent).to_spatial()
        psf = response.intensity().astype(spatial_field.data.real.dtype)

        if self.enforce_nonnegative_psf:
            psf = jnp.maximum(psf, 0.0)
        if self.normalize_psf:
            norm_distance_um = self.normalization_distance_um(spatial_field)
            if abs(norm_distance_um - self.distance_um) < 1e-12:
                sums = jnp.sum(psf, axis=(-2, -1), keepdims=True)
            else:
                ref_propagator = replace(self.propagator, distance_um=norm_distance_um)  # type: ignore[type-var]
                ref_response = ref_propagator.forward(coherent).to_spatial()
                ref_psf = ref_response.intensity().astype(spatial_field.data.real.dtype)
                if self.enforce_nonnegative_psf:
                    ref_psf = jnp.maximum(ref_psf, 0.0)
                sums = jnp.sum(ref_psf, axis=(-2, -1), keepdims=True)
            psf = psf / jnp.maximum(sums, 1e-12)
        return psf

    def forward(self, field: Field) -> Field:
        spatial_field = field.to_spatial()
        self.validate_for(spatial_field)
        psf = _expand_psf_for_field(self.build_psf(spatial_field), spatial_field)
        intensity = spatial_field.intensity().astype(spatial_field.data.real.dtype)

        if self.mode not in ("psf", "otf", "auto"):
            raise ValueError("mode must be one of: psf, otf, auto")
        resolved_mode = self.mode
        if resolved_mode == "auto":
            # FFT-domain imaging becomes favorable for larger kernels/grids.
            resolved_mode = (
                "otf"
                if (spatial_field.grid.nx * spatial_field.grid.ny) >= 4096
                else "psf"
            )

        outputs: list[jnp.ndarray] = []
        if resolved_mode == "otf":
            ky, kx = spatial_field.grid.ny, spatial_field.grid.nx
            full_shape = (2 * ky - 1, 2 * kx - 1)
            otf = jnp.stack(
                [
                    jnp.fft.fftn(
                        psf[0] if psf.shape[0] == 1 else psf[i],
                        s=full_shape,
                        axes=(-2, -1),
                    )
                    for i in range(spatial_field.spectrum.size)
                ],
                axis=0,
            )
            for i in range(spatial_field.spectrum.size):
                outputs.append(
                    _linear_convolve_same_with_otf(
                        intensity[i],
                        otf[i],
                        kernel_shape=(ky, kx),
                    )
                )
        else:
            for i in range(spatial_field.spectrum.size):
                psf_i = psf[0] if psf.shape[0] == 1 else psf[i]
                outputs.append(fftconvolve(intensity[i], psf_i, mode="same", axes=(-2, -1)))

        image = jnp.stack(outputs, axis=0)
        amplitude = jnp.sqrt(jnp.maximum(image, 0.0)).astype(spatial_field.data.real.dtype)
        return Field(
            data=amplitude.astype(spatial_field.data.dtype),
            grid=spatial_field.grid,
            spectrum=spatial_field.spectrum,
            domain="spatial",
            kx_pixel_size_cyc_per_um=spatial_field.kx_pixel_size_cyc_per_um,
            ky_pixel_size_cyc_per_um=spatial_field.ky_pixel_size_cyc_per_um,
        )

    def parameters(self) -> dict[str, jnp.ndarray]:
        params: dict[str, jnp.ndarray] = {
            "distance_um": jnp.asarray(self.distance_um, dtype=jnp.float32),
        }
        for key, value in self.optical_layer.parameters().items():
            params[f"optical_layer.{key}"] = value
        return params
