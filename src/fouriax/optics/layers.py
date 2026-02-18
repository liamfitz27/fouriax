from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import jax.numpy as jnp

from fouriax.optics.interfaces import OpticalLayer, PropagationModel, Sensor
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


@dataclass(frozen=True)
class PropagationLayer(OpticalLayer):
    """Optical layer wrapper around a propagation model and fixed distance."""

    model: PropagationModel
    distance_um: float

    def forward(self, field: Field) -> Field:
        self.validate_for(field)
        if self.distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")
        return self.model.propagate(field, distance_um=self.distance_um)

    def parameters(self) -> dict[str, jnp.ndarray]:
        return {"distance_um": jnp.asarray(self.distance_um, dtype=jnp.float32)}


@dataclass(frozen=True)
class OpticalModule(OpticalLayer):
    """Composable sequence of optical layers."""

    layers: tuple[OpticalLayer | PropagationModel, ...]
    sensor: Sensor | None = None
    auto_apply_na: bool = False
    medium_index: float = 1.0
    na_fallback_to_effective: bool = False

    @staticmethod
    def _propagation_layer_view(layer: OpticalLayer | PropagationModel) -> PropagationLayer | None:
        if isinstance(layer, PropagationLayer):
            return layer
        if isinstance(layer, PropagationModel):
            distance_um = getattr(layer, "distance_um", None)
            if distance_um is None:
                raise ValueError(
                    "PropagationModel used directly in OpticalModule.layers must define "
                    "distance_um, or be wrapped in PropagationLayer"
                )
            if float(distance_um) <= 0:
                raise ValueError("distance_um must be strictly positive")
            return PropagationLayer(model=layer, distance_um=float(distance_um))
        return None

    @staticmethod
    def _spatial_grid_diameter_um(field: Field) -> float:
        return float(min(field.grid.nx * field.grid.dx_um, field.grid.ny * field.grid.dy_um))

    @staticmethod
    def _is_k_layer(layer: OpticalLayer) -> bool:
        return isinstance(layer, (KPhaseMaskLayer, KAmplitudeMaskLayer, KComplexMaskLayer))

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
                z_um += propagation_layer.distance_um
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
                if propagation_layer.distance_um <= 0:
                    raise ValueError("distance_um must be strictly positive")
                z0 = z_um
                z1 = z_um + float(propagation_layer.distance_um)
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

    @staticmethod
    def _layer_with_na_if_supported(
        layer: OpticalLayer | PropagationModel,
        na_limit: float | None,
    ) -> OpticalLayer | PropagationModel:
        propagation_layer = OpticalModule._propagation_layer_view(layer)
        if propagation_layer is None:
            return cast(OpticalLayer | PropagationModel, layer)
        if na_limit is None:
            return propagation_layer
        model = propagation_layer.model
        if not hasattr(model, "na_limit"):
            return propagation_layer
        existing_na = cast(Any, model).na_limit
        merged_na = na_limit if existing_na is None else min(float(existing_na), float(na_limit))
        updated_model = replace(cast(Any, model), na_limit=merged_na)
        return replace(propagation_layer, model=updated_model)

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

    @staticmethod
    def _layer_output_domain(
        layer: OpticalLayer | PropagationModel,
        current_domain: Literal["spatial", "kspace"],
    ) -> Literal["spatial", "kspace"]:
        propagation_layer = OpticalModule._propagation_layer_view(layer)
        if propagation_layer is not None:
            from fouriax.optics.propagation import ASMPropagator, KSpacePropagator, RSPropagator

            if isinstance(propagation_layer.model, (ASMPropagator, RSPropagator)):
                return "spatial"
            if isinstance(propagation_layer.model, KSpacePropagator):
                return "kspace"
            return current_domain
        if isinstance(layer, (PhaseMaskLayer, AmplitudeMaskLayer, ComplexMaskLayer, ThinLensLayer)):
            return "spatial"
        if isinstance(layer, (KPhaseMaskLayer, KAmplitudeMaskLayer, KComplexMaskLayer)):
            return "kspace"
        return current_domain

    @staticmethod
    def _resolve_auto_propagator_layer(
        layer: OpticalLayer | PropagationModel,
        field: Field,
    ) -> OpticalLayer | PropagationModel:
        propagation_layer = OpticalModule._propagation_layer_view(layer)
        if propagation_layer is None:
            return cast(OpticalLayer | PropagationModel, layer)
        from fouriax.optics.propagation import AutoPropagator

        if not isinstance(propagation_layer.model, AutoPropagator):
            return propagation_layer
        resolved_model = propagation_layer.model.resolved_model(
            field=field,
            distance_um=propagation_layer.distance_um,
        )
        return replace(propagation_layer, model=resolved_model)

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
        - Track axial position using `PropagationLayer.distance_um`.
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
class PhaseMaskLayer(OpticalLayer):
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
class AmplitudeMaskLayer(OpticalLayer):
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
class ComplexMaskLayer(OpticalLayer):
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
class KPhaseMaskLayer(OpticalLayer):
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
class KAmplitudeMaskLayer(OpticalLayer):
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
class KComplexMaskLayer(OpticalLayer):
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
class ThinLensLayer(OpticalLayer):
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
