from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp

from fouriax.optics.interfaces import OpticalLayer
from fouriax.optics.layers import KSpaceAmplitudeMask, KSpaceComplexMask, KSpacePhaseMask
from fouriax.optics.model import Field


def propagation_layer_view(layer: OpticalLayer) -> OpticalLayer | None:
    from fouriax.optics.propagation import ASMPropagator, KSpacePropagator, RSPropagator

    if isinstance(layer, (ASMPropagator, KSpacePropagator, RSPropagator)):
        distance_um = getattr(layer, "distance_um", None)
        if distance_um is None:
            raise ValueError("propagation layer in stack must define distance_um")
        if float(distance_um) <= 0:
            raise ValueError("distance_um must be strictly positive")
        return layer
    return None


def _spatial_grid_diameter_um(field: Field) -> float:
    return float(min(field.grid.nx * field.grid.dx_um, field.grid.ny * field.grid.dy_um))


def _is_k_layer(layer: OpticalLayer) -> bool:
    return isinstance(layer, (KSpacePhaseMask, KSpaceAmplitudeMask, KSpaceComplexMask))


def layer_stop_diameter_um(layer: OpticalLayer, field: Field) -> float | None:
    grid_diameter_um = _spatial_grid_diameter_um(field)
    aperture_diameter_um = getattr(layer, "aperture_diameter_um", None)

    if _is_k_layer(layer):
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


def collect_stops(
    layers: tuple[OpticalLayer, ...], field: Field
) -> list[tuple[float, float, float]]:
    z_um = 0.0
    stops: list[tuple[float, float, float]] = []
    for layer in layers:
        propagation_layer = propagation_layer_view(layer)
        if propagation_layer is not None:
            z_um += float(getattr(propagation_layer, "distance_um"))  # noqa: B009
            continue
        stop_diameter_um = layer_stop_diameter_um(layer, field)
        if stop_diameter_um is None:
            continue
        radius_um = float(stop_diameter_um) / 2.0
        stops.append((z_um, radius_um, radius_um))
    return stops


def collect_propagation_segments(
    layers: tuple[OpticalLayer, ...],
) -> dict[int, tuple[float, float]]:
    z_um = 0.0
    segments: dict[int, tuple[float, float]] = {}
    for i, layer in enumerate(layers):
        propagation_layer = propagation_layer_view(layer)
        if propagation_layer is not None:
            distance_um = float(getattr(propagation_layer, "distance_um"))  # noqa: B009
            if distance_um <= 0:
                raise ValueError("distance_um must be strictly positive")
            z0 = z_um
            z1 = z_um + distance_um
            segments[i] = (z0, z1)
            z_um = z1
    return segments


def local_na_for_segment(
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


def effective_na(
    layers: tuple[OpticalLayer, ...],
    field: Field | None = None,
    *,
    medium_index: float = 1.0,
) -> float:
    if medium_index <= 0:
        raise ValueError("medium_index must be strictly positive")

    if field is None:
        return float(medium_index)
    stops = collect_stops(layers, field)

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


def na_schedule(
    layers: tuple[OpticalLayer, ...],
    field: Field,
    *,
    medium_index: float = 1.0,
    fallback_to_effective: bool = False,
) -> dict[int, float]:
    if medium_index <= 0:
        raise ValueError("medium_index must be strictly positive")
    stops = collect_stops(layers, field)
    segments = collect_propagation_segments(layers)
    schedule: dict[int, float] = {}
    for i, (z0, z1) in segments.items():
        local_na = local_na_for_segment(
            z0,
            z1,
            stops,
            medium_index=medium_index,
        )
        if local_na is not None:
            schedule[i] = local_na
    if schedule:
        return schedule
    if not fallback_to_effective:
        return {}
    effective = effective_na(layers=layers, field=field, medium_index=medium_index)
    if effective <= 0:
        return {}
    return {i: effective for i in segments}


def layer_with_na_if_supported(layer: OpticalLayer, na_limit: float | None) -> OpticalLayer:
    propagation_layer = propagation_layer_view(layer)
    if propagation_layer is None:
        return layer
    if na_limit is None:
        return propagation_layer
    if not hasattr(propagation_layer, "na_limit"):
        return propagation_layer
    existing_na = getattr(propagation_layer, "na_limit", None)  # noqa: B009
    merged_na = na_limit if existing_na is None else min(float(existing_na), float(na_limit))
    return replace(propagation_layer, na_limit=merged_na)  # type: ignore[type-var]


def apply_na_limits(
    layers: tuple[OpticalLayer, ...],
    schedule: dict[int, float] | None = None,
) -> tuple[OpticalLayer, ...]:
    if not schedule:
        return layers
    return tuple(
        layer_with_na_if_supported(layer, schedule.get(i))
        for i, layer in enumerate(layers)
    )
