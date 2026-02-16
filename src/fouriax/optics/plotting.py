from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fouriax.optics.layers import PropagationLayer
from fouriax.optics.model import Field

if TYPE_CHECKING:
    from fouriax.optics.layers import OpticalModule


def _field_map(field: Field, mode: str, wavelength_idx: int, log_scale: bool) -> np.ndarray:
    if wavelength_idx < 0 or wavelength_idx >= field.spectrum.size:
        raise ValueError(
            f"wavelength_idx out of range: got {wavelength_idx}, size {field.spectrum.size}"
        )

    if mode == "intensity":
        image = np.asarray(field.intensity()[wavelength_idx])
        if log_scale:
            image = np.log10(np.maximum(image, 1e-12))
        return image
    if mode == "phase":
        return np.asarray(field.phase()[wavelength_idx])
    if mode == "amplitude":
        image = np.asarray(np.abs(field.data[wavelength_idx]))
        if log_scale:
            image = np.log10(np.maximum(image, 1e-12))
        return image
    raise ValueError("mode must be one of: intensity, phase, amplitude")


def _sensor_map(sensor_output: Any, sensor: Any) -> np.ndarray:
    arr = np.asarray(sensor_output)
    if hasattr(sensor, "detector_masks"):
        masks = np.asarray(sensor.detector_masks, dtype=np.float32)
        if masks.ndim == 3:
            values = arr
            if values.ndim == 2:
                values = np.sum(values, axis=0)
            values = values.reshape(-1)
            if values.size == masks.shape[0]:
                return np.asarray(np.sum(masks * values[:, None, None], axis=0), dtype=np.float32)

    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return arr[None, None]
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    return np.asarray(arr[0])


def _phase_mask_image(layer: Any, wavelength_idx: int) -> np.ndarray:
    if not hasattr(layer, "phase_map_rad"):
        raise ValueError("layer does not expose phase_map_rad")
    phase = np.asarray(layer.phase_map_rad)
    if phase.ndim == 0:
        return phase[None, None]
    if phase.ndim == 2:
        return phase
    if phase.ndim == 3:
        if wavelength_idx < 0 or wavelength_idx >= phase.shape[0]:
            raise ValueError(
                f"wavelength_idx out of range for phase_map_rad: got {wavelength_idx}, "
                f"size {phase.shape[0]}"
            )
        return np.asarray(phase[wavelength_idx], dtype=np.float32)
    raise ValueError("phase_map_rad must be scalar, 2D, or 3D")


def plot_field_evolution(
    module: OpticalModule,
    field_in: Field,
    mode: str = "intensity",
    wavelength_idx: int = 0,
    log_scale: bool = False,
    include_input: bool = True,
    include_sensor_output: bool = True,
):
    """Plot field evolution at propagation checkpoints only."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plot_field_evolution") from exc

    items: list[tuple[str, np.ndarray, str]] = []
    phase_items: list[np.ndarray | None] = []
    if include_input:
        items.append(
            (
                "Input",
                _field_map(field_in, mode=mode, wavelength_idx=wavelength_idx, log_scale=log_scale),
                "twilight" if mode == "phase" else "inferno",
            )
        )
        phase_items.append(None)

    output = field_in
    prop_idx = 0
    pending_phase: np.ndarray | None = None
    for layer in module.layers:
        output = layer.forward(output)
        if hasattr(layer, "phase_map_rad"):
            pending_phase = _phase_mask_image(layer, wavelength_idx=wavelength_idx)
        if isinstance(layer, PropagationLayer):
            prop_idx += 1
            items.append(
                (
                    f"After Propagation {prop_idx}",
                    _field_map(
                        output,
                        mode=mode,
                        wavelength_idx=wavelength_idx,
                        log_scale=log_scale,
                    ),
                    "twilight" if mode == "phase" else "inferno",
                )
            )
            phase_items.append(pending_phase)
            pending_phase = None

    if include_sensor_output and module.sensor is not None:
        sensor_output = module.sensor.measure(output)
        items.append(("Sensor Output", _sensor_map(sensor_output, module.sensor), "magma"))
        phase_items.append(None)

    if len(items) == 0:
        raise ValueError("no states to plot")

    n = len(items)
    fig, axes = plt.subplots(2, n, figsize=(max(4.0, 3.2 * n), 6.6), squeeze=False)
    for i in range(n):
        top_ax = axes[0, i]
        title, image, cmap = items[i]
        top_im = top_ax.imshow(image, cmap=cmap)
        top_ax.set_title(title, fontsize=9)
        top_ax.set_xticks([])
        top_ax.set_yticks([])
        plt.colorbar(top_im, ax=top_ax, fraction=0.046, pad=0.03)

        bottom_ax = axes[1, i]
        phase_image = phase_items[i]
        if phase_image is None:
            bottom_ax.axis("off")
            continue
        phase_im = bottom_ax.imshow(phase_image, cmap="twilight")
        bottom_ax.set_title("Phase Mask", fontsize=9)
        bottom_ax.set_xticks([])
        bottom_ax.set_yticks([])
        plt.colorbar(phase_im, ax=bottom_ax, fraction=0.046, pad=0.03)

    scale_text = " (log10)" if log_scale and mode != "phase" else ""
    fig.suptitle(f"Field Evolution: {mode}{scale_text}")
    fig.tight_layout()
    return fig, axes
