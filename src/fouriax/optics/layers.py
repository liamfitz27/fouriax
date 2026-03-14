from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, cast

import jax
import jax.numpy as jnp

from fouriax.fft import fftconvolve, fftconvolve_same_with_otf
from fouriax.linop import LinearOperator
from fouriax.optics.interfaces import IncoherentLayer, Monitor, OpticalLayer, Sensor
from fouriax.optics.model import Field, Grid, Intensity


def _expand_map(
    value: jnp.ndarray | float,
    field: Field,
    *,
    dtype: jnp.dtype,
    name: str,
) -> jnp.ndarray:
    arr = jnp.asarray(value, dtype=dtype)
    if field.is_jones:
        if arr.ndim == 0:
            return arr
        if arr.ndim == 2:
            if arr.shape != (field.grid.ny, field.grid.nx):
                raise ValueError(
                    f"{name} shape mismatch: got {arr.shape}, expected "
                    f"{(field.grid.ny, field.grid.nx)}"
                )
            return arr[None, None, :, :]
        if arr.ndim == 3:
            if arr.shape[1:] != (field.grid.ny, field.grid.nx):
                raise ValueError(
                    f"{name} shape mismatch: got {arr.shape[1:]}, "
                    f"expected {(field.grid.ny, field.grid.nx)}"
                )
            if arr.shape[0] not in (1, 2, field.spectrum.size):
                raise ValueError(
                    f"{name} leading-axis mismatch: got {arr.shape[0]}, expected 1, 2, "
                    f"or {field.spectrum.size}"
                )
            if arr.shape[0] == 2:
                return arr[None, :, :, :]
            return arr[:, None, :, :]
        if arr.ndim == 4:
            if arr.shape[2:] != (field.grid.ny, field.grid.nx):
                raise ValueError(
                    f"{name} shape mismatch: got {arr.shape[2:]}, "
                    f"expected {(field.grid.ny, field.grid.nx)}"
                )
            if arr.shape[0] not in (1, field.spectrum.size):
                raise ValueError(
                    f"{name} wavelength axis mismatch: got {arr.shape[0]}, expected 1 or "
                    f"{field.spectrum.size}"
                )
            if arr.shape[1] not in (1, 2):
                raise ValueError(
                    f"{name} polarization axis mismatch: got {arr.shape[1]}, expected 1 or 2"
                )
            return arr
        raise ValueError(
            f"{name} must be scalar, (ny, nx), (k, ny, nx), or (wavelength, pol, ny, nx)"
        )

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


def _expand_jones_matrix_map(
    jones_matrix: jnp.ndarray,
    field: Field,
    *,
    name: str,
) -> jnp.ndarray:
    arr = jnp.asarray(jones_matrix, dtype=field.data.dtype)
    if not field.is_jones:
        raise ValueError(f"{name} requires polarization_mode='jones'")
    if arr.ndim == 2:
        if arr.shape != (2, 2):
            raise ValueError(f"{name} shape mismatch: got {arr.shape}, expected (2, 2)")
        return arr
    if arr.ndim == 4:
        if arr.shape[:2] != (2, 2) or arr.shape[2:] != (field.grid.ny, field.grid.nx):
            raise ValueError(
                f"{name} shape mismatch: got {arr.shape}, expected (2, 2, ny, nx)"
            )
        return arr[None, :, :, :, :]
    if arr.ndim == 5:
        if arr.shape[0] not in (1, field.spectrum.size):
            raise ValueError(
                f"{name} wavelength axis mismatch: got {arr.shape[0]}, expected 1 or "
                f"{field.spectrum.size}"
            )
        if arr.shape[1:3] != (2, 2) or arr.shape[3:] != (field.grid.ny, field.grid.nx):
            raise ValueError(
                f"{name} shape mismatch: got {arr.shape}, expected (w, 2, 2, ny, nx)"
            )
        return arr
    raise ValueError(
        f"{name} must have shape (2, 2), (2, 2, ny, nx), or (wavelength, 2, 2, ny, nx)"
    )


def _require_domain(
    field: Field,
    *,
    expected: Literal["spatial", "kspace"],
    layer_name: str,
) -> None:
    if field.domain == expected:
        return
    if expected == "spatial":
        raise ValueError(
            f"{layer_name} requires spatial-domain input; "
            "insert InverseFourierTransform before this layer."
        )
    raise ValueError(
        f"{layer_name} requires kspace-domain input; "
        "insert FourierTransform before this layer."
    )


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
        polarization_mode=field.polarization_mode,
        domain=field.domain if domain is None else domain,
        kx_pixel_size_cyc_per_um=field.kx_pixel_size_cyc_per_um,
        ky_pixel_size_cyc_per_um=field.ky_pixel_size_cyc_per_um,
    )


def _expand_psf_for_intensity(psf: Intensity) -> jnp.ndarray:
    psf.validate()
    arr = jnp.asarray(psf.data)
    if arr.ndim == 2:
        if arr.shape != (psf.grid.ny, psf.grid.nx):
            raise ValueError(
                f"psf shape mismatch: got {arr.shape}, expected {(psf.grid.ny, psf.grid.nx)}"
            )
        return arr[None, :, :]
    if arr.ndim == 3:
        if arr.shape[1:] != (psf.grid.ny, psf.grid.nx):
            raise ValueError(
                "psf shape mismatch: got "
                f"{arr.shape[1:]}, expected {(psf.grid.ny, psf.grid.nx)}"
            )
        if arr.shape[0] not in (1, psf.spectrum.size):
            raise ValueError(
                f"psf wavelength axis mismatch: got {arr.shape[0]}, expected 1 or "
                f"{psf.spectrum.size}"
            )
        return arr
    raise ValueError("psf must have shape (ny, nx) or (num_wavelengths, ny, nx)")


def _centered_delta_field(field: Field) -> Field:
    data = jnp.zeros_like(field.data)
    cy = field.grid.ny // 2
    cx = field.grid.nx // 2
    data = data.at[..., cy, cx].set(1.0 + 0.0j)
    return Field(
        data=data,
        grid=field.grid,
        spectrum=field.spectrum,
        polarization_mode=field.polarization_mode,
        domain="spatial",
        kx_pixel_size_cyc_per_um=field.kx_pixel_size_cyc_per_um,
        ky_pixel_size_cyc_per_um=field.ky_pixel_size_cyc_per_um,
    )


def _point_source_field(
    field: Field,
    *,
    distance_um: float,
    source_x_um: float = 0.0,
    source_y_um: float = 0.0,
) -> Field:
    if distance_um <= 0:
        raise ValueError("object_distance_um must be strictly positive")

    x, y = field.grid.spatial_grid()
    r = jnp.sqrt((x - source_x_um) ** 2 + (y - source_y_um) ** 2 + distance_um**2)
    waves = []
    for wavelength_um in field.spectrum.wavelengths_um:
        k_um_inv = (2.0 * jnp.pi) / jnp.asarray(wavelength_um, dtype=field.data.real.dtype)
        waves.append(jnp.exp(1j * k_um_inv * r) / jnp.maximum(r, 1e-12))
    data = jnp.stack(waves, axis=0).astype(field.data.dtype)
    return Field(
        data=data,
        grid=field.grid,
        spectrum=field.spectrum,
        polarization_mode=field.polarization_mode,
        domain="spatial",
        kx_pixel_size_cyc_per_um=field.kx_pixel_size_cyc_per_um,
        ky_pixel_size_cyc_per_um=field.ky_pixel_size_cyc_per_um,
    )


@dataclass(frozen=True)
class OpticalModule(OpticalLayer):
    """Sequential container for optical layers and monitor checkpoints.

    Layers are executed in order via :meth:`forward`.  ``Monitor``
    instances in the layer tuple are skipped during forward propagation
    but can be read via :meth:`observe`.  An optional ``Sensor`` can be
    attached for end-to-end measurement via :meth:`measure`.

    Args:
        layers: Ordered sequence of ``OpticalLayer`` and ``Monitor``
            stages.
        sensor: Optional detector applied after the last layer.
    """

    layers: tuple[OpticalLayer | Monitor, ...]
    sensor: Sensor | None = None

    def forward(self, field: Field) -> Field:
        """Run the non-monitor stages in sequence.

        Args:
            field: Input field for the module.

        Returns:
            Output of the last non-monitor layer. Inline monitors are skipped
            and do not modify the field.
        """
        output = field
        for stage in self.layers:
            if isinstance(stage, Monitor):
                continue
            output = stage.forward(output)
        return output

    def measure(self, field: Field, *, key: jax.Array | None = None) -> jnp.ndarray:
        """Forward through all layers and apply the configured sensor.

        Args:
            field: Input optical field.
            key: Optional PRNG key forwarded to the sensor for noise.

        Raises:
            ValueError: If no sensor is configured.
        """
        if self.sensor is None:
            raise ValueError("OpticalModule has no sensor configured")
        output = self.forward(field)
        return self.sensor.measure(output, key=key)

    def observe(self, field: Field) -> tuple[Field, tuple[jnp.ndarray, ...]]:
        """Forward through all layers and collect monitor readouts.

        Returns:
            A tuple ``(output_field, monitor_readings)`` where
            *monitor_readings* contains one entry per ``Monitor``
            in the layer stack, in encounter order.
        """
        output = field
        observed: list[jnp.ndarray] = []
        for stage in self.layers:
            if isinstance(stage, Monitor):
                observed.append(stage.read(output))
                continue
            output = stage.forward(output)

        return output, tuple(observed)

    def trace(self, field: Field, include_input: bool = True) -> list[Field]:
        """Return the intermediate field after each layer.

        Args:
            field: Input optical field.
            include_input: If ``True`` (default) the input field is
                included as the first element.

        Returns:
            List of ``Field`` objects, one per layer (plus input if
            requested).
        """
        output = field
        states: list[Field] = [output] if include_input else []
        for stage in self.layers:
            if isinstance(stage, Monitor):
                continue
            output = stage.forward(output)
            states.append(output)
        return states

    def parameters(self) -> dict[str, jnp.ndarray]:
        """Collect parameter dictionaries from non-monitor layers.

        Returns:
            Flat dictionary keyed as ``layer_<index>.<parameter_name>``.
        """
        params: dict[str, jnp.ndarray] = {}
        for i, stage in enumerate(self.layers):
            if isinstance(stage, Monitor):
                continue
            for key, value in stage.parameters().items():
                params[f"layer_{i}.{key}"] = value
        return params


@dataclass(frozen=True)
class PhaseMask(OpticalLayer):
    """Spatial-domain phase-only modulation layer.

    Multiplies the field by ``exp(j * phase_map_rad)``.  Requires
    spatial-domain input.

    Args:
        phase_map_rad: Phase in radians.  Accepted shapes: scalar,
            ``(ny, nx)``, or ``(num_wavelengths, ny, nx)``.
    """

    phase_map_rad: jnp.ndarray | float

    def forward(self, field: Field) -> Field:
        """Apply the phase mask to a spatial-domain field."""
        _require_domain(field, expected="spatial", layer_name="PhaseMask")
        self.validate_for(field)
        phase = _expand_map(
            self.phase_map_rad,
            field,
            dtype=jnp.float32,
            name="phase_map_rad",
        )
        return field.apply_phase(phase)

    def parameters(self) -> dict[str, jnp.ndarray]:
        """Return the phase map as the exposed layer parameter."""
        return {"phase_map_rad": jnp.asarray(self.phase_map_rad, dtype=jnp.float32)}


@dataclass(frozen=True)
class AmplitudeMask(OpticalLayer):
    """Spatial-domain amplitude modulation layer.

    Multiplies the field amplitude by a real-valued mask.  Requires
    spatial-domain input.

    Args:
        amplitude_map: Amplitude transmission.  Accepted shapes: scalar,
            ``(ny, nx)``, or ``(num_wavelengths, ny, nx)``.
    """

    amplitude_map: jnp.ndarray | float

    def forward(self, field: Field) -> Field:
        """Apply the amplitude mask to a spatial-domain field."""
        _require_domain(field, expected="spatial", layer_name="AmplitudeMask")
        self.validate_for(field)
        amplitude = _expand_map(
            self.amplitude_map,
            field,
            dtype=field.data.real.dtype,
            name="amplitude_map",
        )
        return field.apply_amplitude(amplitude)

    def parameters(self) -> dict[str, jnp.ndarray]:
        """Return the amplitude map as the exposed layer parameter."""
        return {"amplitude_map": jnp.asarray(self.amplitude_map, dtype=jnp.float32)}


@dataclass(frozen=True)
class ComplexMask(OpticalLayer):
    """Spatial-domain complex-valued modulation layer.

    Applies both amplitude and phase modulation in a single step.
    Requires spatial-domain input.

    Args:
        amplitude_map: Real amplitude transmission (default 1.0).
        phase_map_rad: Phase in radians (default 0.0).
    """

    amplitude_map: jnp.ndarray | float = 1.0
    phase_map_rad: jnp.ndarray | float = 0.0

    def forward(self, field: Field) -> Field:
        """Apply amplitude and phase modulation in one spatial-domain step."""
        _require_domain(field, expected="spatial", layer_name="ComplexMask")
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
        """Return both amplitude and phase maps for optimisation."""
        return {
            "amplitude_map": jnp.asarray(self.amplitude_map, dtype=jnp.float32),
            "phase_map_rad": jnp.asarray(self.phase_map_rad, dtype=jnp.float32),
        }


@dataclass(frozen=True)
class JonesMatrixLayer(OpticalLayer):
    """Spatial-domain Jones matrix modulation layer.

    Applies a 2×2 Jones matrix to each spatial point of a Jones-polarised
    field. Requires spatial-domain input with ``polarization_mode='jones'``.

    Args:
        jones_matrix: Jones matrix with shape ``(2, 2)``,
            ``(2, 2, ny, nx)``, or ``(num_wavelengths, 2, 2, ny, nx)``.
    """

    jones_matrix: jnp.ndarray

    def forward(self, field: Field) -> Field:
        """Mix Jones polarization channels at each spatial sample."""
        _require_domain(field, expected="spatial", layer_name="JonesMatrixLayer")
        self.validate_for(field)
        matrix = _expand_jones_matrix_map(self.jones_matrix, field, name="jones_matrix")
        ex = field.data[..., 0, :, :]
        ey = field.data[..., 1, :, :]
        if matrix.ndim == 2:
            out_ex = matrix[0, 0] * ex + matrix[0, 1] * ey
            out_ey = matrix[1, 0] * ex + matrix[1, 1] * ey
        else:
            out_ex = matrix[..., 0, 0, :, :] * ex + matrix[..., 0, 1, :, :] * ey
            out_ey = matrix[..., 1, 0, :, :] * ex + matrix[..., 1, 1, :, :] * ey
        out = jnp.stack([out_ex, out_ey], axis=-3)
        return _with_field_metadata(out, field)

    def validate_for(self, field: Field) -> None:
        """Require a valid Jones-polarized input field."""
        super().validate_for(field)
        if not field.is_jones:
            raise ValueError("JonesMatrixLayer requires polarization_mode='jones'")

    def parameters(self) -> dict[str, jnp.ndarray]:
        """Return the Jones matrix as the exposed layer parameter."""
        return {"jones_matrix": jnp.asarray(self.jones_matrix, dtype=jnp.complex64)}


@dataclass(frozen=True)
class KSpacePhaseMask(OpticalLayer):
    """k-space phase-only modulation layer.

    Multiplies the k-space field by ``exp(j * phase_map_rad)``.
    Requires k-space-domain input.

    Args:
        phase_map_rad: Phase in radians.  Accepted shapes: scalar,
            ``(ny, nx)``, or ``(num_wavelengths, ny, nx)``.
        aperture_diameter_um: Optional circular aperture diameter in
            micrometers, reported via :meth:`parameters`.
    """

    phase_map_rad: jnp.ndarray | float
    aperture_diameter_um: float | None = None

    def forward(self, field: Field) -> Field:
        """Apply a phase-only modulation to a k-space field."""
        _require_domain(field, expected="kspace", layer_name="KSpacePhaseMask")
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
        """Return the phase map and optional reported aperture diameter."""
        params = {"phase_map_rad": jnp.asarray(self.phase_map_rad, dtype=jnp.float32)}
        if self.aperture_diameter_um is not None:
            params["aperture_diameter_um"] = jnp.asarray(
                self.aperture_diameter_um, dtype=jnp.float32
            )
        return params


@dataclass(frozen=True)
class KSpaceAmplitudeMask(OpticalLayer):
    """k-space amplitude modulation layer.

    Multiplies the k-space field amplitude by a real-valued mask.
    Requires k-space-domain input.

    Args:
        amplitude_map: Amplitude transmission.  Accepted shapes: scalar,
            ``(ny, nx)``, or ``(num_wavelengths, ny, nx)``.
        aperture_diameter_um: Optional circular aperture diameter in
            micrometers.
    """

    amplitude_map: jnp.ndarray | float
    aperture_diameter_um: float | None = None

    def forward(self, field: Field) -> Field:
        """Apply an amplitude-only modulation to a k-space field."""
        _require_domain(field, expected="kspace", layer_name="KSpaceAmplitudeMask")
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
        """Return the amplitude map and optional reported aperture diameter."""
        params = {"amplitude_map": jnp.asarray(self.amplitude_map, dtype=jnp.float32)}
        if self.aperture_diameter_um is not None:
            params["aperture_diameter_um"] = jnp.asarray(
                self.aperture_diameter_um, dtype=jnp.float32
            )
        return params


@dataclass(frozen=True)
class KSpaceComplexMask(OpticalLayer):
    """k-space complex-valued modulation layer.

    Applies both amplitude and phase modulation in k-space.  Requires
    k-space-domain input.

    Args:
        amplitude_map: Real amplitude transmission (default 1.0).
        phase_map_rad: Phase in radians (default 0.0).
        aperture_diameter_um: Optional circular aperture diameter in
            micrometers.
    """

    amplitude_map: jnp.ndarray | float = 1.0
    phase_map_rad: jnp.ndarray | float = 0.0
    aperture_diameter_um: float | None = None

    def forward(self, field: Field) -> Field:
        """Apply amplitude and phase modulation in k-space."""
        _require_domain(field, expected="kspace", layer_name="KSpaceComplexMask")
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
        """Return both modulation maps and any reported aperture diameter."""
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
class KJonesMatrixLayer(OpticalLayer):
    """k-space Jones matrix modulation layer.

    Applies a 2×2 Jones matrix to each frequency component of a
    Jones-polarised field.  Requires k-space-domain input with
    ``polarization_mode='jones'``.

    Args:
        jones_matrix: Jones matrix array (see ``JonesMatrixLayer`` for
            accepted shapes).
        aperture_diameter_um: Optional circular aperture diameter in
            micrometers.
    """

    jones_matrix: jnp.ndarray
    aperture_diameter_um: float | None = None

    def forward(self, field: Field) -> Field:
        """Mix Jones polarization channels at each sampled k-space point."""
        _require_domain(field, expected="kspace", layer_name="KJonesMatrixLayer")
        self.validate_for(field)
        matrix = _expand_jones_matrix_map(self.jones_matrix, field, name="jones_matrix")
        ex = field.data[..., 0, :, :]
        ey = field.data[..., 1, :, :]
        if matrix.ndim == 2:
            out_ex = matrix[0, 0] * ex + matrix[0, 1] * ey
            out_ey = matrix[1, 0] * ex + matrix[1, 1] * ey
        else:
            out_ex = matrix[..., 0, 0, :, :] * ex + matrix[..., 0, 1, :, :] * ey
            out_ey = matrix[..., 1, 0, :, :] * ex + matrix[..., 1, 1, :, :] * ey
        out = jnp.stack([out_ex, out_ey], axis=-3)
        return _with_field_metadata(out, field)

    def validate_for(self, field: Field) -> None:
        """Require a valid Jones-polarized k-space input field."""
        super().validate_for(field)
        if not field.is_jones:
            raise ValueError("KJonesMatrixLayer requires polarization_mode='jones'")

    def parameters(self) -> dict[str, jnp.ndarray]:
        """Return the Jones matrix and optional reported aperture diameter."""
        params: dict[str, jnp.ndarray] = {
            "jones_matrix": jnp.asarray(self.jones_matrix, dtype=jnp.complex64)
        }
        if self.aperture_diameter_um is not None:
            params["aperture_diameter_um"] = jnp.asarray(
                self.aperture_diameter_um, dtype=jnp.float32
            )
        return params


@dataclass(frozen=True)
class ThinLens(OpticalLayer):
    """Ideal thin lens applying a hyperbolic optical-path phase delay.

    The applied phase is::

        phi(x, y) = -k * (sqrt(x**2 + y**2 + f**2) - f)

    which preserves the on-axis phase reference and is valid over a
    larger domain than the paraxial quadratic approximation.  Requires
    spatial-domain input.

    All length quantities are in micrometers.

    Args:
        focal_length_um: Focal length in micrometers (must be positive).
        aperture_diameter_um: Optional circular aperture diameter in
            micrometers applied after the phase delay.
    """

    focal_length_um: float
    aperture_diameter_um: float | None = None

    def forward(self, field: Field) -> Field:
        """Apply thin-lens phase delay and an optional circular aperture."""
        _require_domain(field, expected="spatial", layer_name="ThinLens")
        self.validate_for(field)
        if self.focal_length_um <= 0:
            raise ValueError("focal_length_um must be strictly positive")

        x, y = field.grid.spatial_grid()
        r2 = x * x + y * y
        f_um = jnp.asarray(self.focal_length_um, dtype=r2.dtype)
        # Hyperbolic path-length delay relative to the on-axis ray.
        path_delta_um = jnp.sqrt(r2 + f_um * f_um) - f_um
        phase_stack = []
        for wavelength_um in field.spectrum.wavelengths_um:
            k_um_inv = (2.0 * jnp.pi) / jnp.asarray(wavelength_um, dtype=r2.dtype)
            phi = -k_um_inv * path_delta_um
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
        """Return the focal length and optional aperture diameter."""
        params: dict[str, jnp.ndarray] = {
            "focal_length_um": jnp.asarray(self.focal_length_um, dtype=jnp.float32)
        }
        if self.aperture_diameter_um is not None:
            params["aperture_diameter_um"] = jnp.asarray(
                self.aperture_diameter_um, dtype=jnp.float32
            )
        return params


@dataclass(frozen=True)
class IncoherentImager(IncoherentLayer):
    """Incoherent shift-invariant imager built from coherent optics.

    The PSF is constructed by propagating a calibration source through
    ``optical_layer`` followed by a propagator at ``distance_um`` and
    taking the output intensity. The input intensity is then convolved
    with this PSF.

    Requires spatial-domain, scalar input.

    Args:
        optical_layer: Coherent optical layer defining the imaging optics.
        propagator: Propagation layer (e.g. ``ASMPropagator``) whose
            ``distance_um`` will be overridden.
        distance_um: Imaging distance in micrometers.
        psf_source: Calibration source type (``"impulse"``,
            ``"plane_wave_focus"``, or ``"point_source"``).
        object_distance_um: Object distance in micrometers when
            ``psf_source="point_source"``.
        normalize_psf: Normalise the PSF so it sums to unity.
        enforce_nonnegative_psf: Clamp negative PSF values to zero.
        mode: Convolution backend — ``"psf"``, ``"otf"``, or ``"auto"``.
        normalization_reference: Reference distance strategy for PSF
            normalisation.
        normalization_reference_distance_um: Explicit reference distance
            override in micrometers.
    """

    optical_layer: OpticalLayer
    propagator: OpticalLayer
    distance_um: float
    psf_source: Literal["impulse", "plane_wave_focus", "point_source"] = "impulse"
    object_distance_um: float | None = None
    normalize_psf: bool = True
    enforce_nonnegative_psf: bool = True
    mode: Literal["psf", "otf", "auto"] = "auto"
    normalization_reference: Literal["near_1um", "near_wavelength", "at_imaging_distance"] = (
        "near_wavelength"
    )
    normalization_reference_distance_um: float | None = None

    @classmethod
    def for_far_field(
        cls,
        *,
        optical_layer: OpticalLayer,
        propagator: OpticalLayer,
        image_distance_um: float,
        normalize_psf: bool = True,
        enforce_nonnegative_psf: bool = True,
        mode: Literal["psf", "otf", "auto"] = "auto",
        normalization_reference: Literal[
            "near_1um",
            "near_wavelength",
            "at_imaging_distance",
        ] = "near_wavelength",
        normalization_reference_distance_um: float | None = None,
    ) -> "IncoherentImager":
        """Construct a far-field imager using a plane-wave calibration source."""
        return cls(
            optical_layer=optical_layer,
            propagator=propagator,
            distance_um=image_distance_um,
            psf_source="plane_wave_focus",
            normalize_psf=normalize_psf,
            enforce_nonnegative_psf=enforce_nonnegative_psf,
            mode=mode,
            normalization_reference=normalization_reference,
            normalization_reference_distance_um=normalization_reference_distance_um,
        )

    @classmethod
    def for_finite_distance(
        cls,
        *,
        optical_layer: OpticalLayer,
        propagator: OpticalLayer,
        object_distance_um: float,
        image_distance_um: float,
        normalize_psf: bool = True,
        enforce_nonnegative_psf: bool = True,
        mode: Literal["psf", "otf", "auto"] = "auto",
        normalization_reference: Literal[
            "near_1um",
            "near_wavelength",
            "at_imaging_distance",
        ] = "near_wavelength",
        normalization_reference_distance_um: float | None = None,
    ) -> "IncoherentImager":
        """Construct a finite-distance imager using an on-axis point source."""
        return cls(
            optical_layer=optical_layer,
            propagator=propagator,
            distance_um=image_distance_um,
            psf_source="point_source",
            object_distance_um=object_distance_um,
            normalize_psf=normalize_psf,
            enforce_nonnegative_psf=enforce_nonnegative_psf,
            mode=mode,
            normalization_reference=normalization_reference,
            normalization_reference_distance_um=normalization_reference_distance_um,
        )

    def normalization_distance_um(self, intensity: Intensity) -> float:
        """Choose the reference distance used to normalize the PSF.

        Args:
            intensity: Input intensity whose wavelength range may be used when
                ``normalization_reference="near_wavelength"``.

        Returns:
            Reference distance in micrometers used to compute PSF energy sums.
        """
        if self.normalization_reference_distance_um is not None:
            if self.normalization_reference_distance_um <= 0:
                raise ValueError("normalization_reference_distance_um must be strictly positive")
            return float(self.normalization_reference_distance_um)
        if self.normalization_reference == "near_1um":
            return 1.0
        if self.normalization_reference == "near_wavelength":
            return max(1e-3, float(jnp.min(intensity.spectrum.wavelengths_um)))
        if self.normalization_reference == "at_imaging_distance":
            return float(self.distance_um)
        raise ValueError(
            "normalization_reference must be one of: "
            "near_1um, near_wavelength, at_imaging_distance"
        )

    def build_psf(self, field: Field) -> Intensity:
        """Construct the imaging PSF on the input field grid.

        Args:
            field: Spatial-domain scalar field defining the grid and spectrum.

        Returns:
            Spatial PSF intensity with shape ``(num_wavelengths, ny, nx)``.
        """
        _require_domain(field, expected="spatial", layer_name="IncoherentImager")
        if field.is_jones:
            raise ValueError("IncoherentImager currently supports scalar fields only")
        spatial_field = field
        spatial_field.validate()
        if self.distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")
        if self.psf_source not in ("impulse", "plane_wave_focus", "point_source"):
            raise ValueError("psf_source must be one of: impulse, plane_wave_focus, point_source")

        calibration_field = Field.zeros(
            grid=spatial_field.grid,
            spectrum=spatial_field.spectrum,
            dtype=spatial_field.data.dtype,
        )
        if self.psf_source == "impulse":
            source = _centered_delta_field(calibration_field)
        elif self.psf_source == "plane_wave_focus":
            source = Field.plane_wave(
                grid=spatial_field.grid,
                spectrum=spatial_field.spectrum,
                dtype=spatial_field.data.dtype,
            )
        else:
            if self.object_distance_um is None:
                raise ValueError("object_distance_um must be set when psf_source='point_source'")
            source = _point_source_field(
                calibration_field,
                distance_um=self.object_distance_um,
            )

        coherent = self.optical_layer.forward(source)
        propagator = replace(self.propagator, distance_um=self.distance_um)  # type: ignore[type-var]
        response = propagator.forward(coherent)
        _require_domain(response, expected="spatial", layer_name="IncoherentImager")
        psf = response.to_intensity()

        if self.enforce_nonnegative_psf:
            psf = Intensity(
                data=jnp.maximum(psf.data, 0.0),
                grid=psf.grid,
                spectrum=psf.spectrum,
            )
        if self.normalize_psf:
            norm_distance_um = self.normalization_distance_um(psf)
            if abs(norm_distance_um - self.distance_um) < 1e-12:
                sums = jnp.sum(psf.data, axis=(-2, -1), keepdims=True)
            else:
                ref_propagator = replace(self.propagator, distance_um=norm_distance_um)  # type: ignore[type-var]
                ref_response = ref_propagator.forward(coherent)
                _require_domain(ref_response, expected="spatial", layer_name="IncoherentImager")
                ref_psf = ref_response.to_intensity()
                if self.enforce_nonnegative_psf:
                    ref_psf = Intensity(
                        data=jnp.maximum(ref_psf.data, 0.0),
                        grid=ref_psf.grid,
                        spectrum=ref_psf.spectrum,
                    )
                sums = jnp.sum(ref_psf.data, axis=(-2, -1), keepdims=True)
            psf = Intensity(
                data=psf.data / jnp.maximum(sums, 1e-12),
                grid=psf.grid,
                spectrum=psf.spectrum,
            )
        psf.validate()
        return psf

    def infer_from_paraxial_limit(
        self,
        sensor_grid: Grid,
        paraxial_max_angle_rad: float,
    ) -> Grid:
        """Infer a same-shape input grid from a sensor grid under paraxial limits.

        The returned grid preserves ``sensor_grid.nx`` and ``sensor_grid.ny``.
        Only the pixel pitch changes, according to the imaging geometry:

        - ``plane_wave_focus`` imagers return an image-equivalent far-field grid
          with the same pitch as the sensor grid.
        - ``point_source`` imagers return an object-plane grid whose pitch is
          scaled by the magnification ``|M| = image_distance / object_distance``.

        The supplied ``sensor_grid`` must fit within the paraxial field of view
        set by ``paraxial_max_angle_rad`` at the image/sensor distance.

        Args:
            sensor_grid: Detector-plane grid to map from.
            paraxial_max_angle_rad: Maximum allowed paraxial ray angle in radians.

        Returns:
            A same-shape input grid compatible with the configured imaging mode.

        Raises:
            ValueError: If the paraxial limit or imager geometry is invalid, if
                the sensor grid exceeds the paraxial field of view, or if the
                imager source mode does not define object/sensor geometry.
        """
        sensor_grid.validate()
        if paraxial_max_angle_rad <= 0:
            raise ValueError("paraxial_max_angle_rad must be strictly positive")
        if self.distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")

        max_sensor_half_extent_um = self.distance_um * math.tan(paraxial_max_angle_rad)
        sensor_half_width_um = 0.5 * sensor_grid.nx * sensor_grid.dx_um
        sensor_half_height_um = 0.5 * sensor_grid.ny * sensor_grid.dy_um
        if (
            sensor_half_width_um > max_sensor_half_extent_um
            or sensor_half_height_um > max_sensor_half_extent_um
        ):
            raise ValueError(
                "sensor_grid exceeds the paraxial field of view for the configured "
                f"image distance: half extents ({sensor_half_width_um:.3f}, "
                f"{sensor_half_height_um:.3f}) um exceed "
                f"{max_sensor_half_extent_um:.3f} um"
            )

        if self.psf_source == "plane_wave_focus":
            return Grid.from_extent(
                nx=sensor_grid.nx,
                ny=sensor_grid.ny,
                dx_um=sensor_grid.dx_um,
                dy_um=sensor_grid.dy_um,
            )

        if self.psf_source == "point_source":
            if self.object_distance_um is None or self.object_distance_um <= 0:
                raise ValueError(
                    "object_distance_um must be strictly positive when "
                    "inferring a finite-distance input grid"
                )
            magnification = abs(self.distance_um / self.object_distance_um)
            if magnification <= 0:
                raise ValueError("finite-distance magnification must be strictly positive")

            input_grid = Grid.from_extent(
                nx=sensor_grid.nx,
                ny=sensor_grid.ny,
                dx_um=sensor_grid.dx_um / magnification,
                dy_um=sensor_grid.dy_um / magnification,
            )

            max_input_half_extent_um = self.object_distance_um * math.tan(paraxial_max_angle_rad)
            input_half_width_um = 0.5 * input_grid.nx * input_grid.dx_um
            input_half_height_um = 0.5 * input_grid.ny * input_grid.dy_um
            if (
                input_half_width_um > max_input_half_extent_um
                or input_half_height_um > max_input_half_extent_um
            ):
                raise ValueError(
                    "inferred input grid exceeds the paraxial field of view for the "
                    f"configured object distance: half extents ({input_half_width_um:.3f}, "
                    f"{input_half_height_um:.3f}) um exceed "
                    f"{max_input_half_extent_um:.3f} um"
                )
            return input_grid

        raise ValueError(
            "infer_from_paraxial_limit is only supported for far-field "
            "(plane_wave_focus) and finite-distance (point_source) imagers"
        )

    def forward(self, intensity: Intensity) -> Intensity:
        """Apply shift-invariant incoherent imaging to a spatial intensity.

        Args:
            intensity: Spatial intensity image.

        Returns:
            Spatial intensity after PSF or OTF filtering.
        """
        self.validate_for(intensity)
        template = Field.zeros(
            grid=intensity.grid,
            spectrum=intensity.spectrum,
            dtype=jnp.result_type(intensity.data.dtype, 1j),
        )
        psf = _expand_psf_for_intensity(self.build_psf(template))
        image = self._apply_conv(
            intensity.data,
            psf,
            mode=self.mode,
            adjoint=False,
            conv_grid=intensity.grid,
        ).astype(intensity.data.dtype)
        image_out = Intensity(
            data=image,
            grid=intensity.grid,
            spectrum=intensity.spectrum,
        )
        image_out.validate()
        return image_out

    @staticmethod
    def _apply_conv(
        x: jax.Array,
        psf: jax.Array,
        *,
        mode: Literal["psf", "otf", "auto"] = "auto",
        adjoint: bool = False,
        conv_grid: Grid | None = None,
    ) -> jax.Array:
        """Apply the shared incoherent convolution backend.

        ``x`` is expected to have trailing shape ``(num_wavelengths, ny, nx)`` and optional
        leading batch axes. ``psf`` is expected to have shape ``(1|num_wavelengths, ny, nx)``.
        """
        if mode not in ("psf", "otf", "auto"):
            raise ValueError("mode must be one of: psf, otf, auto")
        if x.ndim < 3:
            raise ValueError(
                "incoherent convolution input must have shape (*batch, num_wavelengths, ny, nx)"
            )
        if psf.ndim != 3:
            raise ValueError("psf must have shape (1|num_wavelengths, ny, nx)")

        num_wavelengths, ny, nx = x.shape[-3:]
        if conv_grid is not None and (conv_grid.ny, conv_grid.nx) != (ny, nx):
            raise ValueError(
                "conv_grid shape mismatch: got "
                f"{(conv_grid.ny, conv_grid.nx)}, expected {(ny, nx)}"
            )
        if psf.shape[-2:] != (ny, nx):
            raise ValueError(
                f"psf spatial shape mismatch: got {psf.shape[-2:]}, expected {(ny, nx)}"
            )
        if psf.shape[0] not in (1, num_wavelengths):
            raise ValueError(
                f"psf wavelength axis mismatch: got {psf.shape[0]}, expected 1 or {num_wavelengths}"
            )

        resolved_mode = mode
        if resolved_mode == "auto":
            resolved_mode = "otf" if (nx * ny) >= 4096 else "psf"

        batch_shape = x.shape[:-3]
        flat_batch = math.prod(batch_shape) if batch_shape else 1
        flat_x = x.reshape((flat_batch, num_wavelengths, ny, nx))

        outputs: list[jax.Array] = []
        if resolved_mode == "otf":
            full_shape = (2 * ny - 1, 2 * nx - 1)
            kernels = [
                psf[0] if psf.shape[0] == 1 else psf[i]
                for i in range(num_wavelengths)
            ]
            if adjoint:
                kernels = [jnp.conj(kernel[::-1, ::-1]) for kernel in kernels]
            otf = jnp.stack(
                [
                    jnp.fft.fftn(
                        kernels[i],
                        s=full_shape,
                        axes=(-2, -1),
                    )
                    for i in range(num_wavelengths)
                ],
                axis=0,
            )
            for i in range(num_wavelengths):
                outputs.append(
                    fftconvolve_same_with_otf(
                        flat_x[:, i],
                        otf[i],
                        kernel_shape=(ny, nx),
                    )
                )
        else:
            for i in range(num_wavelengths):
                kernel = psf[0] if psf.shape[0] == 1 else psf[i]
                if adjoint:
                    kernel = jnp.conj(kernel[::-1, ::-1])
                outputs.append(
                    fftconvolve(
                        flat_x[:, i],
                        kernel,
                        mode="same",
                        axes=(-2, -1),
                    )
                )

        return jnp.stack(outputs, axis=1).reshape((*batch_shape, num_wavelengths, ny, nx))

    def linear_operator(
        self,
        template: Intensity,
        *,
        cache: Literal["psf", "otf", "auto"] = "auto",
        flatten: bool = False,
        conv_grid: Grid | None = None,
    ) -> LinearOperator:
        """Construct a cached linear operator for this imager on ``template``."""
        self.validate_for(template)
        if template.data.ndim != 3:
            raise ValueError(
                "linear_operator currently requires an unbatched template intensity with shape "
                "(num_wavelengths, ny, nx)"
            )
        template_field = Field.zeros(
            grid=template.grid,
            spectrum=template.spectrum,
            dtype=jnp.result_type(template.data.dtype, 1j),
        )
        psf = self.build_psf(template_field)
        target_grid = template.grid if conv_grid is None else conv_grid
        if conv_grid is not None:
            # Reuse detector-array readout semantics to map a high-resolution PSF onto
            # the requested convolution grid before building the operator.
            from fouriax.optics.sensors import DetectorArray

            detector = DetectorArray(
                detector_grid=conv_grid,
                qe_curve=1.0,
                sum_wavelengths=False,
                resample_method="linear",
            )
            detector_op = detector.linear_operator(template, flatten=False)
            psf = Intensity(
                data=detector_op.matvec(psf.data).astype(template.data.dtype),
                grid=conv_grid,
                spectrum=psf.spectrum,
            )
        psf_array = _expand_psf_for_intensity(psf)
        tensor_shape: tuple[int, int, int] = (
            template.spectrum.size,
            target_grid.ny,
            target_grid.nx,
        )
        num_wavelengths, ny, nx = tensor_shape
        resolved_cache = cache
        if resolved_cache == "auto":
            resolved_cache = "otf" if (nx * ny) >= 4096 else "psf"

        kernel_stack = jnp.stack(
            [
                psf_array[0] if psf_array.shape[0] == 1 else psf_array[i]
                for i in range(num_wavelengths)
            ],
            axis=0,
        )

        def apply_cached_kernels(x: jax.Array, kernels: jax.Array) -> jax.Array:
            batch_shape = x.shape[:-3]
            flat_batch = math.prod(batch_shape) if batch_shape else 1
            flat_x = x.reshape((flat_batch, num_wavelengths, ny, nx))
            out = jax.vmap(
                lambda xi, kernel: fftconvolve(
                    xi,
                    kernel,
                    mode="same",
                    axes=(-2, -1),
                ),
                in_axes=(1, 0),
                out_axes=1,
            )(flat_x, kernels)
            return cast(jax.Array, out.reshape((*batch_shape, num_wavelengths, ny, nx)))

        def apply_cached_otf(x: jax.Array, otf_stack: jax.Array) -> jax.Array:
            batch_shape = x.shape[:-3]
            flat_batch = math.prod(batch_shape) if batch_shape else 1
            flat_x = x.reshape((flat_batch, num_wavelengths, ny, nx))
            out = jax.vmap(
                lambda xi, otf: fftconvolve_same_with_otf(
                    xi,
                    otf,
                    kernel_shape=(ny, nx),
                ),
                in_axes=(1, 0),
                out_axes=1,
            )(flat_x, otf_stack)
            return cast(jax.Array, out.reshape((*batch_shape, num_wavelengths, ny, nx)))

        if resolved_cache == "otf":
            full_shape = (2 * ny - 1, 2 * nx - 1)
            forward_otf = jax.vmap(
                lambda kernel: jnp.fft.fftn(kernel, s=full_shape, axes=(-2, -1))
            )(kernel_stack)
            adjoint_otf = jax.vmap(
                lambda kernel: jnp.fft.fftn(
                    jnp.conj(kernel[::-1, ::-1]),
                    s=full_shape,
                    axes=(-2, -1),
                )
            )(kernel_stack)
        else:
            forward_kernels = kernel_stack
            adjoint_kernels = jnp.conj(kernel_stack[:, ::-1, ::-1])

        in_shape: tuple[int, ...]
        out_shape: tuple[int, ...]
        if flatten:
            in_shape = (math.prod(tensor_shape),)
            out_shape = in_shape

            def matvec_fn(x: jax.Array) -> jax.Array:
                x_tensor = x.reshape(tensor_shape)
                if resolved_cache == "otf":
                    out = apply_cached_otf(x_tensor, forward_otf)
                else:
                    out = apply_cached_kernels(x_tensor, forward_kernels)
                return out.reshape(out_shape)

            def rmatvec_fn(y: jax.Array) -> jax.Array:
                y_tensor = y.reshape(tensor_shape)
                if resolved_cache == "otf":
                    out = apply_cached_otf(y_tensor, adjoint_otf)
                else:
                    out = apply_cached_kernels(y_tensor, adjoint_kernels)
                return out.reshape(in_shape)
        else:
            in_shape = tensor_shape
            out_shape = tensor_shape

            def matvec_fn(x: jax.Array) -> jax.Array:
                if resolved_cache == "otf":
                    return apply_cached_otf(x, forward_otf)
                return apply_cached_kernels(x, forward_kernels)

            def rmatvec_fn(y: jax.Array) -> jax.Array:
                if resolved_cache == "otf":
                    return apply_cached_otf(y, adjoint_otf)
                return apply_cached_kernels(y, adjoint_kernels)

        return LinearOperator(
            in_shape=in_shape,
            out_shape=out_shape,
            in_dtype=template.data.dtype,
            out_dtype=template.data.dtype,
            matvec_fn=matvec_fn,
            rmatvec_fn=rmatvec_fn,
        )

    def parameters(self) -> dict[str, jnp.ndarray]:
        """Return imager parameters together with nested optical-layer parameters."""
        params: dict[str, jnp.ndarray] = {
            "distance_um": jnp.asarray(self.distance_um, dtype=jnp.float32),
        }
        if self.object_distance_um is not None:
            params["object_distance_um"] = jnp.asarray(self.object_distance_um, dtype=jnp.float32)
        for key, value in self.optical_layer.parameters().items():
            params[f"optical_layer.{key}"] = value
        return params


@dataclass(frozen=True)
class FourierTransform(OpticalLayer):
    """Explicit spatial → k-space domain transform layer.

    Requires spatial-domain input.  Equivalent to :meth:`Field.to_kspace`
    wrapped as an ``OpticalLayer`` for use inside an ``OpticalModule``.
    """

    def forward(self, field: Field) -> Field:
        """Convert a spatial-domain field to k-space."""
        _require_domain(field, expected="spatial", layer_name="FourierTransform")
        self.validate_for(field)
        return field.to_kspace()


@dataclass(frozen=True)
class InverseFourierTransform(OpticalLayer):
    """Explicit k-space → spatial domain transform layer.

    Requires k-space-domain input.  Equivalent to
    :meth:`Field.to_spatial` wrapped as an ``OpticalLayer``.
    """

    def forward(self, field: Field) -> Field:
        """Convert a k-space field back to spatial domain."""
        _require_domain(field, expected="kspace", layer_name="InverseFourierTransform")
        self.validate_for(field)
        return field.to_spatial()
