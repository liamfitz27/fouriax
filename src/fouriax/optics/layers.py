from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import jax
import jax.numpy as jnp

from fouriax.fft import fftconvolve, fftconvolve_same_with_otf
from fouriax.optics.interfaces import Monitor, OpticalLayer, Sensor
from fouriax.optics.model import Field


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
class IncoherentImager(OpticalLayer):
    """Incoherent shift-invariant imager built from coherent optics.

    The PSF is constructed by propagating a calibration source (impulse
    or plane wave) through ``optical_layer`` followed by a propagator at
    ``distance_um`` and taking the output intensity.  The input field's
    intensity is then convolved with this PSF.

    Requires spatial-domain, scalar input.

    Args:
        optical_layer: Coherent optical layer defining the imaging optics.
        propagator: Propagation layer (e.g. ``ASMPropagator``) whose
            ``distance_um`` will be overridden.
        distance_um: Imaging distance in micrometers.
        psf_source: Calibration source type (``"impulse"`` or
            ``"plane_wave_focus"``).
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
    psf_source: Literal["impulse", "plane_wave_focus"] = "impulse"
    normalize_psf: bool = True
    enforce_nonnegative_psf: bool = True
    mode: Literal["psf", "otf", "auto"] = "auto"
    normalization_reference: Literal["near_1um", "near_wavelength", "at_imaging_distance"] = (
        "near_wavelength"
    )
    normalization_reference_distance_um: float | None = None

    def normalization_distance_um(self, field: Field) -> float:
        """Choose the reference distance used to normalize the PSF.

        Args:
            field: Input scalar field whose wavelength range may be used when
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
            return max(1e-3, float(jnp.min(field.spectrum.wavelengths_um)))
        if self.normalization_reference == "at_imaging_distance":
            return float(self.distance_um)
        raise ValueError(
            "normalization_reference must be one of: "
            "near_1um, near_wavelength, at_imaging_distance"
        )

    def build_psf(self, field: Field) -> jnp.ndarray:
        """Construct the imaging PSF on the input field grid.

        Args:
            field: Spatial-domain scalar field defining the grid and spectrum.

        Returns:
            Real PSF array with shape ``(num_wavelengths, ny, nx)``.
        """
        _require_domain(field, expected="spatial", layer_name="IncoherentImager")
        if field.is_jones:
            raise ValueError("IncoherentImager currently supports scalar fields only")
        spatial_field = field
        self.validate_for(spatial_field)
        if self.distance_um <= 0:
            raise ValueError("distance_um must be strictly positive")
        if self.psf_source not in ("impulse", "plane_wave_focus"):
            raise ValueError("psf_source must be one of: impulse, plane_wave_focus")

        calibration_field = Field.zeros(
            grid=spatial_field.grid,
            spectrum=spatial_field.spectrum,
            dtype=spatial_field.data.dtype,
        )
        if self.psf_source == "impulse":
            source = _centered_delta_field(calibration_field)
        else:
            source = Field.plane_wave(
                grid=spatial_field.grid,
                spectrum=spatial_field.spectrum,
                dtype=spatial_field.data.dtype,
            )

        coherent = self.optical_layer.forward(source)
        propagator = replace(self.propagator, distance_um=self.distance_um)  # type: ignore[type-var]
        response = propagator.forward(coherent)
        _require_domain(response, expected="spatial", layer_name="IncoherentImager")
        psf = response.intensity().astype(spatial_field.data.real.dtype)

        if self.enforce_nonnegative_psf:
            psf = jnp.maximum(psf, 0.0)
        if self.normalize_psf:
            norm_distance_um = self.normalization_distance_um(spatial_field)
            if abs(norm_distance_um - self.distance_um) < 1e-12:
                sums = jnp.sum(psf, axis=(-2, -1), keepdims=True)
            else:
                ref_propagator = replace(self.propagator, distance_um=norm_distance_um)  # type: ignore[type-var]
                ref_response = ref_propagator.forward(coherent)
                _require_domain(ref_response, expected="spatial", layer_name="IncoherentImager")
                ref_psf = ref_response.intensity().astype(spatial_field.data.real.dtype)
                if self.enforce_nonnegative_psf:
                    ref_psf = jnp.maximum(ref_psf, 0.0)
                sums = jnp.sum(ref_psf, axis=(-2, -1), keepdims=True)
            psf = psf / jnp.maximum(sums, 1e-12)
        return psf

    def forward(self, field: Field) -> Field:
        """Apply shift-invariant incoherent imaging to a scalar field.

        Args:
            field: Spatial-domain scalar field.

        Returns:
            Spatial-domain scalar field whose amplitude is the square root of
            the imaged intensity.
        """
        _require_domain(field, expected="spatial", layer_name="IncoherentImager")
        spatial_field = field
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

        batch_shape = spatial_field.batch_shape
        flat_batch = math.prod(batch_shape) if batch_shape else 1
        flat_intensity = intensity.reshape(
            (flat_batch, spatial_field.spectrum.size, spatial_field.grid.ny, spatial_field.grid.nx)
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
                    fftconvolve_same_with_otf(
                        flat_intensity[:, i],
                        otf[i],
                        kernel_shape=(ky, kx),
                    )
                )
        else:
            for i in range(spatial_field.spectrum.size):
                psf_i = psf[0] if psf.shape[0] == 1 else psf[i]
                outputs.append(
                    fftconvolve(flat_intensity[:, i], psf_i, mode="same", axes=(-2, -1))
                )

        image = jnp.stack(outputs, axis=1).reshape(
            (
                *batch_shape,
                spatial_field.spectrum.size,
                spatial_field.grid.ny,
                spatial_field.grid.nx,
            )
        )
        amplitude = jnp.sqrt(jnp.maximum(image, 0.0)).astype(spatial_field.data.real.dtype)
        return Field(
            data=amplitude.astype(spatial_field.data.dtype),
            grid=spatial_field.grid,
            spectrum=spatial_field.spectrum,
            polarization_mode=spatial_field.polarization_mode,
            domain="spatial",
            kx_pixel_size_cyc_per_um=spatial_field.kx_pixel_size_cyc_per_um,
            ky_pixel_size_cyc_per_um=spatial_field.ky_pixel_size_cyc_per_um,
        )

    def parameters(self) -> dict[str, jnp.ndarray]:
        """Return imager parameters together with nested optical-layer parameters."""
        params: dict[str, jnp.ndarray] = {
            "distance_um": jnp.asarray(self.distance_um, dtype=jnp.float32),
        }
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
