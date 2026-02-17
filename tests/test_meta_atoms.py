import jax
import jax.numpy as jnp
import numpy as np

from fouriax.optics import Field, Grid, MetaAtomInterpolationLayer, MetaAtomLibrary, Spectrum


def _single_param_library() -> MetaAtomLibrary:
    wavelengths_um = jnp.array([1.0], dtype=jnp.float32)
    side_um = jnp.array([0.10, 0.20], dtype=jnp.float32)
    # Transmission at side=0.10 -> 1.0, at side=0.20 -> 0.5.
    transmission_complex = jnp.array([[1.0 + 0.0j, 0.5 + 0.0j]], dtype=jnp.complex64)
    return MetaAtomLibrary.from_complex(
        wavelengths_um=wavelengths_um,
        parameter_axes=(side_um,),
        transmission_complex=transmission_complex,
    )


def test_meta_atom_layer_per_pixel_map_forward():
    library = _single_param_library()

    grid = Grid.from_extent(nx=6, ny=4, dx_um=0.7, dy_um=0.7)
    spectrum = Spectrum.from_scalar(1.0)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    # sigmoid(0)=0.5 => bounded side length = 0.15 um everywhere.
    raw_map = jnp.zeros((grid.ny, grid.nx), dtype=jnp.float32)
    layer = MetaAtomInterpolationLayer(
        library=library,
        raw_geometry_params=raw_map,
        min_geometry_params=jnp.array([0.10], dtype=jnp.float32),
        max_geometry_params=jnp.array([0.20], dtype=jnp.float32),
    )

    out = layer.forward(field)
    intensity = np.asarray(out.intensity())

    expected_amp = 0.75  # linear interpolation between 1.0 and 0.5 at midpoint
    expected_intensity = expected_amp**2
    np.testing.assert_allclose(intensity, expected_intensity, atol=1e-6)


def test_meta_atom_layer_gradient_tracks_raw_pixel_map():
    library = _single_param_library()

    grid = Grid.from_extent(nx=5, ny=5, dx_um=0.7, dy_um=0.7)
    spectrum = Spectrum.from_scalar(1.0)
    field = Field.plane_wave(grid=grid, spectrum=spectrum)

    min_bounds = jnp.array([0.10], dtype=jnp.float32)
    max_bounds = jnp.array([0.20], dtype=jnp.float32)

    def loss_fn(raw_map: jnp.ndarray) -> jnp.ndarray:
        layer = MetaAtomInterpolationLayer(
            library=library,
            raw_geometry_params=raw_map,
            min_geometry_params=min_bounds,
            max_geometry_params=max_bounds,
        )
        out = layer.forward(field)
        return -jnp.sum(out.intensity())

    raw_init = 0.05 * jax.random.normal(jax.random.PRNGKey(0), (grid.ny, grid.nx))
    grads = jax.grad(loss_fn)(raw_init)

    assert grads.shape == raw_init.shape
    assert bool(jnp.all(jnp.isfinite(grads)))
    assert float(jnp.sum(jnp.abs(grads))) > 0.0
