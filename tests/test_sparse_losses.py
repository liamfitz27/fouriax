import jax
import jax.numpy as jnp
import numpy as np

from fouriax.optics.losses import (
    build_full_sensing_matrix_dct,
    dct_basis_synthesis_matrix_2d,
    mutual_information_loss_from_matrix,
    total_coherence_loss,
    total_coherence_loss_from_matrix,
)


def test_dct_basis_is_orthonormal():
    psi = dct_basis_synthesis_matrix_2d((4, 5))
    ident = np.asarray(psi.T @ psi)
    np.testing.assert_allclose(ident, np.eye(20), atol=1e-5)


def test_build_full_sensing_matrix_dct_with_identity_forward():
    image_shape = (3, 4)

    def identity_forward(image: jnp.ndarray) -> jnp.ndarray:
        return image

    a_matrix, psi = build_full_sensing_matrix_dct(identity_forward, image_shape)
    np.testing.assert_allclose(np.asarray(a_matrix), np.asarray(psi), atol=1e-6)


def test_total_coherence_zero_for_orthonormal_matrix():
    a_matrix = jnp.eye(8, dtype=jnp.float32)
    loss = total_coherence_loss_from_matrix(a_matrix, num_blocks=2)
    np.testing.assert_allclose(float(loss), 0.0, atol=1e-6)


def test_total_coherence_operator_matches_dense():
    key = jax.random.PRNGKey(123)
    a_matrix = jax.random.normal(key, (10, 8), dtype=jnp.float32)
    dense = total_coherence_loss_from_matrix(a_matrix, num_blocks=3)
    op = total_coherence_loss(
        a_matvec=lambda v: a_matrix @ v,
        n_coeffs=8,
        num_blocks=3,
    )
    np.testing.assert_allclose(float(op), float(dense), atol=1e-5)


def test_mutual_information_loss_scales_with_operator_strength():
    key = jax.random.PRNGKey(0)
    base = jax.random.normal(key, (12, 8), dtype=jnp.float32)
    loss_small = mutual_information_loss_from_matrix(
        0.1 * base,
        key=key,
        rsvd_k=6,
        p=0,
    )
    loss_big = mutual_information_loss_from_matrix(
        base,
        key=key,
        rsvd_k=6,
        p=0,
    )
    assert float(loss_big) < float(loss_small)
