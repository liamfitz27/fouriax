from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import random


def focal_spot_loss(
    intensity: jnp.ndarray,
    target_xy: tuple[int, int],
    window_px: int = 2,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Minimize loss by maximizing power concentration around a target focal spot.

    Args:
        intensity: Either `(ny, nx)` or `(num_channels, ny, nx)` intensity map.
        target_xy: Target pixel `(x, y)` for the focal spot center.
        window_px: Half-width of square integration window around target.
        eps: Numerical stabilizer for total power normalization.

    Returns:
        Scalar loss in `[0, 1]` approximately: `1 - spot_power / total_power`.
    """
    if window_px < 0:
        raise ValueError("window_px must be >= 0")

    arr = jnp.asarray(intensity)
    if arr.ndim == 3:
        # Aggregate multi-channel intensity (e.g., multi-wavelength) into one map.
        arr2d = jnp.sum(arr, axis=0)
    elif arr.ndim == 2:
        arr2d = arr
    else:
        raise ValueError("intensity must have shape (ny, nx) or (num_channels, ny, nx)")

    ny, nx = arr2d.shape
    tx, ty = int(target_xy[0]), int(target_xy[1])

    x = jnp.arange(nx)
    y = jnp.arange(ny)
    xmask = (x >= (tx - window_px)) & (x <= (tx + window_px))
    ymask = (y >= (ty - window_px)) & (y <= (ty + window_px))
    mask = ymask[:, None] & xmask[None, :]

    spot_power = jnp.sum(arr2d * mask.astype(arr2d.dtype))
    total_power = jnp.sum(arr2d)
    return 1.0 - (spot_power / (total_power + eps))


def dct_basis_synthesis_matrix_2d(image_shape: tuple[int, int]) -> jnp.ndarray:
    """
    Build an orthonormal 2D DCT synthesis matrix.

    Returns matrix Psi with shape (n_pixels, n_pixels) such that:
    x_vec = Psi @ c_vec, where c_vec are DCT coefficients.
    """
    ny, nx = image_shape
    if ny <= 0 or nx <= 0:
        raise ValueError("image_shape entries must be positive")

    def _dct_1d_matrix(n: int) -> jnp.ndarray:
        k = jnp.arange(n, dtype=jnp.float32)[:, None]
        m = jnp.arange(n, dtype=jnp.float32)[None, :]
        alpha = jnp.sqrt(2.0 / n) * jnp.ones((n, 1), dtype=jnp.float32)
        alpha = alpha.at[0, 0].set(jnp.sqrt(1.0 / n))
        return alpha * jnp.cos(jnp.pi * (m + 0.5) * k / n)

    cy = _dct_1d_matrix(ny)
    cx = _dct_1d_matrix(nx)
    return jnp.kron(cx.T, cy.T)


def dense_matvec(a_matrix: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    return a_matrix @ x


def dense_rmatvec(a_matrix: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.conjugate(a_matrix).T @ y


def build_full_sensing_matrix_dct(
    forward_image_to_measurement: Callable[[jnp.ndarray], jnp.ndarray],
    image_shape: tuple[int, int],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build full sensing matrix A for a DCT basis from a generic forward operator.

    Args:
        forward_image_to_measurement: maps image (ny, nx) -> measurement tensor.
        image_shape: (ny, nx) image grid shape.

    Returns:
        A: sensing matrix with shape (m_measurement, n_pixels).
        Psi: DCT synthesis basis with shape (n_pixels, n_pixels).
    """
    psi = dct_basis_synthesis_matrix_2d(image_shape)

    def _apply_on_basis_atom(atom_vec: jnp.ndarray) -> jnp.ndarray:
        image = atom_vec.reshape(image_shape)
        measurement = forward_image_to_measurement(image)
        return jnp.ravel(measurement)

    a_matrix = jax.vmap(_apply_on_basis_atom, in_axes=1, out_axes=1)(psi)
    return a_matrix, psi


@partial(
    jax.jit,
    static_argnames=("a_matvec", "a_rmatvec", "m", "n", "k", "p", "q_iter"),
)
def randomized_svd(
    a_matvec: Callable[[jnp.ndarray], jnp.ndarray],
    a_rmatvec: Callable[[jnp.ndarray], jnp.ndarray],
    m: int,
    n: int,
    k: int,
    key: jax.Array,
    p: int = 10,
    q_iter: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Randomized SVD for a linear operator.
    Copied/adapted from cs_metasurface common JAX utilities.
    """
    sketch_size = k + p
    _, subkey = random.split(key)
    omega = random.normal(subkey, (n, sketch_size))

    a_matvec_vmap = jax.vmap(a_matvec, in_axes=1, out_axes=1)
    a_rmatvec_vmap = jax.vmap(a_rmatvec, in_axes=1, out_axes=1)

    y = a_matvec_vmap(omega)
    for _ in range(q_iter):
        y = a_matvec_vmap(a_rmatvec_vmap(y))

    q, _ = jnp.linalg.qr(y)
    b_t = a_rmatvec_vmap(q)
    b = b_t.T

    u_tilde, s_k, vh_k = jnp.linalg.svd(b, full_matrices=False)
    u_k = q @ u_tilde
    return u_k[:, :k], s_k[:k], vh_k[:k, :]


def total_coherence_loss(
    a_matvec: Callable[[jnp.ndarray], jnp.ndarray],
    n_coeffs: int,
    num_blocks: int,
    eps: float = 1e-18,
) -> jnp.ndarray:
    """
    Total coherence loss using only operator-form A matvec.

    Args:
        a_matvec: function mapping coeff vector (n_coeffs,) -> measurement vector (m,).
        n_coeffs: number of sparse coefficients (A columns).
        num_blocks: number of atom blocks for blockwise coherence accumulation.
        eps: numerical stabilizer for normalization.
    """
    if n_coeffs <= 0:
        raise ValueError("n_coeffs must be > 0")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be > 0")
    if eps <= 0:
        raise ValueError("eps must be > 0")

    block_size = (n_coeffs + num_blocks - 1) // num_blocks
    padded_len = block_size * num_blocks - n_coeffs
    atom_indices = jnp.arange(n_coeffs)
    padded_indices = jnp.pad(atom_indices, (0, padded_len), constant_values=-1)
    blocks = padded_indices.reshape((num_blocks, block_size))

    def _sensed_block(indices: jnp.ndarray) -> jnp.ndarray:
        valid = indices != -1
        safe = jnp.where(valid, indices, 0)
        coeffs = jnp.zeros((n_coeffs, block_size), dtype=jnp.float32)
        coeffs = coeffs.at[safe, jnp.arange(block_size)].set(valid.astype(jnp.float32))
        sensed = jax.vmap(a_matvec, in_axes=1, out_axes=1)(coeffs)
        return sensed * valid[None, :]

    total = jnp.asarray(0.0, dtype=jnp.float32)
    for i in range(num_blocks):
        for j in range(i, num_blocks):
            sensed_p = _sensed_block(blocks[i])
            sensed_q = _sensed_block(blocks[j])

            p_norm = jnp.sqrt(jnp.sum(jnp.abs(sensed_p) ** 2, axis=0, keepdims=True) + eps)
            q_norm = jnp.sqrt(jnp.sum(jnp.abs(sensed_q) ** 2, axis=0, keepdims=True) + eps)
            sensed_p = sensed_p / p_norm
            sensed_q = sensed_q / q_norm

            gram_abs = jnp.abs(sensed_p.T @ sensed_q)
            if i == j:
                total = total + jnp.sum(jnp.triu(gram_abs, k=1))
            else:
                total = total + jnp.sum(gram_abs)

    return total


def total_coherence_loss_from_matrix(
    a_matrix: jnp.ndarray,
    num_blocks: int,
    eps: float = 1e-18,
) -> jnp.ndarray:
    """
    Matrix convenience wrapper around operator-form total coherence loss.
    """
    return total_coherence_loss(
        a_matvec=lambda v: dense_matvec(a_matrix, v),
        n_coeffs=int(a_matrix.shape[1]),
        num_blocks=num_blocks,
        eps=eps,
    )


def mutual_information_loss(
    a_matvec: Callable[[jnp.ndarray], jnp.ndarray],
    a_rmatvec: Callable[[jnp.ndarray], jnp.ndarray],
    m_measurement: int,
    n_coeffs: int,
    key: jax.Array,
    rsvd_k: int | None = None,
    rsvd_k_frac: float = 0.1,
    p: int = 10,
    q_iter: int = 0,
) -> jnp.ndarray:
    """
    Mutual-information objective from operator singular values.

    Returns -sum(log(1 + s_i^2)) over top randomized singular values.
    """
    if m_measurement <= 0 or n_coeffs <= 0:
        raise ValueError("m_measurement and n_coeffs must be > 0")
    if p < 0:
        raise ValueError("p must be >= 0")
    if q_iter < 0:
        raise ValueError("q_iter must be >= 0")
    if rsvd_k is not None and rsvd_k <= 0:
        raise ValueError("rsvd_k must be > 0 when provided")
    if rsvd_k_frac <= 0:
        raise ValueError("rsvd_k_frac must be > 0")

    if rsvd_k is None:
        k = max(1, int(rsvd_k_frac * n_coeffs))
        k = min(k, m_measurement - p)
    else:
        k = min(rsvd_k, m_measurement - p)

    if k <= 0:
        raise ValueError("effective randomized SVD rank k must be > 0")

    _, s_k, _ = randomized_svd(
        a_matvec=a_matvec,
        a_rmatvec=a_rmatvec,
        m=m_measurement,
        n=n_coeffs,
        k=k,
        key=key,
        p=p,
        q_iter=q_iter,
    )
    return -jnp.sum(jnp.log1p(s_k**2))


def mutual_information_loss_from_matrix(
    a_matrix: jnp.ndarray,
    key: jax.Array,
    rsvd_k: int | None = None,
    rsvd_k_frac: float = 0.1,
    p: int = 10,
    q_iter: int = 0,
) -> jnp.ndarray:
    """
    Matrix convenience wrapper around operator-form mutual-information loss.
    """
    m_measurement, n_coeffs = a_matrix.shape
    return mutual_information_loss(
        a_matvec=lambda v: dense_matvec(a_matrix, v),
        a_rmatvec=lambda v: dense_rmatvec(a_matrix, v),
        m_measurement=m_measurement,
        n_coeffs=n_coeffs,
        key=key,
        rsvd_k=rsvd_k,
        rsvd_k_frac=rsvd_k_frac,
        p=p,
        q_iter=q_iter,
    )
