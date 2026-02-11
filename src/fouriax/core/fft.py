from functools import partial

import jax
import jax.numpy as jnp
from jax.numpy.fft import fftn, ifftn, irfftn, rfftn


@partial(jax.jit, static_argnames=("mode", "axes", "fft_shape", "sum_fft_axis"))
def fftconvolve(
    input_tensor,
    kernel_tensor,
    mode="same",
    axes=None,
    fft_shape=None,
    sum_fft_axis=None,
):
    """
    Perform nD convolution using FFTs.

    If `sum_fft_axis` is provided, the product of FFTs is summed over that axis
    before inverse FFT to avoid repeated inverse transforms.
    """
    if axes is None:
        axes = range(input_tensor.ndim)

    if fft_shape is None:
        s1 = [input_tensor.shape[i] for i in axes]
        s2 = [kernel_tensor.shape[i] for i in axes]
        fft_shape = tuple(int(s1_i + s2_i - 1) for s1_i, s2_i in zip(s1, s2, strict=True))

    inputs_are_real = jnp.issubdtype(input_tensor.dtype, jnp.floating) and jnp.issubdtype(
        kernel_tensor.dtype, jnp.floating
    )

    if inputs_are_real:
        input_fft = rfftn(input_tensor, s=fft_shape, axes=axes)
        kernel_fft = rfftn(kernel_tensor, s=fft_shape, axes=axes)
        conv_fft = input_fft * kernel_fft
        if sum_fft_axis is not None:
            conv_fft = jnp.sum(conv_fft, axis=sum_fft_axis)
        conv_full = irfftn(conv_fft, s=fft_shape, axes=axes)
    else:
        input_fft = fftn(input_tensor, s=fft_shape, axes=axes)
        kernel_fft = fftn(kernel_tensor, s=fft_shape, axes=axes)
        conv_fft = input_fft * kernel_fft
        if sum_fft_axis is not None:
            conv_fft = jnp.sum(conv_fft, axis=sum_fft_axis)
        conv_full = ifftn(conv_fft, s=fft_shape, axes=axes)

    is_real_output_expected = jnp.issubdtype(input_tensor.dtype, jnp.floating) and jnp.issubdtype(
        kernel_tensor.dtype, jnp.floating
    )

    if mode == "full":
        result = conv_full
    elif mode == "same":
        s1_shape = [input_tensor.shape[i] for i in axes]
        s2_shape = [kernel_tensor.shape[i] for i in axes]
        start_indices = [(k_s - 1) // 2 for k_s in s2_shape]

        crop_slices = [slice(None)] * conv_full.ndim
        for i, axis_idx in enumerate(axes):
            start = start_indices[i]
            end = start + s1_shape[i]
            crop_slices[axis_idx] = slice(start, end)
        result = conv_full[tuple(crop_slices)]
    else:
        raise ValueError("mode must be 'full' or 'same'")

    if is_real_output_expected:
        return result.real
    return result
