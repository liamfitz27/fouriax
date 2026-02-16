import pytest


def _has_cuda_gpu(jax) -> bool:
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        return False
    if not gpu_devices:
        return False

    for dev in gpu_devices:
        kind = str(getattr(dev, "device_kind", "")).lower()
        client = getattr(dev, "client", None)
        platform_version = str(getattr(client, "platform_version", "")).lower()
        if "nvidia" in kind or "cuda" in platform_version:
            return True
    return False


def test_gpu_executes_jax_kernel_when_cuda_available():
    try:
        import jax
        import jax.numpy as jnp
    except ModuleNotFoundError:
        pytest.skip("JAX is not installed in this environment")

    if not _has_cuda_gpu(jax):
        pytest.skip("No CUDA-capable GPU available to JAX")

    gpu = jax.devices("gpu")[0]
    with jax.default_device(gpu):
        x = jnp.arange(1024, dtype=jnp.float32)
        y = jnp.sin(x) + x * x
        y.block_until_ready()

    platforms = {dev.platform for dev in y.devices()}
    assert platforms == {"gpu"}
