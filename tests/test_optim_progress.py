from __future__ import annotations

import jax.numpy as jnp
import optax

from fouriax.optim.data import iter_minibatches
from fouriax.optim.optim import optimize_dataset_optical_module


def test_iter_minibatches_has_len() -> None:
    x = jnp.arange(5)
    batches = iter_minibatches(x, batch_size=2, shuffle=False)
    assert len(batches) == 3
    assert [batch[0].shape[0] for batch in batches] == [2, 2, 1]


def test_dataset_report_uses_epoch_and_batch_progress(capsys) -> None:
    x = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)

    def build_module(params: jnp.ndarray) -> jnp.ndarray:
        return params

    def batch_loss_fn(params: jnp.ndarray, batch: tuple[jnp.ndarray]) -> jnp.ndarray:
        (xb,) = batch
        return jnp.mean((xb + params) ** 2)

    optimize_dataset_optical_module(
        init_params=jnp.array(0.5, dtype=jnp.float32),
        build_module=build_module,
        batch_loss_fn=batch_loss_fn,
        optimizer=optax.adam(1e-2),
        train_data=x,
        batch_size=2,
        epochs=1,
        val_data=x,
        val_every_epochs=1,
        log_every_steps=1,
        jit=False,
        seed=0,
    )

    out = capsys.readouterr().out
    assert "epoch=001/001 batches=2" in out
    assert "  batch=001/002" in out
    assert "  batch=002/002" in out
    assert "->epoch=001/001 val_loss=" in out
    assert "elapsed=" in out
    assert "batch_time=" in out
    assert "epoch_time=" in out
    assert "step=" not in out
