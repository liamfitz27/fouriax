"""Optimization-loop helpers for example scripts and notebooks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Literal, TypeVar, cast

import jax
import jax.numpy as jnp
import optax

from fouriax.optim.data import iter_minibatches

ParamsT = TypeVar("ParamsT")
DecoderParamsT = TypeVar("DecoderParamsT")
OptStateT = TypeVar("OptStateT")
ModuleT = TypeVar("ModuleT")
BatchT = TypeVar("BatchT")
TrainDataT = TypeVar("TrainDataT")
ValDataT = TypeVar("ValDataT")


def apply_optax_updates(
    optimizer: optax.GradientTransformation,
    params: ParamsT,
    opt_state: OptStateT,
    grads: ParamsT,
) -> tuple[ParamsT, OptStateT]:
    """Apply one Optax update step and return `(params, opt_state)`."""
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def should_log_step(step: int, *, every: int, total_steps: int) -> bool:
    """Return `True` on periodic log steps and the final step."""
    if total_steps <= 0:
        return False
    if every <= 0:
        return step == total_steps - 1
    return (step % every == 0) or (step == total_steps - 1)


@dataclass
class BestValueTracker(Generic[ParamsT]):
    """Track the best scalar metric and a copied snapshot of a JAX pytree."""

    mode: Literal["min", "max"] = "min"
    best_value: float = field(init=False)
    best_state: ParamsT | None = field(init=False, default=None)
    best_step: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.best_value = float("inf") if self.mode == "min" else float("-inf")

    def update(self, value: float, state: ParamsT, *, step: int | None = None) -> bool:
        is_better = value < self.best_value if self.mode == "min" else value > self.best_value
        if not is_better:
            return False
        self.best_value = float(value)
        self.best_state = jax.tree_util.tree_map(lambda x: x, state)
        self.best_step = step
        return True


@dataclass
class ModuleOptResult(Generic[ModuleT, ParamsT]):
    """Outputs from `optimize_optical_module`."""

    best_module: ModuleT
    best_params: ParamsT
    history: list[float]
    best_loss: float
    final_loss: float
    best_step: int | None


@dataclass
class ValidationRecord:
    """Validation metrics recorded at a training step/epoch boundary."""

    step: int
    epoch: int
    metrics: dict[str, float]


@dataclass
class DatasetOptResult(Generic[ParamsT]):
    """Outputs from dataset optimization over minibatches."""

    best_params: ParamsT
    final_params: ParamsT
    train_loss_history: list[float]
    val_history: list[ValidationRecord]
    best_metric_name: str
    best_metric_value: float
    best_step: int | None
    best_epoch: int | None
    final_train_loss: float
    final_val_metrics: dict[str, float] | None


@dataclass
class ModuleDatasetOptResult(Generic[ModuleT, ParamsT]):
    """Dataset optimization outputs plus built optical modules."""

    best_module: ModuleT
    final_module: ModuleT
    params_result: DatasetOptResult[ParamsT]


@dataclass
class HybridModuleDatasetOptResult(Generic[ModuleT, ParamsT, DecoderParamsT]):
    """Hybrid dataset optimization outputs for optical module + decoder params."""

    best_module: ModuleT
    final_module: ModuleT
    best_optical_params: ParamsT
    final_optical_params: ParamsT
    best_decoder_params: DecoderParamsT
    final_decoder_params: DecoderParamsT
    params_result: DatasetOptResult[dict[str, Any]]


def _copy_pytree(state: ParamsT) -> ParamsT:
    return cast(ParamsT, jax.tree_util.tree_map(lambda x: x, state))


def _normalize_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        normalized[str(key)] = float(value)
    return normalized


def _as_batch_arrays(data: Any, *, name: str) -> tuple[Any, ...]:
    """Normalize a dataset input into a tuple of aligned arrays."""
    if isinstance(data, tuple):
        arrays = data
    elif isinstance(data, list):
        arrays = tuple(data)
    else:
        arrays = (data,)
    if not arrays:
        raise ValueError(f"{name} must contain at least one array")
    n_items = len(arrays[0])
    if any(len(arr) != n_items for arr in arrays):
        raise ValueError(f"all arrays in {name} must have the same length")
    return arrays


def _uniform_dataset_report(payload: dict[str, Any]) -> None:
    """Default console reporting format used by dataset optimizers."""
    epoch = int(payload.get("epoch", -1)) + 1
    step = int(payload.get("step", -1))
    train_loss = payload.get("train_loss", None)
    train_part = (
        f" train_loss={float(train_loss):.6f}"
        if train_loss is not None
        else ""
    )
    if "val" not in payload:
        print(f"epoch={epoch:03d} step={step:05d}{train_part}")
        return

    metrics = payload["val"]
    if not isinstance(metrics, Mapping):
        print(f"epoch={epoch:03d} step={step:05d}{train_part}")
        return
    metrics_part = " ".join(f"{k}={float(v):.6f}" for k, v in metrics.items())
    best_tag = " [best]" if bool(payload.get("is_best", False)) else ""
    print(f"epoch={epoch:03d} step={step:05d}{train_part} {metrics_part}{best_tag}".strip())


def _default_optical_report(step: int, loss: float) -> None:
    """Default reporter for `optimize_optical_module(...)`."""
    print(f"step={step:03d} loss={loss:.6f}")


def optimize_optical_module(
    *,
    init_params: ParamsT,
    build_module: Callable[[ParamsT], ModuleT],
    loss_fn: Callable[[ParamsT], jax.Array | jnp.ndarray],
    optimizer: optax.GradientTransformation,
    steps: int,
    log_every: int = 50,
    mode: Literal["min", "max"] = "min",
    jit: bool = True,
    reporter: Callable[[int, float], None] | None = None,
) -> ModuleOptResult[ModuleT, ParamsT]:
    """Run an Optax optimization loop and return best module + optimization metadata."""
    if steps <= 0:
        raise ValueError("steps must be > 0")

    value_and_grad = jax.value_and_grad(loss_fn)
    if jit:
        value_and_grad = jax.jit(value_and_grad)

    params = init_params
    opt_state = optimizer.init(params)
    history: list[float] = []
    best = BestValueTracker[ParamsT](mode=mode)
    report_fn = _default_optical_report if reporter is None else reporter

    for step in range(steps):
        loss, grads = value_and_grad(params)
        params, opt_state = apply_optax_updates(optimizer, params, opt_state, grads)
        loss_val = float(loss)
        history.append(loss_val)
        best.update(loss_val, params, step=step)
        if should_log_step(step, every=log_every, total_steps=steps):
            report_fn(step, loss_val)

    if best.best_state is None:
        raise RuntimeError("optimization did not produce a best state")

    return ModuleOptResult(
        best_module=build_module(best.best_state),
        best_params=best.best_state,
        history=history,
        best_loss=best.best_value,
        final_loss=history[-1],
        best_step=best.best_step,
    )


def optimize_dataset_params(
    *,
    init_params: ParamsT,
    optimizer: optax.GradientTransformation,
    train_data: TrainDataT,
    batch_iter_fn: Callable[[TrainDataT], Iterable[BatchT]]
    | Callable[[TrainDataT, int], Iterable[BatchT]]
    | Callable[[TrainDataT, int, Any], Iterable[BatchT]],
    train_loss_fn: Callable[[ParamsT, BatchT], jax.Array | jnp.ndarray],
    epochs: int,
    val_eval_fn: Callable[[ParamsT, ValDataT], Mapping[str, Any]] | None = None,
    val_data: ValDataT | None = None,
    val_every_epochs: int = 1,
    val_every_steps: int = 0,
    log_every_steps: int = 50,
    select_metric: str | None = None,
    select_mode: Literal["min", "max"] = "min",
    jit: bool = True,
    reporter: Callable[[dict[str, Any]], None] | None = None,
    rng: Any | None = None,
) -> DatasetOptResult[ParamsT]:
    """Optimize arbitrary params over minibatches with optional validation tracking.

    `batch_iter_fn` should typically accept `(train_data, epoch, rng)` and return
    an iterable of batches for that epoch. Simpler callables that accept fewer
    positional arguments are also supported.
    """
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if val_every_epochs < 0:
        raise ValueError("val_every_epochs must be >= 0")
    if val_every_steps < 0:
        raise ValueError("val_every_steps must be >= 0")
    if val_eval_fn is not None and val_data is None:
        raise ValueError("val_data is required when val_eval_fn is provided")
    if val_eval_fn is None and select_metric is not None:
        raise ValueError("select_metric requires val_eval_fn")

    value_and_grad = jax.value_and_grad(train_loss_fn)
    if jit:
        value_and_grad = jax.jit(value_and_grad)

    params = init_params
    opt_state = optimizer.init(params)
    train_loss_history: list[float] = []
    val_history: list[ValidationRecord] = []
    best = BestValueTracker[ParamsT](mode=select_mode)
    best_epoch: int | None = None
    last_train_loss: float | None = None
    final_val_metrics: dict[str, float] | None = None
    batch_iter_any = cast(Callable[..., Iterable[BatchT]], batch_iter_fn)

    def iter_batches(epoch: int) -> Iterable[BatchT]:
        try:
            return batch_iter_any(train_data, epoch, rng)
        except TypeError:
            try:
                return batch_iter_any(train_data, epoch)
            except TypeError:
                return batch_iter_any(train_data)

    def maybe_report(payload: dict[str, Any]) -> None:
        if reporter is not None:
            reporter(payload)

    def run_validation(*, step: int, epoch: int) -> None:
        nonlocal final_val_metrics, best_epoch
        if val_eval_fn is None:
            return
        metrics = _normalize_metrics(val_eval_fn(params, val_data))  # type: ignore[arg-type]
        final_val_metrics = metrics
        val_history.append(ValidationRecord(step=step, epoch=epoch, metrics=metrics))
        metric_name = select_metric or next(iter(metrics))
        if metric_name not in metrics:
            raise KeyError(
                f"validation metric {metric_name!r} not found; "
                f"available: {sorted(metrics)}"
            )
        is_best = best.update(metrics[metric_name], params, step=step)
        if is_best:
            best_epoch = epoch
        maybe_report(
            {
                "step": step,
                "epoch": epoch,
                "train_loss": last_train_loss,
                "val": metrics,
                "is_best": is_best,
                "best_metric": best.best_value,
                "best_metric_name": metric_name,
            }
        )

    step = 0
    for epoch in range(epochs):
        saw_batch = False
        for batch in iter_batches(epoch):
            saw_batch = True
            loss, grads = value_and_grad(params, batch)
            params, opt_state = apply_optax_updates(optimizer, params, opt_state, grads)
            last_train_loss = float(loss)
            train_loss_history.append(last_train_loss)

            if val_eval_fn is None:
                best.update(last_train_loss, params, step=step)
                best_epoch = epoch if best.best_step == step else best_epoch

            should_log_train = log_every_steps > 0 and (step % log_every_steps == 0)
            if reporter is not None and should_log_train:
                maybe_report(
                    {
                        "step": step,
                        "epoch": epoch,
                        "train_loss": last_train_loss,
                    }
                )

            if (
                val_eval_fn is not None
                and val_every_steps > 0
                and (step + 1) % val_every_steps == 0
            ):
                run_validation(step=step, epoch=epoch)
            step += 1

        if not saw_batch:
            raise ValueError(f"batch_iter_fn produced no batches for epoch {epoch}")

        if val_eval_fn is not None and val_every_epochs > 0 and (epoch + 1) % val_every_epochs == 0:
            run_validation(step=step - 1, epoch=epoch)

    if not train_loss_history:
        raise RuntimeError("optimization did not run any training steps")

    if best.best_state is None:
        # Happens when validation exists but cadence never fired.
        if val_eval_fn is not None:
            run_validation(step=step - 1, epoch=epochs - 1)
        if best.best_state is None:
            raise RuntimeError("optimization did not produce a best state")

    if val_eval_fn is not None and final_val_metrics is None:
        final_val_metrics = _normalize_metrics(val_eval_fn(params, val_data))  # type: ignore[arg-type]

    metric_name = select_metric if select_metric is not None else "train_loss"
    if val_eval_fn is None:
        metric_name = "train_loss"

    return DatasetOptResult(
        best_params=best.best_state,
        final_params=_copy_pytree(params),
        train_loss_history=train_loss_history,
        val_history=val_history,
        best_metric_name=metric_name,
        best_metric_value=best.best_value,
        best_step=best.best_step,
        best_epoch=best_epoch,
        final_train_loss=train_loss_history[-1],
        final_val_metrics=final_val_metrics,
    )


def optimize_dataset_optical_module(
    *,
    init_params: ParamsT,
    build_module: Callable[[ParamsT], ModuleT],
    batch_loss_fn: Callable[[ParamsT, Any], jax.Array | jnp.ndarray] | None = None,
    sample_loss_fn: Callable[[ParamsT, Any], jax.Array | jnp.ndarray] | None = None,
    optimizer: optax.GradientTransformation,
    train_data: Any,
    batch_size: int,
    epochs: int,
    val_data: Any | None = None,
    val_every_epochs: int = 1,
    val_every_steps: int = 0,
    log_every_steps: int = 0,
    jit: bool = True,
    seed: int = 0,
    drop_last_train: bool = False,
) -> ModuleDatasetOptResult[ModuleT, ParamsT]:
    """Optimize an optical module over minibatches using a shared train/val loss.

    This is a high-level convenience wrapper:
    - minibatching is derived internally from `iter_minibatches(...)`
    - exactly one of `batch_loss_fn` or `sample_loss_fn` must be provided
    - validation uses the same effective batch loss averaged over `val_data`
    - reporting uses a uniform built-in console formatter
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if (batch_loss_fn is None) == (sample_loss_fn is None):
        raise ValueError("provide exactly one of batch_loss_fn or sample_loss_fn")

    train_arrays = _as_batch_arrays(train_data, name="train_data")
    val_arrays = None if val_data is None else _as_batch_arrays(val_data, name="val_data")

    if sample_loss_fn is not None:
        def effective_batch_loss_fn(params: ParamsT, batch: Any) -> jax.Array | jnp.ndarray:
            batch_items = batch if isinstance(batch, tuple) else (batch,)

            def sample_loss_wrapped(*sample_items: Any) -> jax.Array | jnp.ndarray:
                sample = sample_items[0] if len(sample_items) == 1 else sample_items
                return sample_loss_fn(params, sample)

            return jnp.mean(jax.vmap(sample_loss_wrapped)(*batch_items))
    else:
        def effective_batch_loss_fn(params: ParamsT, batch: Any) -> jax.Array | jnp.ndarray:
            assert batch_loss_fn is not None
            return batch_loss_fn(params, batch)

    def batch_iter_fn(data: tuple[Any, ...], epoch: int) -> Iterable[tuple[Any, ...]]:
        return iter_minibatches(
            *data,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
            drop_last=drop_last_train,
        )

    def val_eval_fn(
        params: ParamsT,
        data: tuple[Any, ...],
    ) -> dict[str, float]:
        losses: list[float] = []
        for batch in iter_minibatches(*data, batch_size=batch_size, shuffle=False):
            losses.append(float(effective_batch_loss_fn(params, batch)))
        if not losses:
            raise ValueError("validation data produced no batches")
        return {"val_loss": sum(losses) / len(losses)}

    select_metric = "val_loss" if val_arrays is not None else None

    params_result = optimize_dataset_params(
        init_params=init_params,
        optimizer=optimizer,
        train_data=train_arrays,
        batch_iter_fn=batch_iter_fn,
        train_loss_fn=effective_batch_loss_fn,
        epochs=epochs,
        val_eval_fn=val_eval_fn if val_arrays is not None else None,
        val_data=val_arrays,
        val_every_epochs=val_every_epochs,
        val_every_steps=val_every_steps,
        log_every_steps=log_every_steps,
        select_metric=select_metric,
        select_mode="min",
        jit=jit,
        reporter=_uniform_dataset_report,
    )
    return ModuleDatasetOptResult(
        best_module=build_module(params_result.best_params),
        final_module=build_module(params_result.final_params),
        params_result=params_result,
    )


def optimize_dataset_hybrid_module(
    *,
    init_optical_params: ParamsT,
    init_decoder_params: DecoderParamsT,
    build_module: Callable[[ParamsT], ModuleT],
    batch_loss_fn: (
        Callable[[ParamsT, DecoderParamsT, Any], jax.Array | jnp.ndarray] | None
    ) = None,
    sample_loss_fn: (
        Callable[[ParamsT, DecoderParamsT, Any], jax.Array | jnp.ndarray] | None
    ) = None,
    train_data: Any,
    batch_size: int,
    epochs: int,
    val_data: Any | None = None,
    optimizer: optax.GradientTransformation | None = None,
    optical_optimizer: optax.GradientTransformation | None = None,
    decoder_optimizer: optax.GradientTransformation | None = None,
    val_every_epochs: int = 1,
    val_every_steps: int = 0,
    log_every_steps: int = 0,
    jit: bool = True,
    seed: int = 0,
    drop_last_train: bool = False,
) -> HybridModuleDatasetOptResult[ModuleT, ParamsT, DecoderParamsT]:
    """Optimize optical + decoder params over minibatches using a shared loss.

    Simplified wrapper behavior matches `optimize_dataset_optical_module(...)`:
    - internal minibatching via `iter_minibatches(...)`
    - exactly one of `batch_loss_fn` or `sample_loss_fn` must be provided
    - validation uses the same effective batch loss, averaged over `val_data`
    - built-in uniform reporting

    Optimizer selection:
    - pass `optimizer` directly, or
    - pass both `optical_optimizer` and `decoder_optimizer` to build an
      internal `optax.multi_transform(...)` with separate parameter groups.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if (batch_loss_fn is None) == (sample_loss_fn is None):
        raise ValueError("provide exactly one of batch_loss_fn or sample_loss_fn")

    has_direct_optimizer = optimizer is not None
    has_group_optimizers = optical_optimizer is not None or decoder_optimizer is not None
    if has_direct_optimizer and has_group_optimizers:
        raise ValueError(
            "pass either optimizer or (optical_optimizer and decoder_optimizer), not both"
        )
    if optimizer is None:
        if optical_optimizer is None or decoder_optimizer is None:
            raise ValueError(
                "either optimizer must be provided, or both optical_optimizer "
                "and decoder_optimizer must be provided"
            )
        optimizer = optax.multi_transform(
            transforms={"optical": optical_optimizer, "decoder": decoder_optimizer},
            param_labels={"optical": "optical", "decoder": "decoder"},
        )

    train_arrays = _as_batch_arrays(train_data, name="train_data")
    val_arrays = None if val_data is None else _as_batch_arrays(val_data, name="val_data")

    def batch_iter_fn(data: tuple[Any, ...], epoch: int) -> Iterable[tuple[Any, ...]]:
        return iter_minibatches(
            *data,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
            drop_last=drop_last_train,
        )

    if sample_loss_fn is not None:
        def combined_loss_fn(params: dict[str, Any], batch: Any) -> jax.Array | jnp.ndarray:
            batch_items = batch if isinstance(batch, tuple) else (batch,)

            def sample_loss_wrapped(*sample_items: Any) -> jax.Array | jnp.ndarray:
                sample = sample_items[0] if len(sample_items) == 1 else sample_items
                return sample_loss_fn(params["optical"], params["decoder"], sample)

            return jnp.mean(jax.vmap(sample_loss_wrapped)(*batch_items))
    else:
        def combined_loss_fn(params: dict[str, Any], batch: Any) -> jax.Array | jnp.ndarray:
            assert batch_loss_fn is not None
            return batch_loss_fn(params["optical"], params["decoder"], batch)

    def val_eval_fn(
        params: dict[str, Any],
        data: tuple[Any, ...],
    ) -> dict[str, float]:
        losses: list[float] = []
        for batch in iter_minibatches(*data, batch_size=batch_size, shuffle=False):
            losses.append(float(combined_loss_fn(params, batch)))
        if not losses:
            raise ValueError("validation data produced no batches")
        return {"val_loss": sum(losses) / len(losses)}

    init_params: dict[str, Any] = {
        "optical": init_optical_params,
        "decoder": init_decoder_params,
    }
    select_metric = "val_loss" if val_arrays is not None else None

    params_result = optimize_dataset_params(
        init_params=init_params,
        optimizer=optimizer,
        train_data=train_arrays,
        batch_iter_fn=batch_iter_fn,
        train_loss_fn=combined_loss_fn,
        epochs=epochs,
        val_eval_fn=val_eval_fn if val_arrays is not None else None,
        val_data=val_arrays,
        val_every_epochs=val_every_epochs,
        val_every_steps=val_every_steps,
        log_every_steps=log_every_steps,
        select_metric=select_metric,
        select_mode="min",
        jit=jit,
        reporter=_uniform_dataset_report,
    )

    best_params = params_result.best_params
    final_params = params_result.final_params
    return HybridModuleDatasetOptResult(
        best_module=build_module(best_params["optical"]),
        final_module=build_module(final_params["optical"]),
        best_optical_params=best_params["optical"],
        final_optical_params=final_params["optical"],
        best_decoder_params=best_params["decoder"],
        final_decoder_params=final_params["decoder"],
        params_result=params_result,
    )
