#!/usr/bin/env python3
"""Optimize spectral filters with a D-opt proxy, then end-to-end train an MLP decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from sklearn.decomposition import PCA

from fouriax.optics import AmplitudeMask, Field, Grid, IntensitySensor, OpticalModule, Spectrum

EPS = 1e-6
SPECTRAL_SLICE_START = 40
SPECTRAL_SLICE_STOP = 190
PROXY_STEPS = 3000
PROXY_LOG_EVERY = 25
RECON_LOG_EVERY = 25


class SpectralReconMLP(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(int(hidden_dim))(x)
            x = nn.relu(x)
        return nn.Dense(self.out_dim)(x)


def parse_hidden_dims(spec: str) -> tuple[int, ...]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("hidden-dims must contain at least one positive integer")
    dims = tuple(int(p) for p in parts)
    if any(d <= 0 for d in dims):
        raise ValueError("all hidden-dims must be positive")
    return dims


def load_indian_pines(
    train_path: Path,
    val_path: Path,
    wavelengths_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(np.load(train_path), dtype=np.float32)[
        :, SPECTRAL_SLICE_START:SPECTRAL_SLICE_STOP
    ]
    val = np.asarray(np.load(val_path), dtype=np.float32)[
        :, SPECTRAL_SLICE_START:SPECTRAL_SLICE_STOP
    ]
    wavelengths_nm = np.asarray(np.load(wavelengths_path), dtype=np.float32)[
        SPECTRAL_SLICE_START:SPECTRAL_SLICE_STOP
    ]
    if train.ndim != 2 or val.ndim != 2:
        raise ValueError("train/val spectra must be 2D arrays shaped (n_samples, n_wavelengths)")
    if wavelengths_nm.ndim != 1:
        raise ValueError("wavelengths must be 1D")
    if train.shape[1] != wavelengths_nm.shape[0] or val.shape[1] != wavelengths_nm.shape[0]:
        raise ValueError("spectra wavelength dimension must match wavelengths.npy")
    return train, val, wavelengths_nm


def build_pca_basis(
    train_spectra: np.ndarray, n_components: int, seed: int
) -> tuple[PCA, np.ndarray]:
    pca = PCA(n_components=n_components, svd_solver="full", random_state=seed)
    pca.fit(train_spectra)
    basis = np.asarray(pca.components_.T, dtype=np.float32)  # (L, r)
    return pca, basis


def filter_intensity_matrix(raw_filter: jnp.ndarray) -> jnp.ndarray:
    amp = jax.nn.sigmoid(raw_filter)
    return (amp * amp).astype(jnp.float32)


def add_measurement_noise(
    y: jnp.ndarray,
    *,
    key: jax.Array,
    photon_scale: float,
) -> jnp.ndarray:
    if photon_scale <= 0.0:
        return y
    k_shot = key
    if photon_scale > 0.0:
        shot_std = jnp.sqrt(jnp.maximum(y, 0.0) / jnp.asarray(photon_scale, dtype=y.dtype))
        shot = shot_std * jax.random.normal(k_shot, shape=y.shape, dtype=y.dtype)
    else:
        shot = jnp.zeros_like(y)
    return y + shot


def filter_smoothness(
    raw_filter: jnp.ndarray,
) -> jnp.ndarray:
    a = filter_intensity_matrix(raw_filter)
    smoothness = jnp.mean((a[:, 1:] - a[:, :-1]) ** 2)
    return smoothness


def build_detector_masks(array_size: int) -> jnp.ndarray:
    num_filters = int(array_size * array_size)
    masks = np.zeros((num_filters, array_size, array_size), dtype=np.float32)
    idx = np.arange(num_filters)
    ys = idx // array_size
    xs = idx % array_size
    masks[idx, ys, xs] = 1.0
    return jnp.asarray(masks, dtype=jnp.float32)


def make_optical_measurement(
    *,
    wavelengths_nm: np.ndarray,
    num_filters: int,
) -> dict[str, object]:
    array_size = int(round(np.sqrt(num_filters)))
    if array_size * array_size != int(num_filters):
        raise ValueError(f"num_filters must be a perfect square, got {num_filters}")
    grid = Grid.from_extent(nx=array_size, ny=array_size, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_array(jnp.asarray(wavelengths_nm * 1e-3, dtype=jnp.float32))
    detector_masks = build_detector_masks(array_size)
    sensor = IntensitySensor(sum_wavelengths=True, detector_masks=detector_masks)
    return {
        "array_size": array_size,
        "grid": grid,
        "spectrum": spectrum,
        "sensor": sensor,
    }

def measure_batch_optical(
    batch_spectra: jnp.ndarray,
    a_matrix: jnp.ndarray,
    *,
    array_size: int,
    grid: Grid,
    spectrum: Spectrum,
    sensor: IntensitySensor,
) -> jnp.ndarray:
    amp_map = jnp.sqrt(jnp.clip(a_matrix.T, 0.0, 1.0)).reshape(
        (spectrum.size, array_size, array_size)
    )
    module = OpticalModule(layers=(AmplitudeMask(amplitude_map=amp_map),), sensor=sensor)

    def measure_one(sample_spectrum: jnp.ndarray) -> jnp.ndarray:
        sample_amp = jnp.sqrt(jnp.maximum(sample_spectrum, 0.0)).reshape((spectrum.size, 1, 1))
        field_data = (
            jnp.ones((spectrum.size, array_size, array_size), dtype=jnp.complex64) * sample_amp
        )
        field = Field(data=field_data, grid=grid, spectrum=spectrum, domain="spatial")
        return module.measure(field).astype(jnp.float32)

    return jax.vmap(measure_one)(batch_spectra)


def proxy_loss(
    raw_filter: jnp.ndarray,
    batch_spectra: jnp.ndarray,
    *,
    basis: jnp.ndarray,
    photon_scale: float,
    smoothness_reg: float,
) -> jnp.ndarray:
    a = filter_intensity_matrix(raw_filter)
    ab = a @ basis  # (K, r)
    y = batch_spectra @ a.T  # (B, K)
    noise_var = jnp.zeros_like(y)
    if photon_scale > 0.0:
        noise_var = noise_var + jnp.maximum(y, 0.0) / jnp.asarray(photon_scale, dtype=y.dtype)
    noise_var = jnp.maximum(noise_var, jnp.asarray(EPS, dtype=y.dtype))
    inv_noise_var = 1.0 / noise_var

    fisher = jnp.einsum("kr,bk,ks->brs", ab, inv_noise_var, ab)
    gram = jnp.eye(ab.shape[1], dtype=ab.dtype)[None, :, :] + fisher
    sign, logdet = jax.vmap(jnp.linalg.slogdet)(gram)
    logdet_safe = jnp.where(sign > 0, logdet, -1e12)
    d_opt_score = jnp.mean(logdet_safe)
    info_loss = -d_opt_score

    smoothness = filter_smoothness(raw_filter)
    return info_loss + smoothness_reg * smoothness

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-spectra",
        type=Path,
        default=Path("data/meta_atoms/indian_pines/train_spectra_full.npy"),
    )
    parser.add_argument(
        "--val-spectra",
        type=Path,
        default=Path("data/meta_atoms/indian_pines/val_spectra.npy"),
    )
    parser.add_argument(
        "--wavelengths",
        type=Path,
        default=Path("data/meta_atoms/indian_pines/wavelengths.npy"),
    )
    parser.add_argument("--train-samples", type=int, default=200000)
    parser.add_argument("--val-samples", type=int, default=2048)
    parser.add_argument("--array-size", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)

    parser.add_argument("--filter-lr", type=float, default=1.4e-1)

    parser.add_argument("--smoothness-reg", type=float, default=0.1)
    parser.add_argument("--photon-scale", type=float, default=1000.0)

    parser.add_argument("--recon-steps", type=int, default=2000)
    parser.add_argument("--recon-lr", type=float, default=8e-4)
    parser.add_argument("--hidden-dims", type=str, default="128,256,384,384")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.array_size <= 0:
        raise ValueError("array-size must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.train_samples <= 0 or args.val_samples <= 0:
        raise ValueError("train-samples and val-samples must be positive")
    if args.recon_steps < 0:
        raise ValueError("recon-steps must be >= 0")
    if args.filter_lr <= 0 or args.recon_lr <= 0:
        raise ValueError("filter-lr and recon-lr must be positive")
    if args.photon_scale < 0.0:
        raise ValueError("photon-scale must be >= 0")

    hidden_dims = parse_hidden_dims(args.hidden_dims)

    for p in (args.train_spectra, args.val_spectra, args.wavelengths):
        if not p.exists():
            raise FileNotFoundError(f"missing file: {p}")

    train_full, val_full, wavelengths_nm = load_indian_pines(
        args.train_spectra,
        args.val_spectra,
        args.wavelengths,
    )
    train = train_full[: min(args.train_samples, train_full.shape[0])]
    val = val_full[: min(args.val_samples, val_full.shape[0])]

    n_wavelengths = int(train.shape[1])
    num_filters = int(args.array_size * args.array_size)
    pca_components = min(32, n_wavelengths, train.shape[0])
    pca, basis_np = build_pca_basis(train, pca_components, args.seed)
    basis = jnp.asarray(basis_np, dtype=jnp.float32)
    measurement = make_optical_measurement(
        wavelengths_nm=wavelengths_nm,
        num_filters=num_filters,
    )

    key = jax.random.PRNGKey(args.seed)
    raw_filter = jax.random.normal(key, (num_filters, n_wavelengths), dtype=jnp.float32)

    proxy_value_and_grad = jax.jit(
        jax.value_and_grad(
            lambda rf, xb: proxy_loss(
                rf,
                xb,
                basis=basis,
                photon_scale=args.photon_scale,
                smoothness_reg=args.smoothness_reg,
            )
        )
    )
    proxy_eval_loss = jax.jit(
        lambda rf, xb: proxy_loss(
            rf,
            xb,
            basis=basis,
            photon_scale=args.photon_scale,
            smoothness_reg=args.smoothness_reg,
        )
    )

    optimizer = optax.adam(args.filter_lr)
    opt_state = optimizer.init(raw_filter)

    best_proxy_loss = float("inf")
    best_raw_filter = raw_filter
    train_np = np.asarray(train, dtype=np.float32)
    rng = np.random.default_rng(args.seed)
    proxy_eval_subset = jnp.asarray(
        train_np[: min(max(args.batch_size * 4, args.batch_size), train_np.shape[0])],
        dtype=jnp.float32,
    )

    for step in range(PROXY_STEPS):
        idx = rng.integers(0, train_np.shape[0], size=(min(args.batch_size, train_np.shape[0]),))
        proxy_batch = jnp.asarray(train_np[idx], dtype=jnp.float32)
        proxy_train_loss, grads = proxy_value_and_grad(raw_filter, proxy_batch)
        updates, opt_state = optimizer.update(grads, opt_state, raw_filter)
        raw_filter = optax.apply_updates(raw_filter, updates)

        if step % PROXY_LOG_EVERY == 0 or step == PROXY_STEPS - 1:
            proxy_eval_loss_val = float(proxy_eval_loss(raw_filter, proxy_eval_subset))
            if proxy_eval_loss_val < best_proxy_loss:
                best_proxy_loss = proxy_eval_loss_val
                best_raw_filter = jax.tree_util.tree_map(lambda x: x, raw_filter)
            print(
                f"proxy_step={step:04d} proxy_loss={float(proxy_train_loss):.6f} "
                f"proxy_eval_loss={proxy_eval_loss_val:.6f}"
            )
    raw_filter = best_raw_filter
    recon_model = SpectralReconMLP(
        hidden_dims=hidden_dims,
        out_dim=n_wavelengths,
    )
    key, k_params, train_rng, eval_rng = jax.random.split(key, 4)
    recon_params = recon_model.init(
        {"params": k_params},
        jnp.zeros((1, num_filters), dtype=jnp.float32),
    )
    params = {"raw_filter": raw_filter, "recon": recon_params}

    def measure_noisy_decode(
        p: dict[str, jnp.ndarray],
        batch_spectra: jnp.ndarray,
        *,
        noise_key: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        a = filter_intensity_matrix(p["raw_filter"])
        y_clean = measure_batch_optical(batch_spectra, a, **measurement)
        y_noisy = add_measurement_noise(
            y_clean,
            key=noise_key,
            photon_scale=args.photon_scale,
        )
        x_hat = recon_model.apply(p["recon"], y_noisy)
        return x_hat, y_clean, y_noisy

    def recon_loss_and_metrics(
        p: dict[str, jnp.ndarray],
        batch_spectra: jnp.ndarray,
        *,
        noise_key: jax.Array,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        x_hat, _, _ = measure_noisy_decode(p, batch_spectra, noise_key=noise_key)
        recon_mse = jnp.mean((x_hat - batch_spectra) ** 2)
        smoothness = filter_smoothness(p["raw_filter"])
        loss = recon_mse + args.smoothness_reg * smoothness
        return loss, {
            "recon_mse": recon_mse,
            "smoothness": smoothness,
        }

    recon_value_and_grad = jax.jit(
        jax.value_and_grad(
            lambda p, xb, nk: recon_loss_and_metrics(p, xb, noise_key=nk)[0]
        )
    )
    recon_eval = jax.jit(
        lambda p, xb, nk: recon_loss_and_metrics(p, xb, noise_key=nk)
    )

    recon_optimizer = optax.multi_transform(
        transforms={
            "filter": optax.adam(args.filter_lr),
            "recon": optax.adam(args.recon_lr),
        },
        param_labels={"raw_filter": "filter", "recon": "recon"},
    )
    recon_opt_state = recon_optimizer.init(params)
    history_recon_train: list[float] = []
    history_recon_val: list[float] = []
    history_recon_val_steps: list[int] = []
    best_val = float("inf")
    best_params = params

    val_jnp = jnp.asarray(val, dtype=jnp.float32)
    for step in range(max(args.recon_steps, 1)):
        idx = rng.integers(0, train_np.shape[0], size=(min(args.batch_size, train_np.shape[0]),))
        batch = jnp.asarray(train_np[idx], dtype=jnp.float32)
        if args.recon_steps > 0:
            train_rng, noise_key = jax.random.split(train_rng)
            loss, grads = recon_value_and_grad(params, batch, noise_key)
            updates, recon_opt_state = recon_optimizer.update(grads, recon_opt_state, params)
            params = optax.apply_updates(params, updates)
        else:
            eval_rng, noise_key = jax.random.split(eval_rng)
            loss, _ = recon_eval(params, batch, noise_key)

        if step % RECON_LOG_EVERY == 0 or step == max(args.recon_steps, 1) - 1:
            eval_rng, val_noise_key = jax.random.split(eval_rng)
            val_loss, val_metrics = recon_eval(params, val_jnp, val_noise_key)
            val_loss_val = float(val_loss)
            history_recon_train.append(float(loss))
            history_recon_val.append(val_loss_val)
            history_recon_val_steps.append(step)
            if val_loss_val < best_val:
                best_val = val_loss_val
                best_params = jax.tree_util.tree_map(lambda x: x, params)
            print(
                f"recon_step={step:04d} train_loss={float(loss):.6f} "
                f"val_loss={val_loss_val:.6f} val_mse={float(val_metrics['recon_mse']):.6f}"
            )
        if args.recon_steps == 0:
            break

    params = best_params
    a_opt = np.asarray(filter_intensity_matrix(params["raw_filter"]), dtype=np.float32)
    filt_amp_opt = np.sqrt(np.clip(a_opt, 0.0, 1.0))
    key, k_train_noise, k_val_noise, k_final_noise = jax.random.split(key, 4)
    train_jnp = jnp.asarray(train, dtype=jnp.float32)
    val_jnp = jnp.asarray(val, dtype=jnp.float32)
    x_train_hat, _, _ = measure_noisy_decode(params, train_jnp, noise_key=k_train_noise)
    x_val_hat, _, _ = measure_noisy_decode(params, val_jnp, noise_key=k_val_noise)
    final_x_val_hat, _, _ = measure_noisy_decode(params, val_jnp, noise_key=k_final_noise)

    train_mse = float(jnp.mean((x_train_hat - train_jnp) ** 2))
    val_mse = float(jnp.mean((x_val_hat - val_jnp) ** 2))
    final_val_mse = float(jnp.mean((final_x_val_hat - val_jnp) ** 2))

    svals = np.linalg.svd(a_opt, compute_uv=False)
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "train_spectra_path": str(args.train_spectra),
        "val_spectra_path": str(args.val_spectra),
        "wavelengths_path": str(args.wavelengths),
        "seed": int(args.seed),
        "array_size": int(args.array_size),
        "proxy_objective": "d_opt",
        "proxy_steps": int(PROXY_STEPS),
        "filter_learning_rate": float(args.filter_lr),
        "recon_learning_rate": float(args.recon_lr),
        "recon_steps": int(args.recon_steps),
        "smoothness_reg": float(args.smoothness_reg),
        "pca_components": int(pca_components),
        "pca_cumulative_explained_variance_ratio": float(
            np.cumsum(np.asarray(pca.explained_variance_ratio_, dtype=np.float32))[-1]
        ),
        "mlp_hidden_dims": [int(v) for v in hidden_dims],
        "photon_scale": float(args.photon_scale),
        "proxy_best_loss": float(best_proxy_loss),
        "best_recon_val_loss": float(best_val),
        "train_mse": train_mse,
        "val_mse": val_mse,
        "final_val_mse": final_val_mse,
        "sensing_condition_number": float(svals[0] / max(svals[-1], 1e-12)),
    }
    summary_path = out_dir / "spectral_filter_optimization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved: {summary_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    wavelengths_np = np.asarray(wavelengths_nm)
    n_examples = min(4, val.shape[0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for i in range(num_filters):
        axes[0, 0].plot(wavelengths_np, filt_amp_opt[i], linewidth=1.5)
    axes[0, 0].set_title(
        "D-opt Optimized "
        f"{args.array_size}x{args.array_size} Filter Amplitudes "
        f"({num_filters} channels)"
    )
    axes[0, 0].set_xlabel("Wavelength (nm)")
    axes[0, 0].set_ylabel("Amplitude Transmission")
    axes[0, 0].grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(history_recon_val_steps, history_recon_train, label="Train loss")
    ax.plot(history_recon_val_steps, history_recon_val, label="Val loss")
    ax.set_title("Reconstruction Loss History")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_yscale("log")

    random_indices = jax.random.choice(key, val.shape[0], (n_examples,), replace=False)
    for i in random_indices:
        (line,) = axes[1, 0].plot(
            wavelengths_np,
            val[i],
            alpha=0.9,
            linewidth=0.6,
            label=f"example {i}",
        )
        axes[1, 0].plot(
            wavelengths_np,
            final_x_val_hat[i],
            alpha=0.9,
            linewidth=0.6,
            linestyle="--",
            color=line.get_color(),
        )
    axes[1, 0].set_title("Validation Spectra: target (solid) vs MLP recon (dashed)")
    axes[1, 0].set_xlabel("Wavelength (nm)")
    axes[1, 0].set_ylabel("Reflectance / Intensity")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.98,
        (
            f"Proxy objective: d_opt\n"
            f"PCA comps: {pca_components}\n"
            f"Proxy loss: {float(best_proxy_loss):.4f}\n"
            f"Decoder: Flax MLP (joint finetune)\n"
            f"Val MSE: {val_mse:.6f}\n"
            f"Final val MSE (new noise draw): {final_val_mse:.6f}\n"
            f"Cond #: {float(svals[0] / max(svals[-1], 1e-12)):.2e}"
        ),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    fig.tight_layout()
    plot_path = out_dir / "spectral_filter_optimization_overview.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"saved: {plot_path}")


if __name__ == "__main__":
    main()
