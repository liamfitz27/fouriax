"""Optimize spectral filters and reconstruction MLP for hyperspectral sensing."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn
from sklearn.decomposition import PCA

from fouriax.example_utils import (
    optimize_dataset_hybrid_module,
    optimize_dataset_optical_module,
)
from fouriax.optics import (
    AmplitudeMask,
    Field,
    Grid,
    IntensitySensor,
    OpticalModule,
    Spectrum,
)

ARTIFACTS_DIR = Path("artifacts")
PLOT_PATH = ARTIFACTS_DIR / "spectral_filter_overview.png"
SUMMARY_PATH = ARTIFACTS_DIR / "spectral_filter_summary.json"

SEED = 0
SPECTRAL_SLICE_START = 40
SPECTRAL_SLICE_STOP = 190
PCA_COMPONENTS = 32
PROXY_STEPS = 2000
TRAIN_SAMPLES = 200000
VAL_SAMPLES = 2048
ARRAY_SIZE = 3
BATCH_SIZE = 512
FILTER_LR = 1.4e-1
SMOOTHNESS_REG = 0.1
PHOTON_SCALE = 1000.0
RECON_STEPS = 2000
RECON_LR = 8e-4
HIDDEN_DIMS = (128, 256, 384, 384)


class SpectralReconMLP(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(int(hidden_dim))(x)
            x = nn.relu(x)
        return nn.Dense(self.out_dim)(x)


def load_indian_pines(
    train_path: Path,
    val_path: Path,
    wavelengths_path: Path,
    *,
    spectral_slice_start: int,
    spectral_slice_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(np.load(train_path), dtype=np.float32)[
        :, spectral_slice_start:spectral_slice_stop
    ]
    val = np.asarray(np.load(val_path), dtype=np.float32)[
        :, spectral_slice_start:spectral_slice_stop
    ]
    wavelengths_nm = np.asarray(np.load(wavelengths_path), dtype=np.float32)[
        spectral_slice_start:spectral_slice_stop
    ]
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


def filter_smoothness(raw_filter: jnp.ndarray) -> jnp.ndarray:
    a = filter_intensity_matrix(raw_filter)
    return jnp.mean((a[:, 1:] - a[:, :-1]) ** 2)


def add_measurement_noise(y: jnp.ndarray, *, key: jax.Array, photon_scale: float) -> jnp.ndarray:
    if photon_scale <= 0.0:
        return y
    shot_std = jnp.sqrt(jnp.maximum(y, 0.0) / jnp.asarray(photon_scale, dtype=y.dtype))
    shot = shot_std * jax.random.normal(key, shape=y.shape, dtype=y.dtype)
    return y + shot


def build_detector_masks(array_size: int) -> jnp.ndarray:
    num_filters = int(array_size * array_size)
    masks = np.zeros((num_filters, array_size, array_size), dtype=np.float32)
    idx = np.arange(num_filters)
    masks[idx, idx // array_size, idx % array_size] = 1.0
    return jnp.asarray(masks, dtype=jnp.float32)


def make_optical_measurement(*, wavelengths_nm: np.ndarray, num_filters: int) -> dict[str, object]:
    array_size = int(round(np.sqrt(num_filters)))
    if array_size * array_size != num_filters:
        raise ValueError(f"num_filters must be a perfect square, got {num_filters}")
    grid = Grid.from_extent(nx=array_size, ny=array_size, dx_um=1.0, dy_um=1.0)
    spectrum = Spectrum.from_array(jnp.asarray(wavelengths_nm * 1e-3, dtype=jnp.float32))
    sensor = IntensitySensor(
        sum_wavelengths=True,
        detector_masks=build_detector_masks(array_size),
    )
    return {"array_size": array_size, "grid": grid, "spectrum": spectrum, "sensor": sensor}


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
        sample_amp = jnp.sqrt(jnp.maximum(sample_spectrum, 0.0)).reshape(
            (spectrum.size, 1, 1)
        )
        field_data = (
            jnp.ones((spectrum.size, array_size, array_size), dtype=jnp.complex64)
            * sample_amp
        )
        field = Field(data=field_data, grid=grid, spectrum=spectrum, domain="spatial")
        return module.measure(field).astype(jnp.float32)

    return jax.vmap(measure_one)(batch_spectra)


def proxy_loss(
    raw_filter: jnp.ndarray,
    batch_spectra: jnp.ndarray | tuple[np.ndarray] | tuple[jnp.ndarray],
    *,
    basis: jnp.ndarray,
    photon_scale: float,
    smoothness_reg: float,
) -> jnp.ndarray:
    if isinstance(batch_spectra, tuple):
        (batch_spectra,) = batch_spectra
    batch_spectra = jnp.asarray(batch_spectra, dtype=jnp.float32)
    a = filter_intensity_matrix(raw_filter)
    ab = a @ basis  # (K, r)
    y = batch_spectra @ a.T
    noise_var = jnp.maximum(jnp.maximum(y, 0.0) / jnp.asarray(photon_scale, dtype=y.dtype), 1e-6)
    inv_noise_var = 1.0 / noise_var
    fisher = jnp.einsum("kr,bk,ks->brs", ab, inv_noise_var, ab)
    gram = jnp.eye(ab.shape[1], dtype=ab.dtype)[None, :, :] + fisher
    sign, logdet = jax.vmap(jnp.linalg.slogdet)(gram)
    d_opt = jnp.mean(jnp.where(sign > 0, logdet, -1e12))
    return -d_opt + smoothness_reg * filter_smoothness(raw_filter)


def main() -> None:
    train_path = Path("data/meta_atoms/indian_pines/train_spectra_full.npy")
    val_path = Path("data/meta_atoms/indian_pines/val_spectra.npy")
    wavelengths_path = Path("data/meta_atoms/indian_pines/wavelengths.npy")

    for p in (train_path, val_path, wavelengths_path):
        if not p.exists():
            raise FileNotFoundError(f"missing file: {p}")

    train_full, val_full, wavelengths_nm = load_indian_pines(
        train_path,
        val_path,
        wavelengths_path,
        spectral_slice_start=SPECTRAL_SLICE_START,
        spectral_slice_stop=SPECTRAL_SLICE_STOP,
    )
    train = train_full[: min(TRAIN_SAMPLES, train_full.shape[0])]
    val = val_full[: min(VAL_SAMPLES, val_full.shape[0])]
    train_np = np.asarray(train, dtype=np.float32)
    val_jnp = jnp.asarray(val, dtype=jnp.float32)

    n_wavelengths = int(train.shape[1])
    num_filters = int(ARRAY_SIZE * ARRAY_SIZE)
    pca_components = min(PCA_COMPONENTS, n_wavelengths, train.shape[0])
    pca, basis_np = build_pca_basis(train, pca_components, SEED)
    basis = jnp.asarray(basis_np, dtype=jnp.float32)
    measurement = make_optical_measurement(
        wavelengths_nm=wavelengths_nm,
        num_filters=num_filters,
    )

    key = jax.random.PRNGKey(SEED)
    raw_filter = jax.random.normal(key, (num_filters, n_wavelengths), dtype=jnp.float32)
    proxy_optimizer = optax.adam(FILTER_LR)
    proxy_eval_subset = jnp.asarray(
        train_np[: min(max(BATCH_SIZE * 4, BATCH_SIZE), train_np.shape[0])], dtype=jnp.float32
    )

    def build_filter_module(raw_filter_params: jnp.ndarray) -> OpticalModule:
        a = filter_intensity_matrix(raw_filter_params)
        amp_map = jnp.sqrt(jnp.clip(a.T, 0.0, 1.0)).reshape(
            (measurement["spectrum"].size, measurement["array_size"], measurement["array_size"])
        )
        return OpticalModule(
            layers=(AmplitudeMask(amplitude_map=amp_map),),
            sensor=measurement["sensor"],
        )

    proxy_steps_per_epoch = max(1, int(np.ceil(train_np.shape[0] / BATCH_SIZE)))
    proxy_epochs = max(1, int(np.ceil(PROXY_STEPS / proxy_steps_per_epoch)))

    proxy_result = optimize_dataset_optical_module(
        init_params=raw_filter,
        build_module=build_filter_module,
        loss_fn=partial(
            proxy_loss,
            basis=basis,
            photon_scale=PHOTON_SCALE,
            smoothness_reg=SMOOTHNESS_REG,
        ),
        optimizer=proxy_optimizer,
        train_data=train_np,
        batch_size=BATCH_SIZE,
        epochs=proxy_epochs,
        val_data=proxy_eval_subset,
        seed=SEED,
    )

    raw_filter = proxy_result.params_result.best_params
    best_proxy_loss = proxy_result.params_result.best_metric_value
    recon_model = SpectralReconMLP(hidden_dims=HIDDEN_DIMS, out_dim=n_wavelengths)
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
    ) -> jnp.ndarray:
        a = filter_intensity_matrix(p["raw_filter"])
        y_clean = measure_batch_optical(batch_spectra, a, **measurement)
        y_noisy = add_measurement_noise(y_clean, key=noise_key, photon_scale=PHOTON_SCALE)
        return recon_model.apply(p["recon"], y_noisy)

    train_noise_keys = np.asarray(jax.random.split(train_rng, train_np.shape[0]))
    val_noise_keys = np.asarray(jax.random.split(eval_rng, val.shape[0]))

    def hybrid_loss_fn(
        optical_params: jnp.ndarray,
        decoder_params: dict[str, jnp.ndarray],
        batch: tuple[np.ndarray, np.ndarray] | tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        batch_spectra_raw, batch_noise_keys_raw = batch
        batch_spectra = jnp.asarray(batch_spectra_raw, dtype=jnp.float32)
        batch_noise_keys = jnp.asarray(batch_noise_keys_raw, dtype=jnp.uint32)
        noise_key = batch_noise_keys[0]
        p = {"raw_filter": optical_params, "recon": decoder_params}
        x_hat = measure_noisy_decode(p, batch_spectra, noise_key=noise_key)
        recon_mse = jnp.mean((x_hat - batch_spectra) ** 2)
        return recon_mse + SMOOTHNESS_REG * filter_smoothness(p["raw_filter"])

    steps_per_epoch = max(1, int(np.ceil(train_np.shape[0] / BATCH_SIZE)))
    recon_epochs = max(1, int(np.ceil(RECON_STEPS / steps_per_epoch)))

    recon_result = optimize_dataset_hybrid_module(
        init_optical_params=raw_filter,
        init_decoder_params=recon_params,
        build_module=build_filter_module,
        loss_fn=hybrid_loss_fn,
        train_data=(train_np, train_noise_keys),
        batch_size=BATCH_SIZE,
        epochs=recon_epochs,
        val_data=(val, val_noise_keys),
        optical_optimizer=optax.adam(FILTER_LR),
        decoder_optimizer=optax.adam(RECON_LR),
        seed=SEED,
    )

    history_steps = [record.step for record in recon_result.params_result.val_history]
    history_val = [record.metrics["val_loss"] for record in recon_result.params_result.val_history]
    history_train = [recon_result.params_result.train_loss_history[step] for step in history_steps]
    best_val = recon_result.params_result.best_metric_value
    params = {
        "raw_filter": recon_result.best_optical_params,
        "recon": recon_result.best_decoder_params,
    }
    a_opt = np.asarray(filter_intensity_matrix(params["raw_filter"]), dtype=np.float32)
    filt_amp_opt = np.sqrt(np.clip(a_opt, 0.0, 1.0))
    key, k_train_noise, k_val_noise, k_final_noise = jax.random.split(key, 4)
    train_jnp = jnp.asarray(train, dtype=jnp.float32)
    x_train_hat = measure_noisy_decode(params, train_jnp, noise_key=k_train_noise)
    x_val_hat = measure_noisy_decode(params, val_jnp, noise_key=k_val_noise)
    final_x_val_hat = measure_noisy_decode(params, val_jnp, noise_key=k_final_noise)

    train_mse = float(jnp.mean((x_train_hat - train_jnp) ** 2))
    val_mse = float(jnp.mean((x_val_hat - val_jnp) ** 2))
    final_val_mse = float(jnp.mean((final_x_val_hat - val_jnp) ** 2))

    print("=== Spectral Filter Example ===")
    print(f"array_size={ARRAY_SIZE}x{ARRAY_SIZE}, channels={num_filters}")
    print(f"pca_components={pca_components}")
    print(
        "pca_cumulative_explained_variance="
        f"{float(np.cumsum(np.asarray(pca.explained_variance_ratio_, dtype=np.float32))[-1]):.6f}"
    )
    print(f"proxy_best_loss={best_proxy_loss:.6f}")
    print(f"best_recon_val_loss={best_val:.6f}")
    print(f"train_mse={train_mse:.6f}")
    print(f"val_mse={val_mse:.6f}")
    print(f"final_val_mse={final_val_mse:.6f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    svals = np.linalg.svd(a_opt, compute_uv=False)
    summary = {
        "seed": SEED,
        "array_size": ARRAY_SIZE,
        "num_filters": num_filters,
        "pca_components": pca_components,
        "proxy_steps": PROXY_STEPS,
        "recon_steps": RECON_STEPS,
        "filter_lr": FILTER_LR,
        "recon_lr": RECON_LR,
        "photon_scale": PHOTON_SCALE,
        "smoothness_reg": SMOOTHNESS_REG,
        "best_proxy_loss": best_proxy_loss,
        "best_recon_val_loss": best_val,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "final_val_mse": final_val_mse,
        "condition_number": float(svals[0] / max(svals[-1], 1e-12)),
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved: {SUMMARY_PATH}")

    wavelengths_np = np.asarray(wavelengths_nm)
    n_examples = min(4, val.shape[0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for i in range(num_filters):
        axes[0, 0].plot(wavelengths_np, filt_amp_opt[i], linewidth=1.5)
    axes[0, 0].set_title(
        f"Optimized {ARRAY_SIZE}x{ARRAY_SIZE} Filter Amplitudes "
        f"({num_filters} channels)"
    )
    axes[0, 0].set_xlabel("Wavelength (nm)")
    axes[0, 0].set_ylabel("Amplitude Transmission")
    axes[0, 0].grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(history_steps, history_train, label="Train loss")
    ax.plot(history_steps, history_val, label="Val loss")
    ax.set_title("Reconstruction Loss History")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_yscale("log")

    random_indices = jax.random.choice(key, val.shape[0], (n_examples,), replace=False)
    for i in random_indices:
        idx = int(i)
        (line,) = axes[1, 0].plot(
            wavelengths_np,
            val[idx],
            alpha=0.9,
            linewidth=0.6,
            label=f"example {idx}",
        )
        axes[1, 0].plot(
            wavelengths_np,
            np.asarray(final_x_val_hat[idx]),
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
            f"Proxy loss: {best_proxy_loss:.4f}\n"
            f"Val MSE: {val_mse:.6f}\n"
            f"Final val MSE: {final_val_mse:.6f}\n"
            f"Cond #: {float(svals[0] / max(svals[-1], 1e-12)):.2e}"
        ),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
