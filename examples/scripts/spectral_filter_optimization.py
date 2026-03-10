"""Optimize spectral filters and reconstruction MLP for hyperspectral sensing."""

# %% Imports
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn
from scipy.io import loadmat
from sklearn.decomposition import PCA

import fouriax as fx

# %% Paths and Parameters

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spectral Filter Optimization Example")
    parser.add_argument(
        "--data-path", type=str, default="data/meta_atoms/indian_pines/Indian_pines.mat"
    )
    parser.add_argument(
        "--wavelengths-path", type=str, default="data/meta_atoms/indian_pines/wavelengths.npy"
    )
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spectral-slice-start", type=int, default=40)
    parser.add_argument("--spectral-slice-stop", type=int, default=190)
    parser.add_argument("--array-size", type=int, default=3)
    parser.add_argument("--photon-scale", type=float, default=200.0)

    # Proxy Config
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--proxy-batch-size", type=int, default=512)
    parser.add_argument("--proxy-epochs", type=int, default=100)
    parser.add_argument("--proxy-lr", type=float, default=5e-3)

    # Full Recon Config
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--recon-batch-size", type=int, default=512)
    parser.add_argument("--recon-epochs", type=int, default=100)
    parser.add_argument("--recon-decoder-lr", type=float, default=8e-4)
    parser.add_argument("--recon-filter-lr", type=float, default=8e-2)
    parser.add_argument("--smoothness-reg", type=float, default=0.02)
    parser.add_argument("--unity-reg", type=float, default=1e-5)
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    return parser.parse_args()


ARGS = parse_args()

DATA_PATH = Path(ARGS.data_path)
WAVELENGTHS_PATH = Path(ARGS.wavelengths_path)
ARTIFACTS_DIR = Path(ARGS.artifacts_dir)
PLOT_PATH = ARTIFACTS_DIR / "spectral_filter_overview.png"
SUMMARY_PATH = ARTIFACTS_DIR / "spectral_filter_summary.json"

SEED = ARGS.seed
SPECTRAL_SLICE_START = ARGS.spectral_slice_start
SPECTRAL_SLICE_STOP = ARGS.spectral_slice_stop
ARRAY_SIZE = ARGS.array_size
PHOTON_SCALE = ARGS.photon_scale

PCA_COMPONENTS = ARGS.pca_components
PROXY_BATCH_SIZE = ARGS.proxy_batch_size
PROXY_EPOCHS = ARGS.proxy_epochs
PROXY_LR = ARGS.proxy_lr

VAL_FRACTION = ARGS.val_fraction
RECON_BATCH_SIZE = ARGS.recon_batch_size
RECON_EPOCHS = ARGS.recon_epochs
RECON_DECODER_LR = ARGS.recon_decoder_lr
RECON_FILTER_LR = ARGS.recon_filter_lr
SMOOTHNESS_REG = ARGS.smoothness_reg
UNITY_REG = ARGS.unity_reg
HIDDEN_DIMS = (128, 256, 384, 384)
PLOT = not ARGS.no_plot


# %% MLP Decoder Model
class SpectralReconMLP(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(int(hidden_dim))(x)
            x = nn.relu(x)
        return nn.Dense(self.out_dim)(x)


# %% Helper Functions
def load_indian_pines(
    DATA_PATH: Path,
    WAVELENGTHS_PATH: Path,
    *,
    spectral_slice_start: int,
    spectral_slice_stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    data = loadmat(DATA_PATH)
    if "indian_pines" not in data:
        raise KeyError("expected key 'indian_pines' in Indian_pines.mat")
    cube = np.asarray(data["indian_pines"], dtype=np.float32)
    spectra = cube.reshape(-1, cube.shape[-1])[:, spectral_slice_start:spectral_slice_stop]
    wavelengths_nm = np.asarray(np.load(WAVELENGTHS_PATH), dtype=np.float32)[
        spectral_slice_start:spectral_slice_stop
    ]
    order = np.argsort(wavelengths_nm)
    wavelengths_nm = wavelengths_nm[order]
    spectra = spectra[:, order]
    spectra_norm = spectra / np.maximum(np.max(spectra, axis=1, keepdims=True), 1e-12)
    return spectra_norm, wavelengths_nm


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


def main() -> None:
    # %% Setup
    spectra_full, wavelengths_nm = load_indian_pines(
        DATA_PATH,
        WAVELENGTHS_PATH,
        spectral_slice_start=SPECTRAL_SLICE_START,
        spectral_slice_stop=SPECTRAL_SLICE_STOP,
    )
    spectra_full = spectra_full[np.sum(spectra_full, axis=1) > 0.0]
    split_rng = np.random.default_rng(SEED)
    (train_full,), (val_full,) = fx.optim.train_val_split(
        spectra_full,
        val_fraction=VAL_FRACTION,
        rng=split_rng,
    )
    train = jnp.asarray(train_full, dtype=jnp.float32)
    val = jnp.asarray(val_full, dtype=jnp.float32)

    n_wavelengths = int(train.shape[1])
    num_filters = int(ARRAY_SIZE**2)
    array_size = int(round(np.sqrt(num_filters)))
    grid = fx.Grid.from_extent(nx=array_size, ny=array_size, dx_um=1.0, dy_um=1.0)
    spectrum = fx.Spectrum.from_array(jnp.asarray(wavelengths_nm * 1e-3, dtype=jnp.float32))
    noise_model = fx.PoissonNoise(count_scale=PHOTON_SCALE)

    pca_components = min(PCA_COMPONENTS, n_wavelengths, train.shape[0])
    pca, basis_np = build_pca_basis(train, pca_components, SEED)
    basis = jnp.asarray(basis_np, dtype=jnp.float32)
    signal_prior_cov = jnp.asarray(
        np.asarray(pca.explained_variance_, dtype=np.float32),
        dtype=jnp.float32,
    )

    key = jax.random.PRNGKey(SEED)
    raw_filter = jax.random.normal(key, (num_filters, n_wavelengths), dtype=jnp.float32)
    proxy_optimizer = optax.adam(PROXY_LR)
    proxy_eval_subset = jnp.asarray(
        train[
            : min(
                max(PROXY_BATCH_SIZE * 4, PROXY_BATCH_SIZE),
                train.shape[0],
            )
        ],
        dtype=jnp.float32,
    )

    def build_filter_module(raw_filter_params: jnp.ndarray) -> fx.OpticalModule:
        a = filter_intensity_matrix(raw_filter_params)
        amp_map = jnp.sqrt(jnp.clip(a.T, 0.0, 1.0)).reshape((spectrum.size, array_size, array_size))
        return fx.OpticalModule(
            layers=(),
            sensor=fx.DetectorArray(
                detector_grid=grid,
                filter_mask=fx.AmplitudeMask(amplitude_map=amp_map),
                noise_model=noise_model,
            ),
        )

    def build_sample_field(sample_spectra: np.ndarray | jnp.ndarray) -> fx.Field:
        sample_spectra = jnp.asarray(sample_spectra, dtype=jnp.float32)
        if sample_spectra.ndim == 1:
            sample_spectra = sample_spectra[None, :]
        sample_amp = jnp.sqrt(jnp.maximum(sample_spectra, 0.0)).reshape(
            (sample_spectra.shape[0], spectrum.size, 1, 1)
        )
        field_data = (
            jnp.ones(
                (sample_spectra.shape[0], spectrum.size, array_size, array_size),
                dtype=jnp.complex64,
            )
            * sample_amp
        )
        return fx.Field(data=field_data, grid=grid, spectrum=spectrum, domain="spatial")

    def measure_batch(
        raw_filter: jnp.ndarray,
        batch_spectra: np.ndarray | jnp.ndarray,
        *,
        noise_keys: np.ndarray | jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        module = build_filter_module(raw_filter)
        field = build_sample_field(batch_spectra)
        sensor = module.sensor
        if not isinstance(sensor, fx.DetectorArray):
            raise ValueError("filter module must use fx.DetectorArray for batched measurement")

        image_clean = sensor.expected(field).astype(jnp.float32)
        if noise_keys is None:
            image = image_clean
        else:
            noise_keys = jnp.asarray(noise_keys, dtype=jnp.uint32)
            if noise_keys.ndim == 1:
                noise_keys = noise_keys[None, :]
            image = jax.vmap(
                lambda clean_sample, sample_key: noise_model.sample(clean_sample, key=sample_key)
            )(image_clean, noise_keys)
        return image.reshape((image.shape[0], -1))

    # %% Proxy Loss and Optimization
    def batch_proxy_loss(
        raw_filter: jnp.ndarray,
        batch_spectra: tuple[np.ndarray | jnp.ndarray] | np.ndarray | jnp.ndarray,
    ) -> jnp.ndarray:
        if isinstance(batch_spectra, tuple):
            if len(batch_spectra) != 1:
                raise ValueError(
                    "proxy optimization expects a single spectra array per minibatch"
                )
            batch_spectra = batch_spectra[0]
        a = filter_intensity_matrix(raw_filter)
        ab = a @ basis
        batch_spectra = jnp.asarray(batch_spectra, dtype=jnp.float32)
        batch_coeffs = batch_spectra @ basis
        d_opt = jnp.mean(
            jax.vmap(
                lambda coeffs: fx.analysis.d_optimality(
                    fx.analysis.fisher_information(
                        lambda c: ab @ c,
                        coeffs,
                        noise_model=noise_model,
                    ),
                    prior_covariance=signal_prior_cov,
                )
            )(batch_coeffs)
        )
        regularization = SMOOTHNESS_REG * filter_smoothness(raw_filter) + UNITY_REG * jnp.mean(
            (filter_intensity_matrix(raw_filter) - 1.0) ** 2
        )
        return -d_opt + regularization

    proxy_result = fx.optim.optimize_dataset_optical_module(
        init_params=raw_filter,
        build_module=build_filter_module,
        batch_loss_fn=batch_proxy_loss,
        optimizer=proxy_optimizer,
        train_data=train,
        batch_size=PROXY_BATCH_SIZE,
        epochs=PROXY_EPOCHS,
        val_data=proxy_eval_subset,
        seed=SEED,
    )

    # %% Hybrid Model Loss and Optimization
    raw_filter = proxy_result.params_result.best_params
    best_proxy_loss = proxy_result.params_result.best_metric_value
    recon_model = SpectralReconMLP(hidden_dims=HIDDEN_DIMS, out_dim=n_wavelengths)
    key, k_params, train_rng, eval_rng = jax.random.split(key, 4)
    recon_params = recon_model.init(
        {"params": k_params},
        jnp.zeros((1, num_filters), dtype=jnp.float32),
    )
    params = {"raw_filter": raw_filter, "recon": recon_params}

    def measure_decode_batch(
        raw_filter: jnp.ndarray,
        decoder_params: dict[str, jnp.ndarray],
        batch_spectra: np.ndarray | jnp.ndarray,
        batch_noise_keys: np.ndarray | jnp.ndarray,
    ) -> jnp.ndarray:
        y_noisy = measure_batch(raw_filter, batch_spectra, noise_keys=batch_noise_keys)
        return recon_model.apply(decoder_params, y_noisy)

    train_noise_keys = np.asarray(jax.random.split(train_rng, train.shape[0]))
    val_noise_keys = np.asarray(jax.random.split(eval_rng, val.shape[0]))

    def hybrid_batch_loss(
        optical_params: jnp.ndarray,
        decoder_params: dict[str, jnp.ndarray],
        batch: tuple[np.ndarray, np.ndarray] | tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        batch_spectra_raw, batch_noise_keys_raw = batch
        batch_spectra = jnp.asarray(batch_spectra_raw, dtype=jnp.float32)
        batch_noise_keys = jnp.asarray(batch_noise_keys_raw, dtype=jnp.uint32)
        x_hat = measure_decode_batch(
            optical_params,
            decoder_params,
            batch_spectra,
            batch_noise_keys,
        )
        recon_mse = jnp.mean((x_hat - batch_spectra) ** 2)
        regularization = SMOOTHNESS_REG * filter_smoothness(optical_params) + UNITY_REG * jnp.mean(
            (filter_intensity_matrix(optical_params) - 1.0) ** 2
        )
        return recon_mse + regularization

    recon_result = fx.optim.optimize_dataset_hybrid_module(
        init_optical_params=raw_filter,
        init_decoder_params=recon_params,
        build_module=build_filter_module,
        batch_loss_fn=hybrid_batch_loss,
        train_data=(train, train_noise_keys),
        batch_size=RECON_BATCH_SIZE,
        epochs=RECON_EPOCHS,
        val_data=(val, val_noise_keys),
        optical_optimizer=optax.adam(RECON_FILTER_LR),
        decoder_optimizer=optax.adam(RECON_DECODER_LR),
        seed=SEED,
    )

    # %% Evaluation
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
    train_noise_keys_eval = jax.random.split(k_train_noise, train.shape[0])
    val_noise_keys_eval = jax.random.split(k_val_noise, val.shape[0])
    final_noise_keys_eval = jax.random.split(k_final_noise, val.shape[0])
    x_train_hat = measure_decode_batch(
        params["raw_filter"],
        params["recon"],
        train,
        train_noise_keys_eval,
    )
    x_val_hat = measure_decode_batch(
        params["raw_filter"],
        params["recon"],
        val,
        val_noise_keys_eval,
    )
    final_x_val_hat = measure_decode_batch(
        params["raw_filter"],
        params["recon"],
        val,
        final_noise_keys_eval,
    )

    train_mse = float(jnp.mean((x_train_hat - train) ** 2))
    val_mse = float(jnp.mean((x_val_hat - val) ** 2))
    final_val_mse = float(jnp.mean((final_x_val_hat - val) ** 2))

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
        "proxy_batch_size": PROXY_BATCH_SIZE,
        "proxy_epochs": PROXY_EPOCHS,
        "proxy_lr": PROXY_LR,
        "recon_batch_size": RECON_BATCH_SIZE,
        "recon_epochs": RECON_EPOCHS,
        "recon_filter_lr": RECON_FILTER_LR,
        "recon_decoder_lr": RECON_DECODER_LR,
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

    # %% Plot Results
    if PLOT:
        wavelengths_np = np.asarray(wavelengths_nm)
        n_examples = min(4, val.shape[0])
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        for i in range(num_filters):
            axes[0, 0].plot(wavelengths_np, filt_amp_opt[i], linewidth=1.5)
        axes[0, 0].set_title(
            f"Optimized {ARRAY_SIZE}x{ARRAY_SIZE} Filter Amplitudes ({num_filters} channels)"
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
