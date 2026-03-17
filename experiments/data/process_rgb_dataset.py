"""Preprocess registered RGB datasets into sharded NPZ datasets."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
from functools import partial
from math import ceil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

try:
    from .dataset_registry import DATASET_NAMES, BackgroundMode, get_dataset_spec
except ImportError:  # pragma: no cover - direct script execution fallback
    from dataset_registry import DATASET_NAMES, BackgroundMode, get_dataset_spec


def processed_shard_dir(data_root: Path, *, size: int, dataset_name: str) -> Path:
    spec = get_dataset_spec(dataset_name)
    return spec.processed_shard_dir(Path(data_root), size=size)


def clear_processed_shards(data_root: Path, *, size: int, dataset_name: str) -> None:
    shard_dir = processed_shard_dir(data_root, size=size, dataset_name=dataset_name)
    if not shard_dir.exists():
        return
    for path in shard_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def has_processed_shards(
    data_root: Path,
    *,
    size: int,
    shard_size: int,
    dataset_name: str,
) -> bool:
    spec = get_dataset_spec(dataset_name)
    shard_dir = spec.processed_shard_dir(Path(data_root), size=size)
    manifest_path = spec.manifest_path(Path(data_root), size=size)
    if not manifest_path.exists():
        return False

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False

    if manifest.get("dataset_name") != spec.name:
        return False
    if manifest.get("size") != size or manifest.get("shard_size") != shard_size:
        return False

    for subset_name, expected_count in spec.expected_counts.items():
        count_key = f"{subset_name}_count"
        if manifest.get(count_key) != expected_count:
            return False

        shard_records = manifest.get(f"{subset_name}_shards", [])
        if len(shard_records) != ceil(expected_count / shard_size):
            return False

        processed_count = 0
        for record in shard_records:
            filename = record.get("filename")
            if not filename or not (shard_dir / filename).exists():
                return False
            processed_count += int(record.get("num_samples", 0))
        if processed_count != expected_count:
            return False

    return True


def scan_images(root: Path, valid_extensions: tuple[str, ...]) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in valid_extensions
    )


def replace_white_background(
    image: np.ndarray,
    *,
    white_threshold: int,
    replacement_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    gray = (
        0.299 * image[..., 0].astype(np.float32)
        + 0.587 * image[..., 1].astype(np.float32)
        + 0.114 * image[..., 2].astype(np.float32)
    )
    white_mask = gray >= float(white_threshold)
    labels, num_labels = ndimage.label(white_mask)
    if num_labels == 0:
        return image

    border_labels = set(np.unique(labels[0, :]))
    border_labels.update(np.unique(labels[-1, :]))
    border_labels.update(np.unique(labels[:, 0]))
    border_labels.update(np.unique(labels[:, -1]))
    border_labels.discard(0)
    if not border_labels:
        return image

    background_mask = np.isin(labels, np.fromiter(border_labels, dtype=labels.dtype))
    result = image.copy()
    result[background_mask] = replacement_color
    return result


def preprocess_image(
    image: np.ndarray,
    *,
    background_mode: BackgroundMode,
    white_threshold: int,
) -> np.ndarray:
    if background_mode == "none":
        return image
    if background_mode == "replace_white_border":
        return replace_white_background(image, white_threshold=white_threshold)
    raise ValueError(f"unsupported background_mode={background_mode!r}")


def process_single_image(
    image_path: Path,
    *,
    img_size: tuple[int, int],
    background_mode: BackgroundMode,
    white_threshold: int,
) -> np.ndarray | None:
    try:
        with Image.open(image_path) as img:
            image = np.asarray(img.convert("RGB"))
        image = preprocess_image(
            image,
            background_mode=background_mode,
            white_threshold=white_threshold,
        )
        resized = Image.fromarray(image).resize(img_size, Image.Resampling.LANCZOS)
        return np.asarray(resized, dtype=np.float32) / 255.0
    except Exception as exc:
        print(f"warning: failed to process {image_path}: {exc}")
        return None


def save_shards(
    image_paths: list[Path],
    *,
    dataset_name: str,
    subset_name: str,
    output_dir: Path,
    image_size: int,
    shard_size: int,
    white_threshold: int,
    background_mode: BackgroundMode,
    num_workers: int,
    force: bool,
) -> list[dict[str, object]]:
    if not image_paths:
        print(f"no images to process for subset={subset_name}")
        return []

    spec = get_dataset_spec(dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_files = len(image_paths)
    num_shards = (total_files + shard_size - 1) // shard_size
    num_cpus = num_workers if num_workers > 0 else (mp.cpu_count() or 1)
    process_func = partial(
        process_single_image,
        img_size=(image_size, image_size),
        background_mode=background_mode,
        white_threshold=white_threshold,
    )

    print(
        f"processing dataset={dataset_name} subset={subset_name} images={total_files} "
        f"shards={num_shards} shard_size={shard_size} workers={num_cpus}"
    )
    shard_records: list[dict[str, object]] = []
    for shard_idx in range(num_shards):
        lo = shard_idx * shard_size
        hi = min((shard_idx + 1) * shard_size, total_files)
        shard_files = image_paths[lo:hi]
        shard_name = (
            f"{spec.processed_prefix}_{subset_name}_nx{image_size}_nwvl3_ns{len(shard_files)}"
            f"_shard{shard_idx:03d}.npz"
        )
        save_path = output_dir / shard_name
        if save_path.exists() and not force:
            print(f"skipping existing shard: {save_path}")
            shard_records.append(
                {
                    "subset": subset_name,
                    "shard_index": shard_idx,
                    "filename": save_path.name,
                    "num_samples": len(shard_files),
                    "skipped": True,
                }
            )
            continue

        print(f"  shard {shard_idx + 1}/{num_shards}: files {lo}:{hi}")
        with mp.Pool(processes=num_cpus) as pool:
            results = pool.map(process_func, shard_files)
        valid_results = [result for result in results if result is not None]
        if not valid_results:
            continue

        images = np.stack(valid_results, axis=0).astype(np.float32)
        metadata = {
            "dataset_name": dataset_name,
            "subset": subset_name,
            "image_shape": [image_size, image_size, 3],
            "num_samples_total": total_files,
            "shard_samples": int(images.shape[0]),
            "shard_index": shard_idx,
            "num_shards": num_shards,
            "img_size": [image_size, image_size],
            "background_mode": background_mode,
            "white_threshold": white_threshold,
        }
        np.savez_compressed(save_path, images=images, params_json=np.asarray(json.dumps(metadata)))
        shard_records.append(
            {
                "subset": subset_name,
                "shard_index": shard_idx,
                "filename": save_path.name,
                "num_samples": int(images.shape[0]),
                "skipped": False,
            }
        )
    return shard_records


def _build_downloader(dataset_name: str, data_root: Path, seed: int):
    if dataset_name == "cartoon_set":
        try:
            from .cartoonset_downloader import CartoonSetDownloader
        except ImportError:  # pragma: no cover - direct script execution fallback
            from cartoonset_downloader import CartoonSetDownloader

        return CartoonSetDownloader(Path(data_root), seed=seed)

    if dataset_name == "div2k":
        try:
            from .div2k_downloader import DIV2KDownloader
        except ImportError:  # pragma: no cover - direct script execution fallback
            from div2k_downloader import DIV2KDownloader

        return DIV2KDownloader(Path(data_root))

    raise ValueError(f"unsupported dataset_name={dataset_name!r}")


def ensure_raw_rgb_dataset(
    data_root: Path,
    *,
    dataset_name: str,
    seed: int,
    download_if_missing: bool,
) -> tuple[Path, Path]:
    spec = get_dataset_spec(dataset_name)
    dataset_root = spec.dataset_root(Path(data_root))
    train_dir = spec.subset_dir(dataset_root, "train")
    valid_dir = spec.subset_dir(dataset_root, "valid")
    downloader = _build_downloader(dataset_name, Path(data_root), seed)

    if downloader.dataset_ready():
        return train_dir, valid_dir
    if not download_if_missing:
        raise FileNotFoundError(f"missing dataset train/valid under {dataset_root}")
    if not downloader.download():
        raise RuntimeError(f"failed to download and prepare dataset {dataset_name}")
    return train_dir, valid_dir


def process_rgb_image_dataset(
    *,
    data_root: Path,
    dataset_name: str,
    size: int,
    shard_size: int = 5000,
    white_threshold: int | None = None,
    num_workers: int = 0,
    force: bool = False,
    seed: int = 0,
    download_if_missing: bool = True,
) -> Path:
    spec = get_dataset_spec(dataset_name)
    data_root = Path(data_root)
    output_dir = spec.processed_shard_dir(data_root, size=size)
    resolved_white_threshold = spec.white_threshold if white_threshold is None else white_threshold
    if force or not has_processed_shards(
        data_root,
        size=size,
        shard_size=shard_size,
        dataset_name=dataset_name,
    ):
        clear_processed_shards(data_root, size=size, dataset_name=dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir, valid_dir = ensure_raw_rgb_dataset(
        data_root,
        dataset_name=dataset_name,
        seed=seed,
        download_if_missing=download_if_missing,
    )
    train_paths = scan_images(train_dir, spec.valid_extensions)
    valid_paths = scan_images(valid_dir, spec.valid_extensions)
    print(f"dataset={dataset_name} train_images={len(train_paths)} valid_images={len(valid_paths)}")

    expected_train = spec.expected_counts["train"]
    expected_valid = spec.expected_counts["valid"]
    if len(train_paths) != expected_train or len(valid_paths) != expected_valid:
        raise ValueError(
            f"dataset {dataset_name} has unexpected split sizes: "
            f"train={len(train_paths)} valid={len(valid_paths)} "
            f"(expected {expected_train}/{expected_valid})"
        )

    train_records = save_shards(
        train_paths,
        dataset_name=dataset_name,
        subset_name="train",
        output_dir=output_dir,
        image_size=size,
        shard_size=shard_size,
        white_threshold=resolved_white_threshold,
        background_mode=spec.background_mode,
        num_workers=num_workers,
        force=force,
    )
    valid_records = save_shards(
        valid_paths,
        dataset_name=dataset_name,
        subset_name="valid",
        output_dir=output_dir,
        image_size=size,
        shard_size=shard_size,
        white_threshold=resolved_white_threshold,
        background_mode=spec.background_mode,
        num_workers=num_workers,
        force=force,
    )

    manifest = {
        "dataset_name": dataset_name,
        "data_root": str(data_root),
        "dataset_root": str(spec.dataset_root(data_root)),
        "output_dir": str(output_dir),
        "size": size,
        "shard_size": shard_size,
        "seed": seed,
        "background_mode": spec.background_mode,
        "white_threshold": resolved_white_threshold,
        "train_count": len(train_paths),
        "valid_count": len(valid_paths),
        "train_shards": train_records,
        "valid_shards": valid_records,
    }
    manifest_path = spec.manifest_path(data_root, size=size)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"saved manifest: {manifest_path}")
    return output_dir


def ensure_processed_rgb_dataset(
    *,
    data_root: Path,
    dataset_name: str,
    size: int,
    shard_size: int = 5000,
    white_threshold: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    download_if_missing: bool = True,
) -> Path:
    if has_processed_shards(
        data_root,
        size=size,
        shard_size=shard_size,
        dataset_name=dataset_name,
    ):
        return processed_shard_dir(data_root, size=size, dataset_name=dataset_name)
    return process_rgb_image_dataset(
        data_root=data_root,
        dataset_name=dataset_name,
        size=size,
        shard_size=shard_size,
        white_threshold=white_threshold,
        num_workers=num_workers,
        force=False,
        seed=seed,
        download_if_missing=download_if_missing,
    )


def process_cartoon_faces_dataset(
    *,
    data_root: Path,
    size: int,
    shard_size: int = 5000,
    white_threshold: int = 240,
    num_workers: int = 0,
    force: bool = False,
    seed: int = 0,
    download_if_missing: bool = True,
) -> Path:
    return process_rgb_image_dataset(
        data_root=data_root,
        dataset_name="cartoon_set",
        size=size,
        shard_size=shard_size,
        white_threshold=white_threshold,
        num_workers=num_workers,
        force=force,
        seed=seed,
        download_if_missing=download_if_missing,
    )


def ensure_processed_cartoon_faces_dataset(
    *,
    data_root: Path,
    size: int,
    shard_size: int = 5000,
    white_threshold: int = 240,
    num_workers: int = 0,
    seed: int = 0,
    download_if_missing: bool = True,
) -> Path:
    return ensure_processed_rgb_dataset(
        data_root=data_root,
        dataset_name="cartoon_set",
        size=size,
        shard_size=shard_size,
        white_threshold=white_threshold,
        num_workers=num_workers,
        seed=seed,
        download_if_missing=download_if_missing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process registered RGB datasets into sharded NPZ datasets."
    )
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="cartoon_set")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).resolve().parent),
    )
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--white-threshold", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    process_rgb_image_dataset(
        data_root=Path(args.data_root),
        dataset_name=args.dataset,
        size=args.size,
        shard_size=args.shard_size,
        white_threshold=args.white_threshold,
        num_workers=args.num_workers,
        force=args.force,
        seed=args.seed,
        download_if_missing=not args.no_download,
    )


if __name__ == "__main__":
    main()
