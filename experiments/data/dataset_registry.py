"""Dataset specs for RGB experiment data preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

BackgroundMode = Literal["none", "replace_white_border"]


@dataclass(frozen=True)
class ImageDatasetSpec:
    name: str
    root_dir_name: str
    processed_prefix: str
    expected_counts: dict[str, int]
    background_mode: BackgroundMode
    white_threshold: int = 240
    valid_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

    def dataset_root(self, data_root: Path) -> Path:
        data_root = Path(data_root)
        if data_root.name == self.root_dir_name:
            return data_root
        return data_root / self.root_dir_name

    def subset_dir(self, data_root: Path, subset_name: str) -> Path:
        return self.dataset_root(data_root) / subset_name

    def processed_shard_dir(self, data_root: Path, *, size: int) -> Path:
        return self.dataset_root(data_root) / f"{self.processed_prefix}_nx{size}_shards"

    def manifest_path(self, data_root: Path, *, size: int) -> Path:
        return self.processed_shard_dir(data_root, size=size) / (
            f"{self.processed_prefix}_manifest_nx{size}.json"
        )

    @property
    def expected_total(self) -> int:
        return sum(self.expected_counts.values())


DATASET_SPECS: dict[str, ImageDatasetSpec] = {
    "cartoon_set": ImageDatasetSpec(
        name="cartoon_set",
        root_dir_name="cartoon_set",
        processed_prefix="cartoon_faces",
        expected_counts={"train": 99_000, "valid": 11_000},
        background_mode="replace_white_border",
        white_threshold=240,
    ),
    "div2k": ImageDatasetSpec(
        name="div2k",
        root_dir_name="DIV2K",
        processed_prefix="div2k",
        expected_counts={"train": 800, "valid": 100},
        background_mode="none",
        white_threshold=240,
        valid_extensions=(".png",),
    ),
}

DATASET_NAMES = tuple(sorted(DATASET_SPECS))


def get_dataset_spec(dataset_name: str) -> ImageDatasetSpec:
    try:
        return DATASET_SPECS[dataset_name]
    except KeyError as exc:  # pragma: no cover - defensive programming
        choices = ", ".join(DATASET_NAMES)
        raise ValueError(f"unknown dataset {dataset_name!r}; expected one of: {choices}") from exc


def cartoon_set_root(data_root: Path) -> Path:
    return get_dataset_spec("cartoon_set").dataset_root(Path(data_root))


def div2k_root(data_root: Path) -> Path:
    return get_dataset_spec("div2k").dataset_root(Path(data_root))
