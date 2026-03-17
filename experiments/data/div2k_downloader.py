"""DIV2K downloader for experiments.

Downloads the public DIV2K LR bicubic x2 archives into ``<data_root>/DIV2K``,
extracts them, and organizes the PNGs into ``train`` and ``valid`` directories.

Downloaded zip archives are kept under ``<data_root>/DIV2K/raw`` after a
successful extract so they can be reused later.
"""

from __future__ import annotations

import argparse
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

try:
    from .dataset_registry import div2k_root
except ImportError:  # pragma: no cover - direct script execution fallback
    from dataset_registry import div2k_root

ARCHIVE_SPECS: tuple[tuple[str, str, int], ...] = (
    (
        "train",
        "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip",
        800,
    ),
    (
        "valid",
        "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip",
        100,
    ),
)


def _format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def download_file(
    url: str,
    destination: Path,
    *,
    chunk_size: int = 1024 * 1024,
    progress_interval_s: float = 1.0,
) -> bool:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=30) as response, destination.open("wb") as handle:
            total_size_header = response.headers.get("Content-Length")
            total_size = int(total_size_header) if total_size_header is not None else 0
            downloaded = 0
            start_time = time.monotonic()
            last_report = start_time

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= progress_interval_s:
                    elapsed = max(now - start_time, 1e-6)
                    speed = downloaded / elapsed
                    if total_size > 0:
                        percent = 100.0 * downloaded / total_size
                        print(
                            f"{destination.name}: {percent:5.1f}% "
                            f"({_format_size(downloaded)}/{_format_size(total_size)}) "
                            f"at {_format_size(int(speed))}/s"
                        )
                    else:
                        print(
                            f"{destination.name}: {_format_size(downloaded)} "
                            f"downloaded at {_format_size(int(speed))}/s"
                        )
                    last_report = now

            elapsed = max(time.monotonic() - start_time, 1e-6)
            speed = downloaded / elapsed
            if total_size > 0:
                print(
                    f"{destination.name}: 100.0% "
                    f"({_format_size(downloaded)}/{_format_size(total_size)}) "
                    f"at {_format_size(int(speed))}/s"
                )
            else:
                print(
                    f"{destination.name}: downloaded {_format_size(downloaded)} "
                    f"at {_format_size(int(speed))}/s"
                )
        print(f"downloaded: {destination}")
        return True
    except Exception as exc:
        print(f"failed to download {url}: {exc}")
        if destination.exists():
            destination.unlink()
        return False


class DIV2KDownloader:
    """Downloader and organizer for the DIV2K x2 bicubic LR dataset."""

    def __init__(
        self,
        data_root: Path,
        *,
        archive_specs: tuple[tuple[str, str, int], ...] = ARCHIVE_SPECS,
    ):
        self.data_root = Path(data_root)
        self.div2k_dir = div2k_root(self.data_root)
        self.raw_dir = self.div2k_dir / "raw"
        self.archive_specs = archive_specs
        self.expected_counts = {subset: count for subset, _, count in archive_specs}

    def setup_directories(self) -> None:
        (self.div2k_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.div2k_dir / "valid").mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _count_pngs(self, root: Path) -> int:
        return sum(1 for path in root.glob("*.png") if path.is_file())

    def validate_subset(self, subset_name: str) -> bool:
        subset_dir = self.div2k_dir / subset_name
        return self._count_pngs(subset_dir) == self.expected_counts[subset_name]

    def dataset_ready(self) -> bool:
        return self.validate_subset("train") and self.validate_subset("valid")

    def extract_archive(self, archive_path: Path) -> bool:
        try:
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(self.div2k_dir)
            print(f"extracted: {archive_path}")
            return True
        except Exception as exc:
            print(f"failed to extract {archive_path}: {exc}")
            return False

    def reorganize_extracted_files(self, subset_name: str) -> None:
        source_dir = self.div2k_dir / f"DIV2K_{subset_name}_LR_bicubic" / "X2"
        target_dir = self.div2k_dir / subset_name
        if not source_dir.exists():
            return

        target_dir.mkdir(parents=True, exist_ok=True)
        for png_path in source_dir.glob("*.png"):
            shutil.move(str(png_path), str(target_dir / png_path.name))
        shutil.rmtree(source_dir.parent, ignore_errors=True)

    def _clear_subset(self, subset_name: str) -> None:
        subset_dir = self.div2k_dir / subset_name
        if not subset_dir.exists():
            return
        for path in subset_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def download(self) -> bool:
        self.setup_directories()
        if self.dataset_ready():
            print("DIV2K train/valid already present")
            return True

        for subset_name, url, expected_count in self.archive_specs:
            if self.validate_subset(subset_name):
                print(f"DIV2K {subset_name} already present")
                continue

            existing_count = self._count_pngs(self.div2k_dir / subset_name)
            if existing_count not in {0, expected_count}:
                print(
                    f"found incomplete DIV2K {subset_name} split with {existing_count} images; "
                    "clearing subset directory"
                )
                self._clear_subset(subset_name)

            archive_path = self.raw_dir / Path(url).name
            if not archive_path.exists() and not download_file(url, archive_path):
                return False
            if not self.extract_archive(archive_path):
                return False
            self.reorganize_extracted_files(subset_name)
            if not self.validate_subset(subset_name):
                print(
                    f"DIV2K {subset_name} extraction incomplete: found "
                    f"{self._count_pngs(self.div2k_dir / subset_name)} images, "
                    f"expected {expected_count}"
                )
                return False

        return self.dataset_ready()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and organize DIV2K into train/valid.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data"),
    )
    args = parser.parse_args()

    downloader = DIV2KDownloader(Path(args.data_root))
    if not downloader.download():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
