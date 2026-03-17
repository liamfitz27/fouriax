"""Cartoon Set downloader for experiments.

Downloads the public Cartoon Set archives into ``<data_root>/cartoon_set``,
extracts them, and splits images into ``train`` and ``valid`` subdirectories.

By default the downloaded ``.tgz`` archives are kept under
``<data_root>/cartoon_set/raw`` after a successful extract so they can be
reused or transferred elsewhere later.
"""

from __future__ import annotations

import argparse
import random
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

try:
    from .dataset_registry import cartoon_set_root
except ImportError:  # pragma: no cover - direct script execution fallback
    from dataset_registry import cartoon_set_root

ARCHIVE_SPECS: tuple[tuple[str, str, int], ...] = (
    (
        "cartoonset10k",
        "https://huggingface.co/datasets/cgarciae/cartoonset/resolve/main/data/cartoonset10k.tgz",
        10000,
    ),
    (
        "cartoonset100k",
        "https://huggingface.co/datasets/cgarciae/cartoonset/resolve/main/data/cartoonset100k.tgz",
        100000,
    ),
)
EXPECTED_TOTAL_IMAGES = sum(spec[2] for spec in ARCHIVE_SPECS)
DEFAULT_TRAIN_RATIO = 0.9


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


class CartoonSetDownloader:
    """Downloader and splitter for the public Google Cartoon Set archives."""

    def __init__(
        self,
        data_root: Path,
        *,
        seed: int = 0,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        archive_specs: tuple[tuple[str, str, int], ...] = ARCHIVE_SPECS,
    ):
        self.data_root = Path(data_root)
        self.cartoon_dir = cartoon_set_root(self.data_root)
        self.raw_dir = self.cartoon_dir / "raw"
        self.seed = seed
        self.train_ratio = train_ratio
        self.archive_specs = archive_specs
        self.expected_total = sum(spec[2] for spec in archive_specs)

    def setup_directories(self) -> None:
        (self.cartoon_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.cartoon_dir / "valid").mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def extract_archive(self, archive_path: Path) -> bool:
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=self.cartoon_dir)
            print(f"extracted: {archive_path}")
            return True
        except Exception as exc:
            print(f"failed to extract {archive_path}: {exc}")
            try:
                archive_path.unlink()
            except FileNotFoundError:
                pass
            return False

    def _split_ready(self) -> bool:
        train_count = self._count_images(self.cartoon_dir / "train")
        valid_count = self._count_images(self.cartoon_dir / "valid")
        return (
            train_count > 0
            and valid_count > 0
            and (train_count + valid_count) == self.expected_total
        )

    def dataset_ready(self) -> bool:
        return self._split_ready()

    @staticmethod
    def _count_images(root: Path) -> int:
        return sum(
            1
            for path in root.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

    def _discovered_images(self) -> list[Path]:
        def is_in_split_dirs(path: Path) -> bool:
            return any(part in {"train", "valid", "raw"} for part in path.parts)

        return sorted(
            path
            for path in self.cartoon_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            and not is_in_split_dirs(path)
        )

    def _archive_extract_dir(self, archive_name: str) -> Path:
        return self.cartoon_dir / archive_name

    def _archive_ready(self, archive_name: str, expected_count: int) -> bool:
        return self._count_images(self._archive_extract_dir(archive_name)) == expected_count

    def _download_missing_archives(self) -> bool:
        downloaded_any = False
        for _, url, _ in self.archive_specs:
            archive_path = self.raw_dir / Path(url).name
            if archive_path.exists():
                continue
            if not download_file(url, archive_path):
                return False
            downloaded_any = True
        if downloaded_any:
            print(f"raw archives ready under {self.raw_dir}")
        return True

    def _clear_directory_contents(self, root: Path) -> None:
        if not root.exists():
            return
        for path in root.iterdir():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def _clear_split_dirs(self) -> None:
        self._clear_directory_contents(self.cartoon_dir / "train")
        self._clear_directory_contents(self.cartoon_dir / "valid")

    def _clear_extracted_dirs(self) -> None:
        for archive_name, _, _ in self.archive_specs:
            shutil.rmtree(self._archive_extract_dir(archive_name), ignore_errors=True)

    def _clear_archive_extract_dir(self, archive_name: str) -> None:
        shutil.rmtree(self._archive_extract_dir(archive_name), ignore_errors=True)

    def validate_and_split(self) -> bool:
        all_images = self._discovered_images()
        if not all_images:
            print(f"no extracted images found under {self.cartoon_dir}")
            return False
        if len(all_images) != self.expected_total:
            print(
                f"incomplete extracted cartoon set: found {len(all_images)} images, "
                f"expected {self.expected_total}"
            )
            return False

        rng = random.Random(self.seed)
        rng.shuffle(all_images)
        split_index = int(len(all_images) * self.train_ratio)
        train_files = all_images[:split_index]
        valid_files = all_images[split_index:]
        print(f"splitting into train={len(train_files)} valid={len(valid_files)}")

        for subset_name, subset_files in (("train", train_files), ("valid", valid_files)):
            subset_dir = self.cartoon_dir / subset_name
            subset_dir.mkdir(parents=True, exist_ok=True)
            for src in subset_files:
                dst = subset_dir / src.name
                if dst.exists():
                    continue
                shutil.move(str(src), str(dst))
                sidecar = src.with_suffix(".csv")
                if sidecar.exists():
                    shutil.move(str(sidecar), str(subset_dir / sidecar.name))

        self._cleanup_extracted_dirs()
        return self._split_ready()

    def _cleanup_extracted_dirs(self) -> None:
        for path in self.cartoon_dir.iterdir():
            if path.name in {"train", "valid", "raw"}:
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def download(self) -> bool:
        self.setup_directories()
        if self._split_ready():
            print("cartoon set train/valid already present")
            return self._download_missing_archives()

        split_total = self._count_images(self.cartoon_dir / "train") + self._count_images(
            self.cartoon_dir / "valid"
        )
        if split_total > 0:
            print(
                f"found incomplete train/valid split with {split_total} images; "
                "clearing split directories"
            )
            self._clear_split_dirs()

        extracted_total = len(self._discovered_images())
        if extracted_total not in {0, self.expected_total}:
            print(
                f"found incomplete extracted cartoon set with {extracted_total} images; "
                "clearing extracted directories"
            )
            self._clear_extracted_dirs()
            extracted_total = 0

        if extracted_total == self.expected_total and self.validate_and_split():
            return True

        extracted_any = False
        for archive_name, url, expected_count in self.archive_specs:
            if self._archive_ready(archive_name, expected_count):
                print(f"archive already extracted: {archive_name}")
                extracted_any = True
                continue

            archive_path = self.raw_dir / Path(url).name
            for attempt in range(2):
                if not archive_path.exists() and not download_file(url, archive_path):
                    break
                self._clear_archive_extract_dir(archive_name)
                if self.extract_archive(archive_path) and self._archive_ready(
                    archive_name,
                    expected_count,
                ):
                    extracted_any = True
                    break
                print(
                    f"archive {archive_name} was incomplete after extract "
                    f"(attempt {attempt + 1}/2)"
                )

        if not extracted_any:
            print("no archives were downloaded and extracted successfully")
            return False

        extracted_total = len(self._discovered_images())
        if extracted_total != self.expected_total:
            expected_archives = ", ".join(
                f"{name}={count}" for name, _, count in self.archive_specs
            )
            print(
                f"cartoon set extraction incomplete: found {extracted_total} images, "
                f"expected {self.expected_total} ({expected_archives})"
            )
            return False

        return self.validate_and_split()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and split Cartoon Set into train/valid.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "cartoon_set"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    downloader = CartoonSetDownloader(Path(args.data_root), seed=args.seed)
    if not downloader.download():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
