"""Utility helpers for fouriax examples and notebooks."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root containing ``src/fouriax`` and ``README.md``."""
    start_path = (start or Path.cwd()).resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / "src" / "fouriax").exists() and (candidate / "README.md").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate repository root from current working directory. "
        "Expected to find src/fouriax and README.md in an ancestor."
    )
