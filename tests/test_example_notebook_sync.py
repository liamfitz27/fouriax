from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

CELL_MARKER_RE = re.compile(r"^#\s*%%(?:\s.*)?$", re.MULTILINE)


def _sync_managed_example_scripts(repo_root: Path) -> list[Path]:
    scripts_dir = repo_root / "examples" / "scripts"
    notebooks_dir = repo_root / "examples" / "notebooks"

    managed: list[Path] = []
    for script_path in sorted(scripts_dir.glob("*.py")):
        notebook_path = notebooks_dir / f"{script_path.stem}.ipynb"
        if not notebook_path.exists():
            continue

        text = script_path.read_text(encoding="utf-8")
        if CELL_MARKER_RE.search(text) is None:
            # Not yet migrated to the notebook sync workflow.
            continue

        managed.append(script_path.relative_to(repo_root))
    return managed


def _run_sync_tool(
    repo_root: Path,
    *args: str,
    scripts: list[Path],
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "scripts/sync_example_notebook.py",
        *args,
        *[str(path) for path in scripts],
    ]
    return subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_example_notebooks_are_in_sync_with_scripts():
    repo_root = Path(__file__).resolve().parents[1]
    scripts = _sync_managed_example_scripts(repo_root)

    assert scripts, "No sync-managed example scripts found."

    result = _run_sync_tool(repo_root, "--dry-run", scripts=scripts)

    assert result.returncode == 0, (
        "Notebook sync dry-run failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    stdout = result.stdout
    assert "[would update]" not in stdout, (
        "Example notebooks are out of sync with scripts. Run the sync tool and commit the "
        "notebook updates.\n"
        f"STDOUT:\n{stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    up_to_date_lines = [line for line in stdout.splitlines() if line.startswith("[up to date]")]
    assert len(up_to_date_lines) == len(scripts), (
        "Expected every sync-managed example to report '[up to date]'.\n"
        f"Managed scripts: {len(scripts)}\n"
        f"Reported lines:\n{stdout}"
    )


def test_example_notebook_output_hashes_match_scripts():
    repo_root = Path(__file__).resolve().parents[1]
    scripts = _sync_managed_example_scripts(repo_root)

    assert scripts, "No sync-managed example scripts found."

    result = _run_sync_tool(repo_root, "--check", scripts=scripts)

    assert result.returncode == 0, (
        "Notebook execution hashes are stale. Re-run the notebook sync tool with execution "
        "enabled and commit the updated notebooks.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
