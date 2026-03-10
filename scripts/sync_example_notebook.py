#!/usr/bin/env python3
"""Sync example scripts to notebooks by replacing code cells only.

This preserves notebook markdown cells and updates code cells sequentially.
The intended workflow is to author `examples/scripts/*.py` with `#%%` cell
markers (VS Code interactive cell style), then run this tool to update the
matching `examples/notebooks/*.ipynb`.

Key behavior:
- unwraps a top-level `main()` function and inlines its body
- removes the `if __name__ == "__main__": ...` guard
- preserves notebook markdown and code-cell order
- injects `%matplotlib inline` into the imports cell when matplotlib is used
- for `--create-missing`, creates markdown title cells from `#%% <title>`

`#%%` markers are required so generated cell boundaries match the tutorial
notebook structure.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CELL_MARKER_RE = re.compile(r"^#\s*%%(?:\s.*)?$")
CELL_MARKER_PARSE_RE = re.compile(r"^#\s*%%(?:\s+(.*))?$")
MATPLOTLIB_IMPORT_RE = re.compile(
    r"^\s*(?:import\s+matplotlib(?:\.\w+)?|from\s+matplotlib(?:\.\w+)*\s+import\s+.+)\s*$"
)
PLT_CLOSE_CALL_LINE_RE = re.compile(r"^(?P<indent>\s*)plt\.close\((?P<arg>[^)]*)\)\s*$")
REPO_ROOT_SETUP_LINE = "REPO_ROOT = fx.utils.find_repo_root(_Path.cwd())"


@dataclass
class TransformedScript:
    text: str


@dataclass
class ScriptCell:
    code: str
    title: str | None = None


def _line_range(node: ast.AST) -> tuple[int, int]:
    return (node.lineno, node.end_lineno)  # type: ignore[attr-defined]


def _is_module_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _is_main_guard(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.If):
        return False
    test = stmt.test
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False

    left, right = test.left, test.comparators[0]
    candidates = [(left, right), (right, left)]
    for a, b in candidates:
        if isinstance(a, ast.Name) and a.id == "__name__":
            if isinstance(b, ast.Constant) and b.value == "__main__":
                return True
    return False


def _extract_function_body_text(source: str, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    fn_src = ast.get_source_segment(source, fn)
    if not fn_src:
        start, end = _line_range(fn)
        lines = source.splitlines()
        fn_src = "\n".join(lines[start - 1 : end])

    lines = fn_src.splitlines()
    if not lines:
        return ""

    depth = 0
    header_end_idx = None
    for idx, line in enumerate(lines):
        code = line.split("#", 1)[0]
        for ch in code:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
        if depth == 0 and code.rstrip().endswith(":"):
            header_end_idx = idx
            break

    if header_end_idx is None:
        raise ValueError("Could not locate end of main() function header")

    body_lines = lines[header_end_idx + 1 :]
    body_text = textwrap.dedent("\n".join(body_lines))
    return body_text.rstrip("\n")


def transform_script(source: str) -> TransformedScript:
    module = ast.parse(source)
    module_body = module.body

    main_fn: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    guard_stmt: ast.If | None = None
    docstring_stmt: ast.stmt | None = None

    if module_body and _is_module_docstring(module_body[0]):
        docstring_stmt = module_body[0]

    for stmt in module_body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "main":
            main_fn = stmt
        elif _is_main_guard(stmt):
            guard_stmt = stmt

    source_lines = source.splitlines()
    replacements: dict[int, list[str]] = {}
    skip_ranges: list[tuple[int, int]] = []

    if docstring_stmt is not None:
        skip_ranges.append(_line_range(docstring_stmt))

    if main_fn is not None:
        body_text = _extract_function_body_text(source, main_fn)
        replacements[main_fn.lineno] = body_text.splitlines()
        skip_ranges.append(_line_range(main_fn))

    if guard_stmt is not None:
        skip_ranges.append(_line_range(guard_stmt))

    # Merge ranges for simpler line iteration.
    skip_ranges.sort()
    merged_ranges: list[tuple[int, int]] = []
    for start, end in skip_ranges:
        if not merged_ranges or start > merged_ranges[-1][1] + 1:
            merged_ranges.append((start, end))
        else:
            prev_start, prev_end = merged_ranges[-1]
            merged_ranges[-1] = (prev_start, max(prev_end, end))

    out_lines: list[str] = []
    i = 1
    range_idx = 0
    while i <= len(source_lines):
        if i in replacements:
            out_lines.extend(replacements[i])
        if range_idx < len(merged_ranges):
            start, end = merged_ranges[range_idx]
            if i == start:
                i = end + 1
                range_idx += 1
                continue
        out_lines.append(source_lines[i - 1])
        i += 1

    text = "\n".join(out_lines).strip("\n") + "\n"
    return TransformedScript(text=text)


def compute_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def strip_argparse_and_inline_defaults(code: str) -> str:
    tree = ast.parse(code)
    defaults = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            if not node.args:
                continue
            flag_node = node.args[0]
            if not isinstance(flag_node, ast.Constant) or not isinstance(flag_node.value, str):
                continue
            flag = flag_node.value
            if not flag.startswith("--"):
                continue
            dest = flag.lstrip("-").replace("-", "_")
            
            val_repr = "None"
            for kw in node.keywords:
                if kw.arg == "default":
                    val_repr = ast.unparse(kw.value)
                elif (
                    kw.arg == "action"
                    and isinstance(kw.value, ast.Constant)
                    and getattr(kw.value, "value", None) == "store_true"
                ):
                    val_repr = "False"
                elif (
                    kw.arg == "action"
                    and isinstance(kw.value, ast.Constant)
                    and getattr(kw.value, "value", None) == "store_false"
                ):
                    val_repr = "True"
            defaults[dest] = val_repr

    if not defaults:
        return code

    lines = code.splitlines()
    out = []
    in_parse_args = False
    for line in lines:
        if line.startswith("import argparse"):
            continue
        if line.startswith("def parse_args()"):
            in_parse_args = True
            continue
        if in_parse_args:
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                if not line.startswith("#"):
                    in_parse_args = False
            if in_parse_args:
                continue

        if not in_parse_args:
            if line.startswith("ARGS = parse_args()"):
                continue

            def repl(m: re.Match) -> str:
                return defaults.get(m.group(1), m.group(0))

            new_line = re.sub(r"\bARGS\.([a-zA-Z0-9_]+)\b", repl, line)
            new_line = new_line.replace("not False", "True").replace("not True", "False")

            out.append(new_line)

    cleaned = []
    for line in out:
        if not line.strip() and (not cleaned or not cleaned[-1].strip()):
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip() + "\n"


def _parse_marker_title(line: str) -> str | None:
    match = CELL_MARKER_PARSE_RE.match(line)
    if not match:
        return None
    title = (match.group(1) or "").strip()
    return title or None


def _split_cells_by_markers(text: str) -> list[ScriptCell]:
    cells: list[ScriptCell] = []
    current_lines: list[str] = []
    current_title: str | None = None

    for raw_line in text.splitlines():
        if CELL_MARKER_RE.match(raw_line):
            if current_lines:
                cell_text = "\n".join(current_lines).strip("\n")
                if cell_text.strip():
                    cells.append(ScriptCell(code=cell_text + "\n", title=current_title))
                current_lines = []
            current_title = _parse_marker_title(raw_line)
            continue
        current_lines.append(raw_line)

    if current_lines:
        cell_text = "\n".join(current_lines).strip("\n")
        if cell_text.strip():
            cells.append(ScriptCell(code=cell_text + "\n", title=current_title))

    return cells


def script_to_cell_specs(source: str) -> list[ScriptCell]:
    transformed = transform_script(source)
    if not any(CELL_MARKER_RE.match(line) for line in transformed.text.splitlines()):
        raise ValueError("No codebreaks detected (`#%%`). Add notebook cell markers to the script.")
    cells = _split_cells_by_markers(transformed.text)
    
    for cell in cells:
        cell.code = strip_argparse_and_inline_defaults(cell.code)
        
    cells = inject_repo_root_setup(cells)
    cells = inject_matplotlib_inline(cells)
    return cells


def script_to_code_cells(source: str) -> list[str]:
    return [cell.code for cell in script_to_cell_specs(source)]


def inject_matplotlib_inline(cells: list[ScriptCell]) -> list[ScriptCell]:
    if not cells:
        return cells
    if any("%matplotlib inline" in cell.code for cell in cells):
        return cells
    if not any("matplotlib" in cell.code for cell in cells):
        return cells

    target_idx = None
    for i, cell in enumerate(cells):
        if any(MATPLOTLIB_IMPORT_RE.match(line) for line in cell.code.splitlines()):
            target_idx = i
            break
    if target_idx is None:
        target_idx = 0

    lines = cells[target_idx].code.splitlines()
    insert_at = _import_block_end_line_index(cells[target_idx].code)

    # Insert magic after the complete import block (including multiline imports).
    # Preserve spacing without forcing extra blank lines if one already exists.
    to_insert = ["%matplotlib inline"]
    if insert_at < len(lines) and lines[insert_at].strip():
        to_insert.append("")
    for offset, line in enumerate(to_insert):
        lines.insert(insert_at + offset, line)
    cells[target_idx].code = "\n".join(lines).rstrip("\n") + "\n"
    return cells


def inject_repo_root_setup(cells: list[ScriptCell]) -> list[ScriptCell]:
    """Inject a notebook-only repo-root setup snippet into the imports cell.

    This keeps example scripts free to use repo-root-relative paths (e.g. `Path("data/...")`)
    while making notebooks robust when launched from `examples/notebooks/` or any subdirectory.
    """
    if not cells:
        return cells
    if any(REPO_ROOT_SETUP_LINE in cell.code for cell in cells):
        return cells

    target_idx = None
    for i, cell in enumerate(cells):
        try:
            module = ast.parse(cell.code)
        except SyntaxError:
            continue
        if any(isinstance(stmt, (ast.Import, ast.ImportFrom)) for stmt in module.body):
            target_idx = i
            break
    if target_idx is None:
        target_idx = 0

    lines = cells[target_idx].code.splitlines()
    insert_at = _import_block_end_line_index(cells[target_idx].code)

    snippet_lines = [
        "",
        "import os",
        "from pathlib import Path as _Path",
        "",
        REPO_ROOT_SETUP_LINE,
        "if _Path.cwd() != REPO_ROOT:",
        "    os.chdir(REPO_ROOT)",
        "",
    ]

    for offset, line in enumerate(snippet_lines):
        lines.insert(insert_at + offset, line)
    cells[target_idx].code = "\n".join(lines).rstrip("\n") + "\n"
    return cells


def replace_plt_close_with_show(cells: list[ScriptCell]) -> list[ScriptCell]:
    """Replace line-level `plt.close(...)` calls with `plt.show()`."""
    for cell in cells:
        new_lines: list[str] = []
        changed = False
        for line in cell.code.splitlines():
            match = PLT_CLOSE_CALL_LINE_RE.match(line)
            if match:
                new_lines.append(f"{match.group('indent')}plt.show()")
                changed = True
            else:
                new_lines.append(line)
        if changed:
            cell.code = "\n".join(new_lines).rstrip("\n") + "\n"
    return cells


def _import_block_end_line_index(code: str) -> int:
    """Return 0-based insertion index immediately after the leading import block."""
    try:
        module = ast.parse(code)
    except SyntaxError:
        return 0

    last_end = 0
    for stmt in module.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            last_end = max(last_end, stmt.end_lineno or stmt.lineno)  # type: ignore[attr-defined]
            continue
        break
    return last_end


def _cell_source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def update_notebook_code_cells(
    notebook_path: Path,
    code_cells: list[str],
    *,
    clear_outputs: bool = False,
    execute: bool = False,
    force_match: bool = False,
    check: bool = False,
) -> bool:
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)
        
    nb["nbformat"] = 4
    nb["nbformat_minor"] = max(nb.get("nbformat_minor", 0), 5)

    cells = nb.get("cells", [])
    code_indices = [i for i, cell in enumerate(cells) if cell.get("cell_type") == "code"]
    if len(code_indices) != len(code_cells):
        if check:
            raise ValueError(
                f"Code cell count mismatch in {notebook_path}. "
                f"Notebook has {len(code_indices)} code cells, script has {len(code_cells)}."
            )
        raise ValueError(
            f"Code cell count mismatch for {notebook_path}: "
            f"notebook has {len(code_indices)} code cells, generated {len(code_cells)}. "
            "Add `#%%` markers to the script so the generated cell boundaries match the notebook."
        )

    changed = False
    needs_execute = False
    for cell_idx, code_text in zip(code_indices, code_cells, strict=True):
        cell = cells[cell_idx]
        new_source = _cell_source_lines(code_text)
        new_hash = compute_hash("".join(new_source))

        metadata = cell.setdefault("metadata", {})
        existing_hash = metadata.get("fouriax_source_hash")

        if check:
            if existing_hash != new_hash:
                raise ValueError(
                    f"Hash mismatch in cell {cell_idx} of {notebook_path}. "
                    "Notebook output is stale."
                )

        if cell.get("source") != new_source:
            cell["source"] = new_source
            changed = True

        if "id" not in cell:
            cell["id"] = __import__("uuid").uuid4().hex[:8]
            changed = True

        is_missing_output = cell.get("execution_count") is None and not cell.get("outputs")

        if existing_hash != new_hash:
            needs_execute = True
            if (
                execute
                or force_match
                or clear_outputs
                or (is_missing_output and existing_hash is None)
            ):
                metadata["fouriax_source_hash"] = new_hash
                changed = True
        elif is_missing_output:
            needs_execute = True

        if clear_outputs:
            if cell.get("outputs"):
                cell["outputs"] = []
                changed = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True

    if check:
        return False

    if changed:
        with notebook_path.open("w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
            f.write("\n")

    if execute and needs_execute:
        print(f"[executing] {notebook_path}")
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                str(notebook_path),
            ],
            check=True,
        )
        return True

    return changed

def _markdown_title_cell(index: int, title: str | None) -> dict[str, object]:
    heading = f"## {index}" if not title else f"## {index}  {title}"
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": [heading + "\n"],
    }


def create_notebook(notebook_path: Path, cell_specs: list[ScriptCell]) -> None:
    cells: list[dict[str, object]] = []
    for idx, spec in enumerate(cell_specs, start=0):
        cells.append(_markdown_title_cell(idx, spec.title))
        cells.append(
            {
                "cell_type": "code",
                "id": uuid.uuid4().hex[:8],
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _cell_source_lines(spec.code),
            }
        )

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    with notebook_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")


def resolve_pairs(
    script_args: list[str],
    *,
    examples_scripts_dir: Path,
    examples_notebooks_dir: Path,
) -> list[tuple[Path, Path]]:
    scripts: list[Path]
    if script_args:
        scripts = [Path(p) for p in script_args]
    else:
        scripts = sorted(examples_scripts_dir.glob("*.py"))

    pairs: list[tuple[Path, Path]] = []
    for script in scripts:
        script_path = script if script.is_absolute() else Path.cwd() / script
        if not script_path.exists():
            raise FileNotFoundError(f"missing script: {script}")
        notebook_path = examples_notebooks_dir / f"{script_path.stem}.ipynb"
        pairs.append((script_path, notebook_path))
    return pairs


def _iter_errors(func, items: Iterable[tuple[Path, Path]]) -> int:
    failures = 0
    for script_path, notebook_path in items:
        try:
            func(script_path, notebook_path)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[error] {script_path.name}: {exc}", file=sys.stderr)
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scripts",
        nargs="*",
        help="Script path(s) to sync. Defaults to all files in examples/scripts.",
    )
    parser.add_argument(
        "--examples-scripts-dir",
        type=Path,
        default=Path("examples/scripts"),
        help="Directory containing source example scripts.",
    )
    parser.add_argument(
        "--examples-notebooks-dir",
        type=Path,
        default=Path("examples/notebooks"),
        help="Directory containing destination notebooks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report changes without writing files.",
    )
    parser.add_argument(
        "--create-missing",
        action="store_true",
        help="Create the notebook if it does not already exist.",
    )
    parser.add_argument(
        "--clear-outputs",
        action="store_true",
        help="Clear outputs and execution counts in all updated notebooks.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the notebook after updating it to produce fresh outputs.",
    )
    parser.add_argument(
        "--force-match",
        action="store_true",
        help="Update fouriax_source_hash to match the new code without regenerating outputs.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if any notebook cell's code hash does not match its existing output hash.",
    )
    args = parser.parse_args(argv)

    pairs = resolve_pairs(
        args.scripts,
        examples_scripts_dir=args.examples_scripts_dir,
        examples_notebooks_dir=args.examples_notebooks_dir,
    )

    def _sync_one(script_path: Path, notebook_path: Path) -> None:
        if args.check and not notebook_path.exists():
            raise FileNotFoundError(f"Missing notebook for script: {notebook_path}")

        source = script_path.read_text(encoding="utf-8")
        cell_specs = script_to_cell_specs(source)
        
        cell_specs = replace_plt_close_with_show(cell_specs)
        code_cells = [cell.code for cell in cell_specs]
        if not code_cells:
            raise ValueError("generated 0 code cells")

        if notebook_path.exists():
            if args.dry_run:
                with notebook_path.open("r", encoding="utf-8") as f:
                    nb = json.load(f)
                existing_code_cells = [
                    "".join(cell.get("source", []))
                    for cell in nb.get("cells", [])
                    if cell.get("cell_type") == "code"
                ]
                if len(existing_code_cells) != len(code_cells):
                    raise ValueError(
                        f"Code cell count mismatch for {notebook_path}: "
                        "notebook has "
                        f"{len(existing_code_cells)} code cells, "
                        f"generated {len(code_cells)}"
                    )
                changed = existing_code_cells != code_cells
                status = "would update" if changed else "up to date"
                print(f"[{status}] {script_path} -> {notebook_path}")
                return

            if args.check:
                update_notebook_code_cells(
                    notebook_path, code_cells, check=True
                )
                print(f"[check passed] {script_path} -> {notebook_path}")
                return

            changed = update_notebook_code_cells(
                notebook_path,
                code_cells,
                clear_outputs=args.clear_outputs,
                execute=args.execute,
                force_match=args.force_match,
            )
            status = "updated" if changed else "unchanged"
            print(f"[{status}] {script_path} -> {notebook_path}")
            return

        if not args.create_missing:
            raise FileNotFoundError(
                "No notebook detected for script "
                f"{script_path.name}: expected {notebook_path}. "
                "Use --create-missing to create one."
            )
        if args.dry_run:
            print(f"[would create] {script_path} -> {notebook_path}")
            return
        create_notebook(notebook_path, cell_specs)
        print(f"[created] {script_path} -> {notebook_path}")

    failures = _iter_errors(_sync_one, pairs)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
