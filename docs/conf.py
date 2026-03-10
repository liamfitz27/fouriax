"""Sphinx configuration for the fouriax documentation site."""

from __future__ import annotations

import json
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_NOTEBOOKS_DIR = REPO_ROOT / "examples" / "notebooks"
DOCS_EXAMPLES_DIR = DOCS_DIR / "examples"
DOCS_EXAMPLE_NOTEBOOKS_DIR = DOCS_EXAMPLES_DIR / "notebooks"
DOCS_API_GENERATED_DIR = DOCS_DIR / "api" / "generated"


def _notebook_title(path: Path) -> str:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return path.stem.replace("_", " ")

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
    return path.stem.replace("_", " ")


def _prepare_notebook_copy(src: Path, dst: Path) -> None:
    notebook = json.loads(src.read_text(encoding="utf-8"))

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        source = source.replace("### Outline", "## Outline")
        source = source.replace(
            "(4f_correlator_example.ipynb)",
            "(4f_correlator.ipynb)",
        )
        cell["source"] = source.splitlines(keepends=True)

    dst.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def _sync_example_notebooks_for_docs() -> list[Path]:
    notebooks = sorted(EXAMPLES_NOTEBOOKS_DIR.glob("*.ipynb"))
    DOCS_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_EXAMPLE_NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    notebook_names = {path.name for path in notebooks}
    for src in notebooks:
        _prepare_notebook_copy(src, DOCS_EXAMPLE_NOTEBOOKS_DIR / src.name)
    for existing in DOCS_EXAMPLE_NOTEBOOKS_DIR.glob("*.ipynb"):
        if existing.name not in notebook_names:
            existing.unlink()
    return notebooks


def _clean_generated_api_docs() -> None:
    DOCS_API_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    for path in DOCS_API_GENERATED_DIR.glob("*"):
        if path.is_file() and path.suffix in {".md", ".rst"}:
            path.unlink()


sys.path.insert(0, str(SRC_DIR))

project = "fouriax"
author = "fouriax contributors"
copyright = "2026, fouriax contributors"
root_doc = "index"

_sync_example_notebooks_for_docs()
_clean_generated_api_docs()

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}
add_module_names = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False

myst_heading_anchors = 3
myst_enable_extensions = ["amsmath", "dollarmath"]
nb_execution_mode = "off"

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "fouriax"
html_favicon = "_static/fouriax_logo.png"
html_theme_options = {
    "header_links_before_dropdown": 4,
    "navigation_with_keys": True,
    "navbar_align": "left",
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc"],
    "logo": {
        "image_light": "_static/fouriax_logo.svg",
        "image_dark": "_static/fouriax_logo.svg",
    },
}
