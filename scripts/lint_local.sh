#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PY="$VENV_DIR/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Missing .venv python. Run: scripts/dev_setup.sh"
  exit 1
fi

echo "Running Ruff..."
"$PY" -m ruff check "$ROOT_DIR"

echo "Running mypy..."
"$PY" -m mypy "$ROOT_DIR/src"

echo "Lint checks passed."
