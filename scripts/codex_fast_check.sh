#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Missing .venv python. Run: scripts/dev_setup.sh"
  exit 1
fi

echo "Running Ruff..."
"$PY" -m ruff check "$ROOT_DIR"

if [[ $# -eq 0 ]]; then
  echo "Fast check finished. Pass pytest targets to run focused tests."
  exit 0
fi

echo "Running targeted pytest..."
PYTHONPATH="$ROOT_DIR/src" "$PY" -m pytest -q "$@"
