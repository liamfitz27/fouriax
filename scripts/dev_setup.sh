#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

# Prefer Python 3.12 for dependency compatibility; allow override via PYTHON_BIN.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -e "$ROOT_DIR[dev]"

if "$VENV_DIR/bin/python" -m pre_commit --version >/dev/null 2>&1; then
  "$VENV_DIR/bin/python" -m pre_commit install
  "$VENV_DIR/bin/python" -m pre_commit install --hook-type pre-push
fi

cat <<'EOF'
Dev environment ready.
Run local quality checks with:
  scripts/tests_local.sh
EOF
