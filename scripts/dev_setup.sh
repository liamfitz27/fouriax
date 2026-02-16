#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
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
