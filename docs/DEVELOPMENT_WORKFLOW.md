# Development Workflow

## Setup
```bash
scripts/dev_setup.sh
```
Creates/updates `.venv`, installs dev dependencies, and installs pre-commit hooks.

## Local Quality Gate
```bash
scripts/tests_local.sh
```
Runs:
- `ruff check`
- `mypy src`
- `pytest -q` with `PYTHONPATH=src`

## Git Flow
1. Create a branch:
```bash
git checkout -b feat/<name>
```
2. Implement changes.
3. Run `scripts/tests_local.sh`.
4. Commit.
5. Push and open PR.

## CI Expectations
CI mirrors the same checks:
- lint (`ruff`)
- typing (`mypy`)
- tests (`pytest`)
