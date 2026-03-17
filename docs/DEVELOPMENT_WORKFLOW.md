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

## Remote Experiment Workflow

If you are using a local-edit plus SSH/Slurm-run workflow, use the dedicated
guide in [docs/development/CLUSTER_SYNC_WORKFLOW.md](development/CLUSTER_SYNC_WORKFLOW.md).

That workflow is intentionally different from normal feature development:
- it uses dedicated run branches instead of `main`
- it allows tracked artifacts on those run branches
- it treats the remote clone as execute-only

## CI Expectations
CI mirrors the same checks:
- lint (`ruff`)
- typing (`mypy`)
- tests (`pytest`)
