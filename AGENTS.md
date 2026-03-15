# Fouriax Codex Guide

This file is for repo-specific guidance only. Keep architecture detail in the existing docs.

## Repo map

- `src/fouriax/optics/`: core field model, layers, propagation, sensors, noise, NA planning
- `src/fouriax/optim/`: optimization helpers used by examples
- `src/fouriax/analysis/`: Fisher and sensitivity analysis helpers
- `examples/scripts/`: canonical runnable examples
- `examples/notebooks/`: tutorial notebooks synced from scripts
- `tests/`: pytest suite, including example smoke tests and notebook sync checks

## Read first

- `README.md`
- `docs/DEVELOPMENT_WORKFLOW.md`


## Read for details

- `docs/USER_GUIDE.md`: smaller examples usage
- `docs/ARCHITECTURE.md`: large architecture/structure file, do not read entirely unless needed

## Fast workflow

- Setup: `scripts/dev_setup.sh`
- Full local gate: `scripts/tests_local.sh`
- Fast local gate: `scripts/codex_fast_check.sh`
- Targeted tests: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/<target>.py`

Prefer targeted `pytest` while iterating. Run `scripts/tests_local.sh` before finishing unless the change is docs-only or clearly isolated config.

## Repo-specific rules

- Python is `3.12`; use the repo `.venv` when running checks.
- Import/test commands assume `PYTHONPATH=src`.
- Preserve explicit spatial vs k-space transitions. Do not silently add domain inference where the API currently requires explicit `FourierTransform` or `InverseFourierTransform`.
- Preserve shape conventions. Batch dimensions stay leading, then wavelength, then optional Jones channel, then spatial axes.
- When public behavior changes, update the relevant docs in `README.md` or `docs/USER_GUIDE.md`.

## Examples and notebooks

- Treat `examples/scripts/*.py` as the source of truth.
- If a script uses `#%%` cells and has a matching notebook, sync with `scripts/sync_example_notebook.py`; do not hand-edit code cells in the notebook.
- After touching sync-managed examples, run `.venv/bin/python scripts/sync_example_notebook.py --check`.

## Repo-local skills

Repo-local skills live under `.codex/skills/`. They are not global installs, so open the matching `SKILL.md` directly when the task fits.

- `optics-core-change`: optics-layer, propagation, field-model, or sensor changes
- `example-notebook-sync`: example script or notebook work
- `benchmark-and-profile`: propagator regime, scaling, or performance work

When a request clearly matches one of these, read that skill before editing.
