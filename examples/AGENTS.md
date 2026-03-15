# Examples Guide

`examples/scripts/` is the authoring surface. `examples/notebooks/` is the tutorial artifact.

## Rules

- Prefer editing the script, not the notebook JSON.
- For sync-managed examples, keep `#%%` markers stable so notebook cell boundaries remain predictable.
- If you add or reorder code sections, check whether notebook markdown still matches the script flow.
- Keep example CLIs lightweight and smoke-testable; `tests/test_examples.py` runs reduced-iteration versions.

## Notebook Markdown Style

- Treat the notebook set as a guided tree, not a flat list of unrelated examples.
- Make one notebook in each family the entry point, and write later notebooks as extensions that assume that background.
- Entry-point notebooks should explicitly say:
  - what the notebook does
  - what background it assumes
  - what new ideas it introduces
  - where to go next
- Child notebooks should explicitly say:
  - which earlier notebook they build on
  - which theory is assumed and not repeated
  - what new ingredient changes relative to the parent notebook
- Keep the opening markdown cell focused on scope and navigation, not script provenance boilerplate.
- Put reusable theory in a dedicated concept/setup markdown cell, not hidden in helper-function comments alone.
- Keep `Imports`, `Paths and Parameters`, `Evaluation`, and `Plot Results` markdown concise unless there is real tutorial value in expanding them.
- Use a consistent depth target:
  - entry notebooks: fuller physical framing and assumptions
  - child notebooks: brief reminders plus explanation of the delta
  - standalone advanced notebooks: focused framing with explicit modeling assumptions
- Avoid repeating the same explanations across notebooks when a prerequisite link is enough.
- When a notebook compares against an analytical or classical baseline, say so clearly in the opening cell and again in the results framing.
- End result sections with an interpretation of what the plots demonstrate, not only a description of what is shown.

## Sync Alignment

- Notebook markdown section headers should match the script `#%%` section titles in order.
- Numeric prefixes in notebook headers are fine, but the section names should still align with the script markers.
- If notebook markdown is manually adjusted, verify that `Setup`, `Training Data`, `Evaluation`, and similar headers still sit next to the corresponding synced code cells.
- If a notebook links to another example, use the actual current notebook filename.

## Validation

- Notebook sync: `.venv/bin/python scripts/sync_example_notebook.py --check`
- Example smoke tests: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_examples.py`
- Single example: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_examples.py -k <name>`
