# Development

This section covers local setup and contributor workflow for `fouriax`.

## Local Setup

Install development and docs dependencies:

```bash
pip install -e ".[dev,docs]"
```

## Common Commands

Run the local quality gate:

```bash
scripts/tests_local.sh
```

Build the docs site:

```bash
python -m sphinx -b html docs docs/_build/html
```

```{toctree}
:maxdepth: 1

../DEVELOPMENT_WORKFLOW
../TODO
```
