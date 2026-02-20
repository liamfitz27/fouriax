# Handoff

## Repo State

- Branch: `feat/fourier-hybrid-module-na`
- HEAD: `4a2ab0f` (`feat: add propagator facade and migrate examples`)
- Working tree:
  - modified: `docs/PLANNED_ADDITIONS.md`

## Completed

- Added unified public propagator facade: `Propagator(mode="auto"|"asm"|"rs"|"kspace")`.
- Kept existing specialized propagators (`ASMPropagator`, `RSPropagator`, `KSpacePropagator`, `AutoPropagator`) for low-level use.
- Enabled direct propagator usage inside `OpticalModule.layers` when `distance_um` is defined.
- Added module-level propagation planning to resolve auto choices before execution.
- Added RS `na_limit` support and conservative NA merge behavior (`min(existing, scheduled)`).
- Updated all `scripts/*_example.py` to use the new facade style.

## Open Work

- Primary next goal: add incoherent imaging support with shift-invariant PSF/OTF operators.
- Keep polarization/Jones and rectangular meta-atom additions staged after incoherent imaging core lands.

## Key Files

- `src/fouriax/optics/propagation.py`: propagator implementations + new `Propagator` facade.
- `src/fouriax/optics/layers.py`: module planning + inline propagator handling.
- `src/fouriax/optics/plotting.py`: plotting now uses planned layers.
- `tests/test_propagation_selection.py`: facade mode dispatch and auto-precompute tests.
- `docs/PLANNED_ADDITIONS.md`: roadmap (including coherence phase).

## Runbook

- Setup:

```bash
scripts/dev_setup.sh
```

- Local quality gate:

```bash
scripts/tests_local.sh
```

- Targeted checks used during recent changes:

```bash
.venv/bin/python -m pytest -q tests/test_propagation_selection.py tests/test_optical_module.py tests/test_na_schedule.py
MPLBACKEND=Agg .venv/bin/python scripts/4f_comparison_example.py
MPLBACKEND=Agg .venv/bin/python scripts/lens_optimization_example.py
MPLBACKEND=Agg .venv/bin/python scripts/metaatom_optimization_example.py
```

## Next Tasks

1. Implement incoherent imaging operators (`IncoherentImagingLayer` / `OTFImagingLayer`).
   Done when: PSF and OTF implementations are numerically consistent and covered by parity tests.
2. Integrate incoherent mode into example workflows.
   Done when: at least one example script demonstrates incoherent image formation and saves reproducible artifacts.
3. Add explicit API contract and tests for coherent vs incoherent paths.
   Done when: behavior differences are validated in tests and discoverable from public API docs.

## Risks

- Boundary/padding choices in PSF convolution can bias optimization results.
- Incoherent operators can be confused with coherent k-space propagation unless contracts stay explicit.
- Runtime and memory growth for large kernels/grids.

## Fresh Codex Prompt

```text
Work in /Users/liam/fouriax on branch feat/fourier-hybrid-module-na.
First read docs/HANDOFF.md and docs/PLANNED_ADDITIONS.md.
Use .venv
Primary goal: implement incoherent imaging support (shift-invariant PSF/OTF), starting with real-space imaging scope.
Run pytest -q tests/test_propagation_selection.py tests/test_optical_module.py tests/test_na_schedule.py plus new incoherent-imaging tests.
```
