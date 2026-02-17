# Handoff

## Repo State
- Branch: `feat/metaatom-grid-optimization`
- Recent commits:
  - `a9abedb` feat: add grid-wise meta-atom interpolation workflow
  - `474fc2f` merge: pull request #9 into feature branch
  - `18a4c90` merge: main into feature branch
  - `1299712` chore: local-only pre-commit hooks on branch
  - `7bf3628` feat: add device/objective options and operator-based losses to lens example
- Working tree (not committed on this branch):
  - modified: `docs/DEVELOPMENT_WORKFLOW.md`
  - modified: `docs/PROJECT_STATUS.md`
  - modified: `docs/HANDOFF.md`
  - untracked: `.pre-commit-config.yaml`, `artifacts/`
  - untracked: `src/fouriax.egg-info/`

## Completed
- Core optics composition stack is stable (`OpticalModule`, mask layers, propagation layers, sensors).
- Propagation backends (`ASMPropagator`, `RSPropagator`, `AutoPropagator`) are implemented and integrated.
- Meta-atom LUT interpolation path and optimization example exist with targeted test coverage.
- Plotting and sensor-aware module inspection utilities are available.

## Open Work
- Previous handoff tasks about ONN/planning cleanup are intentionally deprioritized for this cycle.
- Primary focus is now k-space layer planning and implementation:
  - diagonal k-space phase and amplitude operators
  - k-space propagators for uniform slab layers
  - effective angle-dependent specular sheet operators (`T00(kx, ky)`)
- Need explicit NA policy architecture decision:
  - explicit stop/pupil layers vs propagator-level NA limits
  - consistent behavior across `ASM`, `KSpace`, and `RS` paths
- Need explicit hybrid-domain contract definition:
  - where domain conversion is allowed/expected
  - sensor/module boundary behavior (`kspace` input handling)

## Key Files
- `src/fouriax/optics/layers.py`: existing optical layer interfaces; likely insertion point for new k-space layer classes.
- `src/fouriax/optics/propagation.py`: ASM/RS operators and spectral propagation kernels to reuse for slab operators.
- `src/fouriax/optics/field.py`: `Field` and `Spectrum` abstractions for domain-specific operator application.
- `src/fouriax/optics/__init__.py`: public export surface for new k-space APIs.
- `tests/test_optical_module.py`: composition/integration tests to extend for mixed layer stacks.
- `tests/test_propagation.py` (or new `tests/test_kspace_layers.py`): target location for unit tests of diagonal operators.

## Runbook
- Full tests:
```bash
PYTHONPATH=src pytest -q
```
- Optical module tests:
```bash
PYTHONPATH=src pytest -q tests/test_optical_module.py
```
- Lens optimization test:
```bash
PYTHONPATH=src pytest -q tests/test_lens_optimization.py
```
- Lens optimization script:
```bash
MPLBACKEND=Agg PYTHONPATH=src python scripts/lens_optimization_example.py
```
- ONN script:
```bash
MPLBACKEND=Agg PYTHONPATH=src python scripts/onn_mnist_example.py
```
- Meta-atom tests:
```bash
PYTHONPATH=src pytest -q tests/test_meta_atoms.py
```
- Meta-atom script:
```bash
MPLBACKEND=Agg PYTHONPATH=src python scripts/metaatom_optimization_example.py
```
- Propagation-focused tests (recommended during k-space work):
```bash
PYTHONPATH=src pytest -q tests/test_propagation.py tests/test_optical_module.py
```

## Next Tasks
1. Finalize NA policy architecture for hybrid optics.
Done when: NA is represented consistently (layer-based and/or propagator-based), behavior is documented, and parity tests cover at least one `ASM` and one non-`ASM` path.

2. Stabilize hybrid-domain API contract.
Done when: module/sensor domain expectations are explicit, conversion behavior is predictable, and tests cover mixed spatial+k pipelines without ambiguous transitions.

3. Keep 4f parity validation scripts aligned with chosen API contract.
Done when: physical 4f vs k-surrogate vs raw FFT comparison scripts clearly separate profile-parity and throughput-parity checks and produce reproducible outputs in `artifacts/`.

## Risks / Caveats
- JAX tracer safety: avoid Python scalar conversion (`float/int/bool`) inside differentiable paths.
- k-space branch-cut and sign conventions for `kz` can silently introduce phase errors.
- Effective-sheet validity requires subwavelength periodicity and specular-only operation; violations are model errors, not numerical bugs.
- Large spectral grids and multilayer stacks can increase memory/runtime quickly.
- NA behavior may remain confusing if architecture decisions are deferred while adding more layer types.
- Silent domain conversion at module boundaries can hide performance costs and complicate debugging.

## Fresh Codex Prompt
Use this prompt in a fresh instance:

```text
Work in /Users/liam/fouriax on branch feat/metaatom-grid-optimization.
First read docs/HANDOFF.md and docs/PROJECT_STATUS.md.
Primary goal: implement k-space diagonal operators (phase/amplitude), uniform-slab k-space propagators, and effective specular sheet operators.
Immediately resolve NA policy architecture and hybrid-domain contract boundaries before expanding new layer types.
Run PYTHONPATH=src pytest -q tests/test_optical_module.py tests/test_propagation_selection.py tests/test_field_domain.py tests/test_na_schedule.py before finishing.
Do not modify unrelated files (docs/DEVELOPMENT_WORKFLOW.md, .pre-commit-config.yaml, artifacts/) unless explicitly requested.
```
