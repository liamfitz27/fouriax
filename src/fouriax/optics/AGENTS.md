# Optics Core Guide

Changes in this tree should preserve the explicit optics model described in `docs/ARCHITECTURE.md`.

## Invariants

- `Field` carries domain, grid, spectrum, and polarization state explicitly.
- Spatial-domain layers must reject k-space input; k-space layers must reject spatial input.
- Jones support should broadcast or mix channels intentionally, never by accidental reshape.
- Propagation method selection belongs in planning utilities, not hidden inside unrelated layers.
- Keep validation errors specific; many tests assert exact failure modes.

## Working style

- Start from the narrowest affected tests.
- If changing `model.py`, `layers.py`, `propagation.py`, `sensors.py`, or `meta_atoms.py`, read the corresponding tests first.
- Prefer small, local changes over introducing new abstractions unless there is repeated pressure in at least two modules.

## Useful test targets

- `tests/test_field_domain.py`
- `tests/test_optical_module.py`
- `tests/test_propagation_selection.py`
- `tests/test_jones_polarization.py`
- `tests/test_k_layers.py`
- `tests/test_meta_atoms.py`
- `tests/test_camera_sensor.py`
- `tests/test_na_schedule.py`
