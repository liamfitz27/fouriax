"""Sweep one training hyperparameter at a time for `spectral_filter_optimization.py`.

This script treats the current constants in `experiments/spectral_filter_optimization.py`
as the baseline configuration, applies a reduced benchmark override for faster iteration,
then evaluates one candidate value at a time for a single parameter.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TARGET = Path("experiments/spectral_filter_optimization.py")
DEFAULT_SUMMARY = Path("experiments/artifacts/spectral_filter_summary.json")


@dataclass(frozen=True)
class RunResult:
    label: str
    overrides: dict[str, Any]
    best_recon_val_loss: float
    final_val_mse: float
    train_mse: float


def _parse_value(raw: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw


def _parse_assignment(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"expected NAME=VALUE, got {raw!r}")
    name, value = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"invalid assignment {raw!r}")
    return name, _parse_value(value.strip())


def _load_module(module_path: Path, *, run_id: str):
    spec = importlib.util.spec_from_file_location(f"spectral_filter_sweep_{run_id}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_trial(
    *,
    module_path: Path,
    summary_path: Path,
    label: str,
    benchmark_overrides: dict[str, Any],
    param_overrides: dict[str, Any],
    quiet: bool,
    mplconfigdir: str | None,
) -> RunResult:
    if mplconfigdir:
        os.environ["MPLCONFIGDIR"] = mplconfigdir

    module = _load_module(module_path, run_id=label.replace(" ", "_"))

    # Silence the built-in dataset optimizer reporter for sweeps.
    import fouriax.optim.optim as optim_impl

    optim_impl._uniform_dataset_report = lambda payload: None  # type: ignore[assignment]

    for name, value in benchmark_overrides.items():
        setattr(module, name, value)
    for name, value in param_overrides.items():
        setattr(module, name, value)

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout if quiet else sys.stdout):
        module.main()

    if not summary_path.exists():
        raise FileNotFoundError(f"expected summary file: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return RunResult(
        label=label,
        overrides=param_overrides,
        best_recon_val_loss=float(summary["best_recon_val_loss"]),
        final_val_mse=float(summary["final_val_mse"]),
        train_mse=float(summary["train_mse"]),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep one hyperparameter at a time for spectral_filter_optimization.py.",
    )
    parser.add_argument(
        "param",
        help="Name of the module-level parameter to sweep, e.g. RECON_FILTER_LR",
    )
    parser.add_argument(
        "values",
        nargs="+",
        help="Candidate values for the parameter. Values are parsed with ast.literal_eval.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="Path to experiments/spectral_filter_optimization.py",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Path to the summary JSON written by the target script.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=8192,
        help="Reduced benchmark train sample count override.",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=1024,
        help="Reduced benchmark validation sample count override.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Validation fraction override for the reduced benchmark.",
    )
    parser.add_argument(
        "--fixed",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Additional fixed overrides applied to every run.",
    )
    parser.add_argument(
        "--metric",
        choices=("final_val_mse", "best_recon_val_loss"),
        default="final_val_mse",
        help="Metric used to rank candidates.",
    )
    parser.add_argument(
        "--mplconfigdir",
        default="/tmp/.mpl",
        help="Writable MPLCONFIGDIR for subprocess-free script execution.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show the target script stdout instead of suppressing it.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    benchmark_overrides: dict[str, Any] = {
        "TRAIN_SAMPLES": args.train_samples,
        "VAL_SAMPLES": args.val_samples,
        "VAL_FRACTION": args.val_fraction,
    }
    for raw in args.fixed:
        name, value = _parse_assignment(raw)
        benchmark_overrides[name] = value

    baseline = run_trial(
        module_path=args.target,
        summary_path=args.summary,
        label="baseline",
        benchmark_overrides=benchmark_overrides,
        param_overrides={},
        quiet=not args.verbose,
        mplconfigdir=args.mplconfigdir,
    )

    results = [baseline]
    for raw_value in args.values:
        value = _parse_value(raw_value)
        result = run_trial(
            module_path=args.target,
            summary_path=args.summary,
            label=f"{args.param}={value!r}",
            benchmark_overrides=benchmark_overrides,
            param_overrides={args.param: value},
            quiet=not args.verbose,
            mplconfigdir=args.mplconfigdir,
        )
        results.append(result)

    metric_name = args.metric
    ranked = sorted(results, key=lambda r: getattr(r, metric_name))
    best = ranked[0]

    print(f"Baseline {metric_name}: {getattr(baseline, metric_name):.9f}")
    print(f"Swept parameter: {args.param}")
    print("Results (lower is better):")
    for result in results:
        metric_value = getattr(result, metric_name)
        delta = metric_value - getattr(baseline, metric_name)
        print(
            f"  {result.label:>24}  "
            f"{metric_name}={metric_value:.9f}  "
            f"delta={delta:+.9f}"
        )

    if best.label == baseline.label:
        print("No candidate improved on the baseline.")
    else:
        print(
            "Best candidate: "
            f"{best.label} with {metric_name}={getattr(best, metric_name):.9f}"
        )


if __name__ == "__main__":
    main()
