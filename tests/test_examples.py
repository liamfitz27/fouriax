import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path("examples/scripts")
LOGO_PATH = Path("data/logo.jpg")


def run_script(script_name: str, args: list[str]):
    script_path = EXAMPLES_DIR / script_name
    assert script_path.exists(), f"Script {script_path} not found"

    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
    assert result.returncode == 0, f"Script {script_name} failed"


def test_lens_optimization():
    run_script("lens_optimization.py", ["--grid-n", "32", "--steps", "2", "--no-plot"])


def test_basic_propagation():
    run_script("basic_propagation.py", ["--grid-n", "64", "--no-plot"])


def test_4f_correlator():
    run_script("4f_correlator.py", ["--grid-n", "32", "--no-plot"])


def test_4f_edge_optimization():
    run_script(
        "4f_edge_optimization.py",
        [
            "--grid-n",
            "32",
            "--epochs",
            "2",
            "--n-train-scenes",
            "16",
            "--n-test-scenes",
            "8",
            "--batch-size",
            "4",
            "--no-plot",
        ],
    )


def test_hologram_coherent_logo():
    run_script(
        "hologram_coherent_logo.py",
        [
            "--image-path",
            str(LOGO_PATH),
            "--nx",
            "32",
            "--ny",
            "32",
            "--steps",
            "2",
            "--no-plot",
        ],
    )


def test_holography_polarized_dual():
    run_script(
        "holography_polarized_dual.py",
        [
            "--image-path",
            str(LOGO_PATH),
            "--nx",
            "32",
            "--ny",
            "32",
            "--steps",
            "2",
            "--no-plot",
        ],
    )


def test_incoherent_camera():
    run_script(
        "incoherent_camera.py",
        ["--grid-n", "32", "--sensor-size-px", "16", "--no-plot"],
    )


def test_metaatom_optimization():
    run_script("metaatom_optimization.py", ["--grid-n", "5", "--steps", "2", "--no-plot"])


def test_onn_mnist():
    run_script(
        "onn_mnist.py",
        ["--epochs", "1", "--train-samples", "10", "--test-samples", "10", "--no-plot"],
    )


@pytest.mark.skip(reason="Sensitivity analysis example currently exceeds CI memory limits")
def test_sensitivity_analysis():
    run_script("sensitivity_analysis.py", ["--grid-n", "32", "--no-plot"])


def test_spectral_filter_optimization():
    run_script(
        "spectral_filter_optimization.py",
        [
            "--proxy-epochs",
            "1",
            "--recon-epochs",
            "1",
            "--proxy-batch-size",
            "16",
            "--recon-batch-size",
            "16",
            "--no-plot",
        ],
    )
