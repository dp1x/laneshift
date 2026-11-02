"""Report the local inference environment before a LANESHIFT host run.

This is a diagnostic, not a benchmark. It answers the questions that decide
whether a one-click run can work: is Python native ARM64 or running under
emulation, which ONNX Runtime build and execution providers are present, are
the model files where the launcher expects them, and is Godot available?
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIBLING_MODEL_DIR = PROJECT_ROOT.parent / "LANESHIFT2D-Kev-model"
DEFAULT_MODEL = SIBLING_MODEL_DIR / "kev07b_fp16_L256.onnx"
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = Path(r"C:\LANESHIFT2D-Kev-model\kev07b_fp16_L256.onnx")
DEFAULT_TOKENIZER = SIBLING_MODEL_DIR / "tokenizer.json"
if not DEFAULT_TOKENIZER.exists():
    DEFAULT_TOKENIZER = Path(r"C:\LANESHIFT2D-Kev-model\tokenizer.json")


def run(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output or None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--strict", action="store_true", help="exit non-zero when a required check fails")
    args = parser.parse_args(argv)

    processor_identifier = os.environ.get("PROCESSOR_IDENTIFIER", "")
    machine = platform.machine()
    arm_hardware = "ARM" in processor_identifier.upper() or "QUALCOMM" in processor_identifier.upper()
    emulated = machine.upper() in {"AMD64", "X86"} and arm_hardware

    providers: list[str] = []
    ort_version = None
    ort_error = None
    try:
        import onnxruntime as ort
        ort_version = ort.__version__
        providers = ort.get_available_providers()
    except Exception as exc:  # pragma: no cover - diagnostic path
        ort_error = f"{type(exc).__name__}: {exc}"

    model = args.model
    tokenizer = args.tokenizer
    data = model.with_name("kev07b_fp16.data")
    godot = shutil.which("godot") or shutil.which("godot4")
    godot_version = run([godot, "--version"]) if godot else None

    checks = {
        "python_native_arm64": machine.upper() == "ARM64",
        "onnxruntime_imported": ort_error is None,
        "cpu_provider_available": "CPUExecutionProvider" in providers,
        "model_present": model.exists(),
        "external_data_present": data.exists(),
        "tokenizer_present": tokenizer.exists(),
        "godot_on_path": godot is not None,
    }
    warnings = []
    if emulated:
        warnings.append(
            "Python is x64 under ARM64 emulation. Inference will be substantially slower; "
            "install native Windows ARM64 Python and recreate the virtual environment."
        )
    if ort_error:
        warnings.append(f"onnxruntime import failed: {ort_error}")
    elif "QNNExecutionProvider" in providers:
        warnings.append("QNN provider is available. Test it with benchmark_kev.py --provider qnn; "
                        "operator coverage for this graph is not assumed.")
    if not model.exists():
        warnings.append("ONNX model missing. Set LANESHIFT_ONNX_MODEL or place it at the default path.")
    if not data.exists():
        warnings.append("External weights kev07b_fp16.data missing next to the ONNX graph.")
    if not tokenizer.exists():
        warnings.append("Tokenizer missing. Set LANESHIFT_ONNX_TOKENIZER.")

    report = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": machine,
            "processor_identifier": processor_identifier,
            "python_emulated_on_arm64": emulated,
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "machine": machine,
            "implementation": platform.python_implementation(),
            "cpu_count": os.cpu_count(),
        },
        "onnxruntime": {
            "version": ort_version,
            "available_providers": providers,
            "error": ort_error,
        },
        "model": {
            "path": str(model),
            "exists": model.exists(),
            "size_bytes": model.stat().st_size if model.exists() else None,
            "external_data": str(data),
            "external_data_exists": data.exists(),
            "external_data_size_bytes": data.stat().st_size if data.exists() else None,
            "tokenizer": str(tokenizer),
            "tokenizer_exists": tokenizer.exists(),
        },
        "godot": {"path": godot, "version": godot_version},
        "checks": checks,
        "warnings": warnings,
    }
    print(json.dumps(report, indent=2))
    if args.strict and not all(checks.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
