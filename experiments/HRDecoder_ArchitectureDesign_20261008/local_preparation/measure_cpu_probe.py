"""Measure the existing v3 small CPU probe without changing historical evidence.

This covers synthetic 128x128 tensors and the private-path prototype only.
It does not cover full HRNet integration, 2048 training, AMP, or DDP.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import psutil


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "verify_v3_modules.py"


def main() -> None:
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location("v3_resource_probe", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    report = module.run_checks()
    (HERE / "cpu_module_checks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    memory = psutil.Process().memory_info()
    result = {
        "status": report["status"],
        "elapsed_seconds": time.perf_counter() - started,
        "memory_bytes": memory._asdict(),
        "peak_working_set_bytes": getattr(memory, "peak_wset", None),
        "new_parameter_count": report["parameters"]["sum_new_parameters"],
        "python": sys.executable,
        "scope": report["scope"],
        "original_evidence_modified": False,
    }
    (HERE / "cpu_resource_measurement.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
