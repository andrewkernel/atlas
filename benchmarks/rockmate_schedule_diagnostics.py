"""Instrument one completed compatibility case without changing its fixture."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        import torch
        import rockmate
        import rockmate_compatibility_probe as fixture
        stage = "native_reference"
        resets = []
        metadata = {}
        original_reset = torch.cuda.reset_peak_memory_stats
        original_factory = rockmate.from_config

        def reset(device=None):
            resets.append({"stage": stage, "allocated_bytes_before_reset": torch.cuda.memory_allocated(device),
                           "reserved_bytes_before_reset": torch.cuda.memory_reserved(device)})
            return original_reset(device)

        def factory(*values, **settings):
            nonlocal stage
            stage = "planner_construction"
            planned = original_factory(*values, **settings)
            operations = []
            forwards = Counter()
            if planned.op_sched is not None:
                for op in planned.op_sched.op_list:
                    operations.append({"name": op.name, "type": type(op).__name__, "disabled": bool(op.disabled)})
                    if type(op).__name__ == "ComputeOp" and not op.disabled and getattr(op.target, "is_fwd", False):
                        forwards[op.name] += 1
                metadata["reported_schedule_peak_bytes"] = float(planned.op_sched.peak_mem)
                metadata["schedule_with_parameters"] = bool(planned.op_sched.with_parameters)
            metadata["static_operations"] = operations
            metadata["enabled_forward_operation_counts"] = dict(forwards)
            metadata["planned_repeated_forward_count"] = sum(max(count - 1, 0) for count in forwards.values())
            metadata["allocated_after_construction_bytes"] = torch.cuda.memory_allocated()
            stage = "candidate_training"
            return planned

        torch.cuda.reset_peak_memory_stats = reset
        rockmate.from_config = factory
        fixture.CONFIG["budget_bytes"] = 64 * 1024**2
        fixture.worker("rotor_config", args.output)
        result = json.loads(args.output.read_text(encoding="utf-8"))
        for key in ("error_message", "traceback"):
            if key in result:
                result[key] = result[key].replace(str(ROOT), "<atlas>").replace(str(Path.home()), "<profile>")
        result["diagnostics"] = {"reset_observations": resets, **metadata}
        result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["fixture_source_sha256"] = hashlib.sha256(Path(fixture.__file__).read_bytes()).hexdigest()
        result["captured_utc"] = datetime.now(timezone.utc).isoformat()
        result["scope"] = "One instrumented follow-up; schedule counts are planned operations, not kernel traces. Reset wrappers query counters then call the original function. No memory-cause attribution or independent performance replication."
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        return
    directory = ROOT / "artifacts/rockmate-schedule-diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    interpreter = ROOT / ".tools/planner-env/Scripts/python.exe"
    source = ROOT / ".tools/rockmate-review"
    env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
    env.update({"PATH": str(interpreter.parent) + os.pathsep + str(Path(os.environ["SystemRoot"]) / "System32"),
                "PYTHONPATH": str(source / "rockmate/src") + os.pathsep + str(source / "rkgb/src"),
                "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
                "TEMP": str(directory), "TMP": str(directory), "TORCH_HOME": str(directory / "torch-cache"),
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    output = ROOT / "docs/research/experiments/rockmate-schedule-diagnostics.json"
    with (directory / "stdout.txt").open("w", encoding="utf-8") as stdout, (directory / "stderr.txt").open("w", encoding="utf-8") as stderr:
        subprocess.run([str(interpreter), str(Path(__file__).resolve()), "--worker", "--output", str(output)], cwd=directory,
                       env=env, stdout=stdout, stderr=stderr, timeout=60, check=True)


if __name__ == "__main__":
    main()
