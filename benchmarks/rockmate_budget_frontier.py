"""Exploratory intermediate-budget follow-up to the frozen compatibility case."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BUDGETS_MIB = (32, 48)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-mib", type=int, choices=BUDGETS_MIB)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.budget_mib is not None:
        import rockmate
        import rockmate_compatibility_probe as fixture
        original_factory = rockmate.from_config
        summary = {}

        def factory(*values, **settings):
            planned = original_factory(*values, **settings)
            if planned.op_sched is not None:
                counts = Counter(op.name for op in planned.op_sched.op_list
                                 if type(op).__name__ == "ComputeOp" and not op.disabled and getattr(op.target, "is_fwd", False))
                summary.update({"enabled_forward_operation_counts": dict(counts),
                                "planned_repeated_forward_count": sum(max(value - 1, 0) for value in counts.values()),
                                "reported_schedule_peak_bytes": float(planned.op_sched.peak_mem)})
            return planned

        rockmate.from_config = factory
        fixture.CONFIG["budget_bytes"] = args.budget_mib * 1024**2
        fixture.worker("rotor_config", args.output)
        record = json.loads(args.output.read_text(encoding="utf-8"))
        record["schedule_summary"] = summary
        for key in ("error_message", "traceback"):
            if key in record:
                record[key] = record[key].replace(str(ROOT), "<atlas>").replace(str(Path.home()), "<profile>")
        args.output.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        return
    directory = ROOT / "artifacts/rockmate-budget-frontier"
    directory.mkdir(parents=True, exist_ok=True)
    interpreter = ROOT / ".tools/planner-env/Scripts/python.exe"
    source = ROOT / ".tools/rockmate-review"
    env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
    env.update({"PATH": str(interpreter.parent) + os.pathsep + str(Path(os.environ["SystemRoot"]) / "System32"),
                "PYTHONPATH": str(source / "rockmate/src") + os.pathsep + str(source / "rkgb/src"),
                "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
                "TEMP": str(directory), "TMP": str(directory), "TORCH_HOME": str(directory / "torch-cache"),
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    rows = []
    for budget in BUDGETS_MIB:
        output = directory / (str(budget) + ".json")
        started = time.perf_counter()
        with (directory / (str(budget) + ".stdout.txt")).open("w", encoding="utf-8") as stdout, (directory / (str(budget) + ".stderr.txt")).open("w", encoding="utf-8") as stderr:
            subprocess.run([str(interpreter), str(Path(__file__).resolve()), "--budget-mib", str(budget), "--output", str(output)],
                           cwd=directory, env=env, stdout=stdout, stderr=stderr, timeout=60, check=True)
        row = json.loads(output.read_text(encoding="utf-8"))
        row["outer_wall_seconds"] = time.perf_counter() - started
        row["budget_mib"] = budget
        rows.append(row)
        print(budget, row["status"], row.get("schedule_summary", {}).get("planned_repeated_forward_count"), flush=True)
    result = {"captured_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "fixture_source_sha256": hashlib.sha256((ROOT / "benchmarks/rockmate_compatibility_probe.py").read_bytes()).hexdigest(),
              "budgets_mib": BUDGETS_MIB, "trials": rows,
              "scope": "Exploratory fixed intermediate budgets after coarse pilot and static schedule inspection. Original fixture/settings reused; no random workload sample, held-out policy comparison or scientific equivalence."}
    (ROOT / "docs/research/experiments/rockmate-budget-frontier.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
