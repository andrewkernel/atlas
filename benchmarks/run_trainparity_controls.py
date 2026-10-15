"""Bounded launcher for reviewed TrainParity source and original CPU cases.

No external checker import occurs in the outer launcher. Child environments
are constructed from an explicit allowlist; this is not OS isolation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/trainparity-controls.json"
MANIFEST = ROOT / "docs/research/competitors/trainparity-execution-review.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def internal(name: str, output: Path) -> None:
    import torch
    import numpy
    import trainparity
    from trainparity.api import AccumulationExecutionPlan, ToleranceComparison, check_accumulation
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    control = next(item for item in config["controls"] if item["name"] == name)
    started = time.perf_counter()
    result = check_accumulation(
        "trainparity_cases:" + control["case"],
        candidate=AccumulationExecutionPlan(**control["plan"]),
        comparison=ToleranceComparison(rtol=config["rtol"], atol=config["atol"]),
        device="cpu", seed=config["seed"], cwd=ROOT,
        timeout=config["worker_timeout_seconds"], temporary_root=output / "temporary",
        report_path=output / (name + ".json"),
    )
    payload = {
        "name": name, "predicted_outcome": control["predicted_outcome"],
        "wall_seconds": time.perf_counter() - started, "result": result.to_dict(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": numpy.__version__, "trainparity": trainparity.__version__, "device": "cpu"},
    }
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.internal:
        assert args.output is not None
        internal(args.internal, args.output)
        return
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    review = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source = ROOT / ".tools/trainparity-reviewed"
    assert review["revision"] == config["revision"]
    assert review["authorized_scope"] == "reviewed_import_graph_and_accumulation_on_original_cpu_cases"
    for entry in review["files"]:
        assert sha(source / entry["path"]) == entry["sha256"], entry["path"]
    interpreter = ROOT / ".tools/checker-env/Scripts/python.exe"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "artifacts/trainparity-controls" / stamp
    output.mkdir(parents=True)
    env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
    env.update({
        "PATH": str(interpreter.parent) + os.pathsep + str(Path(os.environ["SystemRoot"]) / "System32"),
        "PYTHONPATH": str(source / "src") + os.pathsep + str(ROOT / "benchmarks"),
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "TEMP": str(output), "TMP": str(output),
    })
    rows = []
    for control in config["controls"]:
        started = time.perf_counter()
        completed = subprocess.run(
            [str(interpreter), str(Path(__file__).resolve()), "--internal", control["name"], "--output", str(output)],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=110, check=False,
        )
        (output / (control["name"] + ".stdout.txt")).write_text(completed.stdout, encoding="utf-8")
        (output / (control["name"] + ".stderr.txt")).write_text(completed.stderr, encoding="utf-8")
        if completed.returncode == 0:
            row = json.loads(completed.stdout.strip().splitlines()[-1])
        else:
            row = {"name": control["name"], "predicted_outcome": control["predicted_outcome"],
                   "launcher_failure": True, "returncode": completed.returncode,
                   "stderr_excerpt": completed.stderr[-1500:]}
        row["outer_wall_seconds"] = time.perf_counter() - started
        row["stderr_nonempty"] = bool(completed.stderr.strip())
        rows.append(row)
        print(control["name"], row.get("result", {}).get("outcome", "launcher_failure"), flush=True)
    record = {
        "started_utc": stamp, "source_sha256": sha(Path(__file__)),
        "case_source_sha256": sha(ROOT / "benchmarks/trainparity_cases.py"),
        "config_sha256": sha(CONFIG), "review_sha256": sha(MANIFEST),
        "config": config, "trials": rows,
        "claims": "Investigator-authored bounded controls; not an Atlas policy comparison, independent benchmark, full-training equivalence, or cross-environment timing comparison.",
        "environment_keys": sorted(env),
    }
    destination = ROOT / "docs/research/experiments/trainparity-controls.json"
    destination.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
