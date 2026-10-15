"""Isolate framework precision/export behavior without importing a planner."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODES = ["default", "backend_ieee", "operators_ieee", "legacy_only"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=MODES)
    args = parser.parse_args()
    if args.worker:
        import torch
        mode = args.worker
        if mode in ("backend_ieee", "operators_ieee"):
            torch.backends.fp32_precision = "ieee"
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.fp32_precision = "ieee"
        if mode == "operators_ieee":
            torch.backends.cudnn.conv.fp32_precision = "ieee"
            torch.backends.cudnn.rnn.fp32_precision = "ieee"
        if mode == "legacy_only":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        row = {"mode": mode, "torch": torch.__version__}
        for name, getter in {
            "conv_precision": lambda: torch.backends.cudnn.conv.fp32_precision,
            "rnn_precision": lambda: torch.backends.cudnn.rnn.fp32_precision,
            "backend_precision": lambda: torch.backends.cudnn.fp32_precision,
            "legacy_cudnn_query": lambda: torch.backends.cudnn.allow_tf32,
        }.items():
            try: row[name] = getter()
            except Exception as error: row[name] = {"error": type(error).__name__, "message": str(error)}
        try:
            program = torch.export.export(torch.nn.Linear(2, 1), (torch.ones(1, 2),))
            row["export"] = "completed"
            row["graph_node_count"] = len(list(program.graph.nodes))
        except Exception as error:
            row["export"] = "error"
            row["error_type"] = type(error).__name__
            row["error_message"] = str(error)
        print(json.dumps(row, sort_keys=True), flush=True)
        return
    directory = ROOT / "artifacts/precision-export-control"
    directory.mkdir(parents=True, exist_ok=True)
    interpreter = ROOT / ".tools/planner-env/Scripts/python.exe"
    env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
    env.update({"PATH": str(interpreter.parent) + os.pathsep + str(Path(os.environ["SystemRoot"]) / "System32"),
                "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
                "TEMP": str(directory), "TMP": str(directory), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    rows = []
    for mode in MODES:
        process = subprocess.run([str(interpreter), str(Path(__file__).resolve()), "--worker", mode], cwd=directory,
                                 env=env, capture_output=True, text=True, timeout=30, check=False)
        (directory / (mode + ".stderr.txt")).write_text(process.stderr, encoding="utf-8")
        row = json.loads(process.stdout) if process.returncode == 0 else {"mode": mode, "status": "worker_failure", "returncode": process.returncode}
        row["stderr_nonempty"] = bool(process.stderr.strip())
        rows.append(row)
        print(mode, row.get("export", row.get("status")), flush=True)
    result = {"captured_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "trials": rows,
              "scope": "Exploratory four fresh-process CPU torch.export controls in installed CUDA binary stack; no Rockmate import, GPU training or scientific equivalence claim."}
    (ROOT / "docs/research/experiments/precision-export-control.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
