"""Observe planner-construction state effects and a scoped Torch RNG guard."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        import random
        import numpy as np
        import torch
        import rockmate
        import rockmate_compatibility_probe as fixture
        original_factory = rockmate.from_config
        observations = {}

        def snapshot(model):
            def digest(value):
                return hashlib.sha256(value).hexdigest()
            numpy_state = np.random.get_state()
            numpy_value = [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]
            return {
                "torch_cpu_rng": digest(torch.get_rng_state().numpy().tobytes()),
                "torch_cuda_rng": [digest(value.numpy().tobytes()) for value in torch.cuda.get_rng_state_all()],
                "python_rng": digest(repr(random.getstate()).encode()),
                "numpy_rng": digest(json.dumps(numpy_value, sort_keys=True).encode()),
                "parameter": {name: digest(value.detach().cpu().numpy().tobytes()) for name, value in model.named_parameters()},
                "gradient": {name: None if value.grad is None else digest(value.grad.detach().cpu().numpy().tobytes()) for name, value in model.named_parameters()},
                "buffer": {name: digest(value.detach().cpu().numpy().tobytes()) for name, value in model.named_buffers()},
                "training_mode": model.training,
            }

        def factory(model, *values, **settings):
            observations["before"] = snapshot(model)
            with torch.random.fork_rng(devices=[0]):
                planned = original_factory(model, *values, **settings)
                observations["inside_guard_after_construction"] = snapshot(model)
            observations["after_torch_guard"] = snapshot(model)
            return planned

        rockmate.from_config = factory
        fixture.CONFIG["budget_bytes"] = 64 * 1024**2
        fixture.worker("rotor_config", args.output)
        result = json.loads(args.output.read_text(encoding="utf-8"))
        for key in ("error_message", "traceback"):
            if key in result:
                result[key] = result[key].replace(str(ROOT), "<atlas>").replace(str(Path.home()), "<profile>")
        result["state_observations"] = observations
        if "inside_guard_after_construction" in observations:
            before = observations["before"]
            result["changed_inside_guard"] = {key: before[key] != observations["inside_guard_after_construction"][key] for key in before}
            result["restored_or_unchanged_after_torch_guard"] = {key: before[key] == observations["after_torch_guard"][key] for key in before}
        result["source_sha256"] = sha(Path(__file__))
        result["fixture_source_sha256"] = sha(ROOT / "benchmarks/rockmate_compatibility_probe.py")
        result["review_sha256"] = sha(ROOT / "docs/research/competitors/rockmate-execution-review.json")
        result["captured_utc"] = datetime.now(timezone.utc).isoformat()
        result["scope"] = "One pure-fixture construction observation and Torch global RNG guard. Hashes identify state but contain no recovered random data. No custom generators, opaque state, buffers with mutations, untrusted-code rollback or general transaction guarantee."
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        return
    review_path = ROOT / "docs/research/competitors/rockmate-execution-review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    source = ROOT / ".tools/rockmate-review"
    for item in review["files"]:
        assert sha(source / item["path"]) == item["sha256"], item["path"]
    directory = ROOT / "artifacts/rockmate-state-control"
    directory.mkdir(parents=True, exist_ok=True)
    interpreter = ROOT / ".tools/planner-env/Scripts/python.exe"
    env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
    env.update({"PATH": str(interpreter.parent) + os.pathsep + str(Path(os.environ["SystemRoot"]) / "System32"),
                "PYTHONPATH": str(source / "rockmate/src") + os.pathsep + str(source / "rkgb/src"),
                "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
                "TEMP": str(directory), "TMP": str(directory), "TORCH_HOME": str(directory / "torch-cache"),
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    output = ROOT / "docs/research/experiments/rockmate-state-control.json"
    with (directory / "stdout.txt").open("w", encoding="utf-8") as stdout, (directory / "stderr.txt").open("w", encoding="utf-8") as stderr:
        subprocess.run([str(interpreter), str(Path(__file__).resolve()), "--worker", "--output", str(output)],
                       cwd=directory, env=env, stdout=stdout, stderr=stderr, timeout=60, check=True)


if __name__ == "__main__":
    main()
