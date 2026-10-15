"""Reviewed-source GPU compatibility probe, not an Atlas planner benchmark."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "docs/research/competitors/rockmate-execution-review.json"
CONFIG = {
    "seed": 1801, "blocks": 4, "width": 64, "hidden": 128,
    "input_shape": [128, 64, 64], "budget_bytes": 16 * 1024**2,
    "updates": 8, "warmup_updates": 3, "lr": 0.01, "momentum": 0.9,
    "modes": ["rotor_config", "rockmate_config"],
    "timeout_seconds": 180, "rtol": 1e-5, "atol": 1e-6,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(mode: str, output: Path) -> None:
    import copy
    import gc
    import statistics
    import torch
    from torch import nn

    phase = "environment"
    result: dict = {"mode": mode, "status": "error", "config": CONFIG}
    try:
        torch.set_num_threads(1)
        torch.manual_seed(CONFIG["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        result["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
                                 "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
        phase = "original_fixture"
        blocks = []
        for _ in range(CONFIG["blocks"]):
            blocks.extend([nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()])
        template = nn.Sequential(*blocks)
        initial = {name: value.detach().clone() for name, value in template.state_dict().items()}
        digest = hashlib.sha256()
        for name, value in sorted(initial.items()):
            digest.update(name.encode()); digest.update(value.numpy().tobytes())
        result["initial_state_sha256"] = digest.hexdigest()
        x = torch.randn(CONFIG["input_shape"], device="cuda")
        target = torch.randn_like(x)
        result["input_target_sha256"] = hashlib.sha256(x.cpu().numpy().tobytes() + target.cpu().numpy().tobytes()).hexdigest()

        def train(module, original, *, wrapped: bool) -> tuple[dict, dict]:
            optimizer = torch.optim.SGD(original.parameters(), lr=CONFIG["lr"], momentum=CONFIG["momentum"])
            measurements = []
            losses = []
            first = {}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for index in range(CONFIG["updates"]):
                if wrapped:
                    module.zero_grad(set_to_none=False)
                else:
                    optimizer.zero_grad(set_to_none=False)
                torch.cuda.synchronize()
                started = time.perf_counter()
                prediction = module(x)
                if isinstance(prediction, tuple):
                    if len(prediction) != 1:
                        raise ValueError("Unexpected output structure")
                    prediction = prediction[0]
                loss = (prediction - target).square().mean()
                loss.backward()
                if index == 0:
                    first["gradient"] = {name: None if p.grad is None else p.grad.detach().cpu().clone() for name, p in original.named_parameters()}
                optimizer.step()
                torch.cuda.synchronize()
                measurements.append(time.perf_counter() - started)
                losses.append(float(loss.detach().cpu()))
                if index == 0:
                    first["parameter"] = {name: p.detach().cpu().clone() for name, p in original.named_parameters()}
                    first["momentum"] = {name: optimizer.state[p].get("momentum_buffer", torch.empty(0)).detach().cpu().clone() for name, p in original.named_parameters()}
                del prediction, loss
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            final = {name: p.detach().cpu().clone() for name, p in original.named_parameters()}
            observation = {"first": first, "final": final}
            record = {"losses": losses, "allocated_peak_bytes": allocated, "reserved_peak_bytes": reserved,
                      "update_seconds": measurements, "median_later_update_seconds": statistics.median(measurements[CONFIG["warmup_updates"]:]),
                      "completed_updates": CONFIG["updates"]}
            return record, observation

        phase = "native_reference"
        native = copy.deepcopy(template).cuda()
        reference_record, reference = train(native, native, wrapped=False)
        result["native_reference"] = reference_record
        del native
        gc.collect(); torch.cuda.empty_cache()

        phase = "reviewed_planner_import"
        import rockmate
        model = copy.deepcopy(template).cuda()
        phase = "planner_construction"
        started = time.perf_counter()
        common = {"gpu_optim": None, "cpu_optim": None, "minor_offload_size": 0}
        if mode == "rotor_config":
            config = rockmate.generate_config("rotor")
            config.solver.top.rotor.force_python = True
            config.partitioner.sequence.sub_partitioner = "base"
            candidate = rockmate.from_config(model, x, CONFIG["budget_bytes"], config=config, **common)
        elif mode == "rockmate_config":
            config = rockmate.generate_config("rockmate")
            config.solver.top.rotor.force_python = True
            config.solver.bottom.hilp.time_limit = 2
            config.solver.bottom.hilp.nb_bdg_save = 2
            config.solver.bottom.hilp.nb_bdg_peak = 2
            config.solver.bottom.hilp.nb_total_sched = 5
            candidate = rockmate.from_config(model, x, CONFIG["budget_bytes"], config=config, **common)
        else:
            raise ValueError(mode)
        torch.cuda.synchronize()
        result["construction_seconds"] = time.perf_counter() - started
        result["schedule_present"] = candidate.op_sched is not None
        result["minor_parameter_node_count"] = len(candidate.minor_param_nodes)
        result["constructor_parameter_max_abs_change"] = max(float((p.detach().cpu() - initial[name]).abs().max()) for name, p in model.named_parameters())
        model.load_state_dict(initial)
        phase = "candidate_training"
        candidate_record, observed = train(candidate, model, wrapped=True)
        result["candidate"] = candidate_record
        checks = {}
        for section in ("gradient", "parameter", "momentum"):
            left = reference["first"][section]
            right = observed["first"][section]
            errors = []
            passed = set(left) == set(right)
            for name in left:
                a, b = left[name], right.get(name)
                if a is None or b is None:
                    passed = passed and a is None and b is None
                else:
                    passed = passed and a.shape == b.shape and torch.allclose(a, b, rtol=CONFIG["rtol"], atol=CONFIG["atol"])
                    if a.shape == b.shape and a.numel():
                        errors.append(float((a-b).abs().max()))
            checks["first_" + section] = {"within_declared_tolerance": bool(passed), "max_abs_error": max(errors, default=0.)}
        errors = [float((reference["final"][name] - value).abs().max()) for name, value in observed["final"].items()]
        checks["final_parameter"] = {"max_abs_error": max(errors), "within_declared_tolerance": all(torch.allclose(reference["final"][name], value, rtol=CONFIG["rtol"], atol=CONFIG["atol"]) for name, value in observed["final"].items())}
        result["comparisons"] = checks
        result["status"] = "completed"
    except Exception as error:
        result["failed_phase"] = phase
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)[:2000]
        result["traceback"] = traceback.format_exc()[-6000:]
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=CONFIG["modes"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        assert args.output is not None
        worker(args.worker, args.output)
        return
    import psutil
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    assert review["execution_scope"] == "original_sequential_fixture_activation_only_python_solver_paths"
    source = ROOT / ".tools/rockmate-review"
    for item in review["files"]:
        assert sha(source / item["path"]) == item["sha256"], item["path"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "artifacts/rockmate-compatibility" / stamp
    directory.mkdir(parents=True)
    interpreter = ROOT / ".tools/planner-env/Scripts/python.exe"
    env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
    env.update({"PATH": str(interpreter.parent) + os.pathsep + str(Path(os.environ["SystemRoot"]) / "System32"),
                "PYTHONPATH": str(source / "rockmate/src") + os.pathsep + str(source / "rkgb/src"),
                "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
                "TEMP": str(directory), "TMP": str(directory), "TORCH_HOME": str(directory / "torch-cache"),
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    rows = []
    for mode in CONFIG["modes"]:
        destination = directory / (mode + ".json")
        started = time.perf_counter()
        with (directory / (mode + ".stdout.txt")).open("w", encoding="utf-8") as stdout, (directory / (mode + ".stderr.txt")).open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen([str(interpreter), str(Path(__file__).resolve()), "--worker", mode, "--output", str(destination)], cwd=directory, env=env, stdout=stdout, stderr=stderr)
            timed_out = False
            try:
                process.wait(timeout=CONFIG["timeout_seconds"])
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    parent = psutil.Process(process.pid)
                    descendants = parent.children(recursive=True)
                    for child in reversed(descendants):
                        try: child.kill()
                        except psutil.NoSuchProcess: pass
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass
                process.wait(timeout=10)
        if destination.exists():
            row = json.loads(destination.read_text(encoding="utf-8"))
        else:
            row = {"mode": mode, "status": "timeout" if timed_out else "worker_failure", "returncode": process.returncode}
        row["outer_wall_seconds"] = time.perf_counter() - started
        for key in ("error_message", "traceback"):
            if key in row:
                row[key] = row[key].replace(str(ROOT), "<atlas>").replace(str(directory), "<artifact>").replace(str(Path.home()), "<profile>")
        row["stdout_bytes"] = (directory / (mode + ".stdout.txt")).stat().st_size
        row["stderr_bytes"] = (directory / (mode + ".stderr.txt")).stat().st_size
        rows.append(row)
        print(mode, row["status"], row.get("failed_phase", ""), flush=True)
    record = {"started_utc": stamp, "config": CONFIG, "source_sha256": sha(Path(__file__)),
              "review_sha256": sha(REVIEW), "trials": rows, "environment_keys": sorted(env),
              "scope": "Scoped compatibility/finite numerical control. No Atlas policy, whole-training equivalence, useful OOM rescue or general tool-performance claim."}
    (ROOT / "docs/research/experiments/rockmate-compatibility.json").write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
