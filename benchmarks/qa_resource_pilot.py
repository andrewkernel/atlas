"""Bounded native BERT QA resource segment, with complete attempted-run ledger."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

from board_energy_control import ROOT, child_environment

CONFIG = {"logical_batch": 32, "microbatches": [32, 4], "attention": ["eager", "sdpa"],
          "checkpointing": [False, True], "repetitions": 2, "model_seed": 2501,
          "scientific_rng_seed": 2503, "order_seed": 2502, "warmup_updates": 2,
          "measured_updates": 5, "gradient_clip_norm": 1.0,
          "worker_timeout_seconds": 180, "host_poll_seconds": 0.1,
          "worker_rss_soft_limit_bytes": 8 * 1024**3,
          "system_available_floor_bytes": 4 * 1024**3,
          "worker_log_limit_bytes": 16 * 1024**2}
DIRECTORY = ROOT / "docs/research/workload-screen"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    temporary.replace(path)


def portable(message):
    return str(message).replace(str(ROOT), "<atlas>").replace(str(ROOT).replace("\\", "/"), "<atlas>")


def child(attention, microbatch, checkpointed, output):
    from qa_native_helpers import (build_model, build_optimizer, canonical, load_features,
                                   model_state_digest, tensor_digest, validate_runtime)
    started = time.perf_counter()
    result = {"status": "running", "stage": "validation", "successful_updates": 0,
              "attempted_updates": 0, "history": [], "config": CONFIG,
              "attention": attention, "microbatch": microbatch, "checkpointed": checkpointed,
              "source_sha256": sha(Path(__file__)),
              "helper_source_sha256": sha(ROOT / "benchmarks/qa_native_helpers.py")}
    atomic_json(output, result)
    torch = None

    def memory():
        values = {}
        for name, function in (("allocated_bytes", torch.cuda.memory_allocated),
                               ("reserved_bytes", torch.cuda.memory_reserved),
                               ("peak_allocated_bytes", torch.cuda.max_memory_allocated),
                               ("peak_reserved_bytes", torch.cuda.max_memory_reserved)):
            try:
                values[name] = int(function())
            except Exception as error:
                values[name] = None
                values[name + "_error"] = type(error).__name__
        return values

    try:
        result["runtime_identities"] = validate_runtime()
        import torch as imported_torch
        import transformers
        torch = imported_torch
        if not torch.cuda.is_available():
            result.update({"status": "unavailable", "stage": "cuda_availability"})
            return
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        result["stage"] = "preparation"
        arrays, manifest = load_features()
        updates = 1 + CONFIG["warmup_updates"] + CONFIG["measured_updates"]
        assert manifest["features"] >= updates * CONFIG["logical_batch"]
        model, preparation = build_model(CONFIG["model_seed"], attention, checkpointed, "cuda")
        optimizer, groups = build_optimizer(model)
        properties = torch.cuda.get_device_properties(0)
        result.update({"model_preparation": preparation, "optimizer_group_names_digest": groups,
                       "feature_manifest_canonical_sha256": canonical(manifest),
                       "feature_rows": {"start": 0, "stop_exclusive": updates * CONFIG["logical_batch"]},
                       "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                       "transformers": transformers.__version__,
                                       "gpu": properties.name, "device_total_bytes": properties.total_memory,
                                       "allocator": torch.cuda.memory.get_allocator_backend(),
                                       "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                       "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                                       "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                                       "allocator_cap": None},
                       "dropout_probabilities": sorted({m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)})})
        torch.manual_seed(CONFIG["scientific_rng_seed"])
        result["scientific_start_rng"] = {
            "cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "cuda": hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}
        torch.cuda.synchronize()
        result["preparation_wall_seconds"] = time.perf_counter() - started
        result["prepared_memory"] = memory()
        atomic_json(output, result)

        def update(index):
            result["attempted_updates"] += 1
            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.zeros((), device="cuda")
            begin = index * CONFIG["logical_batch"]
            for offset in range(0, CONFIG["logical_batch"], microbatch):
                result["stage"] = "forward"
                batch = {name: values[begin + offset:begin + offset + microbatch].to("cuda")
                         for name, values in arrays.items()}
                prediction = model(**batch)
                if not bool(torch.isfinite(prediction.loss)):
                    raise ValueError("Nonfinite training loss")
                if index == 0 and offset == 0:
                    with torch.no_grad():
                        nll = -(torch.log_softmax(prediction.start_logits, -1).gather(
                            1, batch["start_positions"][:, None]).sum() +
                            torch.log_softmax(prediction.end_logits, -1).gather(
                            1, batch["end_positions"][:, None]).sum()) / (2 * microbatch)
                        result["first_microbatch_loss_accounting"] = {
                            "builtin_mean": float(prediction.loss.item()), "manual_mean": float(nll.item()),
                            "absolute_difference": float((prediction.loss - nll).abs().item()),
                            "denominator_unit": "feature_examples_per_head",
                            "start_valid": microbatch, "end_valid": microbatch}
                contribution = prediction.loss * (microbatch / CONFIG["logical_batch"])
                total_loss += contribution.detach()
                result["stage"] = "backward"
                contribution.backward()
            result["stage"] = "clip"
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip_norm"],
                                                  error_if_nonfinite=True, foreach=False)
            result["stage"] = "optimizer_step"
            optimizer.step()
            result["stage"] = "synchronize"
            torch.cuda.synchronize()
            result["successful_updates"] += 1
            return {"loss": float(total_loss.item()), "preclip_gradient_norm": float(norm.item()),
                    "none_gradient_parameters": sum(p.grad is None for p in model.parameters())}

        durations = []
        for index in range(updates):
            measured = index >= 1 + CONFIG["warmup_updates"]
            if index in (0, 1 + CONFIG["warmup_updates"]):
                torch.cuda.reset_peak_memory_stats()
            result["stage"] = "update"
            result["next_update_index"] = index
            atomic_json(output, result)
            torch.cuda.synchronize()
            before = time.perf_counter()
            observed = update(index)
            elapsed = time.perf_counter() - before
            result["history"].append({"index": index, "window": "measured" if measured else "first" if index == 0 else "warmup",
                                       "synchronized_update_seconds": elapsed, **observed})
            if measured:
                durations.append(elapsed)
            if index == 0:
                result["first_update_memory"] = memory()
                result["first_state_digest"] = model_state_digest(model)
                result["first_post_clip_gradient_digest"] = tensor_digest(
                    (name, p.grad) for name, p in model.named_parameters() if p.grad is not None)
                result["first_none_gradient_names"] = [name for name, p in model.named_parameters() if p.grad is None]
                result["first_optimizer_state_digest"] = tensor_digest(
                    (name + "/" + key, value) for name, p in model.named_parameters()
                    for key, value in sorted(optimizer.state[p].items()) if torch.is_tensor(value))
            atomic_json(output, result)
        result["measured_update_seconds"] = durations
        result["later_update_memory"] = memory()
        result["allocation_peak_above_nominal_device_bytes"] = (
            result["later_update_memory"]["peak_allocated_bytes"] > properties.total_memory)
        result["final_state_digest"] = model_state_digest(model)
        result.update({"status": "completed", "stage": "finished"})
    except Exception as error:
        is_oom = torch is not None and isinstance(error, torch.cuda.OutOfMemoryError)
        result.update({"status": "oom" if is_oom else "error", "error_type": type(error).__name__,
                       "message": portable(str(error))[:4000]})
        if torch is not None and result.get("environment"):
            result["partial_memory"] = memory()
    finally:
        result["worker_wall_seconds_before_final_write"] = time.perf_counter() - started
        atomic_json(output, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--attention", choices=CONFIG["attention"])
    parser.add_argument("--microbatch", type=int, choices=CONFIG["microbatches"])
    parser.add_argument("--checkpointed", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.child:
        child(args.attention, args.microbatch, args.checkpointed, args.output)
        return
    import psutil
    from qa_native_helpers import canonical
    directory = ROOT / "artifacts/qa-resource-pilot" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory.mkdir(parents=True)
    order = []
    rng = random.Random(CONFIG["order_seed"])
    for repetition in range(CONFIG["repetitions"]):
        plans = [(attention, micro, checkpointed) for attention in CONFIG["attention"]
                 for micro in CONFIG["microbatches"] for checkpointed in CONFIG["checkpointing"]]
        rng.shuffle(plans)
        order.extend((repetition, *plan) for plan in plans)
    result = {"started_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha(Path(__file__)),
              "helper_source_sha256": sha(ROOT / "benchmarks/qa_native_helpers.py"),
              "environment_helper_sha256": sha(ROOT / "benchmarks/board_energy_control.py"),
              "config": CONFIG, "order": order, "trials": [],
              "feature_manifest_canonical_sha256": canonical(json.loads((DIRECTORY / "qa-feature-manifest.json").read_text(encoding="utf-8"))),
              "scope": "Real QA-input resource segment, two same-seed process repetitions; no full fine-tuning, quality, scientific equivalence, primary selection or hard isolation result."}
    interpreter = ROOT / ".tools/qa-env/Scripts/python.exe"
    for number, (repetition, attention, micro, checkpointed) in enumerate(order):
        scratch = directory / str(number)
        scratch.mkdir()
        target = scratch / "worker.json"
        command = [str(interpreter), str(Path(__file__).resolve()), "--child", "--attention", attention,
                   "--microbatch", str(micro), "--output", str(target)]
        if checkpointed:
            command.append("--checkpointed")
        env = child_environment(interpreter, scratch)
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                    "DO_NOT_TRACK": "1", "TOKENIZERS_PARALLELISM": "false"})
        began = time.perf_counter()
        peak_rss = 0
        lifetime_peak = None
        minimum_available = psutil.virtual_memory().available
        stop_reason = None
        stdout_path, stderr_path = scratch / "stdout.txt", scratch / "stderr.txt"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            owned = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    info = owned.memory_info()
                    peak_rss = max(peak_rss, info.rss)
                    if hasattr(info, "peak_wset"):
                        lifetime_peak = max(lifetime_peak or 0, info.peak_wset)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                available = psutil.virtual_memory().available
                minimum_available = min(minimum_available, available)
                if time.perf_counter() - began > CONFIG["worker_timeout_seconds"]:
                    stop_reason = "timeout"
                elif peak_rss > CONFIG["worker_rss_soft_limit_bytes"]:
                    stop_reason = "stopped_for_worker_rss"
                elif available < CONFIG["system_available_floor_bytes"]:
                    stop_reason = "stopped_for_host_pressure"
                elif stdout_path.stat().st_size + stderr_path.stat().st_size > CONFIG["worker_log_limit_bytes"]:
                    stop_reason = "stopped_for_log_limit"
                if stop_reason:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    break
                time.sleep(CONFIG["host_poll_seconds"])
        observed = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {"status": "missing_worker_result"}
        trial = {"repetition": repetition, "attention": attention, "microbatch": micro, "checkpointed": checkpointed,
                 "status": stop_reason or observed["status"], "returncode": process.returncode, "worker_result": observed,
                 "owned_child_wall_seconds": time.perf_counter() - began, "host_sampled_peak_rss_bytes": peak_rss,
                 "host_os_lifetime_peak_working_set_bytes": lifetime_peak,
                 "host_minimum_sampled_available_bytes": minimum_available,
                 "stderr_excerpt": portable(stderr_path.read_text(encoding="utf-8")[-4000:])}
        result["trials"].append(trial)
        atomic_json(DIRECTORY / "qa-resource-pilot.json", result)
        print(json.dumps({"trial": number, "repetition": repetition, "attention": attention,
                          "microbatch": micro, "checkpointed": checkpointed, "status": trial["status"],
                          "updates": observed.get("successful_updates"),
                          "later_peak": observed.get("later_update_memory")}), flush=True)


if __name__ == "__main__":
    main()
