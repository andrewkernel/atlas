"""Local builtin constructor-default check; no weights, forward or optimizer execution."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

from board_energy_control import ROOT, child_environment
from worker_ownership_control import save


def worker():
    from qa_native_helpers import validate_runtime
    identities = validate_runtime()
    import torch
    from transformers import BertConfig, BertForQuestionAnswering
    torch.set_num_threads(1)
    torch.manual_seed(2701)
    config = BertConfig.from_json_file(ROOT / 'weights/bert-base-uncased-pinned/config.json')
    before = config._attn_implementation
    model = BertForQuestionAnswering(config)
    result = {'runtime_identities': identities, 'torch': torch.__version__, 'attention_before_constructor': before,
              'attention_after_constructor': model.config._attn_implementation,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'parameter_devices': sorted({str(p.device) for p in model.parameters()}),
              'scope': 'Builtin CPU constructor with pinned full config and fresh random parameters; no checkpoint load, forward, optimizer, CUDA measurement or training.'}
    print(json.dumps(result, sort_keys=True, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker()
        return
    import psutil
    from windows_job_scope import WindowsJobScope
    interpreter = ROOT / '.tools/qa-env/Scripts/python.exe'
    scratch = ROOT / 'artifacts/qa-default-attention' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    scratch.mkdir(parents=True)
    job = WindowsJobScope()
    result = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'status': 'started',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'helper_sha256': {name: hashlib.sha256((ROOT / 'benchmarks' / name).read_bytes()).hexdigest()
                                for name in ('qa_native_helpers.py', 'windows_job_scope.py', 'board_energy_control.py', 'worker_ownership_control.py')}}
    started = time.monotonic()
    process = None
    try:
        with (scratch / 'stdout.txt').open('w') as stdout, (scratch / 'stderr.txt').open('w') as stderr:
            env = child_environment(interpreter, scratch)
            env.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'})
            process = subprocess.Popen([str(interpreter), str(Path(__file__).resolve()), '--worker'],
                                       cwd=ROOT, env=env, stdout=stdout, stderr=stderr, creationflags=0x00000004)
            root = psutil.Process(process.pid)
            job.assign_owned(root.pid, root.create_time())
            job.resume_owned_single_thread(root)
            process.wait(timeout=30)
        if process.returncode != 0:
            raise RuntimeError('Constructor probe exited with code ' + str(process.returncode))
        result.update({'status': 'completed', 'worker_result': json.loads((scratch / 'stdout.txt').read_text(encoding='utf-8'))})
    except Exception as error:
        result.update({'status': 'error', 'error_type': type(error).__name__, 'message': str(error).replace(str(ROOT), '<atlas>')})
    finally:
        job.terminate(48)
        if process is not None:
            process.wait(timeout=5)
        deadline = time.monotonic() + 2
        while job.members() and time.monotonic() < deadline:
            time.sleep(.02)
        result['job_members_after_cleanup'] = len(job.members())
        job.close()
        result['supervised_probe_seconds'] = time.monotonic() - started
    save(ROOT / 'docs/research/workload-screen/qa-default-attention-control.json', result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
