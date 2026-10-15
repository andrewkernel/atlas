"""Trusted QA diagnostic supervisor using a pre-execution Windows job boundary."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from board_energy_control import ROOT, child_environment
from worker_ownership_control import save

ALLOWED = {'qa_numerical_control.py', 'qa_dispatch_control.py'}


def await_owned_ack(directory, nonce):
    save(directory / 'ready.json', {'pid': os.getpid(), 'ppid': os.getppid(), 'nonce': nonce})
    deadline = time.monotonic() + 10
    while not (directory / 'ack.json').exists() and time.monotonic() < deadline:
        time.sleep(.02)
    if not (directory / 'ack.json').exists():
        raise RuntimeError('No supervisor acknowledgment before bootstrap deadline')
    if json.loads((directory / 'ack.json').read_text(encoding='utf-8'))['nonce'] != nonce:
        raise ValueError('Supervisor acknowledgment identity differs')


def run_worker(script_name, arguments, scratch, timeout_seconds=240):
    import psutil
    from windows_job_scope import WindowsJobScope
    if script_name not in ALLOWED or not (ROOT / 'benchmarks' / script_name).is_file():
        raise ValueError('Worker script is not a registered authored QA diagnostic')
    scratch.mkdir(parents=True)
    interpreter = ROOT / '.tools/qa-env/Scripts/python.exe'
    nonce = uuid.uuid4().hex
    command = [str(interpreter), str(ROOT / 'benchmarks' / script_name), '--worker', str(scratch), '--nonce', nonce, *arguments]
    env = child_environment(interpreter, scratch)
    env.update({'PYTHONPATH': str(ROOT / '.tools/trainparity-reviewed/src') + os.pathsep + str(ROOT / 'benchmarks'),
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                'DO_NOT_TRACK': '1', 'TOKENIZERS_PARALLELISM': 'false'})
    job, retained = WindowsJobScope(), []
    row = {'status': 'supervisor_start', 'actual_worker_sampled_peak_rss_bytes': 0,
           'sampled_minimum_host_available_bytes': psutil.virtual_memory().available,
           'job_peak_commit_bytes': 0, 'worker_timeout_seconds': timeout_seconds,
           'actual_rss_soft_limit_bytes': 8 * 1024**3, 'job_commit_soft_limit_bytes': 12 * 1024**3,
           'host_available_floor_bytes': 4 * 1024**3, 'poll_seconds': .1,
           'supervisor_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    started = time.monotonic()
    stop_reason = None
    launch = None
    try:
        stdout_path, stderr_path = scratch / 'stdout.txt', scratch / 'stderr.txt'
        with stdout_path.open('w') as stdout, stderr_path.open('w') as stderr:
            launch = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, creationflags=0x00000004)
            root = psutil.Process(launch.pid)
            retained.append(root)
            job.assign_owned(root.pid, root.create_time())
            row['launcher_previous_suspend_count'] = job.resume_owned_single_thread(root)
            while not (scratch / 'ready.json').exists() and time.monotonic() - started < 8 and launch.poll() is None:
                time.sleep(.02)
            if not (scratch / 'ready.json').exists():
                raise RuntimeError('Worker did not rendezvous before startup deadline')
            ready = json.loads((scratch / 'ready.json').read_text(encoding='utf-8'))
            retained.extend(root.children(recursive=True))
            matches = [p for p in retained if p.pid == ready['pid']]
            if ready['nonce'] != nonce or len(matches) != 1 or matches[0].ppid() != ready['ppid']:
                raise RuntimeError('Worker rendezvous is not a verified owned identity')
            actual = matches[0]
            members = job.members()
            if set(members) != {p.pid for p in retained} or actual.pid == launch.pid:
                raise RuntimeError('Unexpected job inventory or launcher/interpreter relation')
            row['job_members_before_acknowledgment'] = len(members)
            row['actual_interpreter_inherited_job'] = True
            save(scratch / 'ack.json', {'nonce': nonce})
            while launch.poll() is None:
                try:
                    row['actual_worker_sampled_peak_rss_bytes'] = max(row['actual_worker_sampled_peak_rss_bytes'], actual.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                available = psutil.virtual_memory().available
                row['sampled_minimum_host_available_bytes'] = min(row['sampled_minimum_host_available_bytes'], available)
                row['job_peak_commit_bytes'] = max(row['job_peak_commit_bytes'], job.memory_accounting()['peak_job_commit_bytes'])
                if time.monotonic() - started > timeout_seconds:
                    stop_reason = 'timeout'
                elif row['actual_worker_sampled_peak_rss_bytes'] > row['actual_rss_soft_limit_bytes']:
                    stop_reason = 'stopped_for_actual_worker_rss'
                elif row['job_peak_commit_bytes'] > row['job_commit_soft_limit_bytes']:
                    stop_reason = 'stopped_for_job_commit'
                elif available < row['host_available_floor_bytes']:
                    stop_reason = 'stopped_for_host_pressure'
                elif stdout_path.stat().st_size + stderr_path.stat().st_size > 16 * 1024**2:
                    stop_reason = 'stopped_for_log_limit'
                if stop_reason:
                    job.terminate(44)
                    break
                time.sleep(.1)
            launch.wait(timeout=5)
            target = scratch / 'worker.json'
            row['worker_result'] = json.loads(target.read_text(encoding='utf-8')) if target.exists() else {'status': 'missing_worker_result'}
            row['status'] = stop_reason or (row['worker_result']['status'] if launch.returncode == 0 else 'worker_exit_error')
            row['returncode'] = launch.returncode
            row['stderr_excerpt'] = stderr_path.read_text(encoding='utf-8')[-3000:].replace(str(ROOT), '<atlas>')
    except Exception as error:
        row.update({'status': 'supervisor_error', 'error_type': type(error).__name__,
                    'message': str(error).replace(str(ROOT), '<atlas>')})
    finally:
        job.terminate(45)
        _, alive = psutil.wait_procs(retained, timeout=2)
        row['retained_identities_still_alive'] = len(alive)
        row['job_members_after_cleanup'] = len(job.members())
        job.close()
        if launch is not None:
            launch.wait(timeout=5)
        row['outer_owned_job_seconds'] = time.monotonic() - started
    if row['retained_identities_still_alive'] or row['job_members_after_cleanup']:
        raise RuntimeError('Owned job cleanup incomplete; do not start another trial')
    return row
