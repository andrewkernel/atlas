"""Trusted rendezvous/job lifecycle diagnostic for this Windows environment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from board_energy_control import ROOT, child_environment
from worker_ownership_control import save


def child(directory, nonce):
    save(directory / 'ready.json', {'pid': os.getpid(), 'ppid': os.getppid(), 'nonce': nonce})
    deadline = time.monotonic() + 5
    while not (directory / 'ack.json').exists() and time.monotonic() < deadline:
        time.sleep(.01)
    if not (directory / 'ack.json').exists():
        return
    ack = json.loads((directory / 'ack.json').read_text(encoding='utf-8'))
    if ack['nonce'] != nonce:
        raise ValueError('Rendezvous identity differs')
    allocation = bytearray(64 * 1024**2)
    for offset in range(0, len(allocation), 4096):
        allocation[offset] = 1
    save(directory / 'allocated.json', {'bytes': len(allocation), 'nonce': nonce})
    time.sleep(.25 if ack['mode'] == 'natural' else 2)
    assert allocation[-4096] == 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', type=Path)
    parser.add_argument('--nonce')
    args = parser.parse_args()
    if args.child:
        child(args.child, args.nonce)
        return
    import psutil
    from windows_job_scope import WindowsJobScope
    interpreter = ROOT / '.tools/qa-env/Scripts/python.exe'
    directory = ROOT / 'artifacts/windows-job-control' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory.mkdir(parents=True)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    result = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': digest(Path(__file__)),
              'job_helper_sha256': digest(ROOT / 'benchmarks/windows_job_scope.py'),
              'ownership_helper_sha256': digest(ROOT / 'benchmarks/worker_ownership_control.py'),
              'environment_helper_sha256': digest(ROOT / 'benchmarks/board_energy_control.py'), 'trials': [],
              'scope': 'Trusted bounded local lifecycle control; no hard memory limit, training, hostile-code isolation or all-failure-path guarantee.'}
    for mode in ('natural', 'terminate_job', 'close_job'):
        scratch = directory / mode
        scratch.mkdir()
        nonce = uuid.uuid4().hex
        row, retained = {'mode': mode, 'status': 'started'}, []
        started = time.monotonic()
        job = WindowsJobScope()
        try:
            with (scratch / 'stdout.txt').open('w') as stdout, (scratch / 'stderr.txt').open('w') as stderr:
                launch = subprocess.Popen([str(interpreter), str(Path(__file__).resolve()), '--child', str(scratch), '--nonce', nonce],
                                          cwd=ROOT, env=child_environment(interpreter, scratch), stdout=stdout, stderr=stderr)
                root = psutil.Process(launch.pid)
                retained.append(root)
                while not (scratch / 'ready.json').exists() and time.monotonic() - started < 5 and launch.poll() is None:
                    time.sleep(.01)
                if not (scratch / 'ready.json').exists():
                    raise RuntimeError('No ready rendezvous')
                ready = json.loads((scratch / 'ready.json').read_text(encoding='utf-8'))
                retained.extend(root.children(recursive=True))
                matches = [p for p in retained if p.pid == ready['pid']]
                if len(matches) != 1 or ready['nonce'] != nonce or matches[0].ppid() != ready['ppid']:
                    raise RuntimeError('Unverified actual interpreter identity')
                actual = matches[0]
                try:
                    job.assign_owned(actual.pid, actual.create_time() + 1)
                except ValueError:
                    row['wrong_creation_identity_rejected'] = True
                else:
                    raise RuntimeError('Wrong process identity was accepted')
                for process in retained:
                    row['phase'] = 'assign_launcher' if process.pid == launch.pid else 'assign_actual_interpreter'
                    job.assign_owned(process.pid, process.create_time())
                row['phase'] = 'query_members'
                members = job.members()
                if set(members) != {p.pid for p in retained}:
                    raise RuntimeError('Job inventory differs from retained owned identities')
                row['job_members_before_acknowledgment'] = len(members)
                row['actual_interpreter_is_launcher'] = actual.pid == launch.pid
                save(scratch / 'ack.json', {'mode': mode, 'nonce': nonce})
                while not (scratch / 'allocated.json').exists() and time.monotonic() - started < 7 and launch.poll() is None:
                    time.sleep(.01)
                allocated = json.loads((scratch / 'allocated.json').read_text(encoding='utf-8'))
                if allocated['nonce'] != nonce:
                    raise RuntimeError('Allocation record identity differs')
                row.update({'allocated_bytes': allocated['bytes'], 'launcher_rss_bytes': root.memory_info().rss,
                            'actual_interpreter_rss_bytes': actual.memory_info().rss, 'job_accounting': job.memory_accounting()})
                if mode == 'terminate_job':
                    job.terminate(42)
                elif mode == 'close_job':
                    job.close()
                launch.wait(timeout=max(1, 10 - (time.monotonic() - started)))
                _, alive = psutil.wait_procs(retained, timeout=1)
                row.update({'status': 'completed', 'returncode': launch.returncode,
                            'retained_identities_still_alive': len(alive),
                            'job_members_after_exit': len(job.members()) if job.handle is not None else None})
                if alive or (mode == 'natural' and launch.returncode != 0) or (mode == 'terminate_job' and launch.returncode != 42):
                    raise RuntimeError('Unexpected completion or retained owned process')
        except Exception as error:
            row.update({'status': 'error', 'error_type': type(error).__name__, 'message': str(error)})
        finally:
            if job.handle is not None:
                job.terminate(43)
                job.close()
            try:
                job.members()
            except RuntimeError:
                row['use_after_close_rejected'] = True
            for process in reversed(retained):
                if process.is_running():
                    process.terminate()
            _, alive = psutil.wait_procs(retained, timeout=1)
            for process in alive:
                process.kill()
            _, alive = psutil.wait_procs(retained, timeout=1)
            row['cleanup_retained_identities_still_alive'] = len(alive)
            row['wall_seconds'] = time.monotonic() - started
        result['trials'].append(row)
        save(ROOT / 'docs/research/workload-screen/windows-job-control.json', result)
        print(json.dumps(row), flush=True)
        if row['status'] != 'completed' or row['cleanup_retained_identities_still_alive']:
            raise RuntimeError('Job control did not complete cleanly; stop further cases')


if __name__ == '__main__':
    main()
