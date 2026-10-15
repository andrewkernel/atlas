"""Authored, bounded environment-launcher/actual-interpreter ownership diagnostic."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from board_energy_control import ROOT, child_environment


def save(path, value):
    part = path.with_suffix('.part')
    part.write_bytes((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
    part.replace(path)


def child(scratch):
    allocation = bytearray(64 * 1024**2)
    for offset in range(0, len(allocation), 4096):
        allocation[offset] = 1
    save(scratch / 'ready.json', {'pid': os.getpid(), 'ppid': os.getppid(), 'bytes': len(allocation)})
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not (scratch / 'ack.json').exists():
        time.sleep(.02)
    if not (scratch / 'ack.json').exists():
        return
    mode = json.loads((scratch / 'ack.json').read_text(encoding='utf-8'))['mode']
    time.sleep(.25 if mode == 'natural' else 2)
    assert allocation[-4096] == 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', type=Path)
    args = parser.parse_args()
    if args.child:
        child(args.child)
        return
    import psutil
    interpreter = ROOT / '.tools/qa-env/Scripts/python.exe'
    directory = ROOT / 'artifacts/worker-ownership' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory.mkdir(parents=True)
    result = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'captured_utc': datetime.now(timezone.utc).isoformat(), 'trials': [],
              'scope': 'Trusted bounded diagnostic; no hostile-code isolation or general tree-ownership guarantee.'}
    for mode in ('natural', 'cancel'):
        scratch = directory / mode
        scratch.mkdir()
        command = [str(interpreter), str(Path(__file__).resolve()), '--child', str(scratch)]
        started = time.monotonic()
        row = {'mode': mode, 'status': 'started'}
        retained = []
        with (scratch / 'stdout.txt').open('w') as stdout, (scratch / 'stderr.txt').open('w') as stderr:
            launch = subprocess.Popen(command, cwd=ROOT, env=child_environment(interpreter, scratch), stdout=stdout, stderr=stderr)
            root = psutil.Process(launch.pid)
            retained.append(root)
            try:
                target = scratch / 'ready.json'
                while not target.exists() and time.monotonic() - started < 5 and launch.poll() is None:
                    time.sleep(.02)
                if not target.exists():
                    raise RuntimeError('No owned child rendezvous before deadline')
                ready = json.loads(target.read_text(encoding='utf-8'))
                descendants = root.children(recursive=True)
                retained.extend(descendants)
                matches = [p for p in retained if p.pid == ready['pid']]
                if len(matches) != 1 or matches[0].ppid() != ready['ppid']:
                    raise RuntimeError('Interpreter PID/parent not verified in owned tree')
                actual = matches[0]
                row.update({'interpreter_is_launch_pid': actual.pid == launch.pid,
                            'interpreter_parent_is_launch_pid': ready['ppid'] == launch.pid,
                            'owned_descendant_count': len(descendants), 'child_touched_bytes': ready['bytes'],
                            'launcher_rss_bytes': root.memory_info().rss,
                            'actual_interpreter_rss_bytes': actual.memory_info().rss,
                            'sum_observed_tree_rss_bytes': sum(p.memory_info().rss for p in retained)})
                save(scratch / 'ack.json', {'mode': mode})
                if mode == 'cancel':
                    actual.terminate()
                launch.wait(timeout=max(1, 10 - (time.monotonic() - started)))
                _, alive = psutil.wait_procs(retained, timeout=1)
                row.update({'status': 'completed', 'returncode': launch.returncode,
                            'retained_identities_still_alive': len(alive)})
                if alive:
                    raise RuntimeError('A retained owned identity remains alive')
            except Exception as error:
                row.update({'status': 'error', 'error_type': type(error).__name__, 'message': str(error)})
            finally:
                # psutil retains PID + creation identity; do not select or terminate by process name.
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
        save(ROOT / 'docs/research/workload-screen/worker-ownership-control.json', result)
        print(json.dumps(row), flush=True)
        if row['status'] != 'completed' or row['cleanup_retained_identities_still_alive']:
            raise RuntimeError('Ownership control did not complete cleanly; stop further trials')


if __name__ == '__main__':
    main()
