"""One GPU lock, finite campaigns and fresh suspended-launch owned workers."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

from .common import (ROOT, STORE, campaign_dir, domain, file_digest, frozen_json,
                     public_error, read_json, snapshot, utc, write_json)
from . import profiles


def environment(threads=4):
    env = {k: v for k, v in os.environ.items() if k.upper() in {
        'PATH', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'USERPROFILE', 'LOCALAPPDATA',
        'APPDATA', 'COMSPEC', 'PROGRAMFILES', 'PROGRAMW6432'}}
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    return env


class OwnedChild:
    """Windows job inheritance covers the venv launcher and its actual Python child."""
    def __init__(self, args, log, threads=4):
        if os.name != 'nt':
            raise RuntimeError('Owned runtime is currently validated only on Windows')
        sys.path.insert(0, str(ROOT / 'benchmarks'))
        from windows_job_scope import WindowsJobScope
        self.scope, self.process = WindowsJobScope(), None
        self.observed = {}
        self.closed = False
        self.started = time.monotonic()
        try:
            with log.open('wb') as stream:
                self.process = subprocess.Popen([sys.executable, str(ROOT / 'tune.py'), *args],
                    cwd=ROOT, env=environment(threads), stdout=stream, stderr=subprocess.STDOUT,
                    creationflags=0x08000000 | 0x00000004)
            owner = psutil.Process(self.process.pid)
            self.scope.assign_owned(owner.pid, owner.create_time())
            self.observe()
            self.scope.resume_owned_single_thread(owner)
        except BaseException:
            self.close()
            raise

    def observe(self):
        for pid in self.scope.members():
            try:
                p = psutil.Process(pid)
                self.observed[(pid, p.create_time())] = {'pid': pid, 'created_at': p.create_time()}
            except psutil.NoSuchProcess:
                pass

    def close(self):
        if self.closed:
            return
        try:
            self.observe()
            if self.scope.members():
                self.scope.terminate()
            if self.process:
                self.process.wait(timeout=10)
            until = time.monotonic() + 5
            while self.scope.members() and time.monotonic() < until:
                time.sleep(.05)
            if self.scope.members():
                raise RuntimeError('Owned job still has live processes after termination')
            self.accounting = self.scope.memory_accounting()
            self.closed = True
        finally:
            self.scope.close()

    def evidence(self):
        return {'launcher_pid': self.process.pid, 'observed_processes': list(self.observed.values()),
                'closed': self.closed, 'exit_code': self.process.returncode,
                'elapsed_s': time.monotonic()-self.started, 'memory': self.accounting}


class EndCampaign(RuntimeError):
    def __init__(self, status, reason):
        self.status = status
        super().__init__(reason)


def recover_ram(rules, check):
    """Only called after the offending owned scope has closed."""
    start = time.monotonic()
    samples = []
    while True:
        check()
        available = psutil.virtual_memory().available
        elapsed = time.monotonic()-start
        samples.append({'observed_at': utc(), 'available_bytes': available, 'elapsed_s': elapsed})
        recovered = available >= rules['ram_recovery_available_bytes']
        if recovered or elapsed >= rules['ram_recovery_wall_s']:
            return {'recovered': recovered, 'required_available_bytes': rules['ram_recovery_available_bytes'],
                    'wall_bound_s': rules['ram_recovery_wall_s'], 'elapsed_s': elapsed, 'samples': samples}
        time.sleep(.2)


def run_campaign(ident, home=STORE):
    from filelock import FileLock
    from .registry import freeze, settings, calibration_cases, reference_cases, confirmation_cases
    from .verify import build
    from .policy import nominate
    folder = campaign_dir(ident, home)
    request = settings(read_json(folder / 'request.json'))
    start = time.monotonic()
    state = {'id': ident, 'status': 'preflight', 'created_at': utc(), 'settings': request,
             'controller_process': {'pid': os.getpid(), 'created_at': psutil.Process().create_time()},
             'progress': {'done': 0, 'total': 0, 'message': 'Freezing inputs and execution identities'}}
    ledger = None

    def status(**values):
        state.update(values, elapsed_s=time.monotonic()-start)
        write_json(folder / 'status.json', state)

    def check():
        if profiles.cancelled(ident, home) or (folder / 'cancel').exists():
            raise EndCampaign('cancelled', 'Cancellation accepted; no automatic profile commit')
        if time.monotonic()-start > request['max_wall_s']:
            raise EndCampaign('timed_out', 'Campaign exceeded its declared wall-time bound')

    def collect(cases):
        ledger['cases'].extend({'id': c['id'], 'status': 'queued'} for c in cases)
        write_json(folder / 'execution.json', ledger)
        for case in cases:
            check()
            free = shutil.disk_usage(folder).free
            if free < protocol['rules']['minimum_start_disk_bytes']:
                status(resource_guard={'kind': 'disk_before_worker', 'observed_at': utc(),
                       'available_bytes': free, 'minimum_bytes': protocol['rules']['minimum_start_disk_bytes']})
                raise EndCampaign('resource_limited', 'Available disk space fell below the per-worker output reserve')
            target = folder / 'runs' / case['id']
            target.mkdir(parents=True)
            frozen_json(target / 'config.json', case)
            entry = next(e for e in ledger['cases'] if e['id'] == case['id'])
            entry['status'] = 'running'
            write_json(folder / 'execution.json', ledger)
            status(status='references' if case['stage'] == 'reference' else case['stage'],
                   active_case=case['id'], progress={'done': sum(e['status'] not in ('queued', 'running')
                          for e in ledger['cases']), 'total': len(ledger['cases']), 'message': case['id']})
            worker, forced, reason, cancel_at, guard = None, None, None, None, None
            try:
                worker = OwnedChild(['worker', ident, case['id']], target / 'worker.log', case['profile']['threads'])
                while worker.process.poll() is None:
                    worker.observe()
                    try:
                        check()
                    except EndCampaign as exc:
                        forced, reason = exc.status, str(exc)
                        if cancel_at is None:
                            cancel_at = time.monotonic()
                    if time.monotonic()-worker.started > protocol['rules']['worker_wall_s']:
                        forced, reason = 'timed_out', 'Worker exceeded its registered time bound'
                        break
                    available = psutil.virtual_memory().available
                    free = shutil.disk_usage(folder).free
                    if available < protocol['rules']['minimum_available_ram_bytes']:
                        forced, reason = 'resource_limited', 'Available system RAM fell below the advisory guard'
                        guard = {'kind': 'ram', 'observed_at': utc(), 'available_bytes': available,
                                 'minimum_bytes': protocol['rules']['minimum_available_ram_bytes']}
                        break
                    if free < protocol['rules']['minimum_running_disk_bytes']:
                        forced, reason = 'resource_limited', 'Available disk space fell below the running output reserve'
                        guard = {'kind': 'disk', 'observed_at': utc(), 'available_bytes': free,
                                 'minimum_bytes': protocol['rules']['minimum_running_disk_bytes']}
                        break
                    if cancel_at is not None and time.monotonic()-cancel_at > 5:
                        break
                    time.sleep(.2)
            except Exception as exc:
                forced, reason = 'failed', public_error(exc)
            finally:
                if worker:
                    worker.close()
                    entry['ownership'] = worker.evidence()
            path = target / 'result.json'
            raw = read_json(path) if path.exists() else {'id': case['id'], 'config': case}
            if forced or raw.get('status') in (None, 'running'):
                raw.update(status=forced or 'failed', error=reason or 'Worker exited without terminal evidence',
                           finished_at=utc())
                if guard:
                    raw['resource_guard'] = guard
                write_json(path, raw)
            entry.update(status=raw['status'], sha256=file_digest(path))
            write_json(folder / 'execution.json', ledger)
            if forced == 'resource_limited':
                if guard['kind'] != 'ram':
                    raise EndCampaign(forced, reason)
                status(status='resource_recovery', resource_guard=guard,
                       progress={'done': sum(e['status'] not in ('queued', 'running') for e in ledger['cases']),
                                 'total': len(ledger['cases']), 'message': 'Owned worker closed; checking RAM recovery'})
                try:
                    raw['resource_recovery'] = recover_ram(protocol['rules'], check)
                except EndCampaign as exc:
                    raw['resource_recovery'] = {'recovered': False, 'interrupted': exc.status, 'reason': str(exc)}
                    raise
                finally:
                    write_json(path, raw)
                    entry['sha256'] = file_digest(path)
                    write_json(folder / 'execution.json', ledger)
                if not raw['resource_recovery']['recovered']:
                    raise EndCampaign(forced, 'RAM did not recover within the registered recovery bound')
            check()

    status()
    try:
        with FileLock(str(ROOT / 'artifacts/attention-lab/.gpu.lock'), timeout=0):
            check()
            expected = profiles.state(domain(request), home)
            baseline = 'native-t4'
            if expected['profile']:
                try:
                    baseline = profiles.validate(expected['profile'], home)['execution']['id']
                except (ValueError, OSError, RuntimeError) as exc:
                    status(previous_profile_ignored=public_error(exc))
            protocol = freeze(folder, request, baseline, expected)
            snapshot(folder, protocol['sources'])
            ledger = {'protocol_sha256': file_digest(folder / 'protocol.json'), 'cases': []}
            collect(reference_cases('cal'))
            collect(calibration_cases(protocol))
            status(status='nomination', active_case=None)
            check()
            calibration = build(folder, calibration_only=True)
            nomination = nominate(protocol, calibration['cases'])
            if nomination['profile'] != calibration['nominated'] or nomination['basis_sha256'] != calibration['basis_sha256']:
                raise ValueError('Independent calibration and selector disagree')
            frozen_json(folder / 'nomination.json', nomination)
            ledger['nomination_sha256'] = file_digest(folder / 'nomination.json')
            write_json(folder / 'execution.json', ledger)
            if nomination['profile']:
                collect(reference_cases('hold'))
                collect(confirmation_cases(protocol, nomination['profile']))
            status(status='verification', active_case=None)
            check()
            verified = build(folder)
            write_json(folder / 'verified.json', verified)
            check()
            profile = profiles.save(folder, verified, home)
            applied = None
            if profile and request['auto_apply']:
                try:
                    applied = profiles.activate(profile, expected, home, campaign=ident, before_commit=check)
                except profiles.Conflict as exc:
                    status(activation_conflict=str(exc))
            total = time.monotonic()-start
            saved = verified['decision'].get('median_saved_ms')
            status(status='completed', decision=verified['decision'], profile=profile, applied=applied,
                   evidence_sha256=verified['evidence_sha256'], tuning_cost_s=total,
                   speed_break_even_requests=int(__import__('math').ceil(total*1000/saved)) if saved and saved > 0 else None,
                   progress={'done': len(ledger['cases']), 'total': len(ledger['cases']), 'message': 'Evidence verified'})
    except EndCampaign as exc:
        status(status=exc.status, error=str(exc), active_case=None)
    except BaseException as exc:
        status(status='failed', error=public_error(exc), active_case=None)
    return state
