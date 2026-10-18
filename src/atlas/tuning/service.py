"""Local UI wrapper, retaining the frozen study service unchanged."""
from __future__ import annotations

import atexit
import asyncio
import threading
import time
import uuid
import hashlib
import subprocess

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from atlas.attention import server as old
from .common import STORE, campaign_dir, domain, identifier, public_error, read_json, utc, write_json, source_map, snapshot
from .controller import OwnedChild
from .registry import settings
from . import profiles

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
gpu_admission = asyncio.Lock()
TERMINAL = {'completed', 'failed', 'cancelled', 'timed_out', 'resource_limited', 'interrupted', 'abstained'}
try:
    IMPORTED_CAMPAIGN = read_json(old.ROOT / 'bundle-manifest.json').get('campaign')
except (OSError, ValueError):
    IMPORTED_CAMPAIGN = None


def located(ident, kind='campaign'):
    try:
        folder = campaign_dir(ident) if kind == 'campaign' else STORE / 'private-inference' / identifier(ident)
        if not folder.is_dir():
            raise ValueError('Unknown local run')
        return folder
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


def observed_status(folder, kind='campaign'):
    path = folder / ('status.json' if kind == 'campaign' else 'result.json')
    doc = read_json(path) if path.exists() else {'id': folder.name, 'status': 'queued'}
    doc['record_origin'] = 'imported_reference' if kind == 'campaign' and folder.name == IMPORTED_CAMPAIGN else 'recorded_result'
    doc['owned_job_active'] = jobs.active == folder.name
    if doc['status'] not in TERMINAL and not doc['owned_job_active']:
        import psutil
        process = doc.get('controller_process')
        alive = False
        if process:
            try:
                actual = psutil.Process(process['pid'])
                args = actual.cmdline()
                alive = (actual.create_time() == process['created_at'] and 'controller' in args and folder.name in args
                         and str(old.ROOT / 'tune.py') in args and actual.cwd() == str(old.ROOT))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        doc['external_controller_alive'] = alive
        if not alive:
            doc.update(status='interrupted' if process else 'unknown_owner',
                       error='No matching live service owner can be confirmed; partial evidence cannot activate settings')
    return doc


class Jobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = self.child = self.kind = None
        self.cancel_at = None

    def start(self, config, kind='campaign'):
        request_started = time.monotonic()
        with self.lock, old.jobs.lock:
            if self.active or old.jobs.active:
                raise HTTPException(409, 'A workspace GPU job is already active')
            from filelock import FileLock, Timeout
            try:
                with FileLock(str(old.RUNS / '.gpu.lock'), timeout=0):
                    pass
            except Timeout as exc:
                raise HTTPException(409, 'The workspace GPU lock is held by another local job') from exc
            ident = ('tune-' if kind == 'campaign' else 'generate-') + uuid.uuid4().hex[:12]
            folder = campaign_dir(ident) if kind == 'campaign' else STORE / 'private-inference' / ident
            folder.mkdir(parents=True)
            write_json(folder / 'request.json', config)
            state_path = folder / ('status.json' if kind == 'campaign' else 'result.json')
            write_json(state_path, {'id': ident, 'status': 'queued', 'created_at': utc(), 'settings': config})
            try:
                provenance = {'captured_at': utc(), 'scope': 'Before owned service launch', 'execution_sources': source_map()}
                try:
                    provenance['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=old.ROOT,
                        stderr=subprocess.DEVNULL, creationflags=0x08000000, timeout=5).decode().strip()
                    provenance['git_diff_sha256'] = hashlib.sha256(subprocess.check_output(
                        ['git', 'diff', 'HEAD', '--binary'], cwd=old.ROOT, stderr=subprocess.DEVNULL,
                        creationflags=0x08000000, timeout=5)).hexdigest()
                except (OSError, subprocess.SubprocessError):
                    provenance.update(git_head=None, git_diff_sha256=None)
                snapshot(folder, provenance['execution_sources'])
                write_json(folder / 'repository.json', provenance)
                child = OwnedChild(['controller' if kind == 'campaign' else 'inference-worker', ident], folder / 'controller.log')
            except Exception as exc:
                write_json(state_path, {'id': ident, 'status': 'failed', 'error': public_error(exc)})
                raise HTTPException(500, 'Owned worker could not start; its failed run is retained') from exc
            self.active, self.child, self.kind, self.cancel_at = ident, child, kind, None
            deadline = config['max_wall_s']+60 if kind == 'campaign' else 2400
            threading.Thread(target=self.watch, args=(ident, folder, child, state_path, deadline, request_started, kind), daemon=True).start()
            return {'id': ident, 'status': 'queued'}

    def watch(self, ident, folder, child, path, deadline, request_started, kind):
        forced = None
        try:
            while child.process.poll() is None:
                child.observe()
                if time.monotonic()-child.started > deadline:
                    forced = 'timed_out'; break
                if self.cancel_at and time.monotonic()-self.cancel_at > 12:
                    forced = 'cancelled'; break
                time.sleep(.2)
        finally:
            child.close()
            ownership = child.evidence()
            write_json(folder / 'controller-ownership.json', ownership)
            doc = read_json(path)
            changed = False
            if kind == 'inference':
                doc.update(service_job_total_s=time.monotonic()-request_started,
                           owned_process_total_s=ownership['elapsed_s'],
                           service_timing_scope='After request validation through prelaunch setup, owned process startup/execution/exit and cleanup; excludes browser/network transport')
                changed = True
            if forced or doc['status'] not in TERMINAL:
                doc.update(status=forced or 'failed', error='Owner ended without a normal terminal record', finished_at=utc())
                changed = True
            if changed:
                write_json(path, doc)
            with self.lock:
                if self.active == ident:
                    self.active = self.child = self.kind = self.cancel_at = None

    def cancel(self, ident):
        with self.lock:
            if ident != self.active:
                raise HTTPException(409, 'This job is not active')
            folder = located(ident, self.kind)
            doc = read_json(folder / ('status.json' if self.kind == 'campaign' else 'result.json'))
            if doc['status'] in TERMINAL:
                raise HTTPException(409, 'This job has finished; use Revert for applied settings')
            if self.kind == 'campaign':
                try:
                    profiles.cancel(ident)
                except profiles.Conflict as exc:
                    raise HTTPException(409, str(exc)) from exc
            (folder / 'cancel').touch()
            self.cancel_at = time.monotonic()
            return {'status': 'cancellation_requested'}

    def close(self):
        with self.lock:
            if self.child:
                self.child.close()


jobs = Jobs()
atexit.register(jobs.close)


@app.middleware('http')
async def admission(request: Request, call_next):
    if request.method == 'POST' and request.url.path in (
        '/api/tuning/start', '/api/tuning/generate', '/api/runs', '/api/study/repeat', '/api/validation/repeat'):
        async with gpu_admission:
            if jobs.active or old.jobs.active:
                return JSONResponse({'detail': 'A workspace GPU job is already active'}, status_code=409)
            return await call_next(request)
    return await call_next(request)


@app.get('/api/session')
def session():
    return {**old.session(), 'runtime_tuning': True, 'tuning_active': jobs.active}


@app.get('/api/tuning/state')
def state():
    campaigns = sorted((STORE / 'campaigns').glob('*'), key=lambda p: p.name)
    return {'active_job': jobs.active, 'active_kind': jobs.kind,
            'campaigns': [observed_status(p) for p in campaigns if p.is_dir()],
            'profiles': [{'id': p.stem, **read_json(p)} for p in (STORE / 'profiles').glob('*.json')],
            'events': profiles.history(),
            'readiness': {'model': (old.ROOT / 'weights/qwen3-0.6b/atlas-manifest.json').is_file(),
                          'corpus': (old.ROOT / 'data/wikitext-attention-v2/atlas-manifest.json').is_file()}}


class Start(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    context: int = 2048
    dtype: str = 'float32'
    generation_tokens: int = 16
    goal: str = 'fastest'
    budget_mib: float | None = None
    auto_apply: bool = True
    max_wall_s: int = 14400


@app.post('/api/tuning/start')
def start(value: Start):
    try:
        config = settings(value.model_dump())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return jobs.start(config)


@app.get('/api/tuning/campaigns/{ident}')
def campaign(ident: str):
    folder = located(ident)
    def read(name):
        return read_json(folder / name) if (folder / name).is_file() else None
    protocol = read('protocol.json')
    try:
        compatible = bool(protocol and protocol['sources'] == source_map())
    except (OSError, ValueError):
        compatible = False
    return {'status': observed_status(folder), 'protocol': protocol,
            'nomination': read('nomination.json'), 'verified': read('verified.json'),
            'execution': read('execution.json'), 'source_compatible': compatible,
            'domain': domain(protocol['settings']) if protocol else None,
            'active_state': profiles.state(domain(protocol['settings'])) if protocol else None}


@app.post('/api/tuning/jobs/{ident}/cancel')
def cancel(ident: str):
    return jobs.cancel(ident)


class Change(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    profile: str | None
    revision: int = Field(ge=0)


@app.post('/api/tuning/profiles/{ident}/apply')
def apply(ident: str, expected: Change):
    try:
        return profiles.activate(ident, expected.model_dump())
    except profiles.Conflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(422, public_error(exc)) from exc


@app.post('/api/tuning/domains/{key}/revert')
def revert(key: str, expected: Change):
    if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
        raise HTTPException(422, 'Invalid execution domain')
    try:
        return profiles.revert(key, expected.model_dump())
    except profiles.Conflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(422, public_error(exc)) from exc


class Generate(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    campaign: str
    prompt: str | None = Field(default=None, min_length=1, max_length=200000)
    check_native: bool = False


@app.post('/api/tuning/generate')
def generate(value: Generate):
    folder = located(value.campaign)
    if not (folder / 'protocol.json').is_file():
        raise HTTPException(422, 'A frozen workload is required')
    return jobs.start({'settings': read_json(folder / 'protocol.json')['settings'],
                       'example_campaign': value.campaign, 'prompt': value.prompt,
                       'check_native': value.check_native}, 'inference')


@app.get('/api/tuning/inference/{ident}')
def inference(ident: str):
    return observed_status(located(ident, 'inference'), 'inference')


@app.get('/api/tuning/campaigns/{ident}/export')
def export(ident: str):
    folder = located(ident)
    if observed_status(folder)['status'] != 'completed':
        raise HTTPException(409, 'Only a fully verified terminal campaign can be exported')
    from .bundle import export as export_bundle
    try:
        path = export_bundle(ident)
    except (ValueError, OSError) as exc:
        raise HTTPException(422, public_error(exc)) from exc
    return FileResponse(path, filename=path.name, media_type='application/zip')


app.middleware('http')(old.local_boundary)
app.mount('/', old.app)
