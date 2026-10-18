"""Actual registered timeout/cancellation controls; never kills unrelated processes."""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from atlas.tuning.common import campaign_dir, read_json, utc, write_json, domain
from atlas.tuning.profiles import state
from atlas.tuning.registry import settings


def control(mode, port=8765):
    base = f'http://127.0.0.1:{port}'
    with urllib.request.urlopen(base + '/api/session', timeout=5) as response:
        session = json.load(response)
    if session.get('active_run') or session.get('tuning_active'):
        raise RuntimeError('Finish the active local job before this lifecycle control')

    def api(path, body=None):
        request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'X-Atlas-Token': session['token']},
            method='GET' if body is None else 'POST')
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    config = {'context': 512, 'dtype': 'float32', 'generation_tokens': 16,
              'goal': 'fastest', 'budget_mib': None, 'max_wall_s': 60, 'auto_apply': True}
    initial = state(domain(settings(config)))
    created = api('/api/tuning/start', config)
    ident = created['id']; folder = campaign_dir(ident)
    started = time.monotonic(); accepted = None; live_at_cancel = None
    print(json.dumps({'id': ident, 'mode': mode}), flush=True)
    while time.monotonic()-started < 150:
        doc = api('/api/tuning/campaigns/' + ident)['status']
        if mode == 'cancel' and accepted is None:
            case = doc.get('active_case')
            raw_path = folder / 'runs' / case / 'result.json' if case else None
            if raw_path and raw_path.is_file():
                raw = read_json(raw_path)
                owner = raw.get('execution_process')
                if raw.get('status') == 'running' and raw.get('model') and owner:
                    try:
                        process = psutil.Process(owner['pid'])
                        if process.create_time() == owner['created_at']:
                            live_at_cancel = owner
                            accepted = api('/api/tuning/jobs/' + ident + '/cancel', {})
                    except psutil.NoSuchProcess:
                        pass
        if doc['status'] in ('cancelled', 'timed_out', 'failed', 'completed', 'resource_limited'):
            expected = 'cancelled' if mode == 'cancel' else 'timed_out'
            if doc['status'] != expected:
                raise ValueError('Unexpected lifecycle outcome: ' + doc['status'])
            ownership_path = folder / 'controller-ownership.json'
            if ownership_path.is_file() and not api('/api/session').get('tuning_active'):
                break
        time.sleep(.25)
    else:
        api('/api/tuning/jobs/' + ident + '/cancel', {})
        raise RuntimeError('Lifecycle control exceeded its diagnostic bound')
    ownership = read_json(folder / 'controller-ownership.json')
    live = []
    for owner in ownership['observed_processes']:
        try:
            if psutil.Process(owner['pid']).create_time() == owner['created_at']:
                live.append(owner)
        except psutil.NoSuchProcess:
            pass
    if live or not ownership['closed']:
        raise ValueError('Owned processes did not exit')
    after = state(domain(settings(config)))
    if after != initial:
        raise ValueError('An interrupted control changed active settings')
    result = {'captured_at': utc(), 'campaign': ident, 'mode': mode, 'status': doc['status'],
              'elapsed_s': time.monotonic()-started, 'controller_scope_closed': ownership['closed'],
              'observed_owned_processes': ownership['observed_processes'], 'matching_processes_still_live': live,
              'active_state_before': initial, 'active_state_after': after,
              'accepted_cancel': accepted, 'actual_model_process_live_at_cancel': live_at_cancel,
              'scope': 'Actual registered public-text model run and owned process exit; no hostile-code sandbox claim'}
    write_json(ROOT / 'artifacts/autotuning-qa' / ('actual-' + mode + '.json'), result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['timeout', 'cancel'])
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args(); print(json.dumps(control(args.mode, args.port)))
