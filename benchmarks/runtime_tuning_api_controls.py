"""Actual loopback admission controls. Never records the session credential."""
import argparse
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from atlas.tuning.common import utc, write_json


def controls(port, campaign):
    base = f'http://127.0.0.1:{port}'
    with urllib.request.urlopen(base + '/api/session', timeout=5) as response:
        session = json.load(response)
    if not session.get('tuning_active'):
        raise ValueError('This control requires an actual active tuning job')
    paths = [('/api/runs', {}, True, None, 409),
             ('/api/validation/repeat', {'case_id': 'long-000'}, True, None, 409),
             ('/api/tuning/start', {}, True, None, 409),
             ('/api/tuning/generate', {'campaign': campaign}, True, None, 409),
             ('/api/tuning/start', {}, False, None, 403),
             ('/api/tuning/start', {}, True, 'https://unrelated.invalid', 403)]
    rows = []
    for path, body, auth, origin, expected in paths:
        headers = {'Content-Type': 'application/json'}
        if auth:
            headers['X-Atlas-Token'] = session['token']
        if origin:
            headers['Origin'] = origin
        request = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=headers, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                status = response.status
                detail = json.load(response)
        except urllib.error.HTTPError as exc:
            status = exc.code
            detail = json.load(exc)
        rows.append({'path': path, 'authenticated': auth, 'foreign_origin': bool(origin),
                     'expected': expected, 'observed': status, 'detail': detail})
        if status != expected:
            raise ValueError('Admission control failed: ' + path)
    result = {'captured_at': utc(), 'active_job': session['tuning_active'], 'checks': rows,
              'scope': 'Actual local HTTP refusals; no extra model job should launch'}
    write_json(ROOT / 'artifacts/autotuning-qa/api-busy-controls.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('campaign'); parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    print(json.dumps(controls(args.port, args.campaign)))
