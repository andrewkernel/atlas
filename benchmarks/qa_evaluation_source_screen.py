"""Capture one pinned official metric source as text; never import or execute it."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'eee5fdbf62f8613a7812b03419e6b29617b74fd1'


def main():
    records = []
    # Preserve the failed guessed v1 path, then the actual pinned v2 source.
    for name in ('evaluate-v1.1.py', 'evaluate-v2.0.py'):
        url = f'https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/{REVISION}/{name}'
        request = urllib.request.Request(url, headers={'User-Agent': 'Atlas-research-source-screen'})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(256 * 1024 + 1)
                if len(data) > 256 * 1024:
                    raise ValueError('Source cap exceeded')
                data.decode('utf-8')
                target = ROOT / '.tools/qa-source-screen' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                records.append({'url': url, 'status': 'retrieved', 'bytes': len(data),
                                'sha256': hashlib.sha256(data).hexdigest()})
        except urllib.error.HTTPError as error:
            records.append({'url': url, 'status': 'http_error', 'http_status': error.code})
    bert = ROOT / '.tools/qa-source-screen/bert_run_squad'
    result = {'captured_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'records': records,
              'bert_previously_captured_sha256': hashlib.sha256(bert.read_bytes()).hexdigest(),
              'read_scope': 'Official v2 metric normalization, exact/F1, raw-score/aggregation and threshold/main sections; pinned BERT write_predictions and get_final_text sections. Text inspection only; no evaluator import/execution, dataset acquisition, model evaluation or score.',
              'version_boundary': 'The pinned repository has evaluate-v2.0.py, not the guessed v1.1 path. Its metric code informs a proposed answerable-only evaluation; it is not a replicated official v1.1 protocol.'}
    out = ROOT / 'docs/research/workload-screen/qa-evaluation-source-screen.json'
    out.write_bytes((json.dumps(result, indent=2, sort_keys=True)+'\n').encode('utf-8'))
    print(json.dumps({'records': records}))
    if len(records) != 2 or records[1]['status'] != 'retrieved':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
