"""Build the small browser snapshot from immutable recorded research, without ML imports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / 'frontend'
RESEARCH = ROOT / 'docs/research/workload-screen'
REVISION = '694bf49fd351f1315201f665414e021e224e03f5'
DOCS = f'https://github.com/andrewkernel/atlas/blob/{REVISION}/docs/research/'


def make_snapshot():
    inputs = {}
    def read(name):
        data = (RESEARCH/name).read_bytes()
        inputs[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    resource = read('qa-resource-derived.json')
    costs = read('qa-workflow-costs.json')
    environments = read('post-study-environments.json')
    read('qa-default-attention-control.json')
    assert resource['intended_trials'] == 16 and resource['completed_updates'] == 128
    configurations = []
    for group in resource['groups']:
        attention, micro, checkpoint = group['attention'], group['microbatch'], group['checkpointed']
        peaks = group['later_peak_allocated_bytes_by_repetition']
        assert len(peaks) == 2 and peaks[0] == peaks[1]
        configurations.append({'id': f'{attention}-{micro}-{int(checkpoint)}', 'attention': attention,
            'microbatch': micro, 'checkpoint': checkpoint, 'allocatedGiB': peaks[0] / 1024**3,
            'updateSeconds': group['median_update_seconds_by_repetition'],
            'completed': group['completed'], 'intended': group['intended_process_repetitions'],
            'updatesPerProcess': 8, 'hashRepeat': all(group['same_seed_exact_digest_match_by_phase'].values()),
            'nativeDefault': attention == 'sdpa' and micro == 32 and not checkpoint})
    configurations.sort(key=lambda c: (not c['nativeDefault'], c['attention'] != 'sdpa', c['microbatch'] != 32, c['checkpoint']))
    evidence = [
        {'name': 'Checkpoint replay', 'scope': '3 features · one update', 'status': 'PASS', 'tone': 'pass',
         'description': 'Existing TrainParity captures match for the repeat and checkpoint comparisons in both tested dropout conditions. This supports the captured fields in the reduced fixture; full-training QA quality remains unverified.', 'source': 'qa-numerical-control.md'},
        {'name': 'Microbatch comparison', 'scope': '3 features · one update', 'status': 'FAIL', 'tone': 'fail',
         'description': 'The unchanged finite predicate fails with and without dropout. Selected native AdamW replays reproduce update sensitivity. A literal numerical failure does not establish a scientific-quality failure.', 'source': 'qa-adam-sensitivity-control.md'},
        {'name': 'Reference repeatability', 'scope': 'Logical batch 32 · selected fields', 'status': 'Variation', 'tone': 'unknown',
         'description': 'Unchanged fresh-process references differ. Five and two position-embedding elements exceed the stated diagnostic in the two dropout conditions. The root cause and full-run consequences are unverified.', 'source': 'qa-reference32-control.md'},
        {'name': 'Worker memory', 'scope': 'Historical resource pilot', 'status': 'Invalid', 'tone': 'fail',
         'description': 'The resource pilot observed the environment launcher instead of its actual Python worker. Worker RSS is unavailable. CUDA allocation and synchronized update timers retain their separate scope.', 'source': 'worker-ownership-protocol.md'},
        {'name': 'Owned-worker cleanup', 'scope': 'Suspended-launch Windows job', 'status': 'Observed', 'tone': 'pass',
         'description': 'Later Windows job controls validate normal completion, termination and handle-close cleanup. Exit zero alone cannot establish completion. These controls do not prove Linux ownership or hostile-code isolation.', 'source': 'windows-job-control.md'},
        {'name': 'QA quality', 'scope': 'Full fine-tuning and evaluation', 'status': 'Unverified', 'tone': 'unknown',
         'description': 'No full QA fine-tuning or model evaluation has run. The prospective protocol requires legal answer spans, cross-window decoding, complete prediction coverage and justified paired outcome analysis.', 'source': 'qa-quality-evaluation-spec.md'}
    ]
    profiles = []
    names = {'checker': 'Existing checker', 'core': 'Research core', 'interop': 'Shared integration',
             'original_cuda': 'Native CUDA', 'planner': 'Existing planner', 'plot': 'Plotting',
             'qa': 'BERT QA', 'telemetry': 'Board telemetry'}
    for key, value in environments['profiles'].items():
        profiles.append({'id': key, 'name': names[key], 'python': value['python'],
                         'torch': value['packages'].get('torch'), 'packages': len(value['packages']),
                         'checkedSources': len(value['selected_files']),
                         'matched': not value['prior_version_mismatches'] and not value['additional_distributions'] and all(f['matches_installed_record'] for f in value['selected_files']),
                         'requirements': f'https://github.com/andrewkernel/atlas/blob/{REVISION}/'+value['requirements_file']})
    return {'schemaVersion': 1, 'snapshotDate': '2026-10-15', 'researchRevision': REVISION,
            'sourceInputs': inputs, 'sourceBase': DOCS+'workload-screen/',
            'scope': 'Saved development evidence on one physical GPU. No policy advantage, useful real OOM rescue, full QA quality equivalence or preservation of scientific conclusions is established.',
            'device': {'name': 'NVIDIA GeForce RTX 3060', 'nominalGB': 12, 'cudaTotalGiB': 12603752448/1024**3, 'os': 'Windows'},
            'configurations': configurations, 'evidence': evidence, 'environments': profiles,
            'costs': {key: costs[key] for key in ('attempted_worker_windows', 'bert_model_executions', 'successful_bert_optimizer_updates', 'reported_supervised_window_seconds_sum')},
            'environmentScope': environments['scope']}


if __name__ == '__main__':
    target = FRONTEND/'data/snapshot.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((json.dumps(make_snapshot(), ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n').encode('utf-8'))
    print(f'Wrote {target.stat().st_size:,} bytes; 8 native configurations, 6 scoped evidence entries, 8 environment profiles.')
