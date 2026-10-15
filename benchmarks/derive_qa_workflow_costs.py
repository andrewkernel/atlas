"""Scoped worker-window accounting; never a whole-policy cost or GPU-active timer."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'docs/research/workload-screen'


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def derive(records):
    rows = []
    for name, record in records.items():
        trials = record['trials'] if 'trials' in record else [record['trial']]
        outer_key = 'owned_child_wall_seconds' if name == 'qa-resource-pilot.json' else 'outer_owned_job_seconds'
        completed = [trial for trial in trials if trial['status'] == 'completed']
        updates = executions = 0
        nested_updates = []
        reported_captures = []
        reported_child_lifetimes = []
        for trial in completed:
            worker = trial['worker_result']
            child_key = 'worker_wall_seconds_before_final_write' if name == 'qa-resource-pilot.json' else 'worker_wall_seconds'
            lifetime = worker[child_key]
            assert 0 <= lifetime <= trial[outer_key]
            reported_child_lifetimes.append(lifetime)
            if name == 'qa-numerical-control.json':
                updates += sum(arm['successful_updates'] for arm in worker['arms'])
                executions += len(worker['arms'])
                reported_captures.extend(arm['capture_seconds'] for arm in worker['arms'])
            elif name == 'qa-adam-sensitivity-control.json':
                updates += sum(arm['successful_updates'] for arm in worker['executions'])
                executions += len(worker['executions'])
            else:
                updates += worker.get('successful_updates', worker.get('successful_optimizer_updates', 0))
                executions += 1
            if name == 'qa-resource-pilot.json':
                nested_updates.extend(step['synchronized_update_seconds'] for step in worker['history'])
        outer = math.fsum(trial[outer_key] for trial in trials)
        child = math.fsum(reported_child_lifetimes)
        assert math.fsum(nested_updates) <= child and math.fsum(reported_captures) <= child
        rows.append({'record': name, 'attempted_worker_windows': len(trials), 'completed_workers': len(completed),
                     'bert_model_executions': executions, 'successful_bert_optimizer_updates': updates,
                     'reported_supervised_window_seconds_sum': outer,
                     'reported_child_lifetime_seconds_sum': child,
                     'reported_complete_update_timer_count': len(nested_updates),
                     'reported_complete_update_seconds_sum': math.fsum(nested_updates) if nested_updates else None,
                     'reported_main_capture_timer_count': len(reported_captures),
                     'reported_main_capture_seconds_sum': math.fsum(reported_captures) if reported_captures else None,
                     'cost_scope': 'Declared timer windows only; child/update/capture intervals are nested, not additive to outer.'})
    return {'studies': rows,
            'attempted_worker_windows': sum(row['attempted_worker_windows'] for row in rows),
            'bert_model_executions': sum(row['bert_model_executions'] for row in rows),
            'successful_bert_optimizer_updates': sum(row['successful_bert_optimizer_updates'] for row in rows),
            'reported_supervised_window_seconds_sum': math.fsum(row['reported_supervised_window_seconds_sum'] for row in rows),
            'scope': 'Sum of sequential declared supervised windows across development studies; not a whole-policy/end-to-end research cost, exclusive allocation, GPU-active time, independent sample count or fair performance comparison.',
            'omissions': ['Controller imports/environment construction/job creation outside those timers',
                          'Parent-side array analysis, source retrieval, environment installation, engineering and adjudication effort',
                          'Unreported capture/comparison/negative-control subdivisions cannot be inferred as a residual category'],
            'nonadditivity': 'Child lifetimes contain measured updates/captures. Adding nested categories to outer double counts.'}


def main():
    names = ['qa-resource-pilot.json', 'qa-numerical-control.json', 'qa-dispatch-control.json',
             'qa-adam-sensitivity-control.json', 'qa-reference32-control.json']
    records = {name: json.loads((DIRECTORY / name).read_text(encoding='utf-8')) for name in names}
    result = derive(records)
    result.update({'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'input_canonical_sha256': {name: canonical(record) for name, record in records.items()}})
    (DIRECTORY / 'qa-workflow-costs.json').write_bytes((json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
    print(json.dumps({name: result[name] for name in ('attempted_worker_windows', 'bert_model_executions', 'successful_bert_optimizer_updates', 'reported_supervised_window_seconds_sum')}))


if __name__ == '__main__':
    main()
