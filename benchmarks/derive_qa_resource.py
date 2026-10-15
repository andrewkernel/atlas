"""Independent offline arithmetic/identity audit of the frozen QA resource screen."""
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'docs/research/workload-screen'


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def derive(record, ownership):
    trials = record['trials']
    expected = set(itertools.product(range(2), ('eager', 'sdpa'), (32, 4), (False, True)))
    keys = [(t['repetition'], t['attention'], t['microbatch'], t['checkpointed']) for t in trials]
    assert len(keys) == len(set(keys)) == 16 and set(keys) == expected
    assert [list(k) for k in keys] == record['order']
    assert len(ownership['trials']) == 2
    for row in ownership['trials']:
        assert row['status'] == 'completed'
        assert row['interpreter_is_launch_pid'] is False
        assert row['interpreter_parent_is_launch_pid'] is True
        assert row['actual_interpreter_rss_bytes'] > row['launcher_rss_bytes'] + row['child_touched_bytes']
        assert row['retained_identities_still_alive'] == row['cleanup_retained_identities_still_alive'] == 0
    completed = [t for t in trials if t['status'] == 'completed']
    for trial in completed:
        worker = trial['worker_result']
        assert trial['returncode'] == 0 and worker['status'] == 'completed'
        assert worker['successful_updates'] == worker['attempted_updates'] == len(worker['history']) == 8
        assert worker['source_sha256'] == record['source_sha256']
        assert worker['helper_source_sha256'] == record['helper_source_sha256']
        assert worker['feature_manifest_canonical_sha256'] == record['feature_manifest_canonical_sha256']
        assert (worker['attention'], worker['microbatch'], worker['checkpointed']) == (trial['attention'], trial['microbatch'], trial['checkpointed'])
        assert [h['index'] for h in worker['history']] == list(range(8))
        assert [h['window'] for h in worker['history']] == ['first', 'warmup', 'warmup'] + ['measured'] * 5
        assert worker['measured_update_seconds'] == [h['synchronized_update_seconds'] for h in worker['history'][3:]]
        for history in worker['history']:
            assert math.isfinite(history['loss']) and math.isfinite(history['preclip_gradient_norm'])
            assert history['synchronized_update_seconds'] > 0
        for window in ('first_update_memory', 'later_update_memory'):
            memory = worker[window]
            assert memory['peak_reserved_bytes'] >= memory['peak_allocated_bytes'] > 0
            assert memory['peak_allocated_bytes'] >= memory['allocated_bytes']
        assert worker['allocation_peak_above_nominal_device_bytes'] == (worker['later_update_memory']['peak_allocated_bytes'] > worker['environment']['device_total_bytes'])
        preparation = worker['model_preparation']
        assert preparation['loaded_parameter_tensors'] == 197
        assert preparation['loaded_legacy_layernorm_key_renames'] == 50
        assert preparation['parameter_count'] == 108893186
        assert preparation['checkpointed_layers'] == (12 if trial['checkpointed'] else 0)
        assert preparation['initialized_qa_head_keys'] == ['qa_outputs.bias', 'qa_outputs.weight']
    assert len({t['worker_result']['model_preparation']['initial_state_digest'] for t in completed}) == 1
    assert len({canonical(t['worker_result']['scientific_start_rng']) for t in completed}) == 1
    assert len({t['worker_result']['optimizer_group_names_digest'] for t in completed}) == 1
    assert len({canonical(t['worker_result']['runtime_identities']) for t in completed}) == 1
    groups = []
    for attention, micro, checkpointed in itertools.product(('eager', 'sdpa'), (32, 4), (False, True)):
        pair = [t for t in trials if (t['attention'], t['microbatch'], t['checkpointed']) == (attention, micro, checkpointed)]
        successful = [t['worker_result'] for t in pair if t['status'] == 'completed']
        groups.append({'attention': attention, 'microbatch': micro, 'checkpointed': checkpointed,
                       'intended_process_repetitions': len(pair), 'completed': len(successful),
                       'later_peak_allocated_bytes_by_repetition': [w['later_update_memory']['peak_allocated_bytes'] for w in successful],
                       'later_peak_reserved_bytes_by_repetition': [w['later_update_memory']['peak_reserved_bytes'] for w in successful],
                       'median_update_seconds_by_repetition': [statistics.median(w['measured_update_seconds']) for w in successful],
                       'same_seed_exact_digest_match_by_phase': {name: len({w[name] for w in successful}) == 1 if len(successful) == 2 else None
                                                                for name in ('first_state_digest', 'first_post_clip_gradient_digest', 'first_optimizer_state_digest', 'final_state_digest')},
                       'completed_segment_outer_seconds_by_repetition': [t['owned_child_wall_seconds'] for t in pair]})
    return {'intended_trials': len(keys), 'statuses': dict(Counter(t['status'] for t in trials)),
            'completed_updates': sum(t['worker_result'].get('successful_updates', 0) for t in trials),
            'outer_worker_seconds_sum': math.fsum(t['owned_child_wall_seconds'] for t in trials),
            'allocation_above_device_total_trials': sum(t['worker_result'].get('allocation_peak_above_nominal_device_bytes', False) for t in trials),
            'groups': groups, 'worker_peak_rss_bytes': None,
            'host_worker_memory_validity': 'invalid: Popen PID is Windows environment launcher; actual interpreter is its descendant',
            'host_guard_validity': 'worker RSS threshold and launcher-only cancellation unvalidated for actual training process; no stop triggered in this screen',
            'reference_self_consistency': 'full-microbatch configurations do not repeat bitwise state/gradient/optimizer hashes; this record has no tensor-distance or tolerance decision',
            'scope': 'Offline arithmetic and record consistency only. No OOM rescue, physical residency, full fine-tuning, QA quality, outcome equivalence or primary-policy comparison.'}


def main():
    record = json.loads((DIRECTORY / 'qa-resource-pilot.json').read_text(encoding='utf-8'))
    ownership = json.loads((DIRECTORY / 'worker-ownership-control.json').read_text(encoding='utf-8'))
    result = derive(record, ownership)
    result.update({'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'pilot_canonical_sha256': canonical(record), 'ownership_control_canonical_sha256': canonical(ownership)})
    (DIRECTORY / 'qa-resource-derived.json').write_bytes((json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
    print(json.dumps({key: result[key] for key in ('intended_trials', 'statuses', 'completed_updates', 'outer_worker_seconds_sum', 'allocation_above_device_total_trials')}))


if __name__ == '__main__':
    main()
