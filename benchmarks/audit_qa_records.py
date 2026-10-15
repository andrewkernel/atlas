"""Offline QA artifact provenance checks; no Torch import or asset download."""
import hashlib
import json
from pathlib import Path

try:
    from .derive_qa_resource import canonical, derive
except ImportError:
    from derive_qa_resource import canonical, derive

ROOT = Path(__file__).resolve().parents[1]


def audit(root=ROOT):
    directory = root / 'docs/research/workload-screen'
    load = lambda name: json.loads((directory / name).read_text(encoding='utf-8'))
    digest = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()
    failures = []
    records = {}
    for script, name in (
        ('qa_workload_source_screen.py', 'qa-source-screen.json'),
        ('prepare_qa_assets.py', 'qa-asset-manifest.json'),
        ('prepare_qa_features.py', 'qa-feature-manifest.json'),
        ('qa_compatibility_smoke.py', 'qa-compatibility-smoke.json'),
        ('qa_resource_pilot.py', 'qa-resource-pilot.json'),
        ('worker_ownership_control.py', 'worker-ownership-control.json'),
        ('derive_qa_resource.py', 'qa-resource-derived.json'),
        ('windows_job_control.py', 'windows-job-control.json'),
        ('qa_numerical_control.py', 'qa-numerical-control.json'),
        ('qa_dispatch_control.py', 'qa-dispatch-control.json'),
    ):
        records[name] = load(name)
        if records[name]['source_sha256'] != digest('benchmarks/' + script):
            failures.append('QA recorded source identity stale: ' + script)
    environment = load('qa-environment.json')
    if environment['requirements_sha256'] != digest(environment['requirements_file']):
        failures.append('QA environment requirements identity stale')
    assets = records['qa-asset-manifest.json']
    if assets['source_screen_sha256'] != digest('docs/research/workload-screen/qa-source-screen.json'):
        failures.append('QA acquisition source-screen identity stale')
    features = records['qa-feature-manifest.json']
    if features['asset_manifest_canonical_sha256'] != canonical(assets):
        failures.append('QA feature asset identity stale')
    counts = features['question_counts']
    if len(counts) != features['selected_questions'] or sum(q['features'] for q in counts) != features['features']:
        failures.append('QA feature/question denominators inconsistent')
    if len({q['id_sha256'] for q in counts}) != len(counts) or any(q['answer_containing_features'] < 1 for q in counts):
        failures.append('QA question identity/answer-window coverage inconsistent')
    if sum(q['features'] - q['answer_containing_features'] for q in counts) != features['cls_target_features']:
        failures.append('QA CLS-window denominator inconsistent')
    for name, value in features['arrays'].items():
        expected = [features['features']] + ([384] if name in ('input_ids', 'attention_mask', 'token_type_ids') else [])
        if value['shape'] != expected or value['dtype'] != 'int64':
            failures.append('QA array shape/dtype inconsistent: ' + name)
    runtime = {'environment_canonical_sha256': canonical(environment),
               'asset_manifest_canonical_sha256': canonical(assets),
               'review_sha256': digest('docs/research/workload-screen/qa-execution-review.json')}
    pilot = records['qa-resource-pilot.json']
    helper = digest('benchmarks/qa_native_helpers.py')
    if pilot['helper_source_sha256'] != helper or pilot['environment_helper_sha256'] != digest('benchmarks/board_energy_control.py'):
        failures.append('QA pilot helper source identity stale')
    if pilot['feature_manifest_canonical_sha256'] != canonical(features):
        failures.append('QA pilot feature identity stale')
    for index, trial in enumerate(pilot['trials']):
        worker = trial['worker_result']
        if worker.get('runtime_identities') != runtime:
            failures.append('QA worker runtime identities inconsistent: ' + str(index))
    smoke = records['qa-compatibility-smoke.json']
    if smoke['helper_source_sha256'] != helper or smoke['runtime_identities'] != runtime:
        failures.append('QA compatibility helper/runtime identities stale')
    derived = records['qa-resource-derived.json']
    ownership = records['worker-ownership-control.json']
    if derived['pilot_canonical_sha256'] != canonical(pilot) or derived['ownership_control_canonical_sha256'] != canonical(ownership):
        failures.append('QA derivation input identity stale')
    try:
        recomputed = derive(pilot, ownership)
        for key, value in recomputed.items():
            if derived.get(key) != value:
                failures.append('QA derived arithmetic/validity differs: ' + key)
    except (AssertionError, KeyError, ValueError, TypeError) as error:
        failures.append('QA raw/derived consistency failed: ' + type(error).__name__)
    job = records['windows-job-control.json']
    for key, name in (('job_helper_sha256', 'windows_job_scope.py'),
                      ('ownership_helper_sha256', 'worker_ownership_control.py'),
                      ('environment_helper_sha256', 'board_energy_control.py')):
        if job[key] != digest('benchmarks/' + name):
            failures.append('Windows job helper identity stale: ' + name)
    if [t['mode'] for t in job['trials']] != ['natural', 'terminate_job', 'close_job']:
        failures.append('Windows job lifecycle denominator incomplete')
    for trial in job['trials']:
        if trial['status'] != 'completed' or trial['retained_identities_still_alive'] or trial['cleanup_retained_identities_still_alive']:
            failures.append('Windows job lifecycle completion/cleanup inconsistent')
        if not trial['actual_interpreter_inherited_job'] or not trial['wrong_creation_identity_rejected'] or not trial['use_after_close_rejected']:
            failures.append('Windows job lifecycle identity/control inconsistent')
    numerical = records['qa-numerical-control.json']
    for name, expected in numerical['helper_sha256'].items():
        if expected != digest('benchmarks/' + name):
            failures.append('QA numerical helper identity stale: ' + name)
    if numerical['checker_review_sha256'] != digest('docs/research/competitors/trainparity-execution-review.json'):
        failures.append('QA numerical checker review identity stale')
    if [t['dropout'] for t in numerical['trials']] != numerical['intended_dropout_workers'] or len(numerical['trials']) != 2:
        failures.append('QA numerical worker denominator inconsistent')
    for trial in numerical['trials']:
        worker = trial['worker_result']
        if trial['status'] != 'completed' or trial['retained_identities_still_alive'] or trial['job_members_after_cleanup']:
            failures.append('QA numerical worker completion/cleanup inconsistent')
        if worker['runtime_identities'] != runtime or worker['feature_manifest_canonical_sha256'] != canonical(features):
            failures.append('QA numerical runtime/input identity inconsistent')
        if worker['source_sha256'] != numerical['source_sha256']:
            failures.append('QA numerical worker source identity inconsistent')
        if [a['name'] for a in worker['arms']] != ['reference', 'repeat', 'checkpoint', 'microbatch', 'microbatch_checkpoint']:
            failures.append('QA numerical execution denominator inconsistent')
        if len({a['model_preparation']['initial_state_digest'] for a in worker['arms']}) != 1:
            failures.append('QA numerical initial states differ')
        if len({canonical(a['scientific_start_rng']) for a in worker['arms']}) != 1:
            failures.append('QA numerical scientific-start RNG identities differ')
        if any(a['successful_updates'] != 1 or a['capture_outcome'] != 'PASS' for a in worker['arms']):
            failures.append('QA numerical update/capture incomplete')
        if worker['negative_comparison_to_unperturbed_candidate']['outcome'] != 'FAIL':
            failures.append('QA numerical perturbation control not detected')
    dispatch = records['qa-dispatch-control.json']
    if dispatch['profiler_review_sha256'] != digest('docs/research/workload-screen/qa-profiler-review.json'):
        failures.append('QA dispatch profiler review identity stale')
    for name, expected in dispatch['helper_sha256'].items():
        if expected != digest('benchmarks/' + name):
            failures.append('QA dispatch helper identity stale: ' + name)
    if [t['microbatch'] for t in dispatch['trials']] != dispatch['intended_microbatches'] or len(dispatch['trials']) != 2:
        failures.append('QA dispatch intended-run denominator incomplete')
    for trial in dispatch['trials']:
        worker = trial['worker_result']
        if trial['status'] != 'completed' or trial['retained_identities_still_alive'] or trial['job_members_after_cleanup']:
            failures.append('QA dispatch completion/cleanup inconsistent')
        if worker['runtime_identities'] != runtime or worker['feature_manifest_canonical_sha256'] != canonical(features):
            failures.append('QA dispatch runtime/input identity inconsistent')
        if worker['source_sha256'] != dispatch['source_sha256'] or worker['successful_optimizer_updates'] != 0:
            failures.append('QA dispatch source/update scope inconsistent')
        if not worker['all_present_gradients_finite'] or worker['present_gradient_parameter_tensors'] != 199:
            failures.append('QA dispatch gradient health/coverage inconsistent')
        if sum(worker['cuda_event_names_and_counts'].values()) != worker['cuda_event_count'] and not worker['cuda_event_names_truncated']:
            failures.append('QA dispatch device-event count inconsistent')
    return failures


if __name__ == '__main__':
    observed = audit()
    print(json.dumps({'failures': observed}, indent=2))
    raise SystemExit(bool(observed))
