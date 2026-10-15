"""Offline QA artifact provenance checks; no Torch import or asset download."""
import hashlib
import json
import math
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
        ('qa_adam_sensitivity_control.py', 'qa-adam-sensitivity-control.json'),
        ('qa_reference32_control.py', 'qa-reference32-control.json'),
        ('verify_qa_reference32_artifacts.py', 'qa-reference32-reconstruction.json'),
        ('derive_qa_workflow_costs.py', 'qa-workflow-costs.json'),
        ('plot_qa_resource.py', 'qa-resource-figure.json'),
        ('snapshot_post_continuation_environments.py', 'post-study-environments.json'),
        ('qa_default_attention_control.py', 'qa-default-attention-control.json'),
        ('qa_evaluation_source_screen.py', 'qa-evaluation-source-screen.json'),
        ('qa_metric_coverage_control.py', 'qa-metric-coverage-control.json'),
    ):
        records[name] = load(name)
        if records[name]['source_sha256'] != digest('benchmarks/' + script):
            failures.append('QA recorded source identity stale: ' + script)
    environment = load('qa-environment.json')
    if environment['requirements_sha256'] != digest(environment['requirements_file']):
        failures.append('QA environment requirements identity stale')
    assets = records['qa-asset-manifest.json']
    evaluation = records['qa-evaluation-source-screen.json']
    if evaluation['bert_previously_captured_sha256'] != records['qa-source-screen.json']['records']['bert_run_squad']['sha256']:
        failures.append('QA evaluation decoder source identity inconsistent')
    if [entry['status'] for entry in evaluation['records']] != ['http_error', 'retrieved'] or evaluation['records'][0]['http_status'] != 404:
        failures.append('QA evaluation source retrieval/version boundary changed')
    metric = records['qa-metric-coverage-control.json']
    if metric['reviewed_metric_source_sha256'] != evaluation['records'][1]['sha256']:
        failures.append('QA metric control reviewed source identity inconsistent')
    metric_trials = metric['trials']
    if [t['case'] for t in metric_trials] != ['complete', 'missing_c', 'unexpected_id', 'all_missing']:
        failures.append('QA metric intended-case denominator inconsistent')
    else:
        if [t['coverage_guard']['status'] for t in metric_trials] != ['complete', 'error', 'error', 'error']:
            failures.append('QA metric coverage guard control inconsistent')
        for trial in metric_trials:
            aggregate = trial['aggregate_default']
            if aggregate is not None:
                scores = trial['exact_by_id']
                if aggregate['total'] != len(scores) or aggregate['exact'] != 100. * sum(scores.values()) / len(scores):
                    failures.append('QA metric aggregate arithmetic inconsistent')
        if metric_trials[1]['explicit_ids_error'] != 'KeyError' or metric_trials[3]['aggregation_error'] != 'ZeroDivisionError':
            failures.append('QA metric missing-score exception control inconsistent')
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
    default = records['qa-default-attention-control.json']
    for name, expected in default['helper_sha256'].items():
        if expected != digest('benchmarks/' + name):
            failures.append('QA constructor-default helper identity stale: ' + name)
    worker = default['worker_result']
    if default['status'] != 'completed' or default['job_members_after_cleanup'] != 0:
        failures.append('QA constructor-default completion/cleanup inconsistent')
    if worker['runtime_identities'] != runtime or worker['parameter_devices'] != ['cpu'] or worker['parameter_count'] != 108893186:
        failures.append('QA constructor-default runtime/device/model scope inconsistent')
    if worker['attention_before_constructor'] is not None or worker['attention_after_constructor'] != 'sdpa':
        failures.append('QA recorded constructor-default observation changed')
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
    sensitivity = records['qa-adam-sensitivity-control.json']
    for name, expected in sensitivity['helper_sha256'].items():
        if expected != digest('benchmarks/' + name):
            failures.append('QA sensitivity helper identity stale: ' + name)
    if sensitivity['optimizer_review_sha256'] != digest('docs/research/workload-screen/qa-optimizer-review.json'):
        failures.append('QA sensitivity optimizer review identity stale')
    original = (root / 'benchmarks/qa_owned_worker.py').read_text(encoding='utf-8')
    expected = original.replace("ALLOWED = {'qa_numerical_control.py', 'qa_dispatch_control.py'}", "ALLOWED = {'qa_adam_sensitivity_control.py'}")
    if (root / 'benchmarks/qa_sensitivity_supervisor.py').read_text(encoding='utf-8') != expected:
        failures.append('QA sensitivity supervisor differs beyond declared registry change')
    trial = sensitivity['trial']
    worker = trial['worker_result']
    if trial['status'] != 'completed' or trial['retained_identities_still_alive'] or trial['job_members_after_cleanup']:
        failures.append('QA sensitivity completion/cleanup inconsistent')
    if worker['runtime_identities'] != runtime or worker['source_sha256'] != sensitivity['source_sha256']:
        failures.append('QA sensitivity runtime/source identity inconsistent')
    if [x['name'] for x in worker['executions']] != ['reference', 'microbatch'] or len(worker['selected_components']) != 10:
        failures.append('QA sensitivity execution/selection denominator inconsistent')
    if any(not x['equal_gradient_positive_exact'] for x in worker['replays']):
        failures.append('QA sensitivity equal-gradient control failed')
    reference = records['qa-reference32-control.json']
    for name, expected in reference['helper_sha256'].items():
        if expected != digest('benchmarks/' + name):
            failures.append('QA logical-32 reference helper identity stale: ' + name)
    expected = original.replace("ALLOWED = {'qa_numerical_control.py', 'qa_dispatch_control.py'}", "ALLOWED = {'qa_reference32_control.py'}")
    if (root / 'benchmarks/qa_reference_supervisor.py').read_text(encoding='utf-8') != expected:
        failures.append('QA reference supervisor differs beyond declared registry change')
    order = [[t['dropout'], t['repetition']] for t in reference['trials']]
    if order != reference['order'] or len(order) != 4 or len({tuple(x) for x in order}) != 4:
        failures.append('QA reference intended-run denominator inconsistent')
    expected_keys = {phase + '__' + name for name in reference['selected_parameters']
                     for phase in ('parameter', 'gradient', 'exp_avg', 'exp_avg_sq')}
    for trial in reference['trials']:
        worker = trial['worker_result']
        if trial['status'] != 'completed' or trial['retained_identities_still_alive'] or trial['job_members_after_cleanup']:
            failures.append('QA reference completion/cleanup inconsistent')
        if worker['runtime_identities'] != runtime or worker['source_sha256'] != reference['source_sha256']:
            failures.append('QA reference runtime/source identity inconsistent')
        if worker['feature_manifest_canonical_sha256'] != canonical(features) or worker['successful_updates'] != 1:
            failures.append('QA reference input/update scope inconsistent')
        if set(worker['selected_arrays']) != expected_keys or worker['selected_state_file']['bytes'] >= 32 * 1024**2:
            failures.append('QA reference selected-state coverage/size inconsistent')
    for pair in reference['pairs']:
        trials = [t['worker_result'] for t in reference['trials'] if t['dropout'] == pair['dropout']]
        if len(trials) != 2 or pair['status'] != 'measured' or set(pair['selected_tensor_differences']) != expected_keys:
            failures.append('QA reference pair denominator inconsistent')
            continue
        first, second = trials
        for name, field in (('whole_model_digest_equal', 'post_model_state_digest'),
                            ('whole_gradient_digest_equal', 'post_clip_gradient_digest'),
                            ('scientific_end_rng_equal', 'scientific_end_rng')):
            if pair[name] != (first[field] == second[field]):
                failures.append('QA reference paired identity/equality inconsistent: ' + name)
        for name, metric in pair['selected_tensor_differences'].items():
            if not metric['finite'] or not 0 <= metric['diagnostic_exceeded_elements'] <= metric['elements']:
                failures.append('QA reference field health/count inconsistent: ' + name)
            maximum = abs(metric['reference_at_maximum'] - metric['candidate_at_maximum'])
            if not math.isclose(maximum, metric['maximum_absolute_difference'], rel_tol=1e-12, abs_tol=0.):
                failures.append('QA reference maximum arithmetic inconsistent: ' + name)
            denominator = metric['reference_square_sum']
            value = (metric['difference_square_sum'] / denominator)**.5 if denominator else None
            saved = metric['relative_l2_error']
            if (value is None) != (saved is None) or (value is not None and not math.isclose(value, saved, rel_tol=1e-12, abs_tol=0.)):
                failures.append('QA reference norm arithmetic inconsistent: ' + name)
    reconstruction = records['qa-reference32-reconstruction.json']
    if reconstruction['input_canonical_sha256'] != canonical(reference) or reconstruction['failures']:
        failures.append('QA reference reconstruction input identity/results inconsistent')
    if reconstruction['arrays_checked'] != 32 or reconstruction['elements_checked'] != 7876624:
        failures.append('QA reference reconstruction coverage inconsistent')
    costs = records['qa-workflow-costs.json']
    for name, expected in costs['input_canonical_sha256'].items():
        if expected != canonical(load(name)):
            failures.append('QA cost input identity stale: ' + name)
    if (costs['attempted_worker_windows'], costs['bert_model_executions'], costs['successful_bert_optimizer_updates']) != (25, 34, 144):
        failures.append('QA cost execution/update units inconsistent')
    if not math.isclose(math.fsum(row['reported_supervised_window_seconds_sum'] for row in costs['studies']), costs['reported_supervised_window_seconds_sum'], rel_tol=1e-12):
        failures.append('QA supervised-window cost arithmetic inconsistent')
    figure = records['qa-resource-figure.json']
    for name, expected in figure['input_canonical_sha256'].items():
        if expected != canonical(load(name)):
            failures.append('QA figure input identity stale: ' + name)
    if figure['figure']['sha256'] != digest(figure['figure']['path']) or figure['requirements_sha256'] != digest(figure['requirements_file']):
        failures.append('QA figure/requirements identity stale')
    post = records['post-study-environments.json']
    for name, expected in post['helper_sha256'].items():
        if expected != digest('benchmarks/' + name):
            failures.append('Post-study environment helper identity stale: ' + name)
    prior = json.loads((root / 'docs/research/experiments/final-environment-inventory.json').read_text(encoding='utf-8'))
    if post['earlier_inventory_canonical_sha256'] != canonical(prior) or post['qa_before_canonical_sha256'] != canonical(environment) or post['plot_figure_record_canonical_sha256'] != canonical(figure):
        failures.append('Post-study environment prior input identity stale')
    if post['qa_asset_manifest_canonical_sha256'] != canonical(assets) or post['qa_feature_manifest_canonical_sha256'] != canonical(features):
        failures.append('Post-study asset/feature manifest identity stale')
    if post['failures'] or len(post['profiles']) != 8 or len(post['asset_checks']) != 8:
        failures.append('Post-study reconciliation incomplete/failed')
    for name, profile in post['profiles'].items():
        if profile['requirements_sha256'] != digest(profile['requirements_file']) or profile['prior_version_mismatches'] or profile['additional_distributions']:
            failures.append('Post-study environment requirements/versions inconsistent: ' + name)
        if any(not entry['matches_installed_record'] for entry in profile['selected_files']):
            failures.append('Post-study selected installed source mismatch: ' + name)
    if sum(len(profile['selected_files']) for profile in post['profiles'].values()) != 54:
        failures.append('Post-study selected-source coverage inconsistent')
    if any(not entry['matches_acquisition'] for entry in post['asset_checks'].values()) or post['feature_npz_sha256'] != features['npz_sha256']:
        failures.append('Post-study input byte identity mismatch')
    for label, value in post['reviewed_reference_checks'].items():
        review = json.loads((root / f'docs/research/competitors/{label}-execution-review.json').read_text(encoding='utf-8'))
        if not value['all_match'] or value['review_canonical_sha256'] != canonical(review):
            failures.append('Post-study reviewed source identity mismatch: ' + label)
    return failures


if __name__ == '__main__':
    observed = audit()
    print(json.dumps({'failures': observed}, indent=2))
    raise SystemExit(bool(observed))
