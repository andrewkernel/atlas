"""Authored controls use invented temporary arrays/timings, never model evidence."""
import copy
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from atlas.tuning import common, policy, profiles, registry, verify


def protocol():
    return {'settings': registry.settings({}), 'baseline': 'native-t4',
            'profiles': registry.profiles(2048)}


def calibration(p):
    return [{'config': c, 'status': 'completed', 'passed': True,
             'generation_ms': 80 if c['profile']['id'] == 'native-t1' else 100,
             'peak_allocated_mib': 2400 if c['profile']['method'] == 'hybrid' else 2800}
            for c in registry.calibration_cases(p)]


def confirmations(p, nominee, ratio=1.25):
    rows = []
    for c in registry.confirmation_cases(p, nominee):
        rows.append({'config': c, 'status': 'completed', 'passed': True,
                     'generation_ms': 100 if c['profile']['id'] == p['baseline'] else 100/ratio,
                     'peak_allocated_mib': 2400 if c['profile']['method'] == 'hybrid' else 2800})
    return rows


def test_selection_is_complete_and_controls_gate_every_candidate():
    p = protocol(); rows = calibration(p)
    assert policy.nominate(p, rows)['profile'] == 'native-t1'
    assert verify.calibration_summary(p, rows)[1] == 'native-t1'
    with pytest.raises(ValueError):
        policy.nominate(p, rows[:-1])
    next(r for r in rows if r['config']['id'] == 'cal-0-resident-t4')['passed'] = False
    assert policy.nominate(p, rows)['profile'] is None
    assert verify.calibration_summary(p, rows)[1] is None


@pytest.mark.parametrize('status', ['failed', 'oom', 'timed_out', 'cancelled', 'running'])
def test_unknown_and_failures_cannot_win(status):
    p = protocol(); rows = calibration(p)
    row = next(r for r in rows if r['config']['id'] == 'cal-1-native-t1')
    row.update(status=status, passed=None)
    assert policy.nominate(p, rows)['profile'] != 'native-t1'


def test_every_pair_must_clear_threshold_and_guards():
    p = protocol(); cal = calibration(p)
    table, nominee = verify.calibration_summary(p, cal)
    confirm = confirmations(p, nominee)
    assert verify.decide(p, cal+confirm, table, nominee)['outcome'] == 'confirmed_improvement'
    next(r for r in confirm if r['config']['stage'] == 'confirmation' and
         r['config']['profile']['id'] == nominee)['generation_ms'] = 100/1.05
    assert verify.decide(p, cal+confirm, table, nominee)['outcome'] == 'keep_current'
    next(r for r in confirm if r['config']['stage'] == 'guard')['passed'] = None
    assert verify.decide(p, cal+confirm, table, nominee)['outcome'] == 'inconclusive'
    with pytest.raises(ValueError):
        verify.decide(p, cal+confirm[:-1], table, nominee)


def test_slower_memory_fit_is_distinct_and_failed_candidate_cannot_apply():
    p = protocol(); p['settings'] = registry.settings({'goal': 'budget', 'budget_mib': 2560})
    cal = calibration(p); table, nominee = verify.calibration_summary(p, cal)
    assert nominee.startswith('hybrid')
    confirm = confirmations(p, nominee, ratio=.5)
    result = verify.decide(p, cal+confirm, table, nominee)
    assert result['outcome'] == 'confirmed_memory_fit' and result['median_saved_ms'] is None
    next(r for r in confirm if r['config']['stage'] == 'confirmation' and
         r['config']['profile']['id'] == nominee)['peak_allocated_mib'] = 2561
    assert verify.decide(p, cal+confirm, table, nominee)['profile'] is None


def test_native_nomination_has_six_real_confirmation_workers():
    p = protocol()
    rows = registry.confirmation_cases(p, 'native-t4')
    assert len([r for r in rows if r['stage'] == 'confirmation']) == 6
    assert len({r['id'] for r in rows}) == len(rows)


def test_full_arrays_not_argmax_or_saved_pass_and_hash_tamper(tmp_path):
    expected = np.zeros((32, 5), dtype=np.float32)
    actual = expected.copy(); actual[31, 4] = .001
    checks = verify.compare(actual, expected)
    assert not checks[-1]['passed'] and sum(c['violating_elements'] for c in checks) == 1
    lying = copy.deepcopy(checks); lying[-1]['passed'] = True
    with pytest.raises(ValueError):
        verify.check_saved_checks(checks, lying)
    item = common.save_array(tmp_path, actual)
    assert common.save_array(tmp_path, actual) == item
    assert len(list((tmp_path / 'blobs').glob('*.npy'))) == 1
    path = tmp_path / item['path']; path.write_bytes(path.read_bytes()[:-1] + b'x')
    with pytest.raises(ValueError):
        common.load_array(tmp_path, item)


def test_nonfinite_output_is_a_failure_not_an_unknown_pass():
    actual = np.zeros((32, 5), dtype=np.float32); actual[0, 0] = np.nan
    result = verify.compare(actual, np.zeros_like(actual))
    assert result[0]['passed'] is False and result[0]['finite'] is False


def test_measured_window_cannot_omit_work():
    row = {'samples_ms': [3]*15, 'generation_ms': 55, 'prefill_ms': 5, 'first_token_ms': 6,
           'allocated_peak_mib': 10, 'reserved_peak_mib': 20, 'generated_token_ids': [1]*16}
    verify.check_measurement(row, 16, 100)
    row['generation_ms'] = 49
    with pytest.raises(ValueError):
        verify.check_measurement(row, 16, 100)


def test_transactional_cancel_commit_cas_and_revert(tmp_path):
    key = 'test-domain'; expected = profiles.state(key, tmp_path)
    applied = profiles.transition(key, None, expected, 'test', 'campaign-a', tmp_path)
    with pytest.raises(profiles.Conflict, match='already committed'):
        profiles.cancel('campaign-a', tmp_path)
    profiles.cancel('campaign-b', tmp_path)
    with pytest.raises(profiles.Conflict, match='cancelled'):
        profiles.transition(key, None, applied, 'automatic', 'campaign-b', tmp_path)
    with pytest.raises(profiles.Conflict, match='changed'):
        profiles.transition(key, None, expected, 'test', home=tmp_path)
    reverted = profiles.revert(key, applied, tmp_path)
    assert reverted['revision'] == 2 and len(profiles.history(tmp_path)) == 2
    assert profiles.cancelled('campaign-b', tmp_path)


def test_profile_bytes_and_invalid_activation(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        profiles.validate('missing', tmp_path)
    with pytest.raises(ValueError):
        profiles.read('../unsafe', tmp_path)
    path = tmp_path / 'profiles/profile-invalid.json'
    common.write_json(path, {'schema_version': 1})
    with pytest.raises(ValueError, match='identity'):
        profiles.read('profile-invalid', tmp_path)


@pytest.mark.parametrize('bad', [{'context': True}, {'context': 1024}, {'generation_tokens': 1},
                               {'dtype': 'int4'}, {'goal': 'budget'}, {'goal': 'fastest', 'budget_mib': 1000},
                               {'goal': 'budget', 'budget_mib': float('nan')}, {'os_power_plan': 'fast'},
                               {'max_wall_s': 999999}])
def test_registered_settings_reject_unscoped_options(bad):
    with pytest.raises(ValueError):
        registry.settings(bad)


def test_wrapper_origin_auth_schema_and_conflicts(monkeypatch):
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    from atlas.tuning import service
    client = TestClient(service.app)
    assert client.get('/api/session').json()['runtime_tuning'] is True
    assert client.post('/api/tuning/start', json={}).status_code == 403
    headers = {'x-atlas-token': service.old.TOKEN}
    assert client.post('/api/tuning/start', json={}, headers={**headers, 'origin': 'https://elsewhere.test'}).status_code == 403
    assert client.post('/api/tuning/start', json={'context': True}, headers=headers).status_code == 422
    assert client.post('/api/tuning/start', json={'context': 1000}, headers=headers).status_code == 422
    assert client.get('/api/tuning/campaigns/not-known').status_code == 404
    monkeypatch.setattr(service.jobs, 'active', 'authored-busy-control')
    assert client.post('/api/runs', json={}, headers=headers).status_code == 409
    assert client.post('/api/validation/repeat', json={'case_id': 'long-001'}, headers=headers).status_code == 409
    assert client.post('/api/tuning/start', json={}, headers=headers).status_code == 409


@pytest.fixture
def authored_campaign(tmp_path, monkeypatch):
    """Tiny invented vocabulary exercises reconstruction without publishing fake ML runs."""
    folder = tmp_path / 'authored-fixture'; folder.mkdir()
    p = {**protocol(), 'identity': {'authored': True}, 'sources': {}, 'assets': {}, 'vocabulary_size': 5,
         'inputs': [], 'contract': {'positions': 32, 'atol': 1e-4, 'rtol': 1e-4}}
    for prefix in ('cal', 'hold'):
        for n in range(2):
            item = common.save_array(folder, np.zeros(2080, dtype='<i8'))
            p['inputs'].append({'id': f'{prefix}-{n}', 'artifact': item})
    common.write_json(folder / 'protocol.json', p)
    monkeypatch.setattr(verify, 'verify_protocol', lambda *args: p)
    sha = common.file_digest(folder / 'protocol.json')
    ledger = {'protocol_sha256': sha, 'cases': []}
    logits = common.save_array(folder, np.zeros((32, 5), dtype=np.float32))

    def collect(cases):
        for c in cases:
            measurement = {'samples_ms': [1]*15, 'generation_ms': 80 if c['profile']['id'] == 'native-t1' else 100,
                           'prefill_ms': 1, 'first_token_ms': 2, 'allocated_peak_mib': 2800,
                           'reserved_peak_mib': 3000, 'generated_token_ids': [0]*16}
            proc = {'pid': len(ledger['cases'])+1, 'created_at': len(ledger['cases'])+1}
            raw = {'id': c['id'], 'campaign': folder.name, 'config': c, 'status': 'completed',
                   'protocol_sha256': sha, 'environment': {'identity': p['identity'], 'source_sha256': {}},
                   'configured': {'dtype': 'float32', 'threads': c['profile']['threads'], 'interop_threads': 1,
                                  'tf32': False, 'matmul_precision': 'highest'},
                   'input_artifact': next(i['artifact'] for i in p['inputs'] if i['id'] == c['input']),
                   'execution_process': proc}
            raw['model'] = {'class': 'Qwen3ForCausalLM', 'parameter_count': 596049920, 'parameter_dtypes': ['torch.float32']}
            if c['stage'] == 'reference':
                raw['artifacts'] = {'reference_logits': logits}
            else:
                ref = folder / 'runs' / ('ref-'+c['input']) / 'result.json'
                raw.update(artifacts={'candidate_logits': logits}, reference_sha256=common.file_digest(ref),
                           logit_checks=verify.compare(np.zeros((32,5)), np.zeros((32,5))),
                           correctness={'passed': True}, warmup=measurement, measurement=measurement,
                           phase_times={'warm_end': 1, 'measure_start': 2, 'measure_end': 3, 'validation_start': 4},
                           mean_nll=math.log(5))
            path = folder / 'runs' / c['id'] / 'result.json'; common.write_json(path, raw)
            ledger['cases'].append({'id': c['id'], 'status': 'completed', 'sha256': common.file_digest(path),
                                    'ownership': {'closed': True, 'observed_processes': [proc]}})
        common.write_json(folder / 'execution.json', ledger)

    collect(registry.reference_cases('cal')+registry.calibration_cases(p))
    calibrated = verify.build(folder, calibration_only=True)
    nomination = policy.nominate(p, calibrated['cases'])
    common.write_json(folder / 'nomination.json', nomination)
    ledger['nomination_sha256'] = common.file_digest(folder / 'nomination.json')
    collect(registry.reference_cases('hold')+registry.confirmation_cases(p, nomination['profile']))
    return folder, ledger


def test_independent_complete_reconstruction_and_incomplete_rejection(authored_campaign):
    folder, ledger = authored_campaign
    verified = verify.build(folder)
    assert verified['decision']['outcome'] == 'confirmed_improvement'
    ledger['cases'].pop()
    common.write_json(folder / 'execution.json', ledger)
    with pytest.raises(ValueError, match='incomplete'):
        verify.build(folder)


def test_independent_verifier_rejects_lying_flag_even_with_updated_record_hash(authored_campaign):
    folder, ledger = authored_campaign
    entry = next(e for e in ledger['cases'] if e['id'] == 'cal-0-native-t1')
    path = folder / 'runs' / entry['id'] / 'result.json'
    raw = common.read_json(path); raw['correctness']['passed'] = False
    common.write_json(path, raw); entry['sha256'] = common.file_digest(path)
    common.write_json(folder / 'execution.json', ledger)
    with pytest.raises(ValueError, match='flag'):
        verify.build(folder)


def test_independent_verifier_rejects_bad_nomination_even_with_updated_nomination_hash(authored_campaign):
    folder, ledger = authored_campaign
    saved = common.read_json(folder / 'nomination.json'); saved['profile'] = 'native-t4'
    common.write_json(folder / 'nomination.json', saved)
    ledger['nomination_sha256'] = common.file_digest(folder / 'nomination.json')
    common.write_json(folder / 'execution.json', ledger)
    with pytest.raises(ValueError):
        verify.build(folder)


def test_independent_verifier_requires_fresh_owned_actual_process(authored_campaign):
    folder, ledger = authored_campaign
    ledger['cases'][4]['ownership']['closed'] = False
    common.write_json(folder / 'execution.json', ledger)
    with pytest.raises(ValueError, match='owned process'):
        verify.build(folder)


def test_profile_reconstructs_evidence_and_rejects_drift(authored_campaign, monkeypatch, tmp_path):
    folder, ledger = authored_campaign
    # Temporary fixture lives under the same public campaign structure as production.
    home = tmp_path / 'local-state'
    destination = home / 'campaigns' / folder.name
    import shutil
    shutil.copytree(folder, destination)
    verified = verify.build(destination)
    ident = profiles.save(destination, verified, home)
    monkeypatch.setattr(profiles, 'check_sources', lambda *a: None)
    monkeypatch.setattr(profiles, 'runtime_identity', lambda: {'authored': True})
    monkeypatch.setattr(profiles, 'assets', lambda *a: {})
    saved = profiles.validate(ident, home)
    expected = profiles.state(saved['domain'], home)
    applied = profiles.activate(ident, expected, home)
    assert applied['profile'] == ident
    assert profiles.revert(saved['domain'], applied, home)['profile'] is None
    monkeypatch.setattr(profiles, 'runtime_identity', lambda: {'authored': 'drift'})
    with pytest.raises(ValueError, match='incompatible'):
        profiles.activate(ident, profiles.state(saved['domain'], home), home)


def test_objectives_have_separate_active_domains():
    fastest = registry.settings({})
    budget = registry.settings({'goal': 'budget', 'budget_mib': 2560})
    lower_budget = registry.settings({'goal': 'budget', 'budget_mib': 2400})
    assert len({common.domain(c) for c in (fastest, budget, lower_budget)}) == 3


def test_asset_drift_rejects_a_valid_saved_profile(authored_campaign, monkeypatch, tmp_path):
    folder, _ = authored_campaign
    ident = profiles.save(folder, verify.build(folder), tmp_path)
    monkeypatch.setattr(profiles, 'check_sources', lambda *a: None)
    monkeypatch.setattr(profiles, 'runtime_identity', lambda: {'authored': True})
    monkeypatch.setattr(profiles, 'assets', lambda *a: {'authored': 'changed asset'})
    with pytest.raises(ValueError, match='bytes are incompatible'):
        profiles.validate(ident, tmp_path)


def test_verified_public_array_sharing_preserves_previous_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'STORE', tmp_path)
    first = tmp_path / 'campaigns/first'
    second = tmp_path / 'campaigns/second'
    value = np.arange(100, dtype=np.float32)
    a = common.save_array(first, value); b = common.save_array(second, value)
    assert np.array_equal(common.load_array(first, a), common.load_array(second, b))
    assert a == b
    assert (first / a['path']).samefile(second / b['path'])
    before = (first / a['path']).read_bytes()
    def unavailable(*args):
        raise OSError('Authored hard-link unsupported control')
    monkeypatch.setattr(common.os, 'link', unavailable)
    third = tmp_path / 'campaigns/third'
    c = common.save_array(third, value)
    assert (third / c['path']).read_bytes() == before == (first / a['path']).read_bytes()


def test_public_export_never_traverses_private_requests(authored_campaign, monkeypatch, tmp_path):
    from atlas.tuning import bundle
    import shutil, zipfile, json
    original, _ = authored_campaign
    home = tmp_path / 'local-store'
    folder = home / 'campaigns' / original.name
    shutil.copytree(original, folder)
    common.write_json(home / 'private-inference/do-not-export/request.json', {'prompt': 'authored private sentinel'})
    monkeypatch.setattr(bundle, 'ROOT', tmp_path)
    path = bundle.export(folder.name, home)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('bundle-manifest.json'))
        assert all('private-inference' not in name for name in archive.namelist())
        assert all(b'authored private sentinel' not in archive.read(name) for name in archive.namelist())
        assert manifest['evidence_sha256'] == verify.build(folder)['evidence_sha256']


def test_stale_status_never_proves_owner_liveness(monkeypatch, tmp_path):
    pytest.importorskip('fastapi')
    from atlas.tuning import service
    monkeypatch.setattr(service.jobs, 'active', None)
    folder = tmp_path / 'authored-stale'
    common.write_json(folder / 'status.json', {'status': 'running'})
    assert service.observed_status(folder)['status'] == 'unknown_owner'
    common.write_json(folder / 'status.json', {'status': 'running', 'controller_process':
                      {'pid': 99999999, 'created_at': 1}})
    assert service.observed_status(folder)['status'] == 'interrupted'


def test_import_of_arithmetic_verifier_does_not_load_torch():
    import subprocess
    result = subprocess.run([sys.executable, '-c',
        "import sys;sys.path.insert(0,'src');import atlas.tuning.verify;assert 'torch' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
