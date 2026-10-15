"""Malformed evidence must not silently preserve a completed/valid result."""
import copy
import json
from pathlib import Path
import pytest
from benchmarks.derive_qa_resource import derive
from benchmarks.audit_qa_records import audit

DIRECTORY = Path(__file__).resolve().parents[1] / 'docs/research/workload-screen'


@pytest.fixture
def inputs():
    return (json.loads((DIRECTORY / 'qa-resource-pilot.json').read_text(encoding='utf-8')),
            json.loads((DIRECTORY / 'worker-ownership-control.json').read_text(encoding='utf-8')))


def test_current_provenance_and_host_memory_amendment(inputs):
    assert audit() == []
    derived = derive(*inputs)
    assert derived['worker_peak_rss_bytes'] is None
    assert 'invalid' in derived['host_worker_memory_validity']
    assert derived['intended_trials'] == 16


@pytest.mark.parametrize('mutation', ['duplicate_trial', 'missing_update', 'different_initial_model', 'unverified_worker_identity'])
def test_reject_inconsistent_completion_or_ownership(inputs, mutation):
    pilot, ownership = copy.deepcopy(inputs)
    if mutation == 'duplicate_trial':
        pilot['trials'][1] = pilot['trials'][0]
    elif mutation == 'missing_update':
        pilot['trials'][0]['worker_result']['successful_updates'] = 7
    elif mutation == 'different_initial_model':
        pilot['trials'][0]['worker_result']['model_preparation']['initial_state_digest'] = '0' * 64
    else:
        ownership['trials'][0]['interpreter_parent_is_launch_pid'] = False
    with pytest.raises(AssertionError):
        derive(pilot, ownership)
