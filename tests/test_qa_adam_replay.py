"""Float32 adjacency and the saved selected-component replay controls."""
import importlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def distance(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'benchmarks'))
    return importlib.import_module('qa_adam_sensitivity_control').ulp_distance


@pytest.mark.parametrize('first,second,expected', [(0., -0., 0), (1., 1.0000001192092896, 1),
                                                (-1., -1.0000001192092896, 1)])
def test_float32_adjacent_values_and_signed_zero(distance, first, second, expected):
    assert distance(first, second) == expected
    assert distance(second, first) == expected


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_ulp_observations_are_not_equal(distance, invalid):
    with pytest.raises(ValueError):
        distance(invalid, invalid)


def test_selected_native_replay_and_equal_gradient_positive_controls(distance):
    result = json.loads((ROOT / 'docs/research/workload-screen/qa-adam-sensitivity-control.json').read_text(encoding='utf-8'))
    worker = result['trial']['worker_result']
    assert result['trial']['status'] == worker['status'] == 'completed'
    assert len(worker['selected_components']) == 10
    replay = worker['replays'][0]
    assert replay['eps'] == 1e-8 and replay['all_selected_values_within_two_ulps']
    for selected, reference, candidate in zip(worker['selected_components'], replay['reference'], replay['candidate'], strict=True):
        for side, observed in (('reference', reference), ('candidate', candidate)):
            for key in ('post', 'exp_avg', 'exp_avg_sq'):
                assert distance(selected[side][key], observed[key]) == 0
    assert all(item['equal_gradient_positive_exact'] for item in worker['replays'])


def test_a_parameter_perturbation_does_not_pass_the_replay_bound(distance):
    result = json.loads((ROOT / 'docs/research/workload-screen/qa-adam-sensitivity-control.json').read_text(encoding='utf-8'))
    value = result['trial']['worker_result']['selected_components'][0]['reference']['post']
    assert distance(value, value + .01) > 2
