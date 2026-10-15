"""Selected-artifact identity and explicit diagnostic arithmetic conventions."""
import hashlib
import importlib
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'benchmarks'))
    return importlib.import_module('qa_reference32_control')


def test_independent_diagnostic_absolute_threshold_and_maximum(module):
    first = {'x': np.array([0., 1., 2.], dtype=np.float32)}
    second = {'x': np.array([1e-6, 1. + 1e-6, 2.], dtype=np.float32)}
    result = module.compare_arrays(first, second)['x']
    assert result['diagnostic_exceeded_elements'] == 1
    assert result['maximum_flat_index'] == 0 and result['maximum_absolute_difference'] == pytest.approx(1e-6)
    assert result['reference_square_sum'] == 5 and result['finite']


def test_signed_zero_byte_identity_and_zero_reference_norm_are_distinct(module):
    result = module.compare_arrays({'x': np.array([-0.], dtype=np.float32)},
                                   {'x': np.array([0.], dtype=np.float32)})['x']
    assert not result['bitwise_equal'] and result['numerically_exact_equal']
    assert result['maximum_absolute_difference'] == result['diagnostic_exceeded_elements'] == 0
    assert result['relative_l2_error'] is None and result['relative_l2_reason'] == 'zero_reference_norm'


def test_missing_selected_tensor_is_rejected(module):
    with pytest.raises(AssertionError):
        module.compare_arrays({'x': np.array([0.], dtype=np.float32)}, {})


def test_changed_archive_is_rejected_before_numeric_comparison(module, tmp_path):
    path = tmp_path / 'selected.npz'
    values = np.array([1., 2.], dtype=np.float32)
    np.savez(path, x=values)
    metadata = {'selected_state_file': {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
                'selected_arrays': {'x': {'shape': [2], 'dtype': 'float32', 'sha256': hashlib.sha256(values.tobytes()).hexdigest()}}}
    assert np.array_equal(module.load_selected(path, metadata)['x'], values)
    path.write_bytes(path.read_bytes() + b'x')
    with pytest.raises(AssertionError):
        module.load_selected(path, metadata)
