"""Different declared relations and a finite-precision invariance boundary."""
import pytest
import torch
from benchmarks.qa_gauge_control import run_controls


@pytest.fixture(scope='module')
def controls():
    torch.set_num_threads(1)
    return run_controls()


def test_identical_recipe_repeats(controls):
    observed = controls['cases']['repeat_float64']['comparison']
    assert observed['probabilities_max_abs_difference'] == observed['absolute_loss_difference'] == 0


def test_bias_parameter_change_preserves_finite_float64_functional_observations(controls):
    case = controls['cases']['bias_shift_float64']
    observed = case['comparison']
    assert case['observations']['all_observed_values_finite']
    assert observed['bias_max_abs_difference'] == pytest.approx(.3)
    assert observed['logits_max_abs_difference'] > .29
    assert observed['centered_logits_max_abs_difference'] < 1e-12
    assert observed['probabilities_max_abs_difference'] < 1e-12 and observed['absolute_loss_difference'] < 1e-12
    assert observed['weight_gradient_max_abs_difference'] < 1e-12 and observed['argmax_equal']


def test_unchanged_argmax_does_not_validate_probability_preservation(controls):
    observed = controls['cases']['weight_change_float64']['comparison']
    assert observed['argmax_equal']
    assert observed['probabilities_max_abs_difference'] > 1e-3


def test_real_arithmetic_bias_invariance_does_not_guarantee_fp32_observations(controls):
    case = controls['cases']['large_bias_float32']
    assert case['observations']['all_observed_values_finite']
    assert not case['comparison']['argmax_equal']
    assert case['comparison']['probabilities_max_abs_difference'] > 1e-3
