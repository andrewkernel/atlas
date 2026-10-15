"""Positive repairs, objective errors and a missing-denominator boundary."""
import pytest
import torch
from benchmarks.composite_objective_control import run_controls


@pytest.fixture(scope='module')
def controls():
    torch.set_num_threads(1)
    return run_controls()


@pytest.mark.parametrize('case', ['equal_global_counts', 'unequal_global_counts', 'empty_local_head'])
def test_separate_global_denominators_repair_partial_accumulation(controls, case):
    outcomes = controls[case]['outcomes']
    repaired = outcomes['global_head_sums']
    assert repaired['loss_finite'] and repaired['gradients_finite']
    assert repaired['loss'] == pytest.approx(outcomes['full']['loss'], rel=1e-10, abs=1e-12)
    assert repaired['gradient_max_abs_error'] < 1e-12
    assert repaired['post_parameter_max_abs_error'] < 1e-12
    assert outcomes['naive_example_weighted_means']['gradient_max_abs_error'] > 1e-3


def test_nonfinite_local_mean_can_coexist_with_finite_gradients(controls):
    observed = controls['empty_local_head']['outcomes']['naive_example_weighted_means']
    assert observed['loss'] is None and not observed['loss_finite']
    assert observed['gradients_finite'] and not observed['optimizer_step_performed']
    assert controls['undefined_global_head']['status'] == 'unsupported_undefined_global_head_mean'


def test_forward_equality_does_not_validate_parameter_dependent_normalization(controls):
    observed = controls['parameter_dependent_denominator']
    assert observed['full_loss'] == observed['detached_denominator_loss']
    assert observed['detached_denominator_gradient_max_abs_error'] > 1e-3
    assert observed['streamed_gradient_max_abs_error'] < 1e-12
    assert observed['finite_difference_max_abs_error'] < 1e-8
