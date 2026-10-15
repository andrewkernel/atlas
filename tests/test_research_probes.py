"""Positive controls and deliberate failures for the bounded research probes."""

import pytest
import torch
from hypothesis import given, settings, strategies as st

from benchmarks.semantic_probes import (
    accumulation_probe,
    checkpoint_buffer_probe,
    checkpoint_dropout_probe,
    precision_probe,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


@given(n=st.integers(min_value=4, max_value=20), fraction=st.floats(min_value=0.1, max_value=0.9))
@settings(max_examples=40, deadline=None, derandomize=True)
def test_weighted_accumulation_preserves_gradient_for_unequal_partitions(n, fraction):
    split = max(1, min(n - 1, int(n * fraction)))
    assert accumulation_probe(n=n, split=split)["weighted_gradient_max_abs_error"] < 1e-12


def test_unequal_mean_of_means_is_counterexample():
    assert accumulation_probe()["mean_of_means_gradient_max_abs_error"] > 1e-4


def test_equal_partitions_are_positive_control_for_mean_of_means():
    assert accumulation_probe(n=8, split=4)["mean_of_means_gradient_max_abs_error"] < 1e-12


def test_batchnorm_breaks_accumulation_even_with_weighted_loss():
    assert accumulation_probe(n=8, split=4, batch_norm=True)["weighted_gradient_max_abs_error"] > 1e-4


def test_checkpoint_rng_preservation_is_positive_control():
    result = checkpoint_dropout_probe(True)
    assert result["forward_max_abs_error"] == 0
    assert result["gradient_max_abs_error"] < 1e-12


def test_checkpoint_wrong_rng_can_match_forward_but_fail_gradient():
    result = checkpoint_dropout_probe(False)
    assert result["forward_max_abs_error"] == 0
    assert result["gradient_max_abs_error"] > 1e-4


def test_gradient_parity_does_not_certify_mutable_buffers():
    result = checkpoint_buffer_probe()
    assert result["gradient_max_abs_error"] < 1e-12
    assert result["baseline_batches_tracked"] == 1
    assert result["checkpoint_batches_tracked"] == 2
    assert result["running_mean_max_abs_error"] > 1e-4
    assert result["evaluation_output_max_abs_error"] > 1e-4


def test_operation_order_changes_float32_result():
    assert precision_probe() == {"left_associated": 1.0, "right_associated": 0.0}
