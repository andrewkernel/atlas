from benchmarks.state_replay_probes import batchnorm_recompute_control, stochastic_partition_control


def test_buffer_control_repairs_known_bn_fixture_and_retains_negative_control():
    result = batchnorm_recompute_control()
    native = result['native_checkpoint']
    repaired = result['buffer_restore_checkpoint']
    assert native['gradient_max_abs_error'] < 1e-12
    assert native['batches_tracked'] == 2
    assert native['evaluation_max_abs_error'] > 1e-5
    assert repaired['gradient_max_abs_error'] < 1e-12
    assert repaired['buffer_max_abs_error'] == 0
    assert repaired['batches_tracked'] == 1
    assert repaired['evaluation_max_abs_error'] < 1e-12


def test_two_dropout_layers_can_change_path_even_when_final_rng_matches():
    result = stochastic_partition_control(dropout_layers=2)
    assert result['forward_max_abs_error'] > 1e-5
    assert result['gradient_max_abs_error'] > 1e-5
    # Total draws alone do not identify which sample/layer received each draw.
    assert result['final_rng_states_equal']
