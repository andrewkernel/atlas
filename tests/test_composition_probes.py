from benchmarks.composition_probes import composition_probe


def test_pure_checkpoint_accumulation_composition_preserves_two_adam_updates():
    result=composition_probe()
    assert result['parameter_max_abs_error'] < 1e-12
    assert result['reference'] == result['candidate'] == {'successful_updates':2,'adam_state_steps':[2]}


def test_clipping_each_partial_gradient_changes_composed_update():
    result=composition_probe('clip_per_microbatch')
    assert result['parameter_max_abs_error'] > 1e-5
    assert result['reference']['successful_updates'] == result['candidate']['successful_updates'] == 2


def test_microstep_optimizer_updates_violate_clock_and_parameters():
    result=composition_probe('step_per_microbatch')
    assert result['parameter_max_abs_error'] > 1e-5
    assert result['reference'] == {'successful_updates':2,'adam_state_steps':[2]}
    assert result['candidate'] == {'successful_updates':4,'adam_state_steps':[4]}
