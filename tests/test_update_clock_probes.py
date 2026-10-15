from benchmarks.update_clock_probes import skipped_step_probe, unused_parameter_probe


def test_unused_zero_gradient_changes_adam_state_despite_matching_observed_loss():
    result = unused_parameter_probe()
    left, right = result['set_to_none'], result['zero_tensor']
    assert left['second_loss'] == right['second_loss']
    assert left['shared_after_second'] == right['shared_after_second']
    assert left['conditional_optimizer_steps'] == 1
    assert right['conditional_optimizer_steps'] == 2
    assert result['conditional_parameter_abs_difference'] > .01


def test_scheduler_and_ema_need_successful_update_clock_after_scaler_skip():
    result = skipped_step_probe()
    correct = result['events']['successful_update_clock']
    wrong = result['events']['attempt_clock']
    for events in (correct, wrong):
        assert [event['optimizer_updated'] for event in events] == [True, False, True]
        assert events[-1]['successful_updates'] == 2
    assert correct[1]['lr_after_scheduler'] == correct[0]['lr_after_scheduler']
    assert correct[1]['ema'] == correct[0]['ema']
    assert wrong[1]['lr_after_scheduler'] != correct[1]['lr_after_scheduler']
    assert result['final_parameter_abs_difference'] > .01
