import pytest

from benchmarks.equivalence_calibration import normal_known_sd_power, normal_paired_tost_power, simulate


def test_joint_tost_power_exposes_underpowered_old_approximation():
    assert .60 < normal_known_sd_power(25, .5, 1) < .62
    assert normal_paired_tost_power(35, 0, .5, 1) < .8
    assert normal_paired_tost_power(36, 0, .5, 1) > .8


def test_tost_respects_both_non_equivalence_boundaries():
    for mean in (-.5, .5, 1.):
        assert normal_paired_tost_power(36, mean, .5, 1) <= .05000001
    assert normal_paired_tost_power(36, -.2, .5, 1) == pytest.approx(normal_paired_tost_power(36, .2, .5, 1))


def test_integrated_power_matches_independently_sampled_sufficient_statistics():
    result = simulate(36, 0, .5, 1, 100_000, seed=2485)
    expected = normal_paired_tost_power(36, 0, .5, 1)
    # Fixed seed, generous >5 standard-error tolerance avoids a brittle MC test.
    assert abs(result['fraction'] - expected) < .01


def test_invalid_statistical_design_is_rejected():
    with pytest.raises(ValueError):
        normal_paired_tost_power(1, 0, .5, 1)
