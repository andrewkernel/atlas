from benchmarks.statistical_assumption_audit import synthetic_tost_decisions


def test_normal_boundary_control_and_rare_mixture_counterexample():
    normal = synthetic_tost_decisions('normal', .5, 30_000, seed=5101)
    mixture = synthetic_tost_decisions('rare_mixture', .5, 30_000, seed=5102)
    assert .04 < normal['fraction'] < .06
    assert .65 < mixture['fraction'] < .75
    # Population mean and variance agree, while t-interval validity does not.
    assert normal['parameters']['population_mean'] == mixture['parameters']['population_mean'] == .5
    assert normal['parameters']['population_sd'] == mixture['parameters']['population_sd'] == 1.
