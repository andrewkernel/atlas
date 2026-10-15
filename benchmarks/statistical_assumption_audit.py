"""Synthetic sensitivity audit for the proposed mean-equivalence analysis.

No ML outcomes are used. Rare-event probabilities are deliberately constructed,
not estimates of real training failure prevalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import chi2, t

if __package__:
    from .equivalence_calibration import normal_paired_tost_power
else:
    from equivalence_calibration import normal_paired_tost_power


def minimum_normal_pairs(mean: float, sd: float) -> int:
    lower, upper = 2, 8
    while normal_paired_tost_power(upper, mean, .5, sd) < .8:
        upper *= 2
        if upper > 4096:
            raise ValueError('Design outside this audit range')
    while upper-lower > 1:
        middle = (upper+lower)//2
        if normal_paired_tost_power(middle, mean, .5, sd) >= .8:
            upper = middle
        else:
            lower = middle
    return upper


def synthetic_tost_decisions(distribution: str, mean: float, repetitions: int, seed: int, n: int = 36) -> dict:
    rng = np.random.default_rng(seed)
    if distribution == 'normal':
        differences = rng.normal(mean, 1., (repetitions, n))
        parameters = {'population_mean': mean, 'population_sd': 1.}
    elif distribution == 'rare_mixture':
        probability, noise_sd = .01, .05
        jump = math.sqrt((1-noise_sd**2)/(probability*(1-probability)))
        differences = mean + rng.normal(0, noise_sd, (repetitions, n))
        differences += jump*(rng.binomial(1, probability, (repetitions, n))-probability)
        parameters = {'population_mean': mean, 'population_sd': 1., 'rare_component_probability': probability,
                      'noise_sd': noise_sd, 'jump': jump, 'probability_no_rare_draw': (1-probability)**n}
    else:
        raise ValueError('Unknown distribution')
    sample_mean = differences.mean(axis=1)
    half_width = t.ppf(.95, n-1)*differences.std(axis=1, ddof=1)/math.sqrt(n)
    decisions = (sample_mean-half_width > -.5) & (sample_mean+half_width < .5)
    count = int(decisions.sum())
    fraction = count/repetitions
    return {'distribution': distribution, 'n_pairs': n, 'parameters': parameters, 'seed': seed,
            'repetitions': repetitions, 'equivalence_decisions': count, 'fraction': fraction,
            'monte_carlo_standard_error': math.sqrt(fraction*(1-fraction)/repetitions)}


def run() -> dict:
    sensitivity = [{'true_mean_difference': mean, 'paired_sd': sd, 'minimum_pairs_under_normal_model': minimum_normal_pairs(mean, sd)}
                   for mean in (0., .25) for sd in (.5, 1., 1.5, 2.)]
    cases = [synthetic_tost_decisions(distribution, mean, 100_000, seed=7410+i*10+j)
             for i, distribution in enumerate(('normal', 'rare_mixture')) for j, mean in enumerate((0., -.5, .5))]
    return {'schema_version': '1.0', 'started_utc': datetime.now(timezone.utc).isoformat(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'calibration_source_sha256': hashlib.sha256(Path(__file__).with_name('equivalence_calibration.py').read_bytes()).hexdigest(),
            'environment': {'numpy': np.__version__, 'scipy': scipy.__version__},
            'normal_model': {'symmetric_margin': .5, 'alpha_each_one_sided': .05, 'target_power': .8},
            'normal_sensitivity': sensitivity,
            'illustrative_three_pair_pilot_sd_95_interval_for_observed_sd_1': [math.sqrt(2/chi2.ppf(.975,2)), math.sqrt(2/chi2.ppf(.025,2))],
            'synthetic_distribution_cases': cases,
            'limitation': 'Constructed independent probability models; no observed ML variance or rare-event frequency. Normal-model power does not validate that model or guarantee robustness of t-intervals.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'normal_sensitivity': result['normal_sensitivity'],
                      'cases': [(case['distribution'],case['parameters']['population_mean'],case['fraction']) for case in result['synthetic_distribution_cases']]},indent=2))
