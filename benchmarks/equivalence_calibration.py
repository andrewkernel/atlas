"""Original statistical design audit; synthetic paired differences, no ML training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.stats import chi2, norm, t


def normal_known_sd_power(n: int, margin: float, sd: float, alpha: float = .05) -> float:
    """Joint TOST power at true mean zero with a known population SD."""
    radius = margin * math.sqrt(n) / sd - norm.ppf(1 - alpha)
    return float(max(0., 2 * norm.cdf(radius) - 1))


def normal_paired_tost_power(n: int, mean: float, margin: float, sd: float, alpha: float = .05) -> float:
    """Integrate independent normal sample mean and chi-square sample variance.

    This is exact under iid normal paired differences, up to quadrature error.
    It does not assume the two one-sided rejection events are independent.
    """
    if n < 2 or margin <= 0 or sd <= 0 or not 0 < alpha < .5:
        raise ValueError("Invalid TOST design")
    df = n - 1
    se = sd / math.sqrt(n)
    critical = t.ppf(1 - alpha, df)
    # For larger sample SD, the two rejection regions cannot overlap.
    variance_limit = df * (margin / (critical * se)) ** 2

    def integrand(v: float) -> float:
        half_width = critical * se * math.sqrt(v / df)
        conditional = norm.cdf((margin - half_width - mean) / se) - norm.cdf((-margin + half_width - mean) / se)
        return float(max(0., conditional) * chi2.pdf(v, df))

    value, error = quad(integrand, 0, variance_limit, epsabs=1e-10, epsrel=1e-9)
    if error > 1e-7:
        raise ArithmeticError("Quadrature precision is insufficient")
    return float(value)


def simulate(n: int, mean: float, margin: float, sd: float, repetitions: int, seed: int) -> dict:
    """Draw sufficient statistics; valid specifically for iid normal differences."""
    rng = np.random.default_rng(seed)
    means = rng.normal(mean, sd / math.sqrt(n), repetitions)
    sample_sd = sd * np.sqrt(rng.chisquare(n - 1, repetitions) / (n - 1))
    width = t.ppf(.95, n - 1) * sample_sd / math.sqrt(n)
    accepted = (means - width > -margin) & (means + width < margin)
    count = int(accepted.sum())
    proportion = count / repetitions
    z = norm.ppf(.975)
    denominator = 1 + z*z / repetitions
    center = (proportion + z*z / (2*repetitions)) / denominator
    radius = z * math.sqrt(proportion * (1-proportion)/repetitions + z*z/(4*repetitions**2)) / denominator
    return {"equivalence_decisions": count, "repetitions": repetitions, "fraction": proportion,
            "monte_carlo_wilson_95": [center-radius, center+radius], "seed": seed}


def run(repetitions: int = 200_000) -> dict:
    margin, sd, target = .5, 1., .8
    old_n = math.ceil(((norm.ppf(.95) + norm.ppf(.8))*sd/margin)**2)
    corrected_n = math.ceil(((norm.ppf(.95) + norm.ppf(.9))*sd/margin)**2)
    minimum_n = next(n for n in range(2, 200) if normal_paired_tost_power(n, 0, margin, sd) >= target)
    cases = []
    for n in sorted({3, 5, 10, old_n, corrected_n, minimum_n-1, minimum_n, 50}):
        cases.append({"n_pairs": n, "true_mean": 0., "known_sd_power": normal_known_sd_power(n, margin, sd),
                      "unknown_sd_power": normal_paired_tost_power(n, 0, margin, sd),
                      "simulation": simulate(n, 0, margin, sd, repetitions, seed=8900+n)})
    boundaries = [{"n_pairs": minimum_n, "true_mean": mean,
                   "unknown_sd_probability": normal_paired_tost_power(minimum_n, mean, margin, sd),
                   "simulation": simulate(minimum_n, mean, margin, sd, repetitions, seed=9200+i)}
                  for i, mean in enumerate((margin, -margin, 2*margin))]
    return {"schema_version": "1.0", "started_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
            "assumptions": "iid normal paired differences; SD=1, symmetric margin=.5, each one-sided alpha=.05; no multiplicity adjustment",
            "target_power": target, "old_approximation_n": old_n, "corrected_known_sd_approximation_n": corrected_n,
            "minimum_unknown_sd_n_in_searched_range": minimum_n, "cases": cases, "boundary_controls": boundaries,
            "limitation": "Synthetic power calibration only; no observed ML variance, validated domain margin or training equivalence."}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('old_approximation_n', 'corrected_known_sd_approximation_n', 'minimum_unknown_sd_n_in_searched_range')}, indent=2))
