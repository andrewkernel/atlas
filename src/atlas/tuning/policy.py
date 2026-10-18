"""Nominate from calibration only. No access to confirmation measurements."""
from __future__ import annotations

import math
from .common import digest
from .registry import calibration_cases


def nominate(protocol, cases):
    rows = [r for r in cases if r['config']['stage'] == 'calibration']
    if {r['config']['id'] for r in rows} != {c['id'] for c in calibration_cases(protocol)}:
        raise ValueError('Complete calibration is required before nomination')
    controls = all(any(r['config']['input'] == inp and r['config']['profile']['id'] == name
                       and r['status'] == 'completed' and r['passed'] is True for r in rows)
                   for inp in ('cal-0', 'cal-1') for name in ('native-t4', 'resident-t4'))
    budget = protocol['settings']['budget_mib']
    table = []
    for profile in protocol['profiles']:
        selected = [r for r in rows if r['config']['profile'] == profile]
        numerical = controls and len(selected) == 2 and all(r['status'] == 'completed'
                                                          and r['passed'] is True for r in selected)
        times = [r['generation_ms'] for r in selected if r.get('generation_ms') is not None]
        peaks = [r['peak_allocated_mib'] for r in selected if r.get('peak_allocated_mib') is not None]
        score = math.sqrt(times[0]*times[1]) if len(times) == 2 else None
        peak = max(peaks) if len(peaks) == 2 else None
        eligible = numerical and score is not None and peak is not None and (budget is None or peak <= budget)
        table.append({'profile': profile, 'numerical': numerical, 'eligible': eligible,
                      'generation_geomean_ms': score, 'peak_allocated_mib': peak,
                      'passed_windows': sum(r['passed'] is True for r in selected), 'planned_windows': 2})
    valid = [r for r in table if r['eligible']]
    winner = min(valid, key=lambda r: (r['generation_geomean_ms'],
                    r['profile']['id'] != protocol['baseline'], r['profile']['id'])) if valid else None
    return {'profile': winner['profile']['id'] if winner else None, 'controls_passed': controls,
            'calibration': table, 'basis_sha256': digest(rows),
            'reason': 'Calibration only; confirmation remains required' if winner else
                      'No numerically/control-qualified setting satisfies the objective'}
