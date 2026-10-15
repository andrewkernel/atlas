"""Second scalar/fsum reconstruction route over retained local selected arrays."""
from array import array
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'docs/research/workload-screen'


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def python_statistics(first, second):
    a, b = array('f'), array('f')
    a.frombytes(first.astype('<f4', copy=False).tobytes())
    b.frombytes(second.astype('<f4', copy=False).tobytes())
    if sys.byteorder != 'little':
        a.byteswap()
        b.byteswap()
    assert len(a) == len(b) > 0
    maximum, position, exceeded, finite, equal = -1., 0, 0, True, True
    for index, (x, y) in enumerate(zip(a, b, strict=True)):
        finite = finite and math.isfinite(x) and math.isfinite(y)
        difference = abs(x - y)
        if difference > maximum:
            maximum, position = difference, index
        exceeded += difference > 1e-7 + 1e-5 * abs(y)
        equal = equal and x == y
    difference_square = math.fsum((x - y)**2 for x, y in zip(a, b, strict=True))
    reference_square = math.fsum(x*x for x in a)
    return {'maximum_absolute_difference': maximum, 'maximum_flat_index': position,
            'diagnostic_exceeded_elements': exceeded, 'finite': finite, 'numerically_exact_equal': equal,
            'difference_square_sum': difference_square, 'reference_square_sum': reference_square,
            'relative_l2_error': (difference_square / reference_square)**.5 if reference_square else None}


def main():
    original = json.loads((DIRECTORY / 'qa-reference32-control.json').read_text(encoding='utf-8'))
    result = {'captured_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_canonical_sha256': canonical(original), 'arrays_checked': 0, 'elements_checked': 0,
              'failures': [], 'comparisons': [],
              'scope': 'Post-hoc independent scalar/fsum arithmetic and retained artifact identities; not new training, full-state coverage or independent scientific labels.'}
    loaded = {}
    for index, trial in enumerate(original['trials']):
        assert trial['status'] == 'completed'
        path = (ROOT / trial['selected_state_local_artifact']).resolve()
        assert path.is_relative_to((ROOT / 'artifacts/qa-reference32').resolve())
        worker = trial['worker_result']
        assert path.stat().st_size == worker['selected_state_file']['bytes'] < 32 * 1024**2
        assert hashlib.sha256(path.read_bytes()).hexdigest() == worker['selected_state_file']['sha256']
        with np.load(path, allow_pickle=False) as archive:
            assert set(archive.files) == set(worker['selected_arrays'])
            values = {name: archive[name] for name in archive.files}
        for name, value in values.items():
            expected = worker['selected_arrays'][name]
            assert list(value.shape) == expected['shape'] and str(value.dtype) == expected['dtype'] == 'float32'
            assert hashlib.sha256(value.tobytes()).hexdigest() == expected['sha256']
        loaded[index] = values
    for pair in original['pairs']:
        indices = [i for i, trial in enumerate(original['trials']) if trial['dropout'] == pair['dropout']]
        first, second = (loaded[i] for i in indices)
        for name, expected in pair['selected_tensor_differences'].items():
            observed = python_statistics(first[name], second[name])
            matching = True
            for key, value in observed.items():
                target = expected[key]
                agrees = (math.isclose(value, target, rel_tol=1e-12, abs_tol=0.) if isinstance(value, float) else value == target)
                if not agrees:
                    matching = False
                    result['failures'].append({'dropout': pair['dropout'], 'array': name, 'field': key})
            if expected['bitwise_equal'] != (first[name].tobytes() == second[name].tobytes()):
                matching = False
                result['failures'].append({'dropout': pair['dropout'], 'array': name, 'field': 'bitwise_equal'})
            result['arrays_checked'] += 1
            result['elements_checked'] += first[name].size
            result['comparisons'].append({'dropout': pair['dropout'], 'array': name, 'all_reconstructed_fields_match': matching})
    (DIRECTORY / 'qa-reference32-reconstruction.json').write_bytes((json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
    print(json.dumps({name: result[name] for name in ('arrays_checked', 'elements_checked', 'failures')}))
    raise SystemExit(bool(result['failures']))


if __name__ == '__main__':
    main()
