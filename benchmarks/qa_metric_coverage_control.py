"""Bounded execution of six reviewed official metric functions on invented data."""
import ast
import collections
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import string
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = '710840ce2c2c61716334c056777032f24660b83eb9869a451036e6a24d8103c4'
NAMES = ('normalize_answer', 'get_tokens', 'compute_exact', 'compute_f1', 'get_raw_scores', 'make_eval_dict')


def main():
    path = ROOT / '.tools/qa-source-screen/evaluate-v2.0.py'
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == SOURCE_SHA
    tree = ast.parse(data.decode('utf-8'), filename=str(path))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    assert len(selected) == len(NAMES) and {n.name for n in selected} == set(NAMES)
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {'collections': collections, 're': re, 'string': string}
    # Reviewed source only, fixed digest and function allowlist; not a sandbox.
    exec(compile(module, str(path), 'exec'), namespace)
    dataset = [{'paragraphs': [{'qas': [{'id': q, 'answers': [{'text': text}]}
                                      for q, text in [('a', 'Paris'), ('b', 'blue'), ('c', 'two')]]}]}]
    predictions = {'complete': {'a': 'Paris', 'b': 'wrong', 'c': 'wrong'},
                   'missing_c': {'a': 'Paris', 'b': 'wrong'},
                   'unexpected_id': {'a': 'Paris', 'b': 'wrong', 'c': 'wrong', 'extra': 'ignored'},
                   'all_missing': {}}
    trials = []
    for name, preds in predictions.items():
        warnings = io.StringIO()
        with redirect_stdout(warnings):
            exact, f1 = namespace['get_raw_scores'](dataset, preds)
        try:
            aggregate = namespace['make_eval_dict'](exact, f1)
            aggregation_error = None
        except ZeroDivisionError:
            aggregate, aggregation_error = None, 'ZeroDivisionError'
        try:
            strict_id_aggregate = namespace['make_eval_dict'](exact, f1, ['a', 'b', 'c'])
            strict_id_error = None
        except KeyError:
            strict_id_aggregate, strict_id_error = None, 'KeyError'
        missing = sorted({'a', 'b', 'c'} - preds.keys())
        unexpected = sorted(preds.keys() - {'a', 'b', 'c'})
        trials.append({'case': name, 'predictions': preds, 'exact_by_id': exact, 'f1_by_id': f1,
                       'warnings': warnings.getvalue(), 'aggregate_default': aggregate,
                       'aggregation_error': aggregation_error, 'aggregate_explicit_ids': strict_id_aggregate,
                       'explicit_ids_error': strict_id_error,
                       'coverage_guard': {'missing': missing, 'unexpected': unexpected,
                                          'status': 'complete' if not missing and not unexpected else 'error'}})
    assert trials[0]['aggregate_default']['total'] == 3
    assert trials[1]['aggregate_default'] == {'exact': 50., 'f1': 50., 'total': 2}
    assert trials[1]['explicit_ids_error'] == 'KeyError'
    assert trials[2]['aggregate_default'] == trials[0]['aggregate_default']
    assert trials[3]['aggregation_error'] == 'ZeroDivisionError'
    assert [t['coverage_guard']['status'] for t in trials] == ['complete', 'error', 'error', 'error']
    result = {'captured_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'reviewed_metric_source_sha256': SOURCE_SHA,
              'selected_functions': list(NAMES),
              'selected_ast_sha256': hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest(),
              'git_head_at_execution': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'python': platform.python_version(), 'fixture': dataset, 'trials': trials,
              'scope': 'Invented three-question control of six reviewed v2 functions only. No official CLI, model/data scoring, v1.1 replication, field incidence or upstream defect claim. Fixed-ID guard is an Atlas research wrapper precondition; not a new metric.'}
    out = ROOT / 'docs/research/workload-screen/qa-metric-coverage-control.json'
    out.write_bytes((json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n').encode('utf-8'))
    print(json.dumps({'cases': len(trials), 'complete_score': trials[0]['aggregate_default'],
                      'partial_score': trials[1]['aggregate_default'], 'scope': result['scope']}))


if __name__ == '__main__':
    main()
