"""Offline consistency checks for the research package, without importing workloads."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    'README.md', 'AGENTS.md', 'pyproject.toml', 'uv.lock', '.gitignore',
    'docs/README.md', 'docs/research/README.md',
    *[f'docs/research/{name}.md' for name in (
        '00-executive-assessment', '01-problem-definition', '02-literature-review',
        '03-competitive-landscape', '04-scientific-fidelity', '05-technical-feasibility',
        '06-novelty-analysis', '07-experimental-design', '08-local-llm-evaluation',
        '09-critical-review', '10-research-proposal', '11-open-questions', '12-research-log',
    )],
    *[f'docs/architecture/{name}.md' for name in ('overview','data-flow','components','security','api-design')],
    *[f'docs/roadmap/{name}.md' for name in ('implementation-plan','milestones','mvp-acceptance-criteria')],
    '.github/workflows/research.yml',
]


def main() -> int:
    failures: list[str] = []
    for name in REQUIRED:
        target = ROOT/name
        if not target.is_file() or target.stat().st_size == 0:
            failures.append(f'Missing/empty required file: {name}')

    markdown = [p for p in ROOT.rglob('*.md') if not any(x in p.parts for x in ('.venv','.git','.tools','.pytest_cache','.hypothesis'))]
    links=0
    external=set()
    for path in markdown:
        content=path.read_text(encoding='utf-8')
        prose=re.sub(r'```.*?```','',content,flags=re.S)
        for destination in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',prose):
            destination=destination.strip().strip('<>')
            parsed=urlsplit(destination)
            if parsed.scheme in ('http','https'):
                external.add(destination)
                continue
            if parsed.scheme or not parsed.path:
                continue
            links+=1
            target=(path.parent/unquote(parsed.path)).resolve()
            if not target.is_relative_to(ROOT):
                failures.append(f'Local link escapes repository: {path.relative_to(ROOT)} -> {destination}')
            elif not target.exists():
                failures.append(f'Broken local link: {path.relative_to(ROOT)} -> {destination}')
        if re.search(r'\b(?:TODO|TBD|FIXME)\b',prose):
            failures.append(f'Unresolved placeholder marker: {path.relative_to(ROOT)}')

    for path in (ROOT/'docs/research').rglob('*.json'):
        try:
            json.loads(path.read_text(encoding='utf-8'),parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (ValueError,UnicodeError) as exc:
            failures.append(f'Invalid/nonfinite JSON: {path.relative_to(ROOT)}: {exc}')

    metadata=tomllib.loads((ROOT/'pyproject.toml').read_text(encoding='utf-8'))
    lock=tomllib.loads((ROOT/'uv.lock').read_text(encoding='utf-8'))
    if metadata['project']['requires-python'] != '>=3.12,<3.13':
        failures.append('Unexpected Python scope')
    result=json.loads((ROOT/'docs/research/experiments/cpu-semantic-probes.json').read_text(encoding='utf-8'))
    digest=hashlib.sha256((ROOT/'benchmarks/semantic_probes.py').read_bytes()).hexdigest()
    if result['source_sha256'] != digest:
        failures.append('Recorded probe source hash no longer matches source')
    if result['environment']['device'] != 'cpu':
        failures.append('Unexpected original probe device')
    composition=json.loads((ROOT/'docs/research/experiments/cpu-composition-probes.json').read_text(encoding='utf-8'))
    if composition['source_sha256'] != hashlib.sha256((ROOT/'benchmarks/composition_probes.py').read_bytes()).hexdigest():
        failures.append('Recorded composition source hash no longer matches source')
    for script, evidence in (
        ('equivalence_calibration.py', 'equivalence-calibration.json'),
        ('state_replay_probes.py', 'state-replay-probes.json'),
        ('cuda_memory_probe.py', 'cuda-memory-probe.json'),
        ('cuda_budget_probe.py', 'cuda-budget-probe.json'),
        ('update_clock_probes.py', 'update-clock-probes.json'),
        ('statistical_assumption_audit.py', 'statistical-assumption-audit.json'),
        ('activation_liveness_probe.py', 'activation-liveness-probe.json'),
        ('replay_fashion_control.py', 'fashion-repeatability.json'),
        ('derive_research_summary.py', 'derived-summary.json'),
        ('precision_export_control.py', 'precision-export-control.json'),
        ('rockmate_compatibility_probe.py', 'rockmate-compatibility.json'),
        ('rockmate_schedule_diagnostics.py', 'rockmate-schedule-diagnostics.json'),
        ('rockmate_budget_frontier.py', 'rockmate-budget-frontier.json'),
        ('rockmate_state_control.py', 'rockmate-state-control.json'),
    ):
        recorded=json.loads((ROOT/'docs/research/experiments'/evidence).read_text(encoding='utf-8'))
        if recorded['source_sha256'] != hashlib.sha256((ROOT/'benchmarks'/script).read_bytes()).hexdigest():
            failures.append(f'Recorded source hash no longer matches: {script}')
    if len(list((ROOT/'docs/research/papers').glob('*.md'))) < 25:
        failures.append('Paper notes/index missing')
    cuda_environment=json.loads((ROOT/'docs/research/experiments/cuda-probe-environment.json').read_text(encoding='utf-8'))
    if cuda_environment['requirements_sha256'] != hashlib.sha256((ROOT/'configs/cuda-probe-windows-py312-requirements.txt').read_bytes()).hexdigest():
        failures.append('CUDA requirements identity does not match its environment record')
    budget_result=json.loads((ROOT/'docs/research/experiments/cuda-budget-probe.json').read_text(encoding='utf-8'))
    if budget_result['fixture_source_sha256'] != hashlib.sha256((ROOT/'benchmarks/cuda_memory_probe.py').read_bytes()).hexdigest():
        failures.append('CUDA budget fixture source identity is stale')
    model_metadata=json.loads((ROOT/'docs/research/local-model-metadata.json').read_text(encoding='utf-8'))
    if model_metadata['source_sha256'] != hashlib.sha256((ROOT/'benchmarks/snapshot_local_model_metadata.py').read_bytes()).hexdigest():
        failures.append('Local-model metadata snapshot source identity is stale')
    assumptions=json.loads((ROOT/'docs/research/experiments/statistical-assumption-audit.json').read_text(encoding='utf-8'))
    if assumptions['calibration_source_sha256'] != hashlib.sha256((ROOT/'benchmarks/equivalence_calibration.py').read_bytes()).hexdigest():
        failures.append('Statistical assumption audit calibration identity is stale')
    fashion_manifest=json.loads((ROOT/'docs/research/experiments/fashion-data-manifest.json').read_text(encoding='utf-8'))
    fashion_result=json.loads((ROOT/'docs/research/experiments/fashion-pilot.json').read_text(encoding='utf-8'))
    if fashion_manifest['source_sha256'] != hashlib.sha256((ROOT/'benchmarks/prepare_fashion_data.py').read_bytes()).hexdigest():
        failures.append('Fashion data preparation source identity is stale')
    if fashion_result['source_sha256'] != hashlib.sha256((ROOT/'benchmarks/fashion_pilot.py').read_bytes()).hexdigest():
        failures.append('Fashion pilot source identity is stale')
    if fashion_result['dataset_manifest'] != fashion_manifest:
        failures.append('Fashion pilot data manifest differs from acquisition record')
    expected={(seed,arm) for seed in fashion_result['config']['seeds'] for arm in (False,True)}
    observed={(trial['seed'],trial['candidate']) for trial in fashion_result['trials']}
    if observed != expected or len(observed) != len(fashion_result['trials']):
        failures.append('Fashion pilot intended-run denominator is incomplete/duplicated')
    for seed in fashion_result['config']['seeds']:
        pair=[trial for trial in fashion_result['trials'] if trial['seed']==seed]
        if len(pair)==2 and all(trial['status']=='completed' for trial in pair):
            for key in ('initial_state_digest','split_digest','sample_order_digest'):
                if pair[0][key] != pair[1][key]:
                    failures.append(f'Fashion paired identity mismatch: {seed}/{key}')
    fashion_replay=json.loads((ROOT/'docs/research/experiments/fashion-repeatability.json').read_text(encoding='utf-8'))
    if fashion_replay['fixture_source_sha256'] != hashlib.sha256((ROOT/'benchmarks/fashion_pilot.py').read_bytes()).hexdigest():
        failures.append('Fashion replay fixture identity is stale')
    bibliography=json.loads((ROOT/'docs/research/bibliography-availability-audit.json').read_text(encoding='utf-8'))
    if bibliography['source_sha256'] != hashlib.sha256((ROOT/'benchmarks/audit_primary_bibliography.py').read_bytes()).hexdigest():
        failures.append('Bibliography capture source identity is stale')
    derived=json.loads((ROOT/'docs/research/experiments/derived-summary.json').read_text(encoding='utf-8'))
    checker=json.loads((ROOT/'docs/research/experiments/trainparity-controls.json').read_text(encoding='utf-8'))
    for key,name in (
        ('source_sha256','benchmarks/run_trainparity_controls.py'),
        ('case_source_sha256','benchmarks/trainparity_cases.py'),
        ('config_sha256','configs/trainparity-controls.json'),
        ('review_sha256','docs/research/competitors/trainparity-execution-review.json'),
    ):
        if checker[key] != hashlib.sha256((ROOT/name).read_bytes()).hexdigest():
            failures.append(f'Checker control identity is stale: {name}')
    declared=json.loads((ROOT/'configs/trainparity-controls.json').read_text(encoding='utf-8'))
    if checker['config'] != declared:
        failures.append('Checker control configuration differs from frozen record')
    intended=[item['name'] for item in declared['controls']]
    observed=[item['name'] for item in checker['trials']]
    if intended != observed or len(set(observed)) != len(observed):
        failures.append('Checker control intended-run denominator is incomplete/duplicated')
    checker_environment=json.loads((ROOT/'docs/research/experiments/checker-environment.json').read_text(encoding='utf-8'))
    if checker_environment['requirements_sha256'] != hashlib.sha256((ROOT/checker_environment['requirements_file']).read_bytes()).hexdigest():
        failures.append('External checker requirements identity is stale')
    planner=json.loads((ROOT/'docs/research/experiments/rockmate-compatibility.json').read_text(encoding='utf-8'))
    if planner['review_sha256'] != hashlib.sha256((ROOT/'docs/research/competitors/rockmate-execution-review.json').read_bytes()).hexdigest():
        failures.append('Planner review identity is stale')
    intended={(mode,budget) for mode in planner['config']['modes'] for budget in planner['config']['bracket_budgets_mib']}
    observed={(trial['mode'],trial['budget_mib']) for trial in planner['trials']}
    if intended != observed or len(observed) != len(planner['trials']):
        failures.append('Planner bracket intended-run denominator is incomplete/duplicated')
    for name in ('rockmate-schedule-diagnostics.json','rockmate-budget-frontier.json','rockmate-state-control.json'):
        record=json.loads((ROOT/'docs/research/experiments'/name).read_text(encoding='utf-8'))
        if record['fixture_source_sha256'] != hashlib.sha256((ROOT/'benchmarks/rockmate_compatibility_probe.py').read_bytes()).hexdigest():
            failures.append(f'Planner follow-up fixture identity is stale: {name}')
    planner_environment=json.loads((ROOT/'docs/research/experiments/planner-environment.json').read_text(encoding='utf-8'))
    if planner_environment['requirements_sha256'] != hashlib.sha256((ROOT/planner_environment['requirements_file']).read_bytes()).hexdigest():
        failures.append('Planner environment requirements identity is stale')
    for name,expected_digest in derived['input_canonical_json_sha256'].items():
        value=json.loads((ROOT/'docs/research/experiments'/name).read_text(encoding='utf-8'))
        actual=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if actual != expected_digest:
            failures.append(f'Derived-summary input identity is stale: {name}')
    print(json.dumps({'markdown_files':len(markdown),'local_links_checked':links,'unique_external_links_catalogued':len(external),'lock_package_count':len(lock['package']),'failures':failures},indent=2))
    return bool(failures)


if __name__=='__main__':
    raise SystemExit(main())
