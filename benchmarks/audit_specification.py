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
    print(json.dumps({'markdown_files':len(markdown),'local_links_checked':links,'unique_external_links_catalogued':len(external),'lock_package_count':len(lock['package']),'failures':failures},indent=2))
    return bool(failures)


if __name__=='__main__':
    raise SystemExit(main())
