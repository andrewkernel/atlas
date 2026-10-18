"""Build a local researcher bundle only after independently verifying all outcomes.

No upload or publication. Weights, corpus text, environments and unrelated files
are excluded. The archive retains exact source/raw/array bytes and all failures.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atlas.attention.portable import artifact_path, canonical_bytes, file_digest, read_json, write_json
from verify_attention_v3 import build

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'docs/research/attention-v3'
DOWNLOADS = ROOT/'lab-ui/downloads'

README = '''# Atlas researcher kit

This bundle contains a local CPU/GPU attention experiment, its real recorded
outcomes, exact execution source, full numerical outputs and an independent
NumPy verifier. It includes negative results. It is self-produced evidence;
external reproduction and a bias-free benchmark are not established.

Start with docs/research/attention-v3/report.md and reproduce.md. The protocol
defines the registered cases, original bounds, inputs and source identities.
The execution ledger retains every attempted case. Do not infer correctness
from execution completion or silently discard failed/unknown cases.

Extract to a short directory on Windows: deeply nested project folders can
exceed legacy path limits for the retained source snapshots. A location outside
unrelated Git checkouts also avoids attributing their Git metadata to this kit.
The full papers open from their original sources in the research library;
local paper caches are not redistributed in this archive.

To check the bundle bytes using only Python:

    python benchmarks/verify_researcher_bundle.py .

To independently reconstruct numerical decisions (Python + NumPy required):

    python benchmarks/verify_attention_v3.py --study docs/research/attention-v3 --source-root .

To run fresh experiments, follow the pinned Windows environment and asset
preparation instructions in reproduce.md. Use a new replication ID, preserving
the original study. Weights and original datasets are downloaded from their
pinned official sources; no remote model/dataset Python or pickle is executed.
Different hardware/software can produce different outcomes. No Linux CUDA
execution, CXL/DSA, broad answer-quality equivalence, energy benefit, novelty
or exclusive physical VRAM is established by this kit.

The manifest covers all shipped files except itself. A matching hash verifies
identity and detects accidental changes; it cannot prove author honesty or
independent execution. The outer ZIP hash is supplied separately. No upload or
public release was performed. A repository-wide public license has not been
selected; upstream dependencies/assets retain their own licenses.

Environment, weight/data preparation and fresh-run output directories are
expected to grow. The byte verifier ignores new files only under its explicit
generated-path allowlist; it still checks every shipped file and rejects new
source files. Downloaded assets and new run results require protocol preflight
and numerical verification, respectively; bundle verification does not bless
those new files.
'''


def inventory(root, protocol):
    selected = {root/relative for relative in protocol['sources']}
    # Ship the actual tracked project, including tests and referenced prior-study
    # tools. Never recurse through untracked environments, data, weights or secrets.
    try:
        top=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=root,stderr=subprocess.DEVNULL,text=True).strip())
        names=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode('utf-8').split('\0') if top.resolve()==root.resolve() else []
    except (OSError,subprocess.SubprocessError):names=[]
    if not names and (root/'bundle-manifest.json').exists():names=list(read_json(root/'bundle-manifest.json')['files_sha256'])
    for name in names:
        excluded={'README.md','docs/research/attention-v3/bundle-receipt.json','docs/research/attention-v3/handoff-verification.json'}
        if name and name not in excluded and not name.startswith('lab-ui/downloads/'):
            path=artifact_path(root,name)
            if path.is_file():selected.add(path)
    selected |= {root/'src/atlas/__init__.py', root/'Start-Atlas.ps1', root/'Stop-Atlas.ps1',
                 root/'benchmarks/package_attention_v3.py', root/'benchmarks/verify_researcher_bundle.py',
                 root/'benchmarks/attention_v3_progress.py',
                 root/'benchmarks/report_attention_v3.py',
                 root/'benchmarks/finalize_attention_v3.py',
                 root/'benchmarks/attention_isolation.py',root/'benchmarks/verify_attention_isolation.py',
                 root/'benchmarks/prepare_attention_history.py',
                 root/'docs/attention-lab.md', root/'docs/architecture/hybrid-attention-lab.md',
                 root/'.gitattributes'}
    for relative in ('lab-ui','frontend'):
        for path in (root/relative).rglob('*'):
            if path.is_file() and path.suffix in ('.html','.css','.js','.svg','.woff2') and 'downloads' not in path.relative_to(root).parts:
                selected.add(path)
    selected |= {root/'frontend/data/snapshot.json', root/'frontend/assets/OFL.txt',
                 root/'frontend/assets/font-source.json',root/'docs/roadmap/research-v3-plan.md'}
    for path in (root/'docs/research/attention-v3').rglob('*'):
        if path.is_file() and path.name not in ('bundle-receipt.json','handoff-verification.json') and '__pycache__' not in path.parts:
            selected.add(path)
    selected.add(root/'docs/research/papers/cxl-context.md')
    selected.add(root/'lab-ui/data/isolation-summary.json')
    selected.add(root/'lab-ui/data/reference-history.json')
    for path in (root/'docs/research/attention-isolation').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:selected.add(path)
    for path in (root/'docs/research/attention-v2').glob('*'):
        if path.is_file() and path.suffix in ('.md','.json'):
            selected.add(path)
    return {p.relative_to(root).as_posix():p for p in sorted(selected)}


def package():
    result = build(STUDY, ROOT)
    if not result['all_cases_recorded']:
        raise ValueError('Complete researcher kit requires every registered outcome, including failures')
    if not (STUDY/'report.md').is_file():
        raise ValueError('A researcher-facing report is required')
    write_json(STUDY/'summary.json',result)
    receipt=read_json(STUDY/'report-receipt.json')
    if not receipt.get('all_cases_recorded') or receipt['protocol_sha256']!=result['protocol_sha256']:
        raise ValueError('Final report identity is incomplete or stale')
    for name in ('execution','summary','report'):
        path=STUDY/(name+('.md' if name=='report' else '.json'))
        if file_digest(path)!=receipt[name+'_sha256']:
            raise ValueError('Final report is stale: '+name)
    from verify_attention_isolation import build as build_isolation
    isolation=build_isolation(ROOT/'docs/research/attention-isolation',ROOT)
    if not isolation['all_cases_recorded']:raise ValueError('Isolated diagnostic remains incomplete')
    if receipt.get('isolation_summary_sha256')!=file_digest(ROOT/'docs/research/attention-isolation/summary.json'):
        raise ValueError('Isolated diagnostic and final report do not match')
    write_json(ROOT/'lab-ui/data/isolation-summary.json',{**isolation,
               'base_protocol_sha256':read_json(ROOT/'docs/research/attention-isolation/protocol.json')['base_protocol_sha256']})
    from prepare_attention_history import prepare
    prepare(ROOT)
    protocol = read_json(STUDY/'protocol.json')
    files = inventory(ROOT,protocol)
    for required in ('benchmarks/verify_researcher_bundle.py','src/atlas/__init__.py'):
        if required not in files or not files[required].is_file():
            raise ValueError('Missing package dependency: '+required)
    manifest = {'schema_version':1, 'protocol_sha256':result['protocol_sha256'],
                'recorded_cases':result['recorded_cases'], 'planned_cases':result['planned_cases'],
                'all_cases_recorded':True,
                'verification_scope':result['verification'],
                'files_sha256':{name:file_digest(path) for name,path in files.items()}}
    manifest['files_sha256']['README.md'] = hashlib.sha256(README.encode('utf-8')).hexdigest()
    DOWNLOADS.mkdir(parents=True,exist_ok=True)
    archive = DOWNLOADS/'atlas-researcher-kit.zip'
    temporary = archive.with_suffix('.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as out:
        for name,path in files.items():
            # This explicit inventory never traverses weights/data/.git/environments.
            out.write(path,'atlas-researcher-kit/'+name)
        out.writestr('atlas-researcher-kit/README.md',README.encode('utf-8'))
        out.writestr('atlas-researcher-kit/bundle-manifest.json',canonical_bytes(manifest))
    # Verify every archived byte against the pre-read inventory before exposing it.
    with zipfile.ZipFile(temporary) as bundle:
        for name,expected in manifest['files_sha256'].items():
            with bundle.open('atlas-researcher-kit/'+name) as stream:
                hasher = hashlib.sha256()
                for chunk in iter(lambda:stream.read(8*1024**2),b''):hasher.update(chunk)
            if hasher.hexdigest()!=expected:raise ValueError('Archive changed during construction: '+name)
    temporary.replace(archive)
    publication = {'schema_version':1, 'built_at':datetime.now(timezone.utc).isoformat(),
                   'filename':archive.name, 'sha256':file_digest(archive), 'bytes':archive.stat().st_size,
                   'protocol_sha256':result['protocol_sha256'], 'all_cases_recorded':True,
                   'recorded_cases':result['recorded_cases'], 'planned_cases':result['planned_cases'],
                   'isolated_cases':isolation['recorded_cases'],
                   'files':len(manifest['files_sha256']), 'external_reproduction':False}
    write_json(DOWNLOADS/'manifest.json',publication)
    write_json(DOWNLOADS/'isolation-summary.json',{**isolation,
               'base_protocol_sha256':read_json(ROOT/'docs/research/attention-isolation/protocol.json')['base_protocol_sha256']})
    write_json(STUDY/'bundle-receipt.json',publication)
    return publication


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(package(),indent=2))
