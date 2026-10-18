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
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atlas.attention.portable import canonical_bytes, file_digest, read_json, write_json
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
'''


def inventory(root, protocol):
    selected = {root/relative for relative in protocol['sources']}
    selected |= {root/'src/atlas/__init__.py', root/'Start-Atlas.ps1', root/'Stop-Atlas.ps1',
                 root/'benchmarks/package_attention_v3.py', root/'benchmarks/verify_researcher_bundle.py',
                 root/'benchmarks/attention_v3_progress.py',
                 root/'docs/attention-lab.md', root/'docs/architecture/hybrid-attention-lab.md',
                 root/'.gitattributes'}
    for relative in ('lab-ui','frontend'):
        for path in (root/relative).rglob('*'):
            if path.is_file() and path.suffix in ('.html','.css','.js','.svg','.woff2') and 'downloads' not in path.relative_to(root).parts:
                selected.add(path)
    selected |= {root/'frontend/data/snapshot.json', root/'frontend/assets/OFL.txt',
                 root/'frontend/assets/font-source.json',root/'docs/roadmap/research-v3-plan.md'}
    for path in (root/'docs/research/attention-v3').rglob('*'):
        if path.is_file() and path.name!='bundle-receipt.json' and '__pycache__' not in path.parts:
            selected.add(path)
    for name in ('hgca-2507.03153.pdf','beyond-hpdc2026.pdf','nomad-osdi24.pdf',
                 'miku-2503.17864v2.pdf','dsa-2lm-atc25.pdf'):
        path=root/'artifacts/cxl-papers'/name
        if path.is_file():selected.add(path)
    selected.add(root/'docs/research/papers/cxl-context.md')
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
                   'files':len(manifest['files_sha256']), 'external_reproduction':False}
    write_json(DOWNLOADS/'manifest.json',publication)
    write_json(STUDY/'bundle-receipt.json',publication)
    return publication


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(package(),indent=2))
