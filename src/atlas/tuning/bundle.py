"""Explicit public-campaign export inventory. Private requests are never traversed."""
from pathlib import Path
import zipfile

from .common import ROOT, STORE, campaign_dir, file_digest, frozen_json, read_json, write_json


def export(ident, home=STORE):
    from filelock import FileLock
    (home / 'exports').mkdir(parents=True, exist_ok=True)
    with FileLock(str(home / 'exports' / (ident + '.lock')), timeout=120):
        return _export(ident, home)


def _export(ident, home):
    from .verify import build
    folder = campaign_dir(ident, home)
    verified = build(folder)
    protocol = read_json(folder / 'protocol.json')
    out = home / 'exports' / (ident + '.zip')
    out.parent.mkdir(parents=True, exist_ok=True)
    files = {p.relative_to(ROOT).as_posix(): p for p in folder.rglob('*') if p.is_file() and
             p.suffix in ('.json', '.npy', '.py', '.txt') and p.name != 'request.json'}
    files.update({name: ROOT / name for name in protocol['sources']})
    for name in ('src/atlas/__init__.py', 'src/atlas/attention/__init__.py',
                 'docs/research/autotuning/research-and-design.md', 'docs/architecture/runtime-autotuner.md',
                 'docs/roadmap/autotuning-acceptance.md', 'configs/attention-v2-windows-py312-requirements.txt',
                 'src/atlas/tuning/bundle.py', 'docs/runtime-tuning.md', 'benchmarks/verify_runtime_bundle.py'):
        if (ROOT / name).is_file():
            files[name] = ROOT / name
    # Include the usable local tool as well as the frozen measured CLI sources.
    extra = ['src/atlas/tuning/service.py', 'Start-Atlas.ps1', 'Stop-Atlas.ps1',
             'benchmarks/atlas_service_owner.py', 'benchmarks/prepare_attention_history.py',
             'docs/attention-lab.md', 'docs/architecture/hybrid-attention-lab.md',
             'configs/attention-windows-py312-requirements.txt',
             'lab-ui/index.html', 'lab-ui/app.css', 'lab-ui/app.js', 'lab-ui/tuning.js',
             'frontend/index.html', 'frontend/app.css', 'frontend/app.js', 'frontend/data/snapshot.json']
    extra += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'frontend/assets').glob('*') if p.is_file()]
    for name in extra:
        if (ROOT / name).is_file():
            files[name] = ROOT / name
    for path in (home / 'profiles').glob('*.json'):
        if read_json(path).get('campaign') == ident:
            files[path.relative_to(ROOT).as_posix()] = path
    inventory = {name: file_digest(path) for name, path in files.items()}
    temporary = out.with_suffix('.tmp')
    import json
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr('bundle-manifest.json', json.dumps({'campaign': ident,
            'evidence_sha256': verified['evidence_sha256'], 'files_sha256': inventory,
            'verification': f'python tune.py verify {ident}', 'dependencies': 'Python 3.12 and NumPy; CUDA not needed for arithmetic verification',
            'exclusions': 'No weights, environments, private inference prompts or private outputs'}))
    temporary.replace(out)
    write_json(out.with_suffix('.receipt.json'), {'sha256': file_digest(out), 'bytes': out.stat().st_size,
                                                'files': len(files)+1, 'campaign': ident})
    return out
