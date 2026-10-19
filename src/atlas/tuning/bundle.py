"""Explicit public-campaign export inventory. Private requests are never traversed."""
from pathlib import Path
import zipfile
import hashlib

from .common import ROOT, STORE, artifact_path, campaign_dir, domain, file_digest, read_json, write_json


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
    files = {}

    def keep(relative, required=True):
        path = artifact_path(folder, relative)
        if path.is_file():
            files[path.relative_to(ROOT).as_posix()] = path
        elif required:
            raise ValueError('Missing declared public evidence: ' + relative)

    # Enumerate schema-declared public evidence rather than trusting a directory's
    # contents. Unrelated notes, unreferenced blobs and symlinks outside it cannot leak.
    for name in ('protocol.json', 'execution.json', 'nomination.json', 'verified.json',
                 'status.json', 'repository.json', 'repository-observation.json', 'controller-ownership.json'):
        keep(name, required=name in ('protocol.json', 'execution.json', 'nomination.json'))
    for item in protocol['inputs']:
        keep(item['artifact']['path'])
    for name in protocol['sources']:
        keep('source/' + name)
    for entry in read_json(folder / 'execution.json')['cases']:
        relative = 'runs/' + entry['id'] + '/'
        keep(relative + 'config.json', required=False)
        keep(relative + 'result.json')
        for item in read_json(folder / relative / 'result.json').get('artifacts', {}).values():
            keep(item['path'])
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
             'benchmarks/runtime_tuning_api_controls.py', 'benchmarks/runtime_tuning_lifecycle.py',
             'tests/test_runtime_tuning.py', 'tests/runtime_tuning_ui.test.mjs',
             'docs/attention-lab.md', 'docs/architecture/hybrid-attention-lab.md',
             'configs/attention-windows-py312-requirements.txt',
             'lab-ui/index.html', 'lab-ui/app.css', 'lab-ui/app.js', 'lab-ui/tuning.js',
             'frontend/index.html', 'frontend/app.css', 'frontend/app.js', 'frontend/data/snapshot.json']
    extra += ['frontend/assets/' + name for name in ('atlas.svg', 'geist.woff2', 'font-source.json', 'OFL.txt')]
    extra += ['docs/research/autotuning/' + name for name in (
        'research-and-design.md', 'development-amendment.md', 'report.md', 'fastest-2k.json',
        'budget-2k.json', 'budget-32k.json', 'application-checks.json', 'lifecycle-checks.json',
        'api-admission-checks.json', 'api-activation-checks.json', 'preserved-sources.json',
        'acceptance-audit.md', 'browser-qa.json', 'checks.json', 'reproduce.md')]
    extra += ['docs/research/autotuning/' + name for name in (
        'resource-recovery-amendment.md', '32k-resource-interruption.json',
        'pre-recovery-fastest-2k.json', 'pre-recovery-budget-2k.json')]
    for name in extra:
        if (ROOT / name).is_file():
            files[name] = ROOT / name
    status_path = folder / 'status.json'
    profile_id = read_json(status_path).get('profile') if status_path.is_file() else None
    if profile_id:
        from .profiles import read
        profile = read(profile_id, home)
        chosen = verified['decision']['profile']
        if chosen is None:
            raise ValueError('No verified public profile belongs to this campaign')
        expected_profile = {'schema_version': 1, 'campaign': ident, 'domain': domain(protocol['settings']),
            'settings': protocol['settings'], 'identity': protocol['identity'], 'sources': protocol['sources'],
            'assets': protocol['assets'], 'protocol_sha256': verified['protocol_sha256'],
            'evidence_sha256': verified['evidence_sha256'], 'decision': verified['decision'],
            'execution': next(p for p in protocol['profiles'] if p['id'] == chosen)}
        if profile != expected_profile:
            raise ValueError('Saved public profile differs from the verified campaign schema')
        path = artifact_path(home / 'profiles', profile_id + '.json')
        files[path.relative_to(ROOT).as_posix()] = path
    inventory = {name: file_digest(path) for name, path in files.items()}
    guide = (f'# Atlas runtime campaign {ident}\n\n'
        'This bundle contains one public-input campaign, its full computed arrays, '
        'frozen execution source and the local tuning tool. Other historical studies, '
        'weights, environments, local activation state and private inference records are excluded.\n\n'
        f'For arithmetic verification, extract the archive and run:\n\n'
        f'`python benchmarks/verify_runtime_bundle.py .`\n\n'
        'Use Python 3.12 with NumPy 2.5.4. This checks byte, array and decision consistency '
        'without PyTorch or model execution. It does not establish independent replication.\n\n'
        'For new model execution, follow `docs/runtime-tuning.md` to install the pinned '
        'Windows CUDA environment and allowlisted public assets, then run `Start-Atlas.ps1`. '
        'Imported results remain historical evidence. A different machine must run a fresh '
        'campaign before applying a compatible profile. Legacy dashboard studies not supplied '
        'in this campaign bundle have no local execution history.\n').encode('utf-8')
    inventory['BUNDLE-README.md'] = hashlib.sha256(guide).hexdigest()
    temporary = out.with_suffix('.tmp')
    import json
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr('BUNDLE-README.md', guide)
        archive.writestr('bundle-manifest.json', json.dumps({'campaign': ident,
            'evidence_sha256': verified['evidence_sha256'], 'files_sha256': inventory,
            'verification': f'python tune.py verify {ident}', 'dependencies': 'Python 3.12 and NumPy; CUDA not needed for arithmetic verification',
            'exclusions': 'No weights, environments, private inference prompts or private outputs'}))
    # Verify the actual shipped payload, not just the files observed before writing.
    # A concurrent edit must never publish a manifest/payload mismatch.
    with zipfile.ZipFile(temporary) as archive:
        if set(archive.namelist()) != set(inventory) | {'bundle-manifest.json'}:
            raise ValueError('Archive inventory differs from the declared public files')
        for name, expected in inventory.items():
            checksum = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b''):
                    checksum.update(block)
            if checksum.hexdigest() != expected:
                raise ValueError('File changed while exporting: ' + name)
    temporary.replace(out)
    write_json(out.with_suffix('.receipt.json'), {'sha256': file_digest(out), 'bytes': out.stat().st_size,
                                                'files': len(inventory)+1, 'campaign': ident})
    return out
