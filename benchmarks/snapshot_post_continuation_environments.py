"""Read-only post-study metadata/source checks and a separately labeled runtime read."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.metadata as meta
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from board_energy_control import ROOT, child_environment
from snapshot_final_environments import PROFILES as ORIGINAL_PROFILES, SELECTED, normalized

DIRECTORY = ROOT / 'docs/research/workload-screen'
PROFILES = {name: (executable, requirements) for name, (executable, _, requirements) in ORIGINAL_PROFILES.items()}
PROFILES.update({'qa': ('.tools/qa-env/Scripts/python.exe', 'configs/qa-windows-py311-requirements.txt'),
                 'plot': ('.tools/plot-env/Scripts/python.exe', 'configs/plot-windows-py311-requirements.txt')})


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def worker(profile):
    packages = {normalized(d.metadata['Name']): d.version for d in meta.distributions()}
    selected = {(name, relative) for name, paths in SELECTED.items() if name in packages for relative in paths}
    if profile == 'qa':
        for review_name in ('qa-execution-review.json', 'qa-profiler-review.json', 'qa-optimizer-review.json'):
            review = json.loads((DIRECTORY / review_name).read_text(encoding='utf-8'))
            selected.update((entry['path'].split('/')[0], entry['path']) for entry in review['files'])
    if profile == 'plot':
        selected.update(('matplotlib', path) for path in ('matplotlib/__init__.py', 'matplotlib/pyplot.py', 'matplotlib/backends/backend_svg.py'))
    files = []
    for package, relative in sorted(selected):
        dist = meta.distribution(package)
        entries = {str(f).replace('\\', '/'): f for f in dist.files or []}
        entry = entries.get(relative)
        if entry is None:
            files.append({'distribution': package, 'path': relative, 'matches_installed_record': False, 'status': 'missing_record_entry'})
            continue
        raw = dist.locate_file(entry).read_bytes()
        digest = hashlib.sha256(raw).digest()
        matches = entry.hash is not None and entry.hash.mode == 'sha256' and entry.hash.value == base64.urlsafe_b64encode(digest).decode().rstrip('=')
        files.append({'distribution': package, 'path': relative, 'sha256': digest.hex(), 'matches_installed_record': matches, 'status': 'read'})
    result = {'python': platform.python_version(), 'packages': packages, 'selected_files': files,
              'runtime_read': None, 'scope': 'Distribution metadata and selected Python-source hashes; unsigned RECORD is not a signature or whole-package attestation.'}
    if profile == 'qa' and all(row['matches_installed_record'] for row in files):
        import torch
        result['runtime_read'] = {'observed_after_studies': True, 'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                                  'cudnn': torch.backends.cudnn.version(), 'cuda_context_initialized': torch.cuda.is_initialized(),
                                  'scope': 'Post-study library version read; no model/training runs, not a per-worker contemporaneous record.'}
    print(json.dumps(result, sort_keys=True, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=sorted(PROFILES))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    import psutil
    from windows_job_scope import WindowsJobScope
    earlier = json.loads((ROOT / 'docs/research/experiments/final-environment-inventory.json').read_text(encoding='utf-8'))
    qa_before = json.loads((DIRECTORY / 'qa-environment.json').read_text(encoding='utf-8'))
    plot_before = json.loads((DIRECTORY / 'qa-resource-figure.json').read_text(encoding='utf-8'))
    result = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(Path(__file__)),
              'helper_sha256': {name: sha(ROOT / 'benchmarks' / name) for name in ('snapshot_final_environments.py', 'board_energy_control.py', 'windows_job_scope.py')},
              'earlier_inventory_canonical_sha256': canonical(earlier), 'qa_before_canonical_sha256': canonical(qa_before),
              'plot_figure_record_canonical_sha256': canonical(plot_before), 'profiles': {}, 'failures': [],
              'scope': 'Post-study checks, not contemporaneous attestation or a full supply-chain/driver audit. Only QA runtime read explicitly imports Torch; no model or training runs.'}
    base = ROOT / 'artifacts/post-continuation-environments' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for name, (executable, requirements) in PROFILES.items():
        scratch = base / name
        scratch.mkdir(parents=True)
        interpreter = ROOT / executable
        job = WindowsJobScope()
        process = None
        try:
            with (scratch / 'stdout.txt').open('w') as stdout, (scratch / 'stderr.txt').open('w') as stderr:
                process = subprocess.Popen([str(interpreter), str(Path(__file__).resolve()), '--worker', name],
                                           cwd=ROOT, env=child_environment(interpreter, scratch), stdout=stdout, stderr=stderr,
                                           creationflags=0x00000004)
                root = psutil.Process(process.pid)
                job.assign_owned(root.pid, root.create_time())
                job.resume_owned_single_thread(root)
                process.wait(timeout=30)
            if process.returncode != 0:
                raise RuntimeError('Metadata worker exited with code ' + str(process.returncode))
            row = json.loads((scratch / 'stdout.txt').read_text(encoding='utf-8'))
            row.update({'requirements_file': requirements, 'requirements_sha256': sha(ROOT / requirements)})
            before = earlier['profiles'][name]['packages'] if name in earlier['profiles'] else {
                normalized(p['name']): p['version'] for p in (qa_before['packages'] if name == 'qa' else plot_before['environment']['packages'])}
            row['prior_version_mismatches'] = {package: {'before': version, 'after': row['packages'].get(package)}
                                               for package, version in before.items() if row['packages'].get(package) != version}
            row['additional_distributions'] = sorted(set(row['packages']) - set(before))
            if row['prior_version_mismatches'] or row['additional_distributions'] or any(not f['matches_installed_record'] for f in row['selected_files']):
                result['failures'].append('Environment reconciliation differs: ' + name)
            result['profiles'][name] = row
        except Exception as error:
            result['profiles'][name] = {'status': 'error', 'error_type': type(error).__name__, 'message': str(error).replace(str(ROOT), '<atlas>')}
            result['failures'].append('Environment read failed: ' + name)
        finally:
            job.terminate(47)
            if process is not None:
                process.wait(timeout=5)
            deadline = time.monotonic() + 2
            while job.members() and time.monotonic() < deadline:
                time.sleep(.02)
            if job.members():
                raise RuntimeError('Metadata job cleanup incomplete')
            job.close()
        print(json.dumps({'profile': name, 'status': result['profiles'][name].get('status', 'read'),
                          'mismatches': result['profiles'][name].get('prior_version_mismatches')}), flush=True)
    assets = json.loads((DIRECTORY / 'qa-asset-manifest.json').read_text(encoding='utf-8'))
    result['qa_asset_manifest_canonical_sha256'] = canonical(assets)
    result['asset_checks'] = {}
    for name, expected in assets['files'].items():
        path = ROOT / ('data/squad/' + name.removeprefix('data/') if name.startswith('data/') else
                       'weights/bert-base-uncased-pinned/' + name.removeprefix('model/'))
        matches = path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256']
        result['asset_checks'][name] = {'sha256': sha(path), 'bytes': path.stat().st_size, 'matches_acquisition': matches}
        if not matches:
            result['failures'].append('QA asset differs: ' + name)
    manifest = json.loads((DIRECTORY / 'qa-feature-manifest.json').read_text(encoding='utf-8'))
    result['qa_feature_manifest_canonical_sha256'] = canonical(manifest)
    result['feature_npz_sha256'] = sha(ROOT / 'data/squad/qa-resource-features.npz')
    if result['feature_npz_sha256'] != manifest['npz_sha256']:
        result['failures'].append('QA feature bytes differ')
    result['reviewed_reference_checks'] = {}
    for label, folder in (('trainparity', 'trainparity-reviewed'), ('rockmate', 'rockmate-review')):
        review = json.loads((ROOT / f'docs/research/competitors/{label}-execution-review.json').read_text(encoding='utf-8'))
        matches = all(sha(ROOT / '.tools' / folder / entry['path']) == entry['sha256'] for entry in review['files'])
        result['reviewed_reference_checks'][label] = {'files': len(review['files']), 'all_match': matches, 'review_canonical_sha256': canonical(review)}
        if not matches:
            result['failures'].append('Reviewed source differs: ' + label)
    observed_at = datetime.now(timezone.utc).isoformat()
    driver = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=10, check=False)
    result['post_study_driver_read'] = {'captured_utc': observed_at, 'returncode': driver.returncode,
                                       'csv_columns': ['name', 'driver_version', 'memory_total_MiB'], 'stdout': driver.stdout.strip(),
                                       'scope': 'Post-study vendor utility read; per-QA-worker driver/cuDNN records were not captured and are not retroactively supplied.'}
    (DIRECTORY / 'post-study-environments.json').write_bytes((json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
    print(json.dumps({'profiles': len(result['profiles']), 'selected_source_files': sum(len(p.get('selected_files', [])) for p in result['profiles'].values()),
                      'assets': len(result['asset_checks']), 'failures': result['failures']}))
    raise SystemExit(bool(result['failures']))


if __name__ == '__main__':
    main()
