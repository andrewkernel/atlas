"""Standalone measured-resource figure; optional isolated plotting environment."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'docs/research/workload-screen'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.tools/matplotlib-cache'))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                                'svg.hashsalt': 'atlas-qa-resource-v1', 'axes.spines.top': False,
                                'axes.spines.right': False})
    data = json.loads((DIRECTORY / 'qa-resource-derived.json').read_text(encoding='utf-8'))
    raw = json.loads((DIRECTORY / 'qa-resource-pilot.json').read_text(encoding='utf-8'))
    order = [(attention, micro, checkpointed) for micro, checkpointed in ((32, False), (32, True), (4, False), (4, True))
             for attention in ('eager', 'sdpa')]
    groups = {(g['attention'], g['microbatch'], g['checkpointed']): g for g in data['groups']}
    colors = {'eager': '#b2674d', 'sdpa': '#39789a'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    labels = []
    for index, key in enumerate(order):
        attention, micro, checkpointed = key
        group = groups[key]
        peaks = group['later_peak_allocated_bytes_by_repetition']
        assert len(peaks) == 2 and peaks[0] == peaks[1]
        memory = peaks[0] / 1024**3
        timing = group['median_update_seconds_by_repetition']
        label = ('SDPA' if attention == 'sdpa' else 'Eager') + f'  |  micro {micro}  |  checkpoint ' + ('on' if checkpointed else 'off')
        labels.append(label)
        axes[0].barh(index, memory, height=.55, color=colors[attention])
        axes[0].text(memory + .12, index, f'{memory:.2f}', va='center', fontsize=9)
        axes[1].plot(timing, [index - .08, index + .08], color='#c4c9ce', linewidth=1., zorder=1)
        axes[1].scatter([timing[0]], [index - .08], color=colors[attention], s=38, marker='o', zorder=2)
        axes[1].scatter([timing[1]], [index + .08], facecolor='white', edgecolor=colors[attention], s=36, marker='s', zorder=2)
    nominal = raw['trials'][0]['worker_result']['environment']['device_total_bytes'] / 1024**3
    axes[0].axvline(nominal, color='#707780', linestyle='--', linewidth=1.)
    axes[0].text(nominal - .15, 7.6, f'Nominal CUDA total: {nominal:.2f} GiB', ha='right', va='bottom', fontsize=9)
    axes[0].set_yticks(range(8), labels)
    axes[0].set_xlim(0, 16)
    axes[0].set_ylim(-.6, 8.)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Later CUDA allocated peak (GiB)')
    axes[0].set_title('Allocated peak: same in both repetitions', loc='left', fontsize=11, pad=15)
    axes[1].set_xlim(0, 5.)
    axes[1].set_xlabel('Median synchronized complete-update time (seconds)')
    axes[1].set_title('Two process medians, not confidence intervals', loc='left', fontsize=11, pad=15)
    axes[1].tick_params(axis='y', left=False, labelleft=False)
    for axis in axes:
        axis.grid(axis='x', color='#e8ebed', linewidth=.6)
        axis.set_axisbelow(True)
    fig.subplots_adjust(left=.29, right=.98, top=.78, bottom=.24, wspace=.22)
    fig.text(.04, .95, 'BERT QA: native resource tradeoffs', fontsize=19, weight='bold')
    fig.text(.04, .90, 'RTX 3060 / Windows · logical batch 32 · sequence length 384 · eight updates per process', fontsize=11)
    fig.text(.04, .855, 'Filled circle: repetition 1. Open square: repetition 2. Each timing point summarizes five measured updates.', fontsize=9, color='#40474d')
    fig.text(.04, .155, 'Allocator peaks are not physical residency. Completion above nominal capacity is not an OOM rescue.', fontsize=10)
    fig.text(.04, .11, 'Original worker-RSS readings are invalid; these panels use CUDA allocator counters and synchronized timers.', fontsize=9)
    fig.text(.04, .065, 'No numerical or QA-quality equivalence established. Source: frozen QA resource screen, 16 same-seed process trials.', fontsize=9)
    svg = DIRECTORY / 'qa-resource-figure.svg'
    preview = ROOT / 'artifacts/figures/qa-resource-figure.png'
    preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg, format='svg', metadata={'Date': None})
    fig.savefig(preview, dpi=150)
    plt.close(fig)
    packages = sorted([{'name': d.metadata['Name'], 'version': d.version} for d in importlib.metadata.distributions()], key=lambda x: x['name'].lower())
    result = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_canonical_sha256': {'qa-resource-derived.json': canonical(data), 'qa-resource-pilot.json': canonical(raw)},
              'requirements_file': 'configs/plot-windows-py311-requirements.txt',
              'requirements_sha256': hashlib.sha256((ROOT / 'configs/plot-windows-py311-requirements.txt').read_bytes()).hexdigest(),
              'environment': {'python': platform.python_version(), 'matplotlib': matplotlib.__version__, 'packages': packages},
              'figure': {'path': str(svg.relative_to(ROOT)).replace('\\', '/'), 'bytes': svg.stat().st_size, 'sha256': hashlib.sha256(svg.read_bytes()).hexdigest()},
              'scope': 'Figure of frozen allocator/update-time observations, not scientific equivalence, exclusive residency or population uncertainty.',
              'setup_note': 'Bundled runtime lacked plotting packages. A separate binary/hash-locked 3.11 plotting environment was used. An initial resolver build-interpreter warning was avoided by selecting the existing 3.11 executable explicitly; core/ML environments were unchanged.'}
    (DIRECTORY / 'qa-resource-figure.json').write_bytes((json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
    print(json.dumps({'figure': result['figure'], 'python': platform.python_version(), 'matplotlib': matplotlib.__version__}))


if __name__ == '__main__':
    main()
