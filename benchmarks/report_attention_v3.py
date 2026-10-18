"""Render the researcher claim card from independently reconstructed evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atlas.attention.portable import file_digest,read_json,write_json
from verify_attention_v3 import build

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'docs/research/attention-v3'


def number(value,places=3):
    return '—' if value is None else f'{value:.{places}f}'


def render(summary):
    cases=summary['cases']
    arms=[a for c in cases for a in c['arms']]
    failures=[(c,a) for c in cases for a in c['arms'] if a['passed'] is False]
    unavailable=[(c,a) for c in cases for a in c['arms'] if a['status']!='completed']
    state='complete collection' if summary['all_cases_recorded'] else 'partial collection'
    lines=['# Atlas: long-context memory tradeoffs with inspectable evidence','',
           '**Atlas is a local experiment workspace that asks whether moving part of',
           'an LLM attention cache to CPU memory saves GPU allocation while preserving',
           'a declared numerical relation, and how much generation time that costs.**','',
           f'This report covers {summary["recorded_cases"]}/{summary["planned_cases"]} registered cases ({state}).',
           'Execution completion is separate from numerical eligibility. Failed or',
           'unavailable arms stay in the denominator and block the affected comparison.',
           'The system is a dense CPU/GPU baseline, not a new sparse-attention algorithm',
           'or a CXL/DSA implementation. No external reproduction is established.','',
           '## What was actually tested','',
           '- Pinned Qwen3-0.6B, FP32, one unpadded sequence on an RTX 3060 12 GB',
           '  and Ryzen 5 7600X with 32 GB RAM, Windows/WDDM shared desktop.',
           '- 32,768 and 40,000 token prompts. Six cases revisit three overlapping',
           '  WikiText inputs from v2; twelve use six separate PG-19 validation books.',
           '- Native model execution, a persistent native-SDPA cache and a dense',
           '  hybrid cache keeping 256 recent tokens on GPU and older entries on CPU.',
           '- Separate unchanged native reference; all 32 full-vocabulary predictions',
           '  checked on the same real continuation, with original atol=rtol=1e-4.',
           '  Both controls must pass before any candidate is eligible.',
           '- Six method-order permutations in the book cohort. Identical short',
           '  warmups. One free 16-token generation window per arm; prefill, selection,',
           '  cache updates, transfers and synchronized forwards included.',
           '- Full arrays, token identities, source snapshots, raw timings, memory',
           '  observations and logs retained. A separate NumPy verifier reconstructs',
           '  decisions and arithmetic; it imports no runtime comparator.','',
           '## Results and limits of the claim','',
           '| Input cohort | Context | Qualified pairs / planned | Native time / hybrid time, median | Allocator peak reduction, median | Meaning |',
           '| --- | ---: | ---: | ---: | ---: | --- |']
    for item in summary['comparisons']:
        ratio=item['eligible_ratio_median']
        meaning='No qualified pair' if ratio is None else ('Hybrid took longer' if ratio<1 else 'Hybrid took less time in these qualified pairs')
        reductions=item['eligible_allocator_reduction_percent']
        reduction=number(statistics.median(reductions),1)+'%' if reductions else '—'
        lines.append(f'| {item["cohort"]} | {item["context"]:,} | {item["eligible_pairs"]}/{item["planned"]} | {number(ratio)} | {reduction} | {meaning} |')
    lines+=['','These are descriptive paired observations, not population estimates.',
            'A median from qualifying pairs does not repair failure on other inputs.',
            'Allocator reductions apply to measured PyTorch peaks, not exclusive',
            'physical VRAM residency or a hard memory quota. An OOM rescue requires',
            'a completed, numerically qualified candidate and a failed native arm;',
            'the predeclared two-control gate does not establish such a rescue.','',
            f'{len(failures)} completed arms failed the literal numerical contract; {len(unavailable)} arms have unavailable numerical outcomes.',
            'These are arm counts, not a scientific-validity score. Small distribution',
            'differences or matching greedy tokens do not override failed bounds.','',
            '| Case | Arm | Outcome | Violating elements |',
            '| --- | --- | --- | ---: |']
    for case,arm in failures+unavailable:
        lines.append(f'| {case["case_id"]} | {arm["plan"]} | {"Outside bounds" if arm["passed"] is False else arm["status"]} | {arm.get("violating_elements","—")} |')
    lines+=['','## What another researcher should try to falsify','',
            '1. Verify the shipped bytes and reconstruct every numerical check from',
            '   the full arrays before trusting the summary.',
            '2. Run the fixed registry under a fresh replication ID on another machine.',
            '   Preserve native-control failures, OOMs, timeouts and all attempts.',
            '3. Test whether the memory/time tradeoff persists on other models,',
            '   precisions, documents, continuation lengths and separately scheduled sessions.',
            '4. Investigate whether attention arithmetic or framework propagation',
            '   explains literal failures. Any implementation repair needs a new',
            '   source-bound protocol and fresh cases; do not relax old bounds.',
            '5. Compare with competent native/offload/sparse baselines before claiming',
            '   a policy or algorithm contribution. Actual CXL needs actual hardware.','',
            '## Remaining bias and uncertainty','',
            'One shared desktop session, one modern model family and FP32 do not',
            'establish generalization. Long book prefixes are selected by length and',
            'historical writing has corpus bias. These public texts may be in model',
            'training data. Contexts within a book are nested; prior WikiText windows',
            'overlap and were already inspected. Native reference always precedes',
            'candidate arms. Short warmups do not establish long steady-state timing.',
            'Arms share one loaded model and CUDA process. Device/host allocator',
            'history may affect later arms, including OOM outcomes; clearing the',
            'device cache is not a proof that all host/framework state was reset.',
            'Balanced arm order does not establish per-arm process isolation. These',
            'OOMs therefore do not prove that an arm cannot run in a fresh process.',
            'Recorded OOM exception prefixes also contain implausibly large per-process',
            'memory numbers. Those vendor/framework diagnostics are preserved as',
            'text, not accepted as measured memory. NVIDIA documents per-process',
            'usedGpuMemory as unavailable under WDDM; an unavailable diagnostic is',
            'consistent with these values, but the exact formatting cause was not',
            'independently traced. See the [NVML reference](https://docs.nvidia.com/deploy/nvml-api/latest/api/structnvmlProcessInfo__t.html).',
            'Desktop/UI work and read-only analyses occurred during the shared',
            'study session. Full-array reconstruction was scheduled in native',
            'reference/validation phases, outside timed generation windows. There',
            'is no exclusive resource reservation or calibrated power measurement.',
            'The 32-position relation is not broad language quality, truth, training',
            'fidelity or scientific-conclusion preservation. No zero-bias or universal',
            'correctness claim is justified. Prior FP16/BF16 failures remain intact.','',
            '## Reproduce and inspect','',
            'Follow [reproduce.md](reproduce.md). The frozen [protocol](protocol.json)',
            'binds sources, assets, cases, bounds and analysis. [execution.json](execution.json)',
            'retains outcomes; [summary.json](summary.json) is independently reconstructed.',
            'Full arrays stay outside Git and are included in the local researcher ZIP.',
            'Byte verification cannot prove author honesty or real device execution;',
            'fresh independent execution and human review remain separate.','',
            f'Protocol SHA-256: `{summary["protocol_sha256"]}`.','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    isolation=None;isolation_path=ROOT/'docs/research/attention-isolation'
    if (ROOT/'docs/roadmap/isolation-diagnostic.md').exists() and not (isolation_path/'protocol.json').exists():
        raise ValueError('Isolated-arm diagnostic must be frozen and completed before final handoff')
    if (isolation_path/'protocol.json').exists():
        ledger=read_json(isolation_path/'execution.json') if (isolation_path/'execution.json').exists() else {}
        if not ledger.get('finished_at'):raise ValueError('Isolated-arm diagnostic must finish before final handoff')
        from verify_attention_isolation import build as build_isolation
        isolation=build_isolation(isolation_path,ROOT)
        if not isolation['all_cases_recorded']:raise ValueError('Isolated-arm diagnostic is incomplete')
        write_json(isolation_path/'summary.json',isolation)
    summary=build(STUDY,ROOT)
    if not summary['all_cases_recorded']:raise ValueError('Final report requires all registered outcomes')
    write_json(STUDY/'summary.json',summary)
    content=render(summary)
    if isolation:
        content+='\n## Isolated-process follow-up (post-hoc diagnostic)\n\n'
        content+='Book 2383 was selected after its v3 numerical failure and control OOMs. Each arm/context now uses a fresh process, the same short warmup, and generation before reference preparation. This is a diagnostic of the known failure pair, not an unseen confirmation cohort.\n\n'
        content+='| Context | Arm | Outcome | Literal relation | Generation (s) |\n| ---: | --- | --- | --- | ---: |\n'
        for row in isolation['cases']:
            relation='pass' if row['passed'] is True else 'fail' if row['passed'] is False else 'unknown'
            content+=f'| {row["context"]:,} | {row["method"]} | {row["arm_status"]} | {relation} | {number(row.get("generation_ms",0)/1000) if row.get("generation_ms") is not None else "—"} |\n'
        content+='\nPinned host allocator observations after generation cleanup:\n\n| Context | Arm | Allocator-owned (GiB) | Active (GiB) |\n| ---: | --- | ---: | ---: |\n'
        for row in isolation['cases']:
            stats=row.get('host_allocator',{}).get('after_generation_cleanup',{})
            owned=stats.get('allocated_bytes.current');active=stats.get('active_bytes.current')
            content+=f'| {row["context"]:,} | {row["method"]} | {number(owned/2**30,6) if isinstance(owned,(int,float)) else "—"} | {number(active/2**30,6) if isinstance(active,(int,float)) else "—"} |\n'
        content+='\nThese are runtime allocator counters, not physical-RAM or exclusive-residency measurements. The [PyTorch 2.14 API](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.host_memory_stats.html) distinguishes allocator-owned blocks from active requests and describes block-size rounding. Missing observations remain unknown. Retained inactive blocks would support a process-history concern, without proving which original OOM they caused.\n'
        content+='\nAll original v3 failures remain unchanged. Disappearing OOMs would be a counterexample to a hard feasibility interpretation, not proof of a particular allocator cause. Host allocator observations are retained, not inferred from RSS. Numerical failures still block eligibility. See [diagnostic evidence](../attention-isolation/summary.json) and its separately frozen protocol.\n'
    (STUDY/'report.md').write_bytes(content.encode('utf-8'))
    receipt={
        'protocol_sha256':summary['protocol_sha256'], 'all_cases_recorded':True,
        'execution_sha256':file_digest(STUDY/'execution.json'),
        'summary_sha256':file_digest(STUDY/'summary.json'),
        'report_sha256':file_digest(STUDY/'report.md'),
        'analysis_source_sha256':file_digest(Path(__file__))}
    if isolation:receipt['isolation_summary_sha256']=file_digest(isolation_path/'summary.json')
    write_json(STUDY/'report-receipt.json',receipt)
    print({'report':'docs/research/attention-v3/report.md','cases':summary['recorded_cases']})


if __name__=='__main__':main()
