# Atlas long-context study v3

This is a separately registered extension of v2, not a rewrite of its unknown
pressure results. Self-produced artifacts and local repeats do not establish
external reproduction. Read protocol.json, execution.json and every failure.

## Tested environment and pinned inputs

The tested profile is Windows, Python 3.12.10, RTX 3060 12 GB, 32 GB RAM,
PyTorch 2.14.1+cu130 and Transformers 5.19.0. Installation uses the existing exact
Windows v2 requirements, including hashes and the official CUDA wheel. Linux
execution has not been tested; do not interpret portable JSON/NumPy as evidence
that the CUDA environment has been validated on Linux. No weights or source
dataset text are included in reports.

From the repository root, with an official uv installation:

```powershell
uv venv .tools/attention-env --python 3.12
uv pip install --python .tools/attention-env/Scripts/python.exe --require-hashes -r configs/attention-v2-windows-py312-requirements.txt
.tools/attention-env/Scripts/python.exe lab.py prepare-research --model qwen3-0.6b
.tools/attention-env/Scripts/python.exe lab.py prepare-research --model corpus
.tools/attention-env/Scripts/python.exe benchmarks/attention_v3_study.py prepare
.tools/attention-env/Scripts/python.exe benchmarks/attention_v3_study.py preflight
.tools/attention-env/Scripts/python.exe benchmarks/attention_v3_study.py run --replication-id independent-01
.tools/attention-env/Scripts/python.exe benchmarks/attention_v3_study.py analyze --replication-id independent-01
```

Use `--case-id long-000` together with a fresh replication ID for one case. This
is a real 32k-token model run, not a cheap CPU smoke test. Workers have a 1,200s
wall bound, execute offline builtin Qwen3 code and are serialized on the GPU.
No remote model/dataset Python or pickle weights are executed. The original
study remains unchanged. A completed failed case is not silently retried.

`prepare` restores exactly the frozen original-bucket PG-19 subset using pinned
object generations and byte hashes. It does not reselect books on a new machine.
Manifests are compared semantically, so Windows/Linux JSON line endings cannot
change the identity of the underlying model/data files. Source hashes are exact;
Git checks out Python with LF according to .gitattributes.

## Verify evidence without running the model

The verifier requires Python and NumPy, and imports no PyTorch or Transformers:

```powershell
.tools/attention-env/Scripts/python.exe benchmarks/verify_attention_v3.py --study docs/research/attention-v3 --source-root .
```

For a separate replication pass its output directory instead. It checks raw
record and array hashes, actual token histories, complete full-vocabulary arrays,
all literal tolerance decisions, control eligibility, NLL and timing arithmetic.
It cannot prove that timings came from a real device or that the author was
honest. Independent fresh execution and human inspection remain necessary.

Full arrays are large and stay outside Git; the researcher bundle must include
them for numerical verification. Summaries alone cannot substantiate numerical
claims. Original per-run source snapshots/logs are retained next to arrays.

During a live study, `benchmarks/attention_v3_progress.py` can verify a captured
terminal-case ledger without modifying the original. Run it outside generation
timing windows. It creates a read-only linked snapshot under artifacts and
publishes an explicitly partial summary; running and unattempted cases remain
unverified. Completing the study requires the full analysis command above.
After all outcomes and the report exist, `benchmarks/package_attention_v3.py`
verifies the full study and builds a local ZIP under lab-ui/downloads. This
performs no upload. The archive includes all full arrays, source snapshots,
negative outcomes, pinned setup and the independent standard-library integrity
checker. Verify an extracted kit before attempting fresh execution.

The [isolated-process follow-up](../attention-isolation/reproduce.md) is part of
the final handoff. It separately tests a known failure pair with one arm per
process. Once both studies finish, run `benchmarks/report_attention_v3.py` before
packaging. That report and its receipt bind both verified summaries. The
extracted kit includes the tracked project and tests, registered source, full
arrays and both independent verifiers; it excludes environments, weights and
original corpus text. Full papers open from their original source links.

## Interpretation and bias

The first six cases revisit overlapping WikiText inputs used in v2 and are
explicitly retrospective. The next twelve use six distinct PG-19 validation
books, selected by filename and minimum length before model outcomes. Each book
uses nested 32k/40k prefixes; contexts and tokens are correlated. Native,
persistent native SDPA and 256-token-GPU hybrid arm orders cover all six
permutations once per context in the book cohort. Native reference preparation
always comes first. Warmups are short, not a full long-context steady-state
regime. The machine, desktop session, model, FP32 precision and dated book corpus
limit generalization. No claim of zero bias, CXL, DSA, novelty, language-quality
equivalence, energy savings, exclusive VRAM residency or OOM rescue is implied.

Arms share one model/CUDA process. Device/host allocator history can influence
later measurements; `empty_cache()` does not prove all host/framework state was
reset. Balanced order helps expose position effects but does not replace per-arm
process isolation. In particular, a within-process OOM cannot establish that the
same arm would fail in a fresh process. Treat isolated-arm controls as a further
research target before generalizing resource-feasibility conclusions.

PG-19 source, dataset description and upstream license declaration:
https://github.com/google-deepmind/pg19 . The original public text bucket is
`deepmind-gutenberg`. No upstream research repository is executed.
