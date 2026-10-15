# Chronological research log

Times are observed UTC times unless stated otherwise; local date at session start was 2026-10-14 America/Chicago. The requested five-hour allocation is a prioritization guide. No five-hour claim is made.

## 2026-10-15 04:39 UTC — environment and prior art

- Read the complete attached assignment. Working directory is a ChatGPT project mirror, not a Git checkout. `sources/` contains no files. Existing root AGENTS.md is read-only and was preserved.
- GitHub connector repository search returned an access/search error. Authenticated GitHub CLI worked; no Atlas repository was present in the account listing.
- Initial literature searches found Checkmate, DTR, Rockmate, Alpa and EasyScale. Broad automatic-memory-optimization novelty is not credible; resource-consistency preservation also has established prior art.
- Created a separate local `atlas/` repository. User explicitly authorized creating a GitHub repository named `atlas`; created private `andrewkernel/atlas` successfully.

## 2026-10-15 04:40 UTC — available hardware and initial boundaries

- Observed Windows host, AMD Ryzen 5 7600X, 12 logical processors, 33,412,775,936 bytes system RAM; NVIDIA RTX 3060, 12,288 MiB VRAM, driver 595.95. Availability does not establish CUDA/PyTorch compatibility.
- Python 3.12.10 is installed; system default is Python 3.14.6. PyTorch was absent in both. Python 3.12 already has NumPy, SciPy and pytest available outside the proposed isolated environment.
- Ollama CLI reports no downloaded models. Its listing command started its local service automatically. No model was downloaded or inference benchmark performed.
- PyTorch live `stable` documentation redirected to 2.14; this does not determine the experimental package version. Exact versions must be recorded separately.
- Checkpoint API's default determinism check compares metadata, not values. This is material to the proposed verifier: passing that check cannot certify numerical or scientific fidelity.

## 2026-10-15 04:43 UTC — isolated environment and original probes

- Installed uv into a local virtual environment, then official PyTorch CPU wheel and the small research dependency set. Python 3.12.10 / Torch 2.14.1+cpu; no CUDA build or model weights installed. `uv.lock` resolved and synced successfully.
- At 04:43:58 UTC, original CPU probes found correct weighted accumulation within float64 rounding, denominator/BatchNorm counterexamples, dropout gradient drift without checkpoint RNG preservation, and BatchNorm-buffer drift despite identical gradients. Raw evidence and source hash saved.
- Initial eight tests passed, including 40 deterministic Hypothesis partition examples. These are bounded semantics controls, not full-training validation.
- Found TrainParity and TrainCheck as close prior art. Revised the contribution away from a generic correctness checker toward contract-conditioned plan selection and interactions. Added composed native+checker workflow as an essential baseline.

## 2026-10-15 04:44 UTC — primary-source snapshots

- Captured15 repository heads/archive/license metadata and seven model/dataset revisions. Alpa's repository is archived; current compatibility of historical memory systems remains unknown.
- Qwen2.5-Coder-3B has research terms distinct from the 7B model; excluded it as the default. WikiText's card has conflicting license metadata/prose; SST-2 license metadata is unknown. Recorded these as planned-evaluation prerequisites, not cleared rights.
- No research repository code, public dataset or model weights were executed/downloaded as a workload.

## 2026-10-15 04:49 UTC — second Git checkpoint

- Committed and pushed the prior-art assessment, initial fidelity framework, original probes, tests, environment lock and source metadata on `research/atlas-spec` (commit 4dcab28). Initial repository/source checkpoint is 4a3dcda on main.
- Accepted adapter-based, native finite planning and scoped evidence boundaries. No full Atlas application development started.

## 2026-10-15 04:57 UTC — synthesis and dependency metadata

- Completed the initial architecture/API/security design, benchmark/statistical plan, six research candidate rankings, three explicitly authored adversarial review perspectives and roadmap.
- Inspected primary package metadata for selected/optional components; recorded installed versus metadata-only status. CPU package compatibility does not validate CUDA/API/MLflow integration.
- Selected source inspection later confirmed TrainParity's accumulation phases and baseline/initial-state abstention logic at its pinned head. TrainCheck's pyproject declares Python>=3.10 and Apache-2.0 metadata. This is a targeted code read, not a code audit or replication.

## 2026-10-15 05:05 UTC — composed-update controls and consistency review

- At 05:05:10 UTC, a second original CPU experiment tested two AdamW logical updates under checkpointing plus3+4 accumulation. Correct composition differed in final parameters by 5.55e-17; clipping each partial gradient caused 0.033198 error; stepping each microbatch caused 0.024045 error and doubled update counters.
- Expanded suite: 11 tests passed, including the original 40 generated partition controls. Original evidence files were preserved; the composition experiment has its own code hash and result file.
- Initial offline audit found no missing required documents, broken local links, invalid JSON, stale probe hashes or unresolved placeholder markers. Final validation and repository status are recorded separately at completion.

## Limits and unfinished research

This paragraph records the **initial** investigation's limits. Later continuation sections and [updated recommendation](14-updated-recommendation.md) supersede its execution/read-depth inventory; the original chronology is preserved.

This is a bounded initial research/engineering investigation, not five hours of research. Read depth varies:24 individual paper notes include many abstract-level notes and selected deeper reads of the closest systems. There was no systematic forward-citation review, independent expert review/labeling, competing-system execution, native CUDA study, full-training equivalence study, LLM inference comparison or energy measurement. The scoped specification is complete as an initial blueprint; these evidence gaps remain explicitly open in `11-open-questions.md`.

## 2026-10-15 05:08 UTC — validation and handoff

- Complete specification checkpoint45477f4 was committed and pushed. GitHub Actions run38685042339 completed successfully on Linux and Windows; local11-test suite and offline specification audit passed.
- Tightened the primary question to an explicit safety/coverage comparison with native mechanisms plus existing checks under a fixed initial search budget. Added an expanded-access utility gate, while keeping full scientific-outcome claims separate.
- Final recorded validation clock05:08:45 UTC, about30 minutes after the first04:39:05 clock. No five-hour duration claim. Handoff publication preserves the research branch and fast-forwards main; exact final revision is in Git history.
- [Repository handoff](repository-status.md) records delivered files, evidence limits, CI link and immediate next coding task.

## 2026-10-15 05:10 UTC — published-main validation and CI maintenance

- Main and research branch were confirmed at9d5f687 with a clean local worktree. Main run38685249751 passed on both Linux and Windows.
- CI emitted an actual Node20-action deprecation warning and an upcoming ubuntu-latest image migration notice. Inspected current official release metadata/action manifests; pinned checkout v7.0.1 and setup-python v7.0.0 by commit, both using Node24. Selected ubuntu-24.04 explicitly. This is CI maintenance, not additional scientific evidence.

## 2026-10-15 05:14–05:21 UTC — continued research and independent audit routes

- Owner requested continued research and simultaneous double-checking. Reopened primary sources and switched to the existing research branch; no new task or delegation was created.
- Discovered HiLoRe's September2026 preprint through a search lead; inspected its primary method, conditional exact-recovery argument and evaluation appendix. Narrowed the general novelty assessment again. Anonymous code page supplied no inspectable browser content; no absence-of-code claim.
- Inspected pinned FairScale checkpoint functions. Specialized BatchNorm handling and recomputation guards predate Atlas, so the motivating hazard is not a new finding.
- Found and corrected the original equivalence sample-size approximation. Original integration plus200,000 simulated sufficient-statistic repetitions per condition gives36 pairs for80% power under the illustrative normal model, versus25 in the earlier formula. This does not estimate real ML variance.
- Ran original state/RNG controls: scoped buffer restoration repairs the known BatchNorm fixture; two dropout layers differ after partitioning even with equal final RNG states. Preserved the single-layer case that matches as a positive observation.
-17 local tests passed. All experiments remain synthetic CPU/statistical controls; no primary policy evaluation, CUDA, full training or external framework execution yet. See the amendment ledger for correction dispositions.

## 2026-10-15 05:22–05:28 UTC — native CUDA fixture

- Official CUDA13.0 wheel for PyTorch2.14.1 was verified available for Windows/Python3.12. NVIDIA's compatibility documentation supports driver-family580+ for CUDA13.x; observed host driver595.95. Installed the official binary and its dependencies in a separate ignored environment, without changing the CPU lock or driver. Native CUDA tensor smoke passed.
- Original reviewed residual-MLP fixture completed eight fresh-process trials across microbatch16/4 and checkpoint off/on, two repetitions each. Both transformations lower measured allocation and increase median update time here. Combined allocated peak378.45 MiB versus972.58 MiB reference; no physical-device OOM occurred.
- First-update peaks understate later peaks because persistent Adam state is present in subsequent forwards. Added explicit distinction to the resource evidence.
- Audited the exploratory metric aggregation: integer/step tensors can dominate a combined optimizer norm. Separated Adam moment/step comparisons and added unique trial directories, then repeated the published measurement at05:24:49 UTC. Initial pilot is retained locally and excluded from the published two repetitions.
- First-update parameters, gradients and each Adam state family were compared; smaller microbatches introduce small FP32 differences. No full-training outcome or real-workload rescue claim. Requirements are separately pinned with hashes; raw JSON carries source identity and environment.

## 2026-10-15 05:29–05:37 UTC — constrained resource controls and study-design audit

- A450 MiB PyTorch allocator cap on the same12 GB board caused three plan OOMs; only the combined checkpoint/microbatch plan completed16 fixture updates. Added completed-step instrumentation and repeated the published capped run: checkpoint-only and accumulation-only each complete one update before later failure. This remains an explicitly emulated budget, not another physical card or a real-workload rescue.
- OOM diagnostics printed impossible per-process memory values. Retained local exploratory messages, published non-process excerpts plus invalid-diagnostic flags, and explicitly rejected those strings as authoritative telemetry. The sentinel interpretation is an inference, not a diagnosed driver root cause.
- Added original unused-gradient and scaler-skip CPU controls. Identical observed loss or successful-update count can coexist with differing parameters, scheduler and EMA state.19 local tests passed; these are known API mechanisms, not new framework bugs.
- Refined H1's primary unit/denominators, indeterminate-label bounds and null selective risk for zero acceptance. Gave the composed baseline equal contract/adapter information and required a shared compatible stack for clean runtime comparisons.
- Distinguished emergent interactions from known defects injected into a composed execution. Existing probes establish the latter, not yet the former; narrowed the novelty gate accordingly.
- Checked primary historical issues/fix metadata and additional Adacc, PRAC, TorchProbe and TensorFuzz abstracts. No third-party research code or dataset was executed. Checkpoint2eefc7a's cross-platform run38687066842 passed; new changes require their own validation.

## 2026-10-15 05:38–05:47 UTC — model-capacity, inference assumptions and pilot freeze

- Captured current Qwen3.5-4B/Qwen3-4B/coder7B first-party model/config metadata and the Ollama4B manifest/config without model/projector downloads. Added component/cache/context arithmetic; no model inference or best-model claim.
- Derived the measured four-plan Pareto tradeoff and workflow-cost limits from raw CUDA records. Summed eight child walls39.22 seconds versus6.86 seconds in80 timed update windows; no production planner-cost claim.
- Extended statistical calibration with SD/nonzero-effect sensitivity and a constructed rare-mixture counterexample. Equal population mean/variance do not ensure valid small-sample t-interval inference. The mixture's1% event probability is invented, not estimated from training.20 local tests passed.
- Cross-platform run38687801756 passed checkpointa965bed. Research-only original Fashion-MNIST preparation/training code and a fixed exploratory protocol are now prepared for commit **before** data acquisition/training. Three pilot seeds, five epochs, paired full versus combined native plans; canonical test partition remains untouched. This is not confirmatory equivalence or an application implementation.

## 2026-10-15 05:48–06:00 UTC — frozen real-data pilot and deeper comparator audit

- Committed/pushed protocol and unchanged original training/preparation code in c406bef before data acquisition. Downloaded26,451,395 bytes of official Fashion-MNIST training gzip files from a pinned repository revision; retained MIT notice and compressed/expanded hashes. Canonical test partition was not downloaded.
- At05:48:30 UTC began six randomized-order fresh-process runs: three paired pilot seeds, five epochs each,1955 updates and250,000 sample exposures per run. All completed with matching paired initialization/split/order digests and empty stderr. Combined native plan peaks112.23 versus166.79 MiB allocated; mean update windows14.24 versus8.56 seconds. Paired final validation differences range-.21 to+.05 percentage points. No equivalence test or scientific margin was selected; no physical OOM rescue occurred.
- Deeper pinned TrainParity worker/snapshot/test inspection confirms buffers, keyed/None gradients, optimizer/scheduler/scaler state and multiple accumulation fault controls. No third-party research program ran. This narrows the generic-verifier distinction further.
- Read selected Rockmate graph/alias and implementation/numerical appendix sections, plus pinned current README. Added HiRemate ICML2025 and OffMate preprint context; HAL returned anti-bot content, so OffMate's limited summary uses the first-party team report/README, not inaccessible paper text.
- Published an updated recommendation: continue only to a bounded independently adjudicated strong-baseline comparison. Current evidence improves engineering feasibility, not established novelty or primary-policy efficacy.20 local tests and the offline identity/link audit pass; c406bef's cross-platform run38688775079 passed before the new result checkpoint.
