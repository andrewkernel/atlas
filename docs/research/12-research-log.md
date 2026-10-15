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

## 2026-10-15 06:01 UTC — verification liveness control

- Original analogue compares streamed microbatch backward with constructing all microbatch losses before backward. No external checker runs. Both completed unconstrained arms match each other's first-update observations relative to the same local reference; allocated peaks are506.51 versus922.59 MiB.
- Under an artificial600 MiB allocator cap on the12 GB board, streamed execution completes8 fixture updates while deferred execution OOMs before the first optimizer update. This motivates measuring checker-resource behavior and giving the baseline a fair adaptation; it is not a TrainParity performance result or new accumulation algorithm.
- Added a reviewable integration note covering denominator assumptions, state/RNG effects of extra forwards, reference unavailability and reuse of existing checks. No upstream communication was sent. Checkpoint79d7eeb's cross-platform run38689646999 passed.

## 2026-10-15 06:03–06:13 UTC — repeatability, citation and report audit

- Replayed the lowest frozen pilot seed1101 once per arm in fresh processes, without changing code/data/configuration. Both final state digests and every recorded epoch validation metric exactly match each original arm. These are two replay controls, not two additional independent seeds.
- Checked the first primary URL for32 paper notes:27 HTML titles, two PDF headers and three final direct-retrieval failures. Preserved403/network failures without calling sources dead. Captured selected publisher-deposited DOI metadata, including title/subtitle and the Capuchin first-author discrepancy between the Microsoft overview and publisher record; retained publisher-supported Xuan Peng. Automatic HTML author decoding can contain replacement characters and was not used to override verified names.
- Added conditional reliability/selection/stopping calculations and abstract-level confidence-sequence context. No population field-risk or new sequential inference result is claimed. Added a compact claim-to-evidence map and clarified the strong-baseline gate across the proposal/roadmap.
- Cleaned prose spacing without changing raw results, experiment code, paper-note identities(except the explicitly amended Capuchin source note), or the frozen pilot protocol. Compiled reviewed Python files;20 CPU tests passed in2.97 seconds; audit checked102 Markdown files and384 relative links with no failures. No matching owned research fixture process remained. Exact source/lock digests are in the continuation validation capture.
- Main was fast-forwarded to validated checkpoint79d7eeb, and main run38689991566 passed on Linux/Windows. Later audit controls are saved in the following checkpoint; this chronology does not turn roughly95 minutes of total logged work into five hours.

## 2026-10-15 06:15 UTC — derived arithmetic and renewed allowance

- Standard-library derivation independently recalculated the raw CNN figures:32.7137% lower allocated memory,66.4385% higher synchronized update time and the three paired validation differences. Canonical JSON input hashes bind this arithmetic to raw records without newline sensitivity. This is not new independent experimental replication.
- Checkpoint6618202 passed cross-platform run38691236082. The account usage tool changed from90% consumed in its prior five-hour readout to2% with a new reset timestamp(and a refreshed weekly readout). No reset/credit action was requested by the agent; continued work uses the currently reported allowance rather than claiming exhaustion.

## 2026-10-15 06:18–06:30 UTC — reviewed direct-checker execution

- Downloaded pinned TrainParity source as text, inspected import-time statements and its accumulation execution path, and retained source identities/license. No build hooks, downloaded fixture cases, resume commands or external checkpoints were executed. Ran only original allowlisted CPU cases with an explicit minimal child environment; this is not validated hostile-code isolation.
- Prepared a separate official Python 3.11.17/PyTorch 2.13.0+cpu environment within the checker's declared support. NumPy 2.5.4 is incompatible with Python 3.11; selected 2.4.6 and captured exact installed dependency pins/hashes. The Atlas core lock is unchanged.
- Froze protocol/source/config at c5a0c93. The first attempt supplied instances instead of required class import targets: all 11 setup errors are retained, with no accumulation workers launched. Amendment df967a4 corrected import targets before rerunning unchanged control meanings/parameters.
- The second attempt produced three PASS, six FAIL, one ABSTAIN and one ERROR as predicted; 28 owned worker launches, 120.18 seconds summed check-call wall time and 143.71 seconds outer wall time. Retained the scheduler warning. These are investigator controls, not independent detection estimates, GPU/checkpoint-selection performance or a full Atlas policy study. Generic-checker overlap now has direct execution evidence.
- Checkpoint e2a4fba passed Linux/Windows on both main and research branches (runs 38691514171 and 38691511629).

## 2026-10-15 06:33–06:57 UTC — existing memory-planner execution and causal audit

- Captured pinned Rockmate/rkgb text and performed a scoped import/constructor/generated-code/process/checkpoint review. Built a separate official Python 3.12.10/Torch 2.14.1+cu130 environment with declared dependencies and explicit PyYAML; no source build hook/C extension, external test/model/checkpoint or untrusted workload ran.
- Froze every attempted protocol/source amendment before execution. Retained direct-factory failure, three framework precision failures across the early arms, Dynamo assignment-count failures, JIT parameter-representation failures and no-schedule results at 16 MiB. A four-mode native PyTorch export control reproduces precision failures without any planner import, preventing an incorrect algorithmic attribution.
- A narrow authored forward(x) wrapper removes an exported input-name alias while preserving parameter bytes after prefix removal and identical input/target bytes. No upstream source patch was applied. Both generated Rotor/Rockmate configurations then complete eight updates at 64/128 MiB budgets, with zero measured first gradient/parameter/momentum and final-parameter differences.
- A separately frozen diagnostic finds no repeated forward operations in the 64 MiB Rotor schedule. A subsequent frozen 32/48 MiB exploration records seven/two planned repeated forwards and successful eight-update numerical controls. Reported schedule peaks exceed their nominal 32/48 MiB values by 512 bytes; cause not isolated, no exact-cap claim.
- Whole allocated peaks in these diagnostic workers exceed their native reference, with differing resident-memory/startup windows. A reset-counter observation records the starting allocation difference. No fair isolated performance, expanded-access, full-training or paper-suite replication conclusion is drawn. The external planner is now executable under this adapted narrow interface; novelty remains the untested selection comparison.

## 2026-10-15 07:00–07:04 UTC — preparation-state noninterference control

- Froze an original constructor observer and protocol at e6e491b, then reran one original pure 64 MiB Rotor case with a scoped Torch RNG context. CUDA global RNG changes inside construction; Torch CPU, Python/NumPy RNG, parameters, gradients and mode do not change, and the model has no buffers. The existing context restores the observed CUDA state. Eight-update recorded numerical errors remain zero.
- Updated the proposed execution lifecycle to verify/replay the scientific start state after planning/profiling; process freshness and pure update parity do not establish that relation. This is a known state-management obligation with a finite existing-tool example, not a new restoration algorithm, emergent-transform bug or scientific conclusion.
- Checkpoint 6ee7e48 passed cross-platform run 38695734735.

## 2026-10-15 07:05–07:22 UTC — deeper paper audit and reusable-execution controls

- Downloaded the public HiRemate proceedings PDF as a read-only local reference, recorded its byte hash and selected read scope, and visually checked method/evaluation pages 7, 17, 19 and 26 using the PDF skill. Selected methods and Appendices B/C/F clarify activation accounting, iteration-only evaluation, original stack and reusable-module claims. Verified a matching historical PDF URL after a newer file-history commit returned 404; no full-paper or historical-suite replication is claimed.
- Froze five module-reuse/accumulation controls at e0d8f00. The first two streaming positives pass; a deferred failure is initially masked by an investigator error reporter that requires the home directory excluded by the minimal environment. Preserved the partial/unstarted denominator and froze the reporting-only repair at 95b5cfe before rerun.
- The corrected run has native/planner streaming and one-forward positives with zero measured gradient/parameter/momentum errors; both deferred patterns raise an invalid-gradient shape error. Output-retention follow-up at 6f21091 does not repair them. A larger-budget control frozen at c9e2c6f records zero planned repeated forwards and the same errors, weakening a new rematerialization-interaction explanation. No silent scientific corruption or actual combined TrainParity/Rockmate execution is observed.
- Proposed plan/verification metadata now includes forward/backward scheduling relations. A competent baseline must preserve a supported target execution rather than reject it based only on a different verification executor. Current-code failure is not a refutation of the paper's historical stack, and no upstream communication was sent.

## 2026-10-15 07:27–07:34 UTC — actual compatible-stack baseline integration

- Prepared a separate Python 3.11.17/Torch 2.13.0+cu130/NumPy 2.4.6 environment within both inspected version ranges. Confirmed the official CUDA wheel link/hash before installation and captured all 26 installed dependency pins. Core and historical experiment environments remain unchanged.
- Froze cases/protocol/environment at 23fab8f before importing both reviewed source trees together. Native TrainParity GPU accumulation reports PASS; the conditional planner case reports ERROR with the deferred shape failure. The false initial-verification flag on the error report is an early-return default, not observed initial inequality.
- An authored ordinary streaming executor using the existing lower-level snapshot capture/comparison functions passes initialized and post-update state comparisons, including explicit denominator/loss. A .01 parameter perturbation is detected with FAIL. RNG is deliberately excluded from those pure post-state comparisons. This makes the competent reusable baseline stronger; it is not a new Atlas checker or full scientific verifier.
- Recorded timings have different control work/process counts and do not support a like-for-like speedup. No independently labeled selection study or energy result exists. Checkpoint fb529db passed Linux/Windows run 38698781501.

## 2026-10-15 07:47 UTC — comparison-policy/health boundary audit

- Froze ten deterministic CPU scalar/float64 controls at dcc1818. Existing tensor tolerance uses the candidate argument as relative scale; scalar math.isclose uses a symmetric maximum and combines absolute/relative bounds by max. Constructed cases disagree with the proposed Atlas baseline-additive predicate, including a near-boundary 1e-5 example. No practical prevalence is inferred.
- Identical infinity and selected NaN policies can pass equality while a separate finite-health predicate fails. The default NaN failure's infinite diagnostic summaries are represented as null with explicit field paths. Existing conventions are not labeled upstream bugs; health/configuration checks and exact comparison identity remain caller obligations. Earlier results retain their actual implemented policy.
- Checkpoint bb30bcb passed cross-platform run 38700374277.

## 2026-10-15 07:51–08:05 UTC — energy replay and independent instrument audit

- Read-only NVML availability supports cumulative energy and power on the physical RTX 3060. Inspected wrappers and exact binding/requirements are retained in a separate minimal environment; no settings or driver changes.
- Froze three same-seed randomized-order replay pairs at 8fbab26 before measuring unchanged five-epoch CNN workers. All six complete with identical same-arm final/initial/data/order digests and five-epoch validation/update values. They are measurement repetitions, not additional independent seeds.
- Independently reconstructed energy differences and power integrals using exact integer/rational units. The analytic integration control is 200 J. Gross native/candidate mean counters are 1205.084/1590.857 J over complete-workflow padded windows, not process or whole-host energy.
- Reported-power integration exceeds counter deltas by 23.61–25.00%. A post-observation passive diagnostic frozen at b15c90a also shows 5.726% disagreement over 19.991 seconds. No favored accuracy reference, corrected result or cause is established. Retained all raw traces and strengthened measurement-scope requirements.
- Current summary/security/dependency prose was reconciled with later external execution and energy evidence; historical logs/protocols remain dated observations. Checkpoint 0966144 passed Linux/Windows run 38701101879.

## 2026-10-15 08:06–08:08 UTC — conditional-check prior art and contract synthesis

- Retrieved the official 18-page TrainCheck PDF with byte identity after the earlier landing-page failure. Selected methods/evaluation reading and visual inspection of pages 13/14 clarify precondition inference, conditioned evaluation populations, analyst review and online versus offline cost. No TrainCheck code or paper suite was run. OffMate public metadata access still fails certificate validation; no bypass or full-paper read.
- Consolidated the observed comparison/state/execution/resource boundaries into a clearly labeled proposed JSON illustration and decision walkthrough. It is specification, not an application or completed policy benchmark. Existing conditional checks further weaken a novelty claim based on contracts alone; the primary comparison gate remains.
