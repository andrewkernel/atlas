# Runtime tuning acceptance audit

Completion is not yet proven. This working audit keeps all 17 original
[acceptance gates](../../roadmap/autotuning-acceptance.md) in scope. The full 32k
campaign, final current-source bundles and final rendered handoff remain pending.
Passing authored controls is not a substitute for those model/browser outcomes.
The resource-recovery amendment changes execution identity. Historical cases
remain explicitly identified alongside amended-source results. Both amended 2k
campaigns completed; the larger required workflow still needs current evidence.

| Gate | Current evidence | Assessment |
|---|---|---|
| 1. Research and preservation | [Pre-implementation review](research-and-design.md), architecture/acceptance commits before new results; [26 previous execution files match](preserved-sources.json). | Verified for the stated source-preservation and review scope. No new CXL implementation or novelty claim. |
| 2. Competent finite grid | [Registry](../../../src/atlas/tuning/registry.py) and both 2k summaries contain native threads 1/2/4/6, resident control, built-in Transformers offloading, hybrid threads and two GPU windows. | Verified. This is a finite grid, not a global optimum. |
| 3. Separate selection and confirmation | Campaign protocols retain two train and two separate validation windows with offsets/token hashes. The [controller](../../../src/atlas/tuning/controller.py) freezes nomination before collecting held-out references/pairs; the verifier reconstructs order and nomination. | Verified in completed 2k campaigns. Windows are fixed, same-session and nested across context lengths. |
| 4. Real full-model execution | Each completed 2k campaign has 42 distinct owned workers: four unscored native references and 38 scored candidates/controls. Scored workers run full-shape warmup, synchronized full generation and separate 32-position full-vocabulary replay. | Verified for both completed 2k campaigns. The larger campaign remains required. |
| 5. Controls gate eligibility | Native-repeat/resident checks are required on calibration and confirmation inputs. [Verifier tests](../../../tests/test_runtime_tuning.py) reject missing/failed controls, unknown outcomes, changed arrays, false flags and incomplete cases. | Verified in completed cases and authored failure controls; no failed outcome is relabeled as a pass. |
| 6. Confirmation decision | [2k speed](fastest-2k.json) kept native because not every ratio exceeded 1.05. [2k target](budget-2k.json) confirmed a slower memory fit. Threshold-edge and failed-candidate controls exercise rejection. | Verified. No positive speed improvement is claimed. |
| 7. Fixed workload and honest target | Protocols fix Qwen3-0.6B, FP32, exact context and 16 generated tokens. Eligibility uses maximum observed warmup/generation CUDA allocated peak. [Runtime guide](../../runtime-tuning.md) states that checking/search can exceed the target. | Verified for completed campaigns. This is not a physical VRAM quota or OOM-rescue claim. |
| 8. Independent arithmetic | [NumPy verifier](../../../src/atlas/tuning/verify.py) reconstructs full-vector predicates, NLL, nomination and decision, and checks timing/count/allocation consistency without importing PyTorch. Altered evidence and nomination controls pass. | Verified for completed campaigns. Offline arithmetic cannot recreate hardware timings or authenticate an author's honesty. |
| 9. Transactional profiles | [Profile implementation](../../../src/atlas/tuning/profiles.py) binds source/assets/environment/objective/domain, uses immutable JSON and SQLite revision/CAS events, and serializes accepted cancellation versus commit. [Actual stale activation](api-activation-checks.json) was refused. | Verified by actual apply/revert and authored drift/race controls. |
| 10. Actual application and fallback | [Application observations](application-checks.json) include actual active native/offloaded generation, explicit native fallback after Revert, and constrained abstention without a qualified active profile. Fresh checks passed on the generated examples. | Verified for 2k application. Private raw request arrays are excluded; this public receipt is not standalone proof of those private outputs. |
| 11. Browser and API workflow | Actual browser setup/start/progress/cancel, decision inspection, apply/revert and generation were exercised. [Admission controls](api-admission-checks.json) refuse conflicting jobs and missing/foreign authorization. Seven deterministic [UI controls](../../../tests/runtime_tuning_ui.test.mjs) cover selected-record/export association, imported labels and final clocks after cleanup. Fresh speed-profile generation/Revert/fallback passed in the browser. | Functional paths verified. Final current-source export and rendered handoff remain pending. |
| 12. Owned lifecycle and bounds | [Actual current-source lifecycle](lifecycle-checks.json) confirms deadline/cancellation, closed scopes, no matching live owned processes and unchanged active state. Authored disk/RAM tests refuse preflight or stop the owned attempt; stale status does not establish liveness. | Verified for lifecycle/control scope. RAM observations are advisory; no hard-memory isolation or hostile-code sandbox is established. |
| 13. Required real campaigns | Amended-source `tune-0ac3414ab599` and `tune-523ff98ee883` each completed 42 workers and independent verification: keep-current and slower memory-fit respectively. Registered `tune-eeb99dbc1f00` is the exact 32k / 6,144 MiB workflow; its terminal decision remains. Earlier full 2k cases and interrupted `tune-53a712409265` are preserved. | Pending until the amended-source 32k registered workflow and independent decision finish. No smaller substitute is accepted. |
| 14. Apply, rollback and invalid rejection | Actual profile generation, browser Revert/reapply and constrained abstention; [actual invalid/stale refusals](api-activation-checks.json); authored source/asset-drift, corrupt/incomplete evidence and failure controls. | Verified within that explicit actual-versus-authored scope. Larger-workload application will be exercised if it qualifies. |
| 15. Cost and break-even | Both summaries retain entire tuning cost, warmed generation and no positive speed break-even. Actual requests separately retain load, warmup, numerical-check and total-request cost. | Verified for completed campaigns. No warmed-generation result is promoted to a total-application speed claim. |
| 16. Reproduction and private exclusion | Pinned setup and CLI commands are in the runtime guide. The earlier development ZIP was byte-matched and independently verified from an extracted copy. Current exporter now enumerates declared public evidence and verifies the written payload; authored private-file/profile and concurrent-edit controls pass. | Pending: create and independently verify the final current-source packages with full outputs and the usable local tool. |
| 17. Final checks and rendering | [Current checks](checks.json) pass: 154 CPU tests (15 optional-integration skips), 132 pinned attention/tuning tests, seven Node UI controls, syntax/docs/dashboard-size checks. Actual narrow setup and desktop switching were inspected. | Pending: final desktop/390px result and extracted-tool inspection, clean Git and evidence-linked completion decision. |

## Required remaining work

Run the amended-source required campaigns without changing their frozen execution
sources. Preserve the interrupted 32k attempt and both earlier complete 2k cases.
Record actual numerical/control/resource outcomes and independently verified
decisions, including negative or inconclusive results. Exercise the
larger-workload profile if qualified, or report actual constrained abstention.

Build the final public-input campaign packages after measurement, verify every
shipped byte and the reconstructed array/decision relation from an extracted
copy, and inspect the supplied local tool in the actual browser. Keep private
requests, activation state, environments and weights outside that inventory.
Then finish desktop/narrow inspection, required source/document/tests and a
clean-worktree audit. Only those final observations can close the pending gates.

One machine/model family, finite public windows, shared desktop scheduling,
fixed precision and lack of external replication remain limits after completion.
The fixed public windows recur across objectives/source amendments; fresh
processes do not make them independent datasets or newly unseen confirmation.
Hashes establish consistency; numerical checks cover finite stated relations.
Neither supplies a zero-bias, universal-correctness or universal-performance
guarantee.
