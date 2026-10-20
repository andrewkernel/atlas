# Runtime tuning acceptance audit

This audit checks all 17 original
[acceptance gates](../../roadmap/autotuning-acceptance.md) against the recorded
model, control, browser and extracted-tool evidence below. Passing authored
controls is not a substitute for those actual outcomes. Final archive byte/SHA
receipts are stored outside the archives to avoid a self-referential digest.
The resource-recovery amendment changes execution identity. Historical cases
remain explicitly identified alongside amended-source results. All three
amended-source required campaigns have completed with independently checked decisions.

| Gate | Current evidence | Assessment |
|---|---|---|
| 1. Research and preservation | [Pre-implementation review](research-and-design.md), architecture/acceptance commits before new results; [26 previous execution files match](preserved-sources.json). | Verified for the stated source-preservation and review scope. No new CXL implementation or novelty claim. |
| 2. Competent finite grid | [Registry](../../../src/atlas/tuning/registry.py) and both 2k summaries contain native threads 1/2/4/6, resident control, built-in Transformers offloading, hybrid threads and two GPU windows. | Verified. This is a finite grid, not a global optimum. |
| 3. Separate selection and confirmation | Campaign protocols retain two train and two separate validation windows with offsets/token hashes. The [controller](../../../src/atlas/tuning/controller.py) freezes nomination before collecting held-out references/pairs; the verifier reconstructs order and nomination. Both 2k campaigns confirmed; 32k had no eligible nominee, so its declared conditional workflow did not collect confirmation. | Verified within that scope. No 32k held-out result is claimed. Windows are fixed, reused and nested across contexts. |
| 4. Real full-model execution | Each 2k campaign has 42 distinct owned workers: four unscored native references and 38 scored candidates/controls. The 32k campaign has two references and all 22 registered calibration attempts, including two RAM stops and five literal failures. Complete scored workers retain full-shape warmup, synchronized generation and separate 32-position full-vocabulary replay. | Verified. Interrupted attempts remain unknown, not successful numerical executions. |
| 5. Controls gate eligibility | Native-repeat/resident checks are required on calibration and confirmation inputs. [Verifier tests](../../../tests/test_runtime_tuning.py) reject missing/failed controls, unknown outcomes, changed arrays, false flags and incomplete cases. | Verified in completed cases and authored failure controls; no failed outcome is relabeled as a pass. |
| 6. Confirmation decision | [2k speed](fastest-2k.json) kept native because not every ratio exceeded 1.05. [2k target](budget-2k.json) confirmed a slower memory fit. Threshold-edge and failed-candidate controls exercise rejection. | Verified. No positive speed improvement is claimed. |
| 7. Fixed workload and honest target | Protocols fix Qwen3-0.6B, FP32, exact context and 16 generated tokens. Eligibility uses maximum observed warmup/generation CUDA allocated peak. [Runtime guide](../../runtime-tuning.md) states that checking/search can exceed the target. | Verified for completed campaigns. This is not a physical VRAM quota or OOM-rescue claim. |
| 8. Independent arithmetic | [NumPy verifier](../../../src/atlas/tuning/verify.py) reconstructs full-vector predicates, NLL, nomination and decision, and checks timing/count/allocation consistency without importing PyTorch. Altered evidence and nomination controls pass. | Verified for completed campaigns. Offline arithmetic cannot recreate hardware timings or authenticate an author's honesty. |
| 9. Transactional profiles | [Profile implementation](../../../src/atlas/tuning/profiles.py) binds source/assets/environment/objective/domain, uses immutable JSON and SQLite revision/CAS events, and serializes accepted cancellation versus commit. [Actual stale activation](api-activation-checks.json) was refused. | Verified by actual apply/revert and authored drift/race controls. |
| 10. Actual application and fallback | [Application observations](application-checks.json) include actual active native/offloaded generation, native fallback after Revert, exact-length refusal, and constrained abstention at 2k and 32k. Fresh checks passed on generated 2k examples; the 32k request abstained before loading/generation and has unknown numerical relation. | Verified within that scope. Private raw request arrays are excluded; this public receipt is not standalone proof of private outputs. |
| 11. Browser and API workflow | Actual browser setup/start/progress/cancel, decision inspection, apply/revert, generation and a byte-matched full-evidence download were exercised. [Admission controls](api-admission-checks.json) refuse conflicts and missing/foreign authorization. Seven [UI controls](../../../tests/runtime_tuning_ui.test.mjs) complement [actual desktop/narrow/extracted-tool checks](browser-qa.json). | Verified for the stated local workflow. |
| 12. Owned lifecycle and bounds | [Actual current-source lifecycle](lifecycle-checks.json) confirms deadline/cancellation, closed scopes, no matching live owned processes and unchanged active state. [Actual 32k RAM recovery](32k-resource-recovery.json) confirms an ineligible stopped trial, closed scope, sampled recovery above 4 GiB and collection continuing to the next registered case. Authored disk/RAM controls cover preflight and failure branches. | Verified for lifecycle/control scope. RAM observations are advisory; no hard-memory isolation or hostile-code sandbox is established. |
| 13. Required real campaigns | Amended-source `tune-0ac3414ab599` and `tune-523ff98ee883` each completed 42 workers and independent verification: keep-current and slower memory-fit respectively. Exact 32k / 6,144 MiB `tune-eeb99dbc1f00` completed all 24 required attempts and independently verified no-eligible-profile. Earlier full 2k cases and interrupted `tune-53a712409265` are preserved. | Verified. All contexts, outputs, precision, finite registry and original numerical bounds were retained. |
| 14. Apply, rollback and invalid rejection | Actual profile generation, browser Revert/reapply, explicit fallback, exact-length refusal, 2k and 32k constrained abstention; [actual invalid/stale refusals](api-activation-checks.json); authored source/asset-drift, corrupt/incomplete evidence and failure controls. | Verified within the stated actual-versus-authored scope. No qualified 32k profile exists to apply. |
| 15. Cost and break-even | All three summaries retain scoped controller and parent-owned campaign costs, warmed generation and no positive speed break-even. Actual requests separately retain load, warmup, numerical-check, worker, service and browser intervals where recorded. | Verified. Warm generation is not promoted to total-application speed, and missing historical clocks remain unknown. |
| 16. Reproduction and private exclusion | [Pinned setup and reproduction](reproduce.md), three full public-input archives, streamed payload/inventory checks and unmodified shipped verification from separate extracted roots. [Replay receipts](browser-qa.json) reproduce each evidence digest/decision. Extracted source runs as the usable local tool with imported labels, missing assets, no transferred activation and actual abstention. Export privacy/profile/concurrent-edit controls pass. | Verified for same-machine byte/array/decision consistency. Exact array hardlinks followed full ZIP/source-byte checks; no fresh physical-copy, fresh environment-install or external model replication claim. Private requests, weights, environments and activation state are excluded. |
| 17. Final checks and rendering | [Current checks](checks.json) pass: 154 CPU tests (16 optional-integration skips), 133 pinned attention/tuning tests, seven Node UI controls and syntax/docs/dashboard-size checks. [Actual desktop and 390px results](browser-qa.json) show no page overflow, contained table scrolling and correct decision/profile/export association. The extracted tool was inspected in the browser. | Verified within that scope, with the final archive receipt and clean-worktree check required at handoff. |

## Handoff verification

All required amended-source campaigns and the larger-workload abstention have
completed. Preserve their frozen sources, full outputs and negative outcomes,
including the interrupted 32k attempt and both earlier complete 2k cases.

The final hash-named ZIPs and full SHA receipts live beside the canonical exports
and in the repository's separate bundle receipt. They retain the exact scientific
sources and full declared arrays. The byte, array and decision checks must match
those final files; final Git status must be clean. Runtime-generated state in an
extracted preview is outside the shipped inventory. Private requests, activation
state, environments and weights remain outside all public archives.

One machine/model family, finite public windows, shared desktop scheduling,
fixed precision and lack of external replication remain limits after completion.
The fixed public windows recur across objectives/source amendments; fresh
processes do not make them independent datasets or newly unseen confirmation.
Hashes establish consistency; numerical checks cover finite stated relations.
Neither supplies a zero-bias, universal-correctness or universal-performance
guarantee.
