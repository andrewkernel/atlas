# Runtime autotuning acceptance gates

Owner objective: research and improve the idea, then build a real end-to-end
automatic runtime tuning workflow. This plan records requirements before new
autotuning outcomes. Completion requires evidence for each gate, including
negative decisions; it does not require manufacturing a winning configuration.

1. Primary-source design review and current-code audit precede implementation.
   Preserve frozen v2/v3/isolated execution identities and previous evidence.
2. A new finite registry changes CPU threads, native/builtin-offload/custom cache
   execution and GPU cache window while holding model/workload/precision fixed.
   Include competent native and builtin Transformers offloading comparators.
3. Freeze protocols, input hashes, search/confirmation rules and source identities
   before outcomes. Two disjoint train windows select; two separate validation
   windows confirm. Persist nomination before confirmation measurements.
4. Real full-model workers run in fresh owned processes, with target-shape warmup,
   complete synchronized generation timing and separate 32-position full-vocabulary
   numerical validation. Record startup, warmup and all resource/failure outcomes.
5. Require native-repeat and resident controls. Failed, missing, cancelled,
   timed-out or corrupt evidence cannot qualify or become an invented run.
6. Speed activation requires every predeclared paired confirmation ratio to clear
   1.05 and all bounds/controls. Otherwise retain current or report inconclusive.
   A budget fit is labeled separately and may legitimately be slower.
7. GPU allocation budgets cover measured warmup/generation peaks, not a hard
   physical-memory guarantee. Never substitute shorter context/output or a lower
   precision to claim an optimization of the fixed workload.
8. Independent NumPy verification reconstructs arrays, decisions and selection,
   rejects lying pass flags, bad nominations, incomplete cases and tampered bytes.
9. Save immutable source-bound profiles, apply only compatible domains, and retain
   authoritative transactional state plus apply/revert history. Concurrent state
   changes, accepted cancellation, drift or invalid evidence prevent stale commits.
10. A real model request uses the active profile or an explicitly qualified fallback.
    Preserve inputs. A memory constraint cannot silently fall back above its budget.
    Distinguish prior profile validation from a fresh check on the request.
11. The actual browser supports setup/start/progress/cancel, inspection/export,
    measured decision explanations, apply/revert and actual model generation.
    Keep old app functionality, origin/token protections and one workspace GPU job.
12. Verify actual cancellation/owned process exit and resource/time boundaries.
    Startup recovery never infers process liveness from a stale status file alone.
13. Run real FP32 2k fastest and 2560 MiB budget workflows, plus a registered 32k
    allocation-budget campaign at 6144 MiB. Retain every outcome. Test additional
    precisions/lengths only as declared separate cases, never as fallback claims.
14. Exercise actual saved-profile application, drift/failure fallback or abstention,
    rollback, and rejection of invalid/incomplete activation. Authored fixtures
    cover unsafe/rare cases without being published as model evidence.
15. Show total tuning cost and scoped break-even arithmetic. Report warm generation
    and actual cold request/verification cost separately. No unsupported speedup.
16. Provide reproduction commands, pinned setup, full public-input outputs and a
    standalone verification/bundle check. Exclude private requests and environments.
17. Required repository checks, meaningful controls and actual desktop/narrow-screen
    browser verification pass. Complete an evidence-linked audit before claiming
    completion; retain unresolved generalization/platform/novelty limits.

The allocation budgets in gate 13 were chosen before new outcomes using prior
2k native/hybrid peaks and the known 32k memory tradeoff. They are operational
test objectives, not independent corroboration of the old memory claims.
