# Atlas researcher handoff and long-context validation

Owner objective: real, inspectable experiments that another researcher can rerun,
delivered as a useful local research product. Zero bias cannot be established.
This plan reduces named biases and records unresolved limits; it does not replace
the objective with a test count or a polished dashboard.

## Acceptance gates

1. Preserve all v2 protocols, source identities, failures and unknowns. New execution
   uses a separately frozen v3 protocol. Never retrospectively call v2 pressure
   rows numerically validated by changing their records.
2. Compare full-vocabulary model logits on identical 32k/40k histories, with the
   original FP32 elementwise bounds. Include an unchanged native repeat and a
   persistent native-SDPA cache control. Save the actual logit arrays so a separate
   verifier can recompute every check. Numerical failures remain failures.
3. Replay the three original pressure inputs as a retrospective cohort. Add six
   distinct PG-19 validation books, chosen only by a fixed lexicographic/length
   rule before model outcomes. Test both contexts on each book. Balance all six
   orders of native/resident/hybrid in the book cohort. Nested contexts and the
   old WikiText overlap are explicit; tokens are not independent observations.
4. Separate free-generation performance from teacher-forced numerical validation.
   Warm all arms on the same short input. Include cache management, transfers and
   selection in generation timing. Retain reference failures, OOMs, cancellations,
   timeouts and missing cases. No successful-only speedup summary.
5. Bind protocol, code, model/data bytes, token histories, environment and output
   artifacts. Canonical JSON identities must survive platform newline changes.
   An independent report verifier must reject missing/corrupt artifacts and
   recompute numerical decisions rather than trust saved pass flags.
6. Provide a portable researcher bundle, setup/preflight, exact tested environment,
   pinned downloads and a fresh-replication command. Distinguish local replay,
   software portability and actual independent human/machine reproduction.
7. Product UI must explain the question, show current evidence and limitations,
   start bounded registered work, support cancellation, inspect and export results,
   and avoid recommending unvalidated configurations. Verify the actual browser
   through worker/artifact path, including failure and unknown states.
8. Deliver a concise researcher-facing claim card: what works, where it loses,
   what is unproved and what another researcher should attempt to falsify. CXL,
   DSA, broad language quality, novelty and exclusive VRAM remain separate claims.

## Pre-implementation review

The existing v2 adapter/runtime remain byte-identical. A new validation module
owns the expanded experiment; the existing owned-process supervisor is reused.
Native reference preparation can itself OOM, so its failure must remain a case
outcome and cannot silently switch to a cheaper reference. Timings are not used
to select documents, bounds or method order. Full arrays cost disk space but
allow independent arithmetic instead of unverifiable aggregate pass flags.
The verifier is read-only, refuses unsafe relative artifact paths and loads
NumPy arrays with pickle disabled. Hash agreement proves identity, not honesty,
and the self-produced bundle cannot certify independent execution or zero bias.

## Work ledger

- Initial audit: v2 pressure execution deliberately bypasses model validation;
  its unknown status is correct. Six method-order permutations, separate long
  documents and an unchanged-reference control are missing from that study.
- Implementation and actual outcomes are recorded in the v3 protocol, execution
  ledger and report. This plan alone is not evidence that any gate has passed.
- Implementation checkpoint: 18 cases frozen before model outcomes. Seventeen
  v3 controls and the existing 57 attention controls passed before execution;
  two later standalone bundle-integrity controls also pass. These test fixtures
  are invented controls, never model/hardware run evidence.
- A live partial audit independently reconstructed 14 terminal cases from full
  arrays. It confirmed three hybrid bound failures and three OOM control arms.
  Partial summaries cannot establish the result of the remaining four cases.
  The full researcher archive, extracted-kit replay, final report and complete
  product acceptance audit remain pending while the study continues.
