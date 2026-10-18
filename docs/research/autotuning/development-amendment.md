# Development hardening before final campaigns

2026-10-18. The initial implementation checkpoint `b472406` completed one actual
2k FP32 speed campaign, `tune-55b6b69a48d3`, after a separate actual cancellation
control. Its 42 workers produced a verified **keep current** decision: the
resident-cache nominee did not clear 1.05 in all six confirmation pairs. Its
452.766-second total cost does not have a positive speed break-even. The complete
development source and raw outputs remain preserved in the campaign and its
separately verified export. This is development evidence, not a result from the
later execution identity.

Before the final required speed/budget/long-context campaigns, code review adds:

- Separate active domains for goal and exact allocation target, alongside the
  original model, precision and length bindings. One objective cannot inherit
  another objective's active selection.
- Explicit pinned model/tokenizer/corpus byte checks before every model worker
  and active-profile use, including native fallback. The campaign's initial
  asset check alone cannot protect a later request against changed files.
- Loaded model class, parameter count and parameter dtype records, checked by
  the independent verifier. Shared-device/host observations surround execution
  spans outside scored generation; they are not process energy or residency.
- Verified hard links for identical **newly computed** public arrays across
  campaigns where supported. Portable paths stay local to each campaign and
  every trial still executes. Prior evidence is neither deleted nor rewritten.
  Unsupported linking falls back to writing the same complete bytes.

These changes receive new source hashes and fresh required campaigns. They do
not change numerical tolerances, the input windows, finite grid, confirmation
order, 1.05 rule or allocation-budget definitions. The initial development
campaign cannot activate a profile under the changed execution source. Its
original exported source remains available for independent arithmetic replay.

The development browser download is byte-identical to its export:
`09443ed5c8f679a6952590c4c94d651dca640c310a3d2d7d3d0a09f7ca757ca6`,
156,916,734 bytes. A separate extracted copy checked 180 inventoried files and
reconstructed evidence digest
`f5d9fd78ca42cabc6befe8787ac2ef09a433e9c8e993a0b0ad87f553e15725f5`.
These are consistency checks, not external model replication.
