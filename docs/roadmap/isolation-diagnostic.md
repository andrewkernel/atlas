# Follow-up: isolate process history before the final handoff

The v3 raw records show OOMs in native/resident arms after hybrid execution in
the same process. They do not identify a specific allocator cause or establish
fresh-process feasibility. A useful researcher product must make this boundary
clear and provide an isolated countercheck, rather than market these as rescues.

This is a **post-hoc diagnostic**, selected after v3 outcomes. Preserve all v3
bytes and bounds. Use its PG-19 book 2383 at both 32,768 and 40,000 tokens:
the former had a hybrid bound failure and the latter had native/control OOMs.
Choosing this failure pair is explicit; it is not an unseen confirmatory cohort.

Before executing the diagnostic, freeze six cases: native, resident and hybrid
at each context, each in a fresh owned process. Reuse the measured projections,
cache implementation, pinned assets, token histories and 1e-4 FP32 bounds.
Every arm gets the same short warmup, then timed free generation, then a separate
32-position validation. Native additionally repeats its validation to quantify
reference repeatability. Prepare the native reference after timing so it cannot
give native a long-context warmup that other arms do not receive.

Record complete arrays, process identity/termination, host allocator statistics,
memory/timing, actual sources and all failures. Require unchanged-native-repeat
and resident controls before qualifying hybrid. Separate processes remove
shared framework/CUDA allocator history, not OS caches, desktop interference or
all bias. A disappearance of OOM is a counterexample to a hard feasibility claim,
not proof that a particular allocator caused the original failure.

Final acceptance now includes collecting/verifying these six outcomes, a report
that separates them from v3, inclusion in the researcher kit, an extracted-kit
check and fresh local browser-to-worker execution. The primary v3 study remains
unaltered, including every failed/unknown arm. No CXL or novelty claim follows.
