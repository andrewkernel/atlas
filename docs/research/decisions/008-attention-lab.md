# ADR 008 — Local exact attention engineering track

Status: implemented owner-authorized engineering track, 2026-10-17.

The owner requested a complete consumer-compute attention application with architecture review, actual benchmarks and an interactive dashboard. This authorizes the bounded Attention Lab alongside the historical training evaluation package; it does not retroactively satisfy the independent-label or training-policy research gates in ADR 007.

The distinct, deliberately simple policy is `linear-median-v1`: interpolate validated calibration medians per fixed plan, reject KV-budget-ineligible plans, select minimum predicted latency with a lexical tie break. Evaluate disjoint lengths and a calibrated fixed comparator; charge calibration separately. Exact dense partition fusion is existing mathematics, not a novelty claim.

The persistent GPT-2 adapter owns old CPU and recent GPU caches before framework concatenation. Native SDPA and direct dense GPU attention are both comparators. Pretrained-model checks retain all attention entries; real-arithmetic equivalence does not imply identical float32 logits. Synthetic finite checks cannot certify full-model inference.

The shipped app is local because execution uses the owner's GPU. Cloud publication is outside this implementation. The interface uses Sites design guidance, with a buildless client, FastAPI loopback backend and isolated fixed-registry workers. No external messages were sent and no remote research code executed.

Four curated sweeps and two full pretrained-model comparisons, their raw samples and failures, are in `docs/research/attention-lab/`. The report's selected 214 rows are actual measurements. The package documents that hybrid paths were slower in these model comparisons, and some violate the predeclared logit bound despite matching greedy tokens. No broad CXL, quality-equivalence, OOM-rescue, or standalone novelty claim follows.
