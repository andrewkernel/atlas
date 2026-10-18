# Atlas runtime autotuning: research and design review

Research phase, 2026-10-18. The owner requested an end-to-end runtime tuner with
most effort allocated to researching and improving the design before building.
This document records decisions before new autotuning model outcomes. It is not
evidence that the tuner has been implemented or that a configuration wins.

## Product question

Can a finite, measured search choose CPU thread count and attention-cache
placement for a fixed local model workload, confirm the choice independently
of selection, and apply a reversible profile without promoting failed or
unknown numerical evidence?

The first adapter uses the existing pinned Qwen3-0.6B and Windows CUDA environment.
Settings belong to Atlas's model process. The tuner does not configure other
applications, change model weights, overclock hardware or require administrative
access. Each campaign fixes model, precision, context, output length, numerical
bounds and input windows. Faster answers and lower allocation are separate goals.
No new algorithm, global optimum, CXL result or universal correctness is claimed.

## Prior evidence and consequences

| Observed evidence | Design consequence |
| --- | --- |
| V3 hybrid saves allocator space but loses generation time in qualifying pairs | Saving memory alone cannot justify a speed claim. A no-change decision is useful. |
| Four literal hybrid failures, seven unavailable OOM arms | Failed or unknown checks exclude a setting; never silently relax bounds or retry under the same identity. |
| Isolated processes remove the known diagnostic OOMs but retain its 32k numerical failure | Own one fresh CUDA process per profile/input/trial. Process isolation does not reset OS state or explain a specific OOM cause. |
| Local replay's resident generation takes 235s versus much shorter original observations | Search noise and allocator/desktop history require fresh confirmation; never activate the fastest single observation. |
| Existing selector explicitly abstains outside layer-attend-only identity | Build a separate full-model tuner; do not promote layer calibration into a model policy. |
| Historical 2k FP32 native peaks near 2836 MiB and hybrid near 2444 MiB | Predeclare a 2560 MiB operational budget for a new 2k test. This is prior-informed, not an unseen research result. |

The design input inspection found 2,517,232 train tokens and 262,210 validation
tokens with the pinned tokenizer. At each registered context (512, 2048, 8192,
32768), two disjoint train windows and two disjoint validation windows fit.
The offsets are 5003 and `5003 + context + 32 + 997`. Their token hashes are
computed before execution. Windows across contexts are nested and must not be
pooled as independent evidence. Public text may overlap model training data.

## Primary sources reviewed

1. [PyTorch CPU threads](https://docs.pytorch.org/docs/2.14/generated/torch.set_num_threads.html)
   requires setting intra-op threads before model work. The
   [inter-op setting](https://docs.pytorch.org/docs/2.14/generated/torch.set_num_interop_threads.html)
   is set once, before parallel work. Fresh processes avoid changing this global
   state between candidate settings in one model process.
2. [CUDA execution semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
   explains asynchronous dispatch and synchronization for timing. Atlas measures
   synchronized wall time over complete generation, including selection, input
   construction and cache work. Its existing forward loop is reused where possible.
3. [PyTorch benchmark guidance](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark)
   motivates warmups and repeated measurements. Full-model requests are expensive;
   the screening stage nominates, while fresh process repetitions confirm.
4. [Transformers cache strategies](https://huggingface.co/docs/transformers/kv_cache#cache-offloading)
   describes built-in CPU offloading and its memory/throughput tradeoff. Include
   this competent implementation instead of comparing only Atlas methods. The
   installed 5.19.0 DynamicCache and Qwen3 source were inspected: an explicit
   `DynamicCache(config=model.config, offloading=True)` can be passed through
   the builtin model forward path.
5. [Optuna reproducibility guidance](https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results)
   distinguishes seeded suggestion order from noisy objectives and parallelism.
   For this small space, use a transparent sequential finite grid rather than
   add a Bayesian-optimization dependency. Preserve every attempted outcome.
6. [Ansor](https://www.usenix.org/system/files/osdi20-zheng.pdf) is prior art for
   search-based tensor optimization. Atlas tunes existing runtime configurations;
   it does not reproduce that compiler search or establish autotuning novelty.
7. [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html) explains
   selection bias from optimizing a noisy finite criterion. Freeze a nominee
   before opening confirmation measurements. Confirmation is not another search.
8. [Kalibera and Jones](https://kar.kent.ac.uk/33611/) motivates treating process
   and iteration variability explicitly. Atlas retains process/trial identities;
   its small, shared-session observations do not establish population confidence.

These sources motivate the engineering decisions. They do not endorse Atlas
or prove that its specific configuration grid will improve performance.

## Search and confirmation contract

Search native execution with 1/2/4/6 CPU threads; resident-cache native SDPA with
4 threads as a control; builtin Transformers offloading with 4 threads; dense
hybrid with a 256-token GPU window and 1/2/4/6 threads; and a 1024-token hybrid
window with 4 threads when shorter than the prompt. This is at most 11 profiles,
not an exhaustive search over all settings. Duplicated/invalid windows are omitted
by a deterministic registry rule before outcomes.

FP32 is the primary tested research precision. FP16/BF16 can be separately
registered with a matched native reference; they are never compared against
FP32 as a free optimization. Retain the strict elementwise atol=rtol=1e-4
relation and reject failures. No low-precision repair is assumed.

Prepare native full-vocabulary references in separate processes. For each
candidate/input/trial: configure threads before model work, load the fixed model,
perform a full target-shape warmup, time the complete fixed-length generation,
then perform the separate 32-position teacher-forced replay. Retain cold/warmup
costs and memory peaks. Do not launch a polling CLI observer inside timed windows.
Read device/host observations around them and record controller/owned-job limits.

Rank eligible settings using only two train windows, with equal weight in a
geometric mean of positive generation durations. Both native repeat and resident
controls must pass. A failed or unavailable setting cannot be rescued by a fast
timing on the other input. Infeasible settings may be pruned only by a frozen rule;
unattempted work stays explicitly unattempted, never a fake timing or failure.

Persist the nominee and its evidence digest before confirmation. Confirmation
uses two separate validation windows and three fresh process repetitions per
window. Pair current/nominee executions with alternating order. If the nominee
is current, repeat current instead of inventing a second configuration. Repeat
the resident numerical control on each confirmation window.

For the speed goal, every required numerical/control result must pass and all
six paired generation ratios must exceed the predeclared 1.05 threshold before
automatic change. Report the raw ratios, not a p-value or a confidence guarantee.
Otherwise retain current settings or report inconclusive evidence.

For a GPU-allocation goal, require every measured warmup/generation allocation
peak of the nominee to fit the selected threshold. If current already fits,
require the same speed-improvement gate to change it. If current does not fit,
a confirmed feasible nominee can meet the memory goal while being slower;
label that result as memory fit and give existing offloading its credit. This
is not a hard process/physical VRAM quota or a validated OOM rescue.

## Optimization overhead and deployment

Store newly computed arrays by their exact NumPy-file content hash within the
campaign. Repeated deterministic outputs can share one immutable blob, while
each actual trial retains its own configuration, process identity, timing and
array descriptor. Deduplication happens after computation; it must never skip
an execution or substitute a previous result. This reduces storage/export cost
without dropping positions or changing precision. Shared source snapshots are
also checked against each worker's recorded executed-source identities.

Save a profile once and reuse it for compatible requests. Record reference,
search, confirmation, validation, startup and warmup costs. Show a descriptive
speed break-even count only when measured savings are positive; a slower memory
profile has no speed break-even. Distinguish warm generation from cold request
latency and include verification/startup overhead in actual application records.

Profiles bind model/tokenizer/data revisions, runtime source, precision, exact
context/output length, hardware fingerprint, driver and packages. A new context,
precision, model or incompatible environment cannot inherit a green result.
Application uses a validated profile or an explicitly reported native fallback;
it never shortens a request to appear faster. Under a memory constraint, do not
silently fall back to an unqualified higher-memory setting.

Finite held-out numerical checks do not certify every future prompt. Application
records must distinguish profile evidence from a fresh check on the current
request. An optional native replay can verify the request at an explicit added
cost; failed checks retain the candidate evidence and use an eligible baseline
or abstain. User-prompt records are excluded from research bundles by default.

## Reliability review

Campaign protocols and inputs are immutable once frozen. Settings execute only
through registered adapters. Preserve v2/v3/isolation source bytes and their
negative outcomes; put new implementation under a separate package and entry
point. A wrapper service can add routes while retaining the old application.

Use the validated suspended-launch Windows job boundary and retain both launcher
and actual CUDA PID/creation identities. Hold the shared workspace GPU lock over
the campaign. Cancellation, timeout or resource pressure ends only owned work;
an interrupted file is not evidence that a process is still live or dead.
Declare per-worker and campaign deadlines, disk checks and advisory RAM headroom.

Use transactional profile state and an append-only activation/revert journal.
Cancel and profile commit need one serialized decision point: accepted cancel
prevents later automatic activation; after commit, report that Revert is required.
Compare expected active profile when committing so a concurrent user decision is
not overwritten. Corruption, identity drift and incomplete verification prevent
activation. A separate NumPy verifier recomputes checks and selection; it must not
trust stored pass flags, nomination or status labels.

Residual limits include one machine/model family, finite grid, fixed public-text
windows, nested contexts, shared desktop/OS caches, firmware/kernel variability,
small confirmation blocks and lack of independent external reproduction. No
zero-bias or zero-mistake guarantee is possible; the response is explicit bounds,
negative controls, real execution and an inspectable completion audit.
