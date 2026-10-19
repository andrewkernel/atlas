# Runtime tuning: measured results and validation status

Validation is in progress. The completed 2k speed/memory campaigns and application
checks below are actual observations. The required 32k memory-target campaign and
final acceptance audit remain unfinished; this is not an end-to-end completion
claim. The [development amendment](development-amendment.md) preserves the
earlier implementation and its negative speed decision.

The [resource-recovery amendment](resource-recovery-amendment.md) requires fresh
2k speed, 2k allocation and 32k allocation campaigns. The two amended-source 2k
campaigns below have completed. Pre-recovery observations are labeled historical
and retained under their original identities. Both old active 2k pointers were
reverted before changing execution source; immutable profiles, full raw evidence
and journals remain.

## Amended-source speed campaign

`tune-0ac3414ab599` completed all 42 fresh workers under the amended controller
and the same fixed 2,048-token FP32 workload. All 38 scored candidate/control
workers passed the 32-position full-vocabulary relation. Independent NumPy
reconstruction matched the saved evidence and decision; the owned controller
scope closed and all matching observed processes exited. See the
[current summary](fastest-2k.json).

Resident cache again won calibration. Its six native/candidate confirmation
ratios were 1.03688, 1.04218, 1.03810, 1.04613, 1.03723 and 1.04085. None exceeded
the required 1.05 threshold, so the verified decision was **keep current:
native-t4**. Total tuning cost was 495.953 seconds, with no positive speed
break-even. A qualified native profile was saved and activated; this records
the same execution setting rather than a demonstrated speed improvement.
The amended-source 32k campaign and final end-to-end audit remain pending.
The parent-owned campaign interval was 496.781 seconds. Controller campaign
cost and parent startup/cleanup are distinct clocks; neither includes browser
transport or service prelaunch setup.

The saved native profile drove actual request `generate-4d900c9c8b06`; its fresh
32-position native comparison passed. Warmed generation took 0.974805 seconds,
worker total 14.398072 seconds, service total including launch/cleanup 15.703
seconds, and browser-observed submission-to-receipt 16.1947 seconds. These
different clocks do not establish an application-level speed improvement.
After browser Revert, `generate-e13bc014ea1a` used explicit native fallback and
passed a fresh check (0.943667-second generation, 10.725879-second worker,
11.891-second service, 13.247-second browser observation).

An actual authored short-prompt request, `generate-1356190895b0`, was refused:
four tokens cannot use this exact 2,048-token profile. Its submitted request
remained intact and no tokens were generated. The
[application receipt](application-checks.json) contains sanitized outcomes and
scoped clocks; the raw request text/output/arrays remain private. Seven authored
asynchronous UI controls also cover the selected-record association, imported
labels and waiting for owned cleanup before displaying final request clocks.

## Amended-source 2k allocation-target campaign

`tune-523ff98ee883` completed 42 fresh workers on the same FP32, 2,048-token
prompt / 16-token output workload with the predeclared 2,560 MiB target. All 38
scored candidate/control workers passed the full 32-position numerical relation.
Independent reconstruction matched the recorded decision; the owned controller
scope closed and all matching observed processes exited. See the
[current memory summary](budget-2k.json).

Built-in Transformers offloading was confirmed within the allocation target:
its maximum observed warmup/generation allocated peak was 2,404.407 MiB, compared
with 2,836.407 MiB for native. Confirmation native/candidate time ratios ranged
from 0.65499 to 0.74771. This is a slower **memory fit**, not a speed improvement
or demonstrated OOM rescue. The total tuning cost was 503.172 seconds; no
positive speed break-even is claimed. The saved offloaded profile was activated
only after complete independent verification.
The parent-owned campaign interval was 504.016 seconds, separately retained from
the 503.172-second controller campaign measurement.

Actual request `generate-8b076acdf516` used the saved offloaded profile and passed
its fresh native comparison. Warmed generation took 1.271639 seconds, worker
total 15.808157 seconds, service total 17.093 seconds and browser observation
18.9261 seconds. Its measured warmup/generation peak stayed at 2,404.407 MiB.
Browser Revert restored the default pointer; `generate-5696d9d88984` then
abstained without generating or loading a model because no qualified memory
profile was active. Manual reapplication succeeded. Actual stale-revision,
historical-source, unknown-profile, invalid-context and incomplete-export
requests were refused without changing active state. Portable response copies
redact local checkout prefixes; the local raw observations are retained.

## Registered amended-source 32k validation

`tune-eeb99dbc1f00` is the amended-source FP32, 32,768-token prompt / 16-token
output campaign with the original 6,144 MiB allocation target and all 11
registered profiles. Its execution source hashes match both completed amended
2k campaigns. The terminal decision is pending in this report; consult the live
owned-process state before inferring liveness from a recorded status. No 32k
profile has been qualified or applied by the observations reported here.

Its first built-in offloading trial reached the unchanged advisory RAM guard:
1,482,407,936 bytes available, below the 2 GiB threshold. The owned trial scope
closed, matching observed trial processes exited, and three recovery samples
recorded available RAM reaching 5,089,566,720 bytes after 0.406 seconds, above the
required 4 GiB. The controller then launched the next registered native trial.
The stopped case remains numerically unknown and ineligible. The
[actual recovery receipt](32k-resource-recovery.json) checks this scalar record
and cleanup; it does not replace full campaign verification or prove the cause
of memory pressure. No candidate was removed from the remaining registry.

## Historical speed campaign

`tune-dc64d8a72091` ran Qwen3-0.6B in FP32 with a 2,048-token prompt and fixed
16-token output on the existing Ryzen 5 7600X / RTX 3060 Windows machine. Source
identities were frozen before measurements. Two train windows select; two
separate validation windows confirm. The finite grid covers CPU threads 1/2/4/6,
native and resident caches, Transformers offloading and 256/1,024-token hybrid
GPU windows. See the [preserved machine-readable result](pre-recovery-fastest-2k.json).

All 42 fresh workers completed. All 38 candidate/control workers passed their
32-position full-vocabulary literal checks against separately computed native
references. This finite relation uses `atol=rtol=1e-4`; it is not a broad answer
quality score or a guarantee for future prompts.

The resident cache won calibration. Confirmation native/candidate ratios were
1.05379, 1.01506, 1.05462, 1.03446, 1.04978 and 1.04774. Because every ratio must
exceed 1.05 under the frozen rule, the decision was **keep current: native-t4**.
No faster execution setting was applied. Total tuning cost was 501.110 seconds;
there is no positive speed break-even for this decision. These are descriptive
same-session observations, not a formal population confidence interval.

## Historical model application

The saved native profile `profile-0f163d62d6c6ca8523e7ddc1` drove an actual
registered-example request, `generate-280168c02e63`. The fresh 32-position check
passed. Warmed generation took 0.933745 seconds, versus 14.410382 seconds for
the worker, including compatibility/evidence checks, loading, warmup and
native replay. A generation-only objective does not establish lower total
application latency.

Revert restored the default active pointer. Actual request
`generate-07baf7f55ddb` explicitly used native fallback and passed a fresh check;
its generation/worker-total times were 0.951360/10.552232 seconds. The saved profile was
then applied again through the browser. Private inference records remain outside
the public research inventory even when the selected prompt was a public example.

## Historical 2k allocation-target campaign

`tune-af0f569e5042` completed another 42 fresh workers for the same FP32 workload
with a 2,560 MiB measured-allocation target. All 38 candidate/control workers
passed their finite numerical checks. Two separate validation windows and six
paired repetitions confirmed `offloaded-t4`, the built-in Transformers cache
offloading comparator. Maximum warmup/generation allocated peaks were
2,404.407 MiB for this nominee and 2,836.407 MiB for native-t4. See the
[preserved complete summary](pre-recovery-budget-2k.json).

The decision is **confirmed memory fit**, not a speed improvement. Confirmation
native/candidate time ratios ranged from 0.73433 to 0.85639, meaning the chosen
setting was slower. Total tuning cost was 510.640 seconds, with no positive
speed break-even. This target was chosen using earlier memory observations;
it is not evidence of an actual OOM rescue or a physical GPU-memory quota.

The resulting `profile-65d8c31ded10874616f69f50` drove actual request
`generate-64e5f69ccf49`, passed its fresh check and generated 16 tokens in
1.289850 seconds (15.763138 seconds for the worker). After Revert, constrained request
`generate-3df5a74a491e` abstained before model execution because no qualified
profile was active. Reapplication succeeded. The
[application receipt](application-checks.json) records these local observations;
private request arrays are not supplied as public arithmetic proof.

## Controls and unfinished validation

The earlier real cancellation control, `tune-1cb8fb2880fc`, stopped its owned
launcher and model processes without applying a profile. Its raw result and
Windows job records remain available; they belong to the earlier development
execution identity. Controls under the pre-recovery source also completed: the 60-second
deadline run `tune-7b170f96d0e8` timed out and finished owned cleanup in 63.672
seconds; `tune-287d3ea2c3e4` accepted cancellation while its actual model process
was live. Both owned scopes closed, all matching observed processes exited and
active settings stayed unchanged. See the [lifecycle receipt](lifecycle-checks.json).

Actual local HTTP checks during `tune-af0f569e5042` refused four conflicting
experiment/generation starts with 409, and refused missing-token and
foreign-origin requests with 403. Actual stale-revision activation returned 409;
unknown profiles and invalid settings returned 422 without changing state. See
the [admission](api-admission-checks.json) and
[activation](api-activation-checks.json) receipts. Authored temporary tests cover full-array
failures, altered flags/nomination, incomplete cases, process ownership, immutable
profiles, objective separation, asset drift, stale state, cancellation/commit
serialization, rollback and exclusion of private records. Invented test arrays
and timings are not model evidence.

The amended-source lifecycle controls also passed: `tune-ec8f268f90b9` reached
its 60-second deadline and completed owned cleanup in 64.485 seconds;
`tune-ee144274ea5e` accepted cancellation while its actual model process was
confirmed live and completed cleanup in 17.844 seconds. Both scopes closed,
matching observed processes exited, and the exact active profile/revision stayed
unchanged. The lifecycle receipt retains both historical and amended controls.
The [latest authored checks](checks.json) passed 132 pinned-environment
attention/tuning tests, 154 CPU tests (15 optional-integration skips), seven
asynchronous UI controls, syntax, document/link and dashboard-size checks.

The 32k/6,144 MiB campaign, `tune-53a712409265`, stopped at the available-RAM guard
during its first built-in offloading calibration trial after 11 completed workers.
The [interruption receipt](32k-resource-interruption.json) records closed owned
processes and no matching live processes. It selected and applied no profile.
The exact low-RAM sample was not retained; peak commit is not RSS or a proved cause.
The amended controller retains exact guard/recovery samples and can reject that
candidate and continue only after owned cleanup and sufficient RAM recovery.
All three required workflows must run under its new source identity. Memory targets are
selection constraints on measured warmup/generation
CUDA allocated peaks, not hard physical-memory quotas. A confirmed memory fit
may be slower, and unavailable/failed evidence cannot qualify.

The [research review](research-and-design.md), [runtime guide](../../runtime-tuning.md)
and [acceptance gates](../../roadmap/autotuning-acceptance.md) define scope and
reproduction. One machine/model family, small fixed windows, nested contexts,
shared desktop scheduling, process variability and public pretraining overlap
limit generalization. No CXL, OS-wide tuning, external reproduction or new
algorithmic superiority is established.

The same fixed public-text windows are reused across objectives and execution
amendments. Fresh worker processes do not create new or independent evaluation
data. Separate confirmation means separate from the campaign's calibration
windows; it does not mean unseen by earlier studies or the model's pretraining.
