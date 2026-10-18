# Atlas runtime tuning

Atlas can search a finite set of CPU thread counts and cache strategies for its
Qwen3-0.6B adapter. This changes compatible future Atlas model requests. It does
not tune Windows, change another program, implement CXL, or guarantee a speedup.

The [research review](research/autotuning/research-and-design.md) and
[implementation contract](architecture/runtime-autotuner.md) explain selection,
independent verification, application and rollback. The
[acceptance gates](roadmap/autotuning-acceptance.md) separate implemented paths
from the real execution evidence required before declaring completion.

## Local workflow

Use the existing pinned CUDA environment and public assets from the
[Attention Lab setup](attention-lab.md). Start `./Start-Atlas.ps1` and open
**Tune performance**. An idle older service from this exact checkout upgrades
automatically; another application's listener or an active older job is refused.

1. Choose the exact prompt length, output length, precision and objective.
2. Start measurement. Keep competing work off the shared GPU/CPU when practical.
   The finite grid includes native execution, a resident cache, Transformers'
   built-in cache offloading, and Atlas hybrid cache windows. The model, input and
   precision stay fixed within a campaign.
3. Atlas freezes public text windows and source identities before observations,
   measures calibration in fresh processes, freezes a nominee, and confirms it
   on separate validation text. Every failed or unavailable attempt remains.
4. The NumPy verifier reconstructs full-vector checks and the decision. A speed
   change requires all six paired ratios to exceed 1.05. Otherwise Atlas can keep
   the current setting, report a memory fit, or abstain.
5. Automatic application is optional. A qualified saved setting can also be
   applied manually; **Revert last change** restores the prior active pointer.
   The journal records both actions. A concurrent change prevents a stale apply.
6. **Generate with active setting** executes a real compatible workload. The
   public example is convenient for reproduction. A custom prompt must have the
   exact registered token count; requests are never silently shortened.

Cancel stops only the campaign's owned processes. Accepted cancellation prevents
an automatic commit. Once a decision has committed, use Revert. A cancelled,
interrupted or failed campaign cannot become fresh successful evidence by
editing its status.

## What the measurements mean

Generation time includes prefill, greedy token selection, input preparation,
cache maintenance, transfers, synchronized forward passes and cache teardown.
It is measured after a full-shape warmup. Loading, warmup, reference generation,
numerical checking and total tuning time remain separate recorded costs. The
objective is warmed generation; actual cold requests show their total cost too.

The allocation target uses the maximum measured CUDA allocated peak across
warmup and scored generation. Search and validation may exceed it. It is not a
physical VRAM quota or a guarantee against OOM. A memory-fit setting may be
slower. A descriptive speed break-even is shown only for positive confirmed
generation savings; it assumes compatible repeated requests and stable savings.

Profile evidence covers finite campaign windows. **Check against native outputs**
adds a fresh 32-position, full-vocabulary numerical replay on the request's native
greedy history. That check has an explicit extra cost and does not establish
answer quality or every future output. Without it the current request's numerical
relation remains unknown. A failed candidate is retained; an unconstrained
request can explicitly fall back to native. A constrained request abstains rather
than silently returning an unqualified setting above its target.

Profiles bind execution sources, pinned inputs/model, precision, exact lengths,
driver, packages and machine identity. Retune after incompatible changes. The
initial validated execution boundary is Windows/NVIDIA; portable verification
does not validate another platform's CUDA runtime.

## CLI and files

```powershell
.tools/attention-env/Scripts/python.exe tune.py start --context 2048
.tools/attention-env/Scripts/python.exe tune.py start --context 2048 --budget-mib 2560
.tools/attention-env/Scripts/python.exe tune.py start --context 32768 --budget-mib 6144
.tools/attention-env/Scripts/python.exe tune.py status CAMPAIGN_ID
.tools/attention-env/Scripts/python.exe tune.py verify CAMPAIGN_ID
.tools/attention-env/Scripts/python.exe tune.py cancel CAMPAIGN_ID
.tools/attention-env/Scripts/python.exe tune.py profiles
.tools/attention-env/Scripts/python.exe tune.py generate CAMPAIGN_ID --check-native
.tools/attention-env/Scripts/python.exe tune.py apply PROFILE_ID
.tools/attention-env/Scripts/python.exe tune.py revert DOMAIN_SHA256
.tools/attention-env/Scripts/python.exe tune.py export CAMPAIGN_ID
```

Replace IDs with those printed or displayed by Atlas. `start` runs in the
foreground; the browser service owns background work. The default campaign
deadline is four hours and each model worker is bounded to 20 minutes. These
are lifecycle limits, not a hostile-code sandbox or hard memory isolation.

Public campaign protocols, full arrays, exact execution sources and terminal
outcomes live under `artifacts/atlas-tuning/campaigns/`. Identical computed arrays
share lossless content-addressed payloads; every trial still executes. Immutable
profiles and authoritative SQLite state are local. User prompts and inference
outputs live separately under `private-inference/` and are excluded from research
exports. Export verification uses Python 3.12 and NumPy without importing PyTorch.
Weights and environments are excluded. No external service receives these files.
