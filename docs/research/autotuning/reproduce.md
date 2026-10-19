# Reproduce and inspect a runtime tuning campaign

Each exported ZIP contains one public-text campaign, its full output arrays,
exact measured source snapshots, the local tool and a byte inventory. It does
not contain model weights, a Python environment, private requests or the local
activation database. Recorded outcomes are historical evidence on the stated
machine; opening a bundle does not execute a new experiment.

## Verify the supplied observations without a GPU

Extract the ZIP into a new folder. Use Python 3.12 and NumPy 2.5.4, then run from
that folder:

```powershell
python benchmarks/verify_runtime_bundle.py .
```

The supplied verifier checks every inventoried file, starts the shipped CLI in
a separate process, reconstructs the full-vector numerical predicates, checks
the frozen selection/confirmation sequence and recomputes the decision. It does
not import PyTorch for that arithmetic. A successful check establishes consistency
of the supplied bytes and arithmetic. It does not recreate hardware timings,
authenticate the author or constitute independent model replication.

The byte inventory includes the local UI/service and the measured execution
sources. Read `BUNDLE-README.md` for the campaign ID and `bundle-manifest.json`
for its exact evidence digest. Failed and resource-limited candidate attempts
remain in the execution ledger; their unknown numerical relation cannot qualify
them. No output positions are removed to make a bundle smaller.

## Execute a fresh local campaign

Use Windows and a compatible NVIDIA CUDA setup. Follow the bundled
[runtime guide](../../runtime-tuning.md) to create the separate hash-pinned
environment and download the allowlisted public Qwen3-0.6B and WikiText assets.
The environment/asset downloads require internet access; the registered model
workers subsequently run offline. The tested host has 32 GiB RAM and a 12 GiB
RTX 3060. That observation is not a minimum-hardware guarantee for every method.

```powershell
./Start-Atlas.ps1
```

Open **Tune performance**, inspect the imported recorded result, then choose
the fixed workload and start a new campaign. No active setting is transferred
in the bundle. Another machine must qualify a new local profile; matching source
alone does not establish hardware/package/asset compatibility. A memory target
limits profile selection, not physical GPU allocation during search or checking.

The acceptance workloads are separately registered:

```powershell
.tools/attention-env/Scripts/python.exe tune.py start --context 2048
.tools/attention-env/Scripts/python.exe tune.py start --context 2048 --budget-mib 2560
.tools/attention-env/Scripts/python.exe tune.py start --context 32768 --budget-mib 6144
```

Run them sequentially. Keep the full context, FP32 precision and 16-token output;
using a smaller workload does not reproduce the registered question. Timing is
affected by other applications, thermal/clock state and system caches. Preserve
all attempts and record those conditions instead of selecting only favorable
observations. The finite grid is not a global optimum or a zero-bias guarantee.
Fixed windows are reused across objectives/source amendments and nested across
context lengths. Fresh processes are not independent datasets, and confirmation
text may have appeared in earlier studies or model pretraining.

After a qualified campaign, **Generate with active setting** executes a real
compatible request. Its optional native comparison covers 32 positions and has
an explicit extra cost. **Revert last change** restores the preceding qualified
pointer; without a qualifying memory profile, a constrained request abstains.
Custom prompts must match the exact registered token length and are never
silently truncated. Request records are private and excluded from research ZIPs.

The old dashboard's unrelated research histories are outside a single-campaign
bundle. Missing historical studies remain unavailable; they are not fabricated
as fresh runs. All supplied numerical and performance conclusions retain the
model, input-window, precision, source, hardware and timing scopes in the protocol.
