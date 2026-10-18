# Isolated-process diagnostic

This post-hoc study investigates the known v3 book-2383 failure pair. It is not
an unseen confirmation cohort or an independent external replication. Each of
six method/context cases owns a fresh CUDA process. The original v3 remains
byte-identical. Original FP32 bounds remain atol=rtol=1e-4.

Use the pinned Windows environment and asset preparation in
[the v3 reproduction guide](../attention-v3/reproduce.md), then:

```powershell
New-Item -ItemType Directory -Force artifacts/attention-lab | Out-Null
.tools/attention-env/Scripts/python.exe benchmarks/attention_isolation.py run --replication-id independent-01
.tools/attention-env/Scripts/python.exe benchmarks/attention_isolation.py analyze --replication-id independent-01
```

This writes to artifacts/attention-isolation-replications/independent-01.
The directory preparation creates the shared lock location; the main runner
normally creates it when starting its first experiment. Completed failures are
retained rather than silently rerun. An interrupted controller
retains its raw evidence and requires authoritative process inspection; preserve
that attempt and use a distinct replication ID for a fresh attempt. Do not infer
that a saved `running` flag proves a process is still live.

The primary diagnostic waits for v3 to finish before measuring. New replications
should run with no competing GPU experiment. Workers share the original GPU
lock, have 1,200s wall limits and reuse the validated owned-process scope. They
execute only the fixed registered script, builtin Qwen3 and local safetensors.
No arbitrary project code, remote model code or synthetic timings are used.

Native references are prepared after timed generation; native also repeats the
validation. Other workers verify and use that native reference on the exact
same token history. Source, process identity, full arrays, all numerical checks,
allocator observations and timing windows are retained. Both native-repeat and
resident controls must pass before hybrid qualifies.

Read-only numerical verification, without PyTorch execution:

```powershell
python benchmarks/verify_attention_isolation.py --study docs/research/attention-isolation --source-root .
```

Requires NumPy. For a new replication, pass its output directory instead.
This checks internal artifact consistency and arithmetic, not author honesty or
independent real-device execution. Separate processes remove shared framework
allocator state, but do not reset OS caches or reserve the desktop. A changed OOM
outcome alone does not identify a particular allocator cause. Host allocator
statistics are observations from the installed runtime, not physical-residency
measurements. No CXL, DSA, universal quality, energy or novelty claim follows.
