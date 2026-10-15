# Security and reliability model

## MVP policy

**Trusted-code-only execution.** Static file inspection is allowed without import. Target imports, model construction, profiling, FX tracing, dependency builds and checkpoint loading can execute code. The initial system accepts only reviewed builtin adapters and explicitly registered trusted local workloads. It must refuse newly downloaded/untrusted programs by default. A user acknowledgment does not create strong isolation.

The initial research executed original reviewed probes. Later controls additionally execute pinned, scoped-reviewed TrainParity and Rockmate/rkgb source on authored allowlisted cases, plus inspected read-only NVML wrappers. Review manifests and exact execution scopes are retained. No foreign fixture/model program, source build hook, external checkpoint or arbitrary workload was executed. Official binary dependencies were installed into separate local environments. The real-data pilot loaded pinned Fashion-MNIST data with byte hashes and its retained notice. No elevated privileges or remote installer scripts were used.

## Threats and controls

| Risk | Proposed control | Residual limitation |
|---|---|---|
| Malicious code reads files or credentials | No arbitrary execution in MVP; reviewed adapters; minimal environment; isolated workspace if later enabled | A normal child process can still access host files as the user |
| Destructive filesystem writes | Immutable source snapshot; restricted allowlisted output roots; reviewed code; no source mounts writable in container mode | Process conventions are not OS filesystem isolation |
| Network exfiltration | Offline data preparation; container network disabled where practical; no inherited credentials | Native trusted runner cannot guarantee network restriction |
| Dependency compromise/build hooks | Exact locks/hashes, known wheel origin; review before new dependencies; no automatic project installation | Registry/native-wheel supply-chain risk remains |
| Host/GPU denial of service | Wall timeout, log/disk caps, child-process supervision, declared RAM/CPU budgets | Consumer GPU VRAM/compute isolation is not reliably hard-enforced by ordinary processes |
| Unsafe checkpoint deserialization | Trust provenance; prefer safetensors for external weights, reviewed state-loading paths | Tensor-only loaders reduce some risks, not malicious Python from imports |
| Local API abuse | Loopback binding, local token, origin/host validation, no wildcard CORS, registered IDs only | Other same-user local software remains in threat model |
| LLM prompt injection | Treat source/comments as data; schema validation; no tool/shell authority | Explanations can still be misleading; independent checks required |
| Worker crash/partial update | Atomic evidence writes, explicit terminal states, disposable process per trial | Hard device resets may need manual recovery; never claim successful cleanup without checking |

## Optional hardened container runner

When implemented and tested, use digest-pinned images, non-root user, read-only root filesystem, dropped capabilities, no-new-privileges, default/restricted seccomp, network none, PID/CPU/RAM limits and bounded writable scratch. Mount only the specific reviewed source read-only, specific data read-only and one run-output directory writable. Never mount the Docker socket, home directory, SSH keys, cloud configs or credential stores. Dependency/data preparation is a separate reviewed step.

[Docker security guidance](https://docs.docker.com/engine/security/) explains configuration and daemon attack surfaces. [Resource constraints](https://docs.docker.com/engine/containers/resource_constraints/) control CPU/host memory; they do not by themselves provide arbitrary hard per-container GPU VRAM quotas. GPU passthrough exposes driver/kernel surface and shared hardware. Docker Desktop/WSL2 add platform-specific boundaries that must be tested. Strong hostile GPU-code isolation is outside the undergraduate MVP.

## Termination and reliability

On Windows, use Job Objects or an equivalently tested process-tree owner; on Linux, process groups plus cgroups/container controls where available. Send graceful cancellation, wait a bounded grace period, then terminate owned children. Test a deliberately hanging worker and one spawning a child; verify no owned child remains and device memory is reclaimed if GPU support is enabled. Do not stop unrelated user processes.

Memory/RSS polling is detection rather than a hard sandbox. On a native trusted runner, advertise soft limits accurately. Reject hard-isolation requests that the backend cannot enforce. On timeout/OOM, preserve partial logs and mark checks incomplete; do not retry without a bounded plan and clean process.

## Approval record

Before a later untrusted isolated run, show exact source identity, dependency preparation, command/adapter, mounts, network, resource limits and residual isolation limits. Approval is scoped to that immutable record and expires after changes. The LLM/API caller cannot approve an untrusted run on the user's behalf. The MVP may simply refuse this mode until the isolation tests pass.

This is a proposed threat model, not a validated sandbox. No safe-arbitrary-code claim is justified by this session.
