# Open questions and unfinished work

Priorities: **P0** changes the credibility of the research question; **P1** is required before implementation/evaluation claims; **P2** is optional scope.

| Priority | Unresolved question | Why it matters | Concrete next evidence / stop rule |
|---|---|---|---|
| P0 | Is selection over composed memory plans new enough beyond native mechanisms plus TrainCheck/TrainParity? | Determines whether this is research or integration | Deep code review, shared cases, forward literature search and advisor review; pivot if composition matches |
| P0 | Do eligible plans exist for useful real workloads at acceptable cost? | Conservative checks may reject everything; persistent state may dominate | One physical-GPU pilot with full-update profiling and finite-set oracle; report no-plan rate |
| P0 | Can verification cost be justified for users with little compute? | Repeated references may cost more than saved execution | Measure total search/check/full-run budget and reuse scope; narrow claim if cost dominates |
| P1 | Which real failure families are missing from authored probes? | Mutation benchmark can be self-confirming | Audit historical issues, independently label cases and hold out families; no labels obtained yet |
| P1 | What can be said when full baseline OOMs locally? | Fidelity comparison is missing, not automatically preserved | Trusted reference artifact/CPU study with limitations or explicit unverified result |
| P1 | What are defensible scientific outcome margins? | Convenient tolerances can manufacture equivalence | Domain/advisor input and preregistered estimands/power; remain inconclusive if unsupported |
| P1 | Which GPU/PyTorch/CUDA versions work on the 12 GB host? | CPU lock does not validate CUDA | Separate explicit CUDA environment, native smoke, allocator/timing check; not done here |
| P1 | Can TrainParity be compared without stack confounding? | Inspected metadata excludes Python 3.12/Torch 2.14 | Separate compatible Python 3.11/Torch environment or shared-trace capability analysis; do not force incompatible install |
| P1 | Which WikiText terms apply to pinned bytes? | Card has conflicting3.0/GFDL versus4.0 metadata/prose | Resolve authoritative provenance/terms before data acquisition/redistribution; current plan conditional |
| P1 | How much adapter authoring does a new workload require? | Access benefit can be lost to integration | Record adaptation effort on held-out workload; no user study here |
| P1 | Can process trees, RSS limits and cancellation work reliably on Windows/Linux? | OOM/timeouts must leave recoverable state | Tested native lifecycle runner; current probes are simple synchronous CPU scripts |
| P1 | Are advanced checkpoint/compile APIs stable enough for a baseline? | Internal knobs can change | Pin version/API and validate behavior; allow baseline incompatibility with an explicit explanation |
| P2 | Does an LLM improve ranking in a finite registry? | It may only add overhead | Equal-budget held-out ablation; no models downloaded or benchmarked |
| P2 | Are6–8/16/24 GB hardware tiers accessible? | Cannot generalize from one board | Obtain physical access; allocator caps remain emulation |
| P2 | Does energy decrease? | VRAM/time are inadequate proxies | Supported board energy or whole-system meter; no energy measurement here |
| P2 | Can hostile GPU workloads be isolated safely? | Trusted processes/Docker are insufficient | Dedicated security work; trusted-code-only MVP remains |

## Evidence maturity

The repository contains a complete **initial blueprint**, a source-backed scoping review and bounded semantic demonstrations. It does not contain an exhaustive literature review, implemented application, confirmed academic novelty, validated benchmark corpus or completed primary study. Many paper notes are abstract-level; their read depth is disclosed. The three reviewer perspectives are authored critiques, not independent expert reviews.

Do not describe the required future evaluation as a missing positive result that can be filled with estimates. Negative and inconclusive results are valid outcomes. No task should silently turn an unknown into a preserved-science badge.
