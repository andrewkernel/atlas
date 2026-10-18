# Prospective resource-recovery amendment

2026-10-18, before any campaigns using this amended source identity. The original
32k attempt `tune-53a712409265` stopped after 11 completed workers and one
resource-limited offloading trial. It did not select, confirm or apply a profile.
The [interruption receipt](32k-resource-interruption.json) confirms closed owned
processes and recovered available RAM. Its exact low-RAM sample was not retained;
peak process/job commit is not RSS and does not establish the allocation's cause.

The controller previously aborted the entire search when one worker hit the RAM
guard. That prevents a finite tuner from rejecting an infeasible candidate and
evaluating the remaining settings. The corrected policy keeps the 2 GiB available
RAM guard, closes the offending owned worker, records the sampled headroom, and
waits up to 30 seconds for at least 4 GiB available RAM before continuing the
remaining registered cases. Cancellation/deadline checks still apply while
waiting. Failure to recover stops the campaign. A resource-limited case has
unknown numerical eligibility and cannot win.

The full-case-grid storage estimate was also too restrictive for repeated
content-addressed output storage on this machine. Replace the up-front whole-grid
free-space requirement with an explicit rolling reserve before each worker: two
full 32-by-151,936 FP32 arrays, 16 MiB metadata allowance and 128 MiB free-space
headroom. During a worker, require one full array plus the same 128 MiB headroom.
Keep the undeduplicated whole-grid estimate visible. A disk-headroom failure
stops collection; it cannot skip arrays or qualify incomplete evidence. These are
advisory space checks, not an exclusive storage reservation against other apps.

All registered methods, contexts, precisions, input windows, full-vocabulary
outputs, numerical tolerances, warmup/timing definitions and confirmation rules
remain fixed. Every trial still executes before lossless deduplication. No
candidate is removed because it failed. New protocols declare the recovery and
rolling-reserve rules, and affected required 2k/32k workflows run again with new
IDs and source hashes. Earlier complete and interrupted records remain historical
evidence under their original frozen sources.

Before changing execution source, the two previously active 2k profile pointers
were explicitly reverted to their prior default state. Their immutable artifacts
and apply/revert journals remain. Historical profiles cannot silently activate
under the amended source identity.
