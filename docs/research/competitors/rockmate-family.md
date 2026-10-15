# Rockmate family — update the comparator, not just the label

The [captured repository head](repository-snapshot.json), `c2f678455393c5d872a1adb099949fe25f682bd0`, has a [README](https://github.com/topal-team/rockmate/blob/c2f678455393c5d872a1adb099949fe25f682bd0/README.md) describing three relevant implementations:

|Member |Primary documented scope |Implication for Atlas |
|---|---|---|
|PureRockmate /2023 paper |Block/sequence rematerialization for supported structures |Original single-device scheduling comparator |
|HiRemate /[ICML2025](../papers/hiremate.md) |Hierarchical partition/solve/merge for broader graphs |Do not rely on the original sequential-like limitation as a family-wide gap |
|OffMate /[2024 preprint context](../papers/offmate.md) |Activation/weight offload plus recomputation and optimizer integration |Relevant if Atlas later targets persistent-state/offload bottlenecks |

These are documented capabilities, not tested compatibility. The examples mix names/imports(e.g. Hiremate versus PureHiremate) and omit some used names from shown imports. That is a reason to inspect actual exports/configuration, not evidence that the algorithms do not work. The README also supports configuration generation and construction from a configuration; a version-pinned integration needs more than copy/paste from an example.

The original Rockmate paper's activation-memory budget is not automatically a total process/device budget. Compare actual complete-update peaks, optimizer/host state and workspaces under explicit measurement definitions. The selected paper appendices handle aliases/views/in-place dependencies and RNG state; generic claims that Atlas alone considers recomputation semantics would be wrong.

No family implementation was installed or executed during this session. The team report supports the limited OffMate summary because the HAL paper page was inaccessible; it does not replace a future full-paper/code replication. Keep original captured metadata rather than silently pretending these details were inspected at the initial abstract-only stage.
