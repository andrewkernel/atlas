# Dynamic Tensor Rematerialization

- **Authors:** Marisa Kirisame, Steven Lyubomirsky, Altan Haan, Jennifer Brennan, Mike He, Jared Roesch, Tianqi Chen, Zachary Tatlock.
- **Year / venue:** 2021 (preprint 2020); ICLR 2021.
- **Verified primary URL:** [Dynamic Tensor Rematerialization](https://ztatlock.net/pubs/2021-iclr-dtr/).
- **Read depth:** Author abstract and project metadata inspected.
- **Objective:** Train with bounded activation memory.
- **Technical method:** Online tensor eviction and recomputation using lightweight metadata.
- **Main finding used:** Dynamic models need not rely on a static checkpoint schedule.
- **Limitations / Atlas assessment:** Prototype integration and observed schedules do not imply arbitrary Python side-effect safety.
- **Relevance:** Dynamic execution is existing prior art, not an Atlas invention.
- **Code status:** Author project links code; current build not audited.
- **Open question:** Which dynamic conditions defeat conservative contract checks?
- **Requires further verification:** Reproduce on current PyTorch before claiming deployability.

Retrieved in this research session; no independent replication is implied.
