# Fresh-process repeatability control

After the six pilot runs, [original replay orchestration](../../../benchmarks/replay_fashion_control.py) repeated the **lowest frozen seed,1101**, once for each arm with the unchanged training code, inputs and CUDA environment. This checks a previously unresolved repeatability concern; it adds no new independent seeds. [Raw replay records](fashion-repeatability.json) bind both orchestration and training source hashes.

Both replays completed five epochs. Each matches its original arm's initialization, split, sample-order and final parameter digests exactly. Every recorded epoch validation NLL, accuracy, correct count and successful-update count also matches exactly. The baseline and combined arms still differ from **each other** as reported in the pilot; repeatability within an arm does not imply equivalence between arms.

This is evidence of reproducibility for these two specific runs on the recorded environment, not all seeds, future releases, different kernels/platforms, arbitrary stochastic models or a full trajectory proof. Timing and memory context need not be identical, and timing equality is not asserted. The independent pilot count remains three pairs. No confidence interval, new equivalence margin or scientific-conclusion claim is created by repeating a seed.
