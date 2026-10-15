# TrainCheck: Training with Confidence

Yuxuan Jiang, Ziming Zhou, Boyu Xu, Beijie Liu, Runhui Xu and Peng Huang. OSDI 2025, pp. 313–329. [Publisher page](https://www.usenix.org/conference/osdi25/presentation/jiang), [official PDF](https://www.usenix.org/system/files/osdi25-jiang.pdf), [implementation](https://github.com/OrderLab/TrainCheck). The [deeper-read capture](traincheck-deeper-read.json) identifies PDF bytes and selected pages; pages 13/14 were also visually inspected. No TrainCheck execution occurred.

It learns context-sensitive semantic invariants from sample traces, then checks target execution. Section 3.6 defines safe preconditions relative to observed passing/failing examples; §3.7 filters superficial invariants. This already overlaps explicit conditional checks, not universal scientific certification.

The authors detect 18/20 reproduced errors. Two misses concern intended training-step primitives and checkpoint-local state. The 63-program false-positive study differs from selection-policy risk; transferability excludes invariants previously raising false positives. Section 5.5 conditions detection analysis on errors detected earlier. Keep those populations explicit.

Section 5.6 evaluates 100 sampled invariants per program; Figure 10 includes roughly 1.6× slowdown on a toy. Offline inference reaches 38 hours in §5.7. Online overhead is not total acquisition/inference cost.

Atlas should compare/reuse applicable checking and charge its declared costs fairly. These paper-reported results do not establish current compatibility, local performance, held-out policy superiority or preserved scientific conclusions. Full replication remains open.
