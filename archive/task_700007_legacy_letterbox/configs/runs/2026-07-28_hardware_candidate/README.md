# 2026-07-28 OMX hardware-candidate training

This campaign contains one full-budget policy candidate:

- immutable GOP15 OMX dataset;
- batch 32 and workers 8;
- two randomly selected brightness/contrast/saturation transforms in
  `[0.8, 1.2]`;
- relative arm actions and absolute gripper actions;
- seed 42, LR `1e-4`, and 20,000 optimizer steps;
- 640,000 sampled examples;
- clean held-out evaluation every 160,000 samples;
- checkpoints at 320,000 and 640,000 samples.

Expected workstation training time is 4–4.5 hours; the timeout is 5.5 hours.
The validated run launched detached at `2026-07-28T00:27:34Z` (launcher PID
`2025731`, W&B run `sqgym53u`). A successful checkpoint is not approved for
powered hardware until repeated held-out evaluation, offline action checks,
Jetson loading/latency, shadow mode, and guarded low-speed trials pass.
