# 2026-07-27 overnight information-gain campaign

This folder is a confirmation-only, matched-sample proposal. Every numbered
training run processes 240,000 examples and evaluates at 60k, 120k, 180k, and
240k sample positions. `00_common.yaml` is inheritance-only and must not be
launched directly.

Comparison graph:

- `01` versus `02`: GOP250 source video versus GOP15, both batch 8.
- `02` versus `03` versus `05`: batch 8/16/32 on the same GOP15 data.
- `03` versus `04`: no augmentation versus the corrected weekend brightness/contrast/saturation augmentation.

Only `03`, `04`, and `05` retain endpoint checkpoints. The two batch-8 control
runs retain W&B learning curves but no 24 GiB checkpoint, limiting projected
new storage to roughly 72 GiB.

The queue runs a 20-step augmentation smoke test first, then `01`, `02`, `03`,
`04`, and finally stretch candidate `05`.

Launch status: started at `2026-07-27T08:54:40Z` under detached supervisor PID `2272799`. The photometric smoke completed 20/20 steps with exit code 0, finite loss/gradient, and 36.12 GiB reported GPU memory. The queue then advanced to run `01`.

Completion status: all six entries completed successfully at
`2026-07-27T20:54:25Z`, after 11h59m45s. Results and recommendations are in
`docs/archive/2026-07-28_groot_n17_overnight_information_gain_results.md`.
