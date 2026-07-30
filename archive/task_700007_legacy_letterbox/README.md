# Task 700007 legacy letterbox lineage

This directory preserves the superseded OMX Insert dataset lineage and its
reproducibility evidence. It is historical reference only. Do not use its
configs, processors, or checkpoints for new Task 700008 data.

The old source used a 1280×720 head camera and converted it to 640×480 with
letterboxing. New recordings use native 640×480 head and wrist cameras at
15 fps, so applying this processor would create the wrong camera contract.

Contents:

- `configs/`: original dataset, run, and queue YAMLs;
- `tooling/`: original letterbox preparation scripts;
- `weekend_queue/`: original queue wrapper scripts;
- `docs/`: status notes retained with the lineage;
- `datasets/`: 1.1 GB of preserved local payloads, ignored by Git;
- `run_outputs/`: 286 GB of generated logs/checkpoints, ignored by Git.

The ignored payloads were moved here without deletion on 2026-07-30. Their
dated project records remain under the workspace-level `docs/archive/`.
