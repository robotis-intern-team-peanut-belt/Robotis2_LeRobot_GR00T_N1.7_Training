# Archived OMX Insert letterbox pipeline

This directory preserves the legacy `1280x720 -> 640x360 -> 640x480`
letterbox processor used by the 2026-07-24 dataset lineage. It is not the
current preprocessing path. New F2 data uses the native-640x480 identity
processor in `data_pipeline/prepare_omx_insert.py`.

Recommended layout:

```text
OMX_Insert/
├── datasets/   # Immutable downloaded/converted dataset versions
├── models/     # Selected complete deployable policies
├── runs/       # Training logs and intermediate checkpoints
└── download_hf_dataset.sh
```

Download the dataset:

```bash
cd /NHNHOME/WORKSPACE/26motir001_D/intern_team/training/archive/omx_insert_letterbox_v1
./download_hf_dataset.sh RobotisSW/Task_700007_OMX_Insert_MCAP_lerobot_v30
```

For reproducible work, pass a Hugging Face commit hash as the optional second
argument instead of using the mutable `main` revision:

```bash
./download_hf_dataset.sh RobotisSW/Task_700007_OMX_Insert_MCAP_lerobot_v30 COMMIT_SHA
```

Train into a new versioned directory under `runs/`. Copy only the selected complete
model, processor pipelines, statistics, configuration, dataset revision, and runtime
manifest into a versioned directory under `models/`.

## Reproduce the legacy LeRobot GR00T N1.7 derivative

```bash
./prepare_lerobot_groot_dataset.sh \
  --source ../../artifacts/datasets/Task_700007_OMX_Insert_MCAP_lerobot_v30 \
  --output ../../artifacts/datasets/Task_700007_OMX_Insert_MCAP_lerobot_v30_arms16_head-letterbox640x480_taskfix
```

Pass explicit `--source` and `--output` paths under `training/artifacts/datasets/`.
The result ends in `_arms16_head-letterbox640x480_taskfix`; the source is never
overwritten. Its `DERIVATION_RECEIPT.json` records the complete transform and
source revision.
