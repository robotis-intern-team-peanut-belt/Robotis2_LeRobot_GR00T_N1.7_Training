# Task 700008/700009 native-camera lineage

This directory preserves the superseded Task 700008 and Task 700009 training
lineages. They are connection and historical evidence only; do not use their
dataset registrations or run configs for the next task dataset.

Contents:

- `configs/datasets/`: former active LeRobot v3 dataset registrations;
- `configs/runs/`: former training and three-policy connection-smoke configs;
- `configs/queues/`: former ordered Task 700009 campaigns;
- `datasets/`: preserved source/prepared/native-Isaac payloads, ignored by Git;
- `run_outputs/`: preserved logs, metrics, and any retained outputs, ignored by Git.

The payloads were moved without deletion on 2026-08-04. The reusable `trainctl`
backend recipes, Isaac v3-to-v2.1 adapter, and absolute-action modality tooling
remain active under `training/`; a future dataset must receive a new registration
and task contract before training.
