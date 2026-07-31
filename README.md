# GR00T N1.7 native-camera training

This repository owns the workstation path from a converted Cyclo LeRobot v3
dataset to a reproducible GR00T N1.7 training run. The active workflow targets
the current F2 OMX Insert robot/task contract. The superseded Task 700007 letterbox lineage is
preserved under `archive/task_700007_legacy_letterbox/`.

Do not run a newly trained policy on powered hardware from this workstation.
Dataset validation, offline action checks, a matching complete processor package,
shadow mode, bounded commands, an E-stop operator, and guarded low-speed trials
remain mandatory.

## Forward contract and defaults

- head, left wrist, and right wrist: native 640×480 H.264 at 15 fps;
- camera geometry: identity; no resize, crop, padding, or letterbox;
- state/action: exact named 16-D dual-arm and gripper order;
- stored and trained actions: absolute joint-position targets by default;
- video derivative: maximum GOP 15, CRF 18, preset `medium`;
- training images: uint8 plus at most two randomly ordered brightness, contrast,
  or saturation jitters in `[0.8, 1.2]`;
- model: GR00T N1.7 3B, BF16 autocast, FP32 trainable parameters, frozen language
  and vision backbones, horizon 40, action window 12 (pending robot-side latency
  and shadow validation);
- optimizer: AdamW `1e-4`, weight decay `1e-5`, cosine schedule, 5% warmup;
- workstation full-run default: batch 32, 8 workers, 20,000 steps/640,000 sampled
  examples, evaluation every 5,000 steps, checkpoint every 10,000 steps;
- tracking is disabled unless a run explicitly enables it.

GOP15 improved measured input throughput by 26.2% at batch 8 without changing
four-point mean held-out loss. Photometric augmentation improved clean held-out
loss at every measured batch-16 evaluation point, averaging 6.3%. Batch 32 reached
45.86 samples/s and the best four-point mean in the matched-sample comparison.
These are useful defaults, not evidence of closed-loop task success.

## Repository map

```text
training/
├── grootctl                       # registry/launch entry point
├── registry.py                    # dataset, config, queue, and result logic
├── data_pipeline/                 # native-format preparation and audits
├── scripts/                       # activation-aware wrappers
├── artifacts/                     # ignored local datasets/models
├── configs/
│   ├── datasets/                  # immutable dataset identities/contracts
│   ├── base/                      # forward defaults
│   ├── runs/                      # small experiment overrides
│   └── queues/                    # ordered run plans
├── templates/                     # dataset/run/queue starting points
├── runs/                          # ignored generated output
├── archive/                       # obsolete lineages; never active defaults
└── tests/
```

## 1. Choose one source workflow and register

The server supports two distinct sources. Both end in a tracked YAML under
`training/configs/datasets/`, and both use the same caption audit, immutable
16-D/GOP15 preparation, and decoded validation after registration. Never reuse a
registration name for different bytes or a different source revision.

### Workflow A: robot uploads to Hugging Face, server downloads

The robot-side Hub upload is outside this guide. After the dataset exists on the
Hub, run these commands from the server workspace root. Public datasets need no
login; for a private dataset, authenticate via the token prompt or `HF_TOKEN` and
never store a token in YAML:

```bash
hf auth login
hf auth whoami
training/grootctl dataset register-hf <dataset_name> --repo-id User/Repo
```

`register-hf` resolves `main` to the exact immutable 40-character Hub commit. A
branch, tag, or commit can be given explicitly with `--revision`. The server does
not trust a moving branch after registration.

### Workflow B: copy converted LeRobot v3.0 directly from the robot

After Cyclo finishes conversion on the robot, copy the completed LeRobot directory
to a new server directory. A normal `rsync` is sufficient; no Hugging Face upload,
checksum manifest, source commit, or UTC handoff ID is required.

For example, on the robot:

```bash
rsync -a --partial -e 'ssh -i /home/robotis/Robotis2_key -p 34301' \
  /home/robotis/cyclo_intelligence/docker/workspace/lerobot/<dataset_name>/ \
  k_humanoid_3@59.150.32.1:/NHNHOME/WORKSPACE/26motir001_D/intern_team/training/artifacts/datasets/<new_server_dataset_directory>/
```

Use a new destination directory rather than mixing a new dataset into an older
one. Once `rsync` finishes, register it on Robotis2:

```bash
training/grootctl dataset register-local <dataset_name> \
  --source-root /NHNHOME/WORKSPACE/26motir001_D/intern_team/training/artifacts/datasets/<new_server_dataset_directory>
```

That is the complete required registration command. It checks that the directory
and LeRobot `meta/info.json` exist, records the observed version/episode/frame
counts and current robot/task defaults, and creates
`training/configs/datasets/<dataset_name>.yaml`. It does not calculate or compare a
producer checksum.

If useful, ordinary provenance notes may be added without a naming convention:

```bash
training/grootctl dataset register-local <dataset_name> \
  --source-root <server_dataset_directory> \
  --source-revision <optional_label_or_commit> \
  --handoff <optional_existing_notes_file>
```

The earlier Task 700008 sample came from
`/home/robotis/cyclo_intelligence/docker/workspace/lerobot/Task_700008_OMX_Insert_F2_Intern_MCAP_lerobot_v30`.
That historical transfer used a stricter partial-directory and checksum procedure,
but those steps are not required for routine direct registration now.

### Shared registration options

Inspect the current F2 robot/task defaults before either workflow:

```bash
training/grootctl dataset defaults
```

The defaults and evidence are in
`training/configs/base/f2_omx_insert_contract.yaml`. For a same-robot, same-task
dataset, omit `--robot-revision` and `--task`. Override them when the dataset uses a
different contract. `--task` is the exact full-episode instruction delivered to
GR00T.

For Hub registration, `--repo-id` must identify a dataset repository and
`--revision` is a Hub branch/tag/commit. For direct registration,
`--source-revision` and `--handoff` are optional notes; neither is required or
validated against the copied bytes.

Both commands create `training/configs/datasets/<dataset_name>.yaml` without
processing the source. Review it before continuing:

```bash
training/grootctl dataset show <dataset_name>
```

## 2. Acquire or recheck the source, audit, prepare, and validate

The normal command is:

```bash
training/grootctl dataset pipeline <dataset_name>
```

It runs source acquisition, dataset preflight, task-specific caption audit,
preparation, and prepared-data validation in order, stopping at the first failure. It
is safe to repeat: a registered local source is reused, and an existing prepared
derivative is reused only when its derivation receipt matches the registration.

### `dataset download`

```bash
training/grootctl dataset download <dataset_name>
```

For a Hub registration, this reads `repo_id` and the pinned commit, downloads that
exact snapshot into `training/artifacts/datasets/<dataset_name>_source`, and writes:

- a per-file SHA-256 manifest under `training/artifacts/manifests/`;
- `.groot_registry_download.json` inside the source, recording repository, commit,
  manifest digest, file count, and byte count.

For a direct registration, it performs no network or checksum operation. It confirms
that the directory is the one recorded by `register-local` and reuses it.

Neither mode edits the LeRobot payload. The later preflight, caption, preparation,
and decode stages still validate the dataset structure needed by this training
pipeline.

### Dataset preflight

Run the detection-only check directly against the registered source or prepared
training derivative:

```bash
training/grootctl dataset preflight <dataset_name> --stage source
training/grootctl dataset preflight <dataset_name>
```

It rejects leading/trailing or repeated caption spaces, tabs, newlines, non-breaking
spaces, and other non-ASCII whitespace across vocabularies, episode metadata, and
annotation JSON. It also verifies globally contiguous frame indices, per-episode
frame indices, declared counts, and that every episode's metadata length and global
`dataset_from_index`/`dataset_to_index` bounds match its actual data rows. It reports
the first exact location and never edits data. `dataset pipeline`, `dataset validate`,
`train check`, and `train start` run this guard automatically.

### Caption audit (`audit_omx_insert_captions.sh`)

The pipeline runs this audit automatically from the `validation.caption_audit` block.
To diagnose it separately:

```bash
training/scripts/audit_omx_insert_captions.sh \
  --dataset training/artifacts/datasets/<dataset_name>_source \
  --language-mode full_episode \
  --expected-subtasks-per-episode 5 \
  --expected-overall-task "<approved full-episode instruction>"
```

The audit is read-only. It cross-checks `meta/tasks.parquet`,
`meta/subtasks.parquet`, frame-level `task_index`/`subtask_index`, episode metadata,
and each episode annotation JSON. The required current mapping is:

- every frame resolves through `task_index` to the one approved episode prompt;
- each episode contains five contiguous subtask ranges in order `0,1,2,3,4`;
- subtask text and half-open frame ranges agree in parquet, episode metadata, and
  annotation JSON.

The MP4 files contain images, not captions. Combining episodes into an MP4 does not
remove frame-level language lookup. The five subtask strings are retained annotations;
they are not switched into the GR00T prompt. On any missing, reordered, overlapping,
or mismatched range/text, the pipeline stops before processing.

### `dataset prepare`

```bash
training/grootctl dataset prepare <dataset_name>
```

This creates a separate immutable derivative ending in `_arms16_gop15`; it never
modifies the registered Hub or direct-transfer source. The processor:

- selects the exact named 16-D dual-arm/gripper state and absolute action fields;
- verifies all three camera features and encoded streams are 640×480 at 15 fps;
- preserves geometry exactly—no resize, crop, padding, or letterbox;
- copies an already-compliant H.264 stream byte-for-byte, otherwise re-encodes it
  with maximum GOP 15, CRF 18, and preset `medium`;
- writes `DERIVATION_RECEIPT.json` with source commit/manifest, dimensions, names,
  task text, camera inventory, decoded frame counts, GOPs, and copy/transcode counts.

If the output already exists, `grootctl` requires its receipt to match the source
path, pinned revision, and manifest before reusing it. It never overwrites a
mismatched derivative.

### `dataset validate --decode`

```bash
training/grootctl dataset validate <dataset_name> --decode
```

This checks the prepared metadata against the registered contract: schema version,
episode/frame counts when declared, 15 fps, camera key order and shapes, and exact
state/action dimensions and joint-name order. `--decode` also opens the dataset with
TorchCodec, decodes a real sample, and requires finite 16-D state/action values.

A pass proves that files connect correctly to the trainer. It does not judge
teleoperation quality, task success, collisions, timing quality, or whether a policy
is safe to deploy; those still require human dataset review and later evaluation.

## 3. Create, check, launch, and monitor training

The normal training flow is four commands:

```bash
training/grootctl train init <run_name> --dataset <dataset_name>
training/grootctl train check <run_name>
training/grootctl train start <run_name>
training/grootctl train status <run_name>
```

### `train init`

This creates `training/configs/runs/<run_name>.yaml`. The file is intentionally small:
it points to the dataset registration and inherits the reviewed defaults from
`configs/base/groot_n17.yaml`. It does not launch training. Replace the generated
`REPLACE ME` description, hypothesis, and annotation notes with dataset-specific
goals, a falsifiable expectation, evaluation criteria, caveats, and intended
comparisons. The generated owner is `intern-team`.

`train init` does not emit run-level tags. The resolved run inherits
`tags: [groot-n17]` from the base. Tags are registry metadata retained in the
resolved configuration and covered by its manifest hash; they are not passed to `lerobot-train`.
YAML lists replace inherited lists rather than adding to them, so add a run-level
`tags` list only when deliberately replacing the base tags. Ordinary baseline use
does not require copying policy or optimizer settings into the run file.

The defaults are absolute actions, GOP15/native-camera input, photometric augmentation,
batch 32, 8 workers, BF16 compute with FP32 trainable parameters, 20,000 steps, and
tracking disabled. A run YAML should override a field only for a deliberate
experiment. For example, a lower-memory run can add:

```yaml
train:
  batch_size: 16
  num_workers: 8
```

### `train check`

This resolves the base YAML and dataset registration, injects the exact camera and
joint contract, reruns the detection-only dataset preflight, rejects inconsistent
steps/learning rates/action settings, checks free disk, renders the final
`lerobot-train` command, and prints future tmux/log/status commands. It is a dry run:
no model is loaded, no GPU training starts, and no tmux session is created.

Run `train check` again after every YAML edit.

### `train start`

This repeats validation, checks visible GPU memory, and launches the exact checked
command in a detached tmux session. Training continues if SSH disconnects. The command
prints how to attach and detach, tail the log, inspect status, and intentionally stop
the session. Generated run state lives under `training/runs/<run_name>/`, including:

- the source and fully resolved configs;
- the exact command and source/dataset manifest;
- `train.log`;
- `result.yaml` with exit state, final logged metrics, and saved checkpoint endpoint;
- checkpoints and serialized pre/postprocessors under `output/`.

Do not run `train start` twice with the same output directory. Use a new run name, or
use the lower-level resume option only after checking the saved training state.

### `train status`

This prints the durable `result.yaml`. While training is active it reports `running`;
a normal wrapper exit records `completed` or `failed`, the exit code, timing, final
logged loss/throughput/memory, and the last saved checkpoint. For live progress, use
the `tail -f` or `tmux attach` command printed by `train start`.

W&B tracking is enabled by default for the
[`robotis-intern-team-peanut-belt/gr00t-omx-insert`](https://wandb.ai/robotis-intern-team-peanut-belt/gr00t-omx-insert)
project. Before the first tracked launch, activate LeRobot and authenticate locally:

```bash
cd /NHNHOME/WORKSPACE/26motir001_D/intern_team/lerobot
source ./activate_lerobot.sh
wandb login
```

`train check` renders the inherited entity, project, and online mode without contacting
W&B. `train start` requires valid W&B authentication and network access. To disable
tracking deliberately for a bounded smoke or other local-only run, add:

```yaml
tracking:
  backend: none
  enable: false
  mode: disabled
```

Hugging Face and W&B authentication are separate. Never store either token in tracked
configuration.

### Sequential multi-run queues

Validate a queue and every resolved run without launching training:

```bash
training/grootctl queue Task_700009_model_0_then_model_1_chunk20 --dry-run
```

The queue runner is foreground-only. For an SSH-safe launch, run it inside tmux:

```bash
cd /NHNHOME/WORKSPACE/26motir001_D/intern_team
tmux new-session -s task700009-chunk-compare
# After tmux opens:
training/grootctl queue Task_700009_model_0_then_model_1_chunk20
# Detach without stopping it: Ctrl-b, then d
```

Monitor the queue state and the active attempt from another shell:

```bash
cat training/runs/_queues/Task_700009_model_0_then_model_1_chunk20/queue_status.yaml
tail -f training/runs/<active_run_name>/attempt-1/train.log
```

The inherited multi-queue defaults allow one attempt per run, require 100 GiB free,
use a 24-hour queue budget, and stop if the first run fails. The current queue runner
has no dedicated stop command; do not kill its tmux session casually while a trainer
may still be active.

The lower-level `config`, `run`, `run-tmux`, `queue`, and `result` commands remain
available for experiments and automation; new operators normally need only the
commands above.

## Outputs and safety boundary

A completed workstation run is still only a training/offline-evaluation artifact.
Before any robot use, load the exact checkpoint with its saved processors, verify
finite actions and the 16-D/camera contract, package checksums, and follow the robot
handoff, shadow, bounded-command, E-stop, and guarded low-speed gates in `AGENTS.md`.
Training loss alone is never deployment evidence.

## Environment and tests

`grootctl` and the wrappers activate the repository-local LeRobot environment.
For an interactive shell:

```bash
cd /NHNHOME/WORKSPACE/26motir001_D/intern_team/lerobot
source ./activate_lerobot.sh
```

Run repository checks from `training/`:

```bash
python3 -m pytest -q
python -m compileall -q data_pipeline registry.py
```

## Git and large artifacts

Source, configs, templates, tests, small receipts, and documentation are tracked.
Datasets, checkpoints, generated runs, caches, and the large legacy payloads are
ignored. Review with `git status --short --ignored` before making your own commit.
This repository has intentionally not been committed by the setup agent.
