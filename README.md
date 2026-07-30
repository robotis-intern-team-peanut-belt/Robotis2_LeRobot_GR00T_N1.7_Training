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

## 1. Connect Hugging Face and register the dataset

Run commands from the workspace root. Public datasets can be read anonymously. For
a private dataset, authenticate once; use the token prompt or `HF_TOKEN`, and never
write a token into a YAML file:

```bash
hf auth login
hf auth whoami
```

For another dataset from the same F2 robot/task lineage, the normal registration
needs only a local name and the Hugging Face dataset repository:

```bash
training/grootctl dataset register-hf <dataset_name> --repo-id User/Repo
```

The command prints the pinned Hub commit, robot revision, task, where each value came
from, and the next commands. Before using it, you can inspect the defaults:

```bash
training/grootctl dataset defaults
```

The defaults are stored with comments and their evidence in
`training/configs/base/f2_omx_insert_contract.yaml`. For the usual same-robot,
same-task dataset, leave them unchanged.

The arguments mean:

- `dataset_name` is a short local name used by later commands. Choose a new name for
  every distinct Hub dataset/commit.
- `--repo-id User/Repo` is the Hugging Face dataset repository, not a model repository.
- `--revision` is a Hugging Face branch, tag, or commit—not a robot revision. Usually
  omit it: it defaults to `main`, which `grootctl` immediately resolves and records as
  an immutable 40-character commit.
- `--robot-revision` identifies the robot-side source/config that recorded the data.
  Usually omit it: the current approved F2 value comes from the defaults file. Override
  it only when a producer handoff explicitly gives a different revision.
- `--task` is the exact full-episode instruction supplied to GR00T. Usually omit it for
  the same OMX Insert task. Override it only when the task contract truly changed.

An unusual dataset with a different producer contract can override the optional
values explicitly:

```bash
training/grootctl dataset register-hf <dataset_name> \
  --repo-id User/Repo \
  --revision <branch-tag-or-commit> \
  --robot-revision <revision-from-producer-handoff> \
  --task "<approved full-episode instruction>"
```

Registration creates `training/configs/datasets/<dataset_name>.yaml`; it does not
transfer or process data. The YAML records whether robot revision/task came from the
project defaults or command line, plus the pinned Hub commit and current F2 contract:
three native 640×480/15 fps cameras, absolute named 16-D state/action, GOP15 processing,
and five ordered subtask annotations. Review it with:

```bash
training/grootctl dataset show <dataset_name>
```

Do not point an existing registration at replacement data; use a new dataset name.

## 2. Download, audit, prepare, and validate

The normal command is:

```bash
training/grootctl dataset pipeline <dataset_name>
```

It runs the four stages below in order and stops at the first failure. It is safe to
repeat: an existing source or derivative is reused only when its provenance marker
or derivation receipt matches the registration.

### `dataset download`

```bash
training/grootctl dataset download <dataset_name>
```

This reads `repo_id` and the pinned commit from the dataset YAML, downloads that exact
Hub snapshot into `training/artifacts/datasets/<dataset_name>_source`, and writes:

- a per-file SHA-256 manifest under `training/artifacts/manifests/`;
- `.groot_registry_download.json` inside the source, recording repository, commit,
  manifest digest, file count, and byte count.

It refuses a non-empty directory without a matching marker, which prevents a partial
or unrelated upload from being mistaken for the registered dataset. It does not edit
the downloaded LeRobot files.

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
modifies the Hub source. The processor:

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
`configs/base/groot_n17.yaml`. It does not launch training. The generated description,
hypothesis, tags, and absolute-action settings form a usable baseline. Edit the human
fields to add dataset-specific goals and evaluation criteria; ordinary baseline use
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
joint contract, rejects inconsistent steps/learning rates/action settings, checks the
dataset and free disk, renders the final `lerobot-train` command, and prints the
future tmux/log/status commands. It is a dry run: no model is loaded, no GPU training
starts, and no tmux session is created.

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

Tracking is disabled by default. To use Weights & Biases, activate LeRobot and log in:

```bash
cd /NHNHOME/WORKSPACE/26motir001_D/intern_team/lerobot
source ./activate_lerobot.sh
wandb login
```

Then set `tracking.backend: wandb`, `tracking.enable: true`, `tracking.mode: online`,
and the project/entity in the run YAML before `train check`. Hugging Face and W&B
authentication are separate. Never store either token in tracked configuration.

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
