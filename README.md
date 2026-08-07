# Three-policy training pipeline

`training/` provides one operator interface for three baselines on the same
canonical LeRobot v3 dataset:

| Policy | Trainer | Inputs | Action contract | Default horizon |
| --- | --- | --- | --- | --- |
| LeRobot GR00T N1.7 | LeRobot | 3 cameras, state, language | absolute 16-D | 40 |
| Isaac GR00T N1.7 | NVIDIA Isaac-GR00T | 3 cameras, state, language | absolute 16-D | 40 |
| LeRobot ACT | LeRobot | 3 cameras and state | absolute 16-D | 100 |

The recipes use each implementation's reviewed recommended defaults. They are
not a hyperparameter-matched architecture experiment: batch sizes, augmentation,
objectives, optimizer details, and ACT's horizon differ. Compare held-out task
outcomes and failure categories; do not compare raw training loss across policy
families.

Never run a new package on powered hardware directly from this server. Dataset
validation, offline inference and command checks, shadow mode, bounded commands,
an E-stop operator, and guarded low-speed trials remain separate gates.

## Layout

```text
training/
├── trainctl                         # operator CLI
├── grootctl                         # backwards-compatible legacy CLI
├── registry.py                     # registry, launch, and result lifecycle
├── backends/                       # backend-specific command rendering
├── checkpoints.py                  # freeze, checksum, transfer, acknowledge
├── configs/
│   ├── contracts/                  # robot/task contract
│   ├── modalities/                 # native Isaac modality mapping
│   ├── policies/                   # one recommended recipe per policy
│   ├── comparisons/                # comparison intent and limitations
│   ├── datasets/                   # immutable dataset registrations
│   ├── runs/                       # small experiment overrides
│   └── queues/                     # ordered campaigns
├── data_pipeline/                  # v3 validation and Isaac v2.1 adapter
├── scripts/                        # isolated backend activation
├── artifacts/                     # ignored datasets/packages/receipts
├── runs/                           # ignored generated training results
└── tests/
```

Run commands from the workspace root as `training/trainctl ...`, or from this
directory as `./trainctl ...`.

## Local W&B authentication

Paste the API key into `training/.env.secret`:

```bash
WANDB_API_KEY="paste-your-key-here"
```

The file is Git-ignored, restricted to the local workspace, and loaded by both
`training/activate_groot.sh` and `training/activate_lerobot.sh`. Training launched
through `trainctl` also passes through those activation scripts. This avoids a
user-wide `wandb login` entry in `~/.netrc`. A run still follows its resolved
`tracking.mode`; an explicitly offline run does not become online just because a
key is present.

## 1. Register one canonical LeRobot v3 dataset

For a dataset already uploaded to this server:

```bash
training/trainctl dataset register-local <dataset_name> \
  --source-root /absolute/path/to/lerobot_v3_dataset
```

For a Hugging Face dataset:

```bash
training/trainctl dataset register-hf <dataset_name> --repo-id User/Repo
```

Both commands pin source identity and the exact F2 camera/state/action order.
Use a new registration name and directory whenever bytes or the robot/task
contract change.

## 2. Prepare every backend

```bash
training/trainctl dataset pipeline <dataset_name>
```

The default `--for all` flow:

1. validates and prepares the immutable canonical LeRobot v3 derivative used by
   LeRobot GR00T and ACT;
2. creates a separate native Isaac-GR00T v2.1 derivative;
3. adds `meta/modality.json`, a dedicated language annotation column, and native
   statistics; and
4. decodes a real sample in each owning runtime and checks the three cameras,
   exact 16-D state/action order, finite values, language, all-absolute action
   config, and horizon 40.


The native derivative reports LeRobot codebase version `v2.1`, contains
`meta/modality.json`, and is launched with embodiment tag
`NEW_EMBODIMENT`. The training-owned Python modality config registers that tag
with the same camera keys and four all-absolute action groups. In other words,
the v2.1 conversion and embodiment binding are both created/checked by this
`dataset pipeline` step, before native training is allowed.
The source tree is never converted in place. Native preparation is receipt-bound,
locked per destination, idempotent, and atomically promoted. Its isolated Python
3.11 converter environment is created at `training/.venv-isaac-converter`; the
NVIDIA-pinned converter currently pulls a full older LeRobot dependency set and
uses about 5.5 GiB. This is an upstream compatibility cost, not training data.

Target one side when debugging:

```bash
training/trainctl dataset pipeline <dataset_name> --for lerobot
training/trainctl dataset prepare-isaac <dataset_name>
```

## 3. Create comparable runs

```bash
training/trainctl run init <name> --dataset <dataset_name> --policy lerobot-groot
training/trainctl run init <name> --dataset <dataset_name> --policy isaac-groot
training/trainctl run init <name> --dataset <dataset_name> --policy lerobot-act
```

Each generated YAML inherits one file under `configs/policies/`; edit only the
run purpose, hypothesis, and deliberate overrides. ACT deliberately ignores
language. Both GR00T recipes consume the full-episode instruction. All three
consume recorded absolute targets; no delta conversion is enabled.

Validate and launch:

```bash
training/trainctl run check <run>
training/trainctl run start <run>
training/trainctl run status <run>
```

`start` uses detached tmux and prints attach, log, status, and explicit stop
commands. Use `training/trainctl exec <run> --dry-run` for command inspection and
`training/trainctl exec <run>` only for short foreground debugging.

## 4. Pull a model from the robot

`trainctl` does not connect to or push models to the robot. Run the robot-side
command with the absolute path of a complete saved model on this server:

```bash
modelctl pull /absolute/server/path/to/model
```

`modelctl` selects the files needed for inference, shows rsync progress, resumes
an interrupted copy, and installs the result in the appropriate robot model
directory:

- LeRobot GR00T and ACT: `/workspace/model/lerobot/<model>`
- native Isaac GR00T: `/workspace/model/groot/<model>`

Pass the directly loadable model root; a deployment manifest or checksum package
is not required. See `/home/robotis/MODELCTL.md` on the robot for examples and
the exact file-selection rules.

The server-side package and verify commands remain available when an immutable
archive is specifically needed, but they are not part of the normal download
workflow:

```bash
training/trainctl checkpoint package <run> --name <immutable_package_name>
training/trainctl checkpoint verify <immutable_package_name>
```

## Known limits

- Native Isaac currently uses seed 42 because NVIDIA's unmodified launcher does
  not expose a seed option; `trainctl` rejects other native seed values instead
  of silently ignoring them.
- Native Isaac resume is intentionally rejected by `trainctl` until its exact
  checkpoint/resume semantics are integrated and tested.
- The native v2.1 tree is a generated backend derivative; LeRobot v3 remains the
  only accepted source of truth.
- NVIDIA's unmodified N1.7 launcher sets its global `use_relative_action`
  capability to `True`. Conversion still runs only for modality groups marked
  `RELATIVE`; this pipeline declares and validates every F2 action group as
  `ABSOLUTE`, so the capability is inactive. The Isaac-GR00T source stays clean.
- Recommended recipes are operational baselines, not controlled scientific
  ablations. A matched-budget comparison should be a separate named comparison.
- The Task 700008/700009 datasets, task-specific run/queue configs, and run
  evidence are historical-only under
  `training/archive/task_700008_700009_native_camera/`.
