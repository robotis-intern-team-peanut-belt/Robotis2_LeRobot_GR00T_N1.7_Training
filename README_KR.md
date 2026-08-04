# 세 가지 policy 학습 pipeline

`training/`은 하나의 canonical LeRobot v3 dataset으로 세 baseline을 학습하는
단일 operator interface를 제공한다.

| Policy | Trainer | 입력 | Action 계약 | 기본 horizon |
| --- | --- | --- | --- | --- |
| LeRobot GR00T N1.7 | LeRobot | camera 3개, state, language | absolute 16-D | 40 |
| Isaac GR00T N1.7 | NVIDIA Isaac-GR00T | camera 3개, state, language | absolute 16-D | 40 |
| LeRobot ACT | LeRobot | camera 3개와 state | absolute 16-D | 100 |

각 구현에서 검토한 recommended default를 사용한다. Batch, augmentation,
objective, optimizer 세부값, ACT horizon이 다르므로 hyperparameter-matched
architecture 실험은 아니다. Policy family 사이의 raw training loss가 아니라
동일한 held-out task 결과와 failure category를 비교한다.

새 package를 이 server에서 곧바로 powered hardware에 사용하면 안 된다.
Dataset validation, offline inference/command 검사, shadow mode, bounded command,
E-stop 담당자, 저속 guarded trial은 별도 gate다.

## 구조

```text
training/
├── trainctl                         # operator CLI
├── grootctl                         # 기존 command 호환
├── registry.py                     # registry/launch/result lifecycle
├── backends/                       # backend별 command 생성
├── checkpoints.py                  # freeze/checksum/transfer/acknowledge
├── configs/{contracts,modalities,policies,comparisons,datasets,runs,queues}/
├── data_pipeline/                  # v3 검증과 Isaac v2.1 adapter
├── scripts/                        # 격리된 backend activation
├── artifacts/                     # Git 제외 dataset/package/receipt
├── runs/                           # Git 제외 학습 결과
└── tests/
```

Workspace root에서는 `training/trainctl ...`, 이 directory에서는
`./trainctl ...`를 실행한다.

## 1. Canonical LeRobot v3 dataset 등록

Server에 이미 upload한 dataset:

```bash
training/trainctl dataset register-local <dataset_name> \
  --source-root /absolute/path/to/lerobot_v3_dataset
```

Hugging Face dataset:

```bash
training/trainctl dataset register-hf <dataset_name> --repo-id User/Repo
```

두 command 모두 source identity와 정확한 F2 camera/state/action 순서를
고정한다. Byte 또는 robot/task 계약이 바뀌면 새 registration 이름과 directory를
사용한다.

## 2. 모든 backend 준비

```bash
training/trainctl dataset pipeline <dataset_name>
```

기본 `--for all`은 다음을 수행한다.

1. LeRobot GR00T와 ACT가 공유하는 immutable canonical LeRobot v3 derivative를
   검증/준비한다.
2. 별도 native Isaac-GR00T v2.1 derivative를 만든다.
3. `meta/modality.json`, 전용 language annotation column, native statistics를
   추가한다.
4. 각 backend runtime에서 실제 sample을 decode하여 camera 3개, 정확한 16-D
   state/action 순서, finite value, language, all-absolute config, horizon 40을
   확인한다.


Native derivative는 LeRobot codebase version `v2.1`을 기록하고
`meta/modality.json`을 포함하며 embodiment tag `NEW_EMBODIMENT`로 launch된다.
Training-owned Python modality config가 같은 camera key와 네 개 all-absolute action
group을 이 tag에 등록한다. 즉 v2.1 변환과 embodiment binding는 native training
전에 이 `dataset pipeline` 단계에서 함께 생성/검증된다.
Source tree는 in-place 변환하지 않는다. Native 준비는 receipt 기반이고 destination
단위 lock, idempotent 실행, atomic promotion을 사용한다. 격리된 Python 3.11
converter는 `training/.venv-isaac-converter`에 생성된다. NVIDIA가 pin한 converter는
현재 이전 LeRobot 전체 dependency를 가져와 약 5.5 GiB를 사용한다. 이는 training
data가 아니라 upstream compatibility 비용이다.

Debug 시 한 쪽만 지정할 수 있다.

```bash
training/trainctl dataset pipeline <dataset_name> --for lerobot
training/trainctl dataset prepare-isaac <dataset_name>
```

## 3. 비교 run 생성

```bash
training/trainctl run init <name> --dataset <dataset_name> --policy lerobot-groot
training/trainctl run init <name> --dataset <dataset_name> --policy isaac-groot
training/trainctl run init <name> --dataset <dataset_name> --policy lerobot-act
```

생성된 YAML은 `configs/policies/`의 policy recipe 하나를 상속한다. Run 목적,
hypothesis, 의도한 override만 수정한다. ACT는 language를 의도적으로 무시하고,
두 GR00T는 full-episode instruction을 사용한다. 세 policy 모두 기록된 absolute
target을 사용하며 delta conversion은 꺼져 있다.

검증과 실행:

```bash
training/trainctl run check <run>
training/trainctl run start <run>
training/trainctl run status <run>
```

`start`는 detached tmux를 사용하고 attach/log/status/명시적 stop command를
출력한다. Command 검사에는 `training/trainctl exec <run> --dry-run`, 매우 짧은
foreground debug에만 `training/trainctl exec <run>`를 사용한다.

## 4. Model package 동결과 전달

완료 상태이며 전체 model/processor를 저장한 run만 package한다.

```bash
training/trainctl checkpoint package <run> --name <immutable_package_name>
training/trainctl checkpoint verify <immutable_package_name>
```

Package는 `training/artifacts/models/<immutable_package_name>/`에 생성되는 직접
load 가능한 model root이며 `deployment_manifest.json`과 `SHA256SUMS`를 포함한다.
LeRobot package에는 pre/postprocessor와 statistics가, native Isaac package에는
model, `processor/`, `experiment_cfg/`가 있어야 한다.

Operator는 `rsync`를 직접 입력하지 않는다. Transfer command가 내부적으로 hidden
staging path와 rsync checksum/partial 기능을 사용하고 robot에서 모든 SHA-256을
검증한 뒤에만 atomically promote한다.

```bash
training/trainctl checkpoint transfer <immutable_package_name> \
  --target robotis@<robot-host> \
  --port <ssh-port> \
  --identity-file /path/to/key
```

`--dry-run`으로 계획만 확인할 수 있다. 기본 host workspace는
`/home/robotis/cyclo_intelligence/docker/workspace`이며, 비표준 설치에서만
`--robot-workspace`로 override한다. Staging directory 생성 전에 `trainctl`은
`/home/robotis/cyclo_intelligence`가 active mountpoint인지 확인하고, 선택한
workspace, home alias, `/mnt/ssd/cyclo_intelligence/workspace`의 device/inode가
동일한지 검증한다. 불일치하면 fail closed하고 검증된 identity를 receipt에 기록한다.

- LeRobot GR00T/ACT: `<robot-workspace>/model/lerobot/<package>`
  (container `/workspace/model/lerobot/<package>`)
- native Isaac GR00T: `<robot-workspace>/model/groot/<package>`
  (container `/workspace/model/groot/<package>`)

Transfer 성공은 consumer acknowledgment나 deployment 승인이 아니다. Robot이 만든
receipt는 다음으로 기록한다.

```bash
training/trainctl checkpoint acknowledge <package> \
  --consumer-receipt /path/to/robot_receipt.json
```

## 현재 한계

- Native Isaac은 NVIDIA 원본 launcher가 seed option을 노출하지 않으므로 현재
  seed 42를 사용한다. `trainctl`은 다른 native seed를 조용히 무시하지 않고
  거부한다.
- Native Isaac resume은 정확한 checkpoint/resume semantics를 통합 검증할 때까지
  `trainctl`이 의도적으로 거부한다.
- Native v2.1은 생성된 backend derivative이며 LeRobot v3만 source of truth다.
- NVIDIA 원본 N1.7 launcher는 global `use_relative_action` capability를 `True`로
  설정한다. 실제 변환은 `RELATIVE`로 표시된 modality group에만 적용된다. 이
  pipeline은 F2 action group 전부를 `ABSOLUTE`로 선언하고 검증하므로 해당
  capability는 비활성이다. Isaac-GR00T source는 수정하지 않는다.
- Recommended recipe 비교는 운영 baseline이지 controlled ablation이 아니다.
  Matched-budget 비교는 별도 이름의 comparison으로 만든다.
- Task 700008/700009 dataset, task-specific run/queue config, run evidence는
  `training/archive/task_700008_700009_native_camera/` 아래의 historical
  evidence다.
