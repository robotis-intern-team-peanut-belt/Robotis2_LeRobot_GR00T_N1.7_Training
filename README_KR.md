# GR00T N1.7 native-camera 학습

이 저장소는 Cyclo가 변환한 LeRobot v3 데이터셋에서 재현 가능한 GR00T
N1.7 학습까지의 워크스테이션 경로를 관리한다. 활성 workflow는 현재 F2 OMX Insert robot/task 계약을 대상으로 한다. 폐기된 Task 700007 letterbox
lineage는 `archive/task_700007_legacy_letterbox/`에 보존했다.

이 워크스테이션에서 만든 새 policy를 곧바로 powered hardware에서 실행하면
안 된다. 데이터셋 검증, offline action 검사, 일치하는 전체 processor package,
shadow mode, bounded command, E-stop 담당자, 저속 guarded trial이 필수다.

## 앞으로의 계약과 기본값

- head/left wrist/right wrist: native 640×480 H.264, 15 fps
- camera geometry: identity; resize/crop/padding/letterbox 없음
- state/action: 이름과 순서가 고정된 16-D dual-arm/gripper
- 저장 및 학습 action: 기본적으로 absolute joint-position target
- video derivative: 최대 GOP 15, CRF 18, preset `medium`
- training image: uint8, brightness/contrast/saturation `[0.8, 1.2]` 중 최대
  두 개를 임의 순서로 적용
- model: GR00T N1.7 3B, BF16 autocast, FP32 trainable parameter, language와
  vision backbone freeze, horizon 40, action window 12(robot-side latency와
  shadow validation 전에는 powered execution 금지)
- optimizer: AdamW `1e-4`, weight decay `1e-5`, cosine, warmup 5%
- full-run 기본값: batch 32, worker 8, 20,000 step/640,000 sample,
  5,000 step마다 evaluation, 10,000 step마다 checkpoint
- run에서 명시적으로 켜지 않는 한 tracking 비활성화

이전 실험에서 GOP15는 batch 8 입력 처리량을 26.2% 높이면서 네 evaluation
평균 loss를 유지했다. Photometric augmentation은 batch 16의 모든 clean
held-out 지점에서 개선되어 평균 6.3% 향상했다. Batch 32는 45.86 samples/s와
matched-sample 비교의 최저 네 지점 평균을 기록했다. 이는 좋은 기본값의
근거이지 closed-loop 성공의 증거가 아니다.

## 저장소 구조

```text
training/
├── grootctl                       # registry/launch 진입점
├── registry.py                    # dataset/config/queue/result 로직
├── data_pipeline/                 # native 준비와 audit
├── scripts/                       # 환경 활성화 wrapper
├── artifacts/                     # Git에서 제외되는 dataset/model
├── configs/{datasets,base,runs,queues}/
├── templates/
├── runs/                          # 생성 output, Git 제외
├── archive/                       # 폐기 lineage, 활성 기본값으로 사용 금지
└── tests/
```

## 1. Hugging Face 연결과 데이터셋 등록

Workspace root에서 실행한다. Public dataset은 인증 없이 읽을 수 있다. Private
dataset은 token prompt 또는 `HF_TOKEN`을 사용하고 token을 YAML에 기록하지 않는다.

```bash
hf auth login
hf auth whoami
```

같은 F2 robot/task lineage의 새 dataset은 local 이름과 Hugging Face dataset repo만
입력하면 된다.

```bash
training/grootctl dataset register-hf <dataset_name> --repo-id User/Repo
```

명령은 pinned Hub commit, robot revision, task, 각 값의 출처와 다음 명령을
출력한다. 사용 전에 현재 기본값을 확인할 수 있다.

```bash
training/grootctl dataset defaults
```

기본값과 evidence는 comment가 있는
`training/configs/base/f2_omx_insert_contract.yaml`에 있다. 같은 robot과 같은 task의
일반 dataset은 이 값을 변경하지 않는다.

Argument의 의미는 다음과 같다.

- `dataset_name`: 이후 명령에서 사용할 짧은 local 이름이다. 다른 Hub
  dataset/commit마다 새 이름을 사용한다.
- `--repo-id User/Repo`: model repo가 아니라 Hugging Face dataset repo다.
- `--revision`: robot revision이 아니라 Hugging Face branch/tag/commit이다. 보통
  생략한다. Default `main`을 즉시 immutable 40자리 commit으로 resolve하여 기록한다.
- `--robot-revision`: 데이터를 기록한 robot-side source/config 식별자다. 보통
  생략하여 current F2 project default를 사용한다. Producer handoff가 다른 값을
  명시할 때만 override한다.
- `--task`: GR00T에 제공할 정확한 full-episode instruction이다. 같은 OMX Insert
  task이면 생략한다. Task 계약이 실제로 변경된 경우에만 override한다.

Producer 계약이 다른 특수 dataset은 optional 값을 명시할 수 있다.

```bash
training/grootctl dataset register-hf <dataset_name> \
  --repo-id User/Repo \
  --revision <branch-tag-or-commit> \
  --robot-revision <revision-from-producer-handoff> \
  --task "<approved full-episode instruction>"
```

Registration은 `training/configs/datasets/<dataset_name>.yaml`만 만들며 data를
transfer/process하지 않는다. YAML에는 robot revision/task가 project default인지
command line인지, pinned Hub commit, 세 native 640×480/15 fps camera, absolute named
16-D state/action, GOP15 processing, 다섯 ordered subtask annotation 계약이 기록된다.
다음으로 검토한다.

```bash
training/grootctl dataset show <dataset_name>
```

기존 registration의 source를 교체하지 말고 다른 dataset 이름을 사용한다.

## 2. Download, caption audit, prepare, validate

일반적으로 다음 한 명령을 사용한다.

```bash
training/grootctl dataset pipeline <dataset_name>
```

아래 네 단계를 순서대로 실행하며 하나라도 실패하면 즉시 중단한다. Source
marker와 derivative receipt가 registration과 일치할 때만 기존 결과를 재사용하므로
동일 명령을 안전하게 다시 실행할 수 있다.

### `dataset download`

```bash
training/grootctl dataset download <dataset_name>
```

Dataset YAML의 `repo_id`와 pinned commit을 읽어 정확한 Hub snapshot을
`training/artifacts/datasets/<dataset_name>_source`에 받는다. 또한 다음을 만든다.

- `training/artifacts/manifests/`의 파일별 SHA-256 manifest
- source 안의 `.groot_registry_download.json`: repo, commit, manifest digest,
  file count, byte count 기록

일치하는 marker가 없는 non-empty directory는 거부한다. Partial upload나 다른
payload를 등록 dataset으로 오인하지 않기 위해서다. Downloaded LeRobot 파일은
수정하지 않는다.

### Caption audit (`audit_omx_insert_captions.sh`)

Pipeline은 dataset YAML의 `validation.caption_audit` 설정으로 자동 실행한다.
문제를 따로 확인할 때는 다음을 실행한다.

```bash
training/scripts/audit_omx_insert_captions.sh \
  --dataset training/artifacts/datasets/<dataset_name>_source \
  --language-mode full_episode \
  --expected-subtasks-per-episode 5 \
  --expected-overall-task "<approved full-episode instruction>"
```

Audit는 read-only다. `meta/tasks.parquet`, `meta/subtasks.parquet`, frame-level
`task_index`/`subtask_index`, episode metadata, episode annotation JSON을 서로
비교한다. 현재 필수 mapping은 다음과 같다.

- 모든 frame의 `task_index`가 승인된 하나의 전체-episode prompt로 resolve된다.
- 각 episode의 subtask 범위가 gap/overlap 없이 `0,1,2,3,4` 순서다.
- Parquet, episode metadata, annotation JSON의 subtask text와 half-open frame
  range가 일치한다.

MP4에는 영상만 있고 caption은 없다. 여러 episode가 한 MP4에 있어도 frame의
language lookup은 사라지지 않는다. 다섯 subtask 문자열은 annotation으로
보존하며 GR00T prompt로 전환하지 않는다. 누락, 순서 오류, overlap, range/text
불일치가 있으면 prepare 전에 중단한다.

### `dataset prepare`

```bash
training/grootctl dataset prepare <dataset_name>
```

Hub source를 변경하지 않고 `_arms16_gop15` suffix의 별도 immutable derivative를
만든다. Processor는 다음을 수행한다.

- 정확한 이름의 16-D dual-arm/gripper state와 absolute action 선택
- 세 camera feature와 encoded stream의 640×480/15 fps 확인
- resize/crop/padding/letterbox 없이 geometry identity 보존
- H.264가 이미 GOP15이면 byte-for-byte copy, 아니면 최대 GOP15, CRF18,
  preset `medium`으로 transcode
- source commit/manifest, dimension/name, task, camera inventory, decoded frame 수,
  GOP, copy/transcode 수를 `DERIVATION_RECEIPT.json`에 기록

Output이 이미 있으면 receipt의 source path, pinned revision, manifest가 일치해야
재사용한다. 불일치 derivative는 덮어쓰지 않는다.

### `dataset validate --decode`

```bash
training/grootctl dataset validate <dataset_name> --decode
```

Prepared metadata의 schema, 선언된 episode/frame 수, 15 fps, camera key 순서와
shape, state/action dimension과 joint-name 순서를 registration과 비교한다.
`--decode`는 TorchCodec으로 실제 sample을 열어 finite 16-D state/action도 확인한다.

통과는 trainer와 파일 연결이 맞다는 의미다. Teleoperation 품질, task 성공,
collision, timing 품질, 배포 안전성을 증명하지 않으므로 사람의 dataset review와
별도 evaluation이 필요하다.

## 3. 학습 생성, 검사, 시작, 모니터링

일반적인 학습 흐름은 네 명령이다.

```bash
training/grootctl train init <run_name> --dataset <dataset_name>
training/grootctl train check <run_name>
training/grootctl train start <run_name>
training/grootctl train status <run_name>
```

### `train init`

`training/configs/runs/<run_name>.yaml`을 만든다. 이 작은 파일은 dataset을
참조하고 `configs/base/groot_n17.yaml`의 검토된 기본값을 상속한다. 학습은
시작하지 않는다. 생성된 description, hypothesis, tag, absolute-action 설정은
바로 사용할 수 있는 baseline이다. Dataset-specific goal과 evaluation criteria를
human field에 추가하면 되며 일반 baseline을 위해 policy/optimizer 설정을 run
파일에 복사할 필요는 없다.

기본값은 absolute action, native/GOP15 camera, photometric augmentation, batch 32,
worker 8, BF16 compute/FP32 trainable parameter, 20,000 step, tracking off다. 명확한
실험 목적이 있을 때만 override한다. 예를 들어 memory를 줄이면 다음을 추가한다.

```yaml
train:
  batch_size: 16
  num_workers: 8
```

### `train check`

Base YAML과 dataset registration을 resolve하고 정확한 camera/joint 계약을
주입한다. Step/LR/action 설정 불일치, dataset 계약, free disk를 검사하고 최종
`lerobot-train` command와 tmux/log/status 명령을 출력한다. Dry run이므로 model을
load하지 않고 GPU 학습이나 tmux session을 시작하지 않는다. YAML을 수정할 때마다
다시 실행한다.

### `train start`

검사를 반복하고 visible GPU memory를 확인한 뒤 detached tmux에서 동일 command를
실행한다. SSH가 끊겨도 학습은 계속된다. Attach/detach, log tail, status, 의도적인
stop 명령을 출력한다. `training/runs/<run_name>/`에는 source/resolved config,
정확한 command와 manifest, `train.log`, `result.yaml`, output checkpoint와 저장된
pre/postprocessor가 생성된다.

같은 output directory로 두 번 시작하지 않는다. 새 run name을 사용하며 resume은
training state를 직접 확인한 뒤 lower-level option으로만 수행한다.

### `train status`

지속되는 `result.yaml`을 출력한다. 실행 중에는 `running`, 정상 wrapper 종료 후에는
`completed` 또는 `failed`, exit code, timing, 마지막 loss/throughput/memory,
마지막 checkpoint endpoint를 보여 준다. Live progress는 `train start`가 출력한
`tail -f` 또는 `tmux attach` 명령을 사용한다.

Tracking은 기본적으로 꺼져 있다. W&B를 사용하려면 LeRobot 환경에서 로그인한다.

```bash
cd /NHNHOME/WORKSPACE/26motir001_D/intern_team/lerobot
source ./activate_lerobot.sh
wandb login
```

그 뒤 run YAML에서 `tracking.backend: wandb`, `tracking.enable: true`,
`tracking.mode: online`, project/entity를 설정하고 `train check`를 실행한다.
Hugging Face와 W&B 인증은 별개이며 token을 tracked config에 저장하지 않는다.

실험/자동화를 위한 lower-level `config`, `run`, `run-tmux`, `queue`, `result` 명령은
유지한다. 새 사용자는 보통 위 명령만 사용하면 된다.

## Output과 safety boundary

완료된 workstation run도 training/offline-evaluation artifact일 뿐이다. Robot에서
사용하기 전에 정확한 checkpoint와 saved processor를 load하고 finite action,
16-D/camera 계약, checksum package를 확인한다. `AGENTS.md`의 handoff, shadow,
bounded command, E-stop, guarded low-speed gate를 따라야 한다. Training loss만으로
배포할 수 없다.

## 환경과 테스트

`grootctl`과 wrapper가 LeRobot 환경을 활성화한다. Interactive shell은 다음과
같이 시작한다.

```bash
cd /NHNHOME/WORKSPACE/26motir001_D/intern_team/lerobot
source ./activate_lerobot.sh
```

`training/`에서 저장소 검사를 실행한다.

```bash
python3 -m pytest -q
python -m compileall -q data_pipeline registry.py
```

## Git과 대형 artifact

Source, config, template, test, 작은 receipt, 문서는 추적한다. Dataset,
checkpoint, 생성 run, cache, 대형 legacy payload는 제외한다. 직접 commit하기
전에 `git status --short --ignored`로 검토한다. Setup agent는 commit하지 않는다.
