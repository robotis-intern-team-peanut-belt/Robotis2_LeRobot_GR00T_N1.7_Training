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

## 1. Source workflow 선택과 등록

Server는 서로 구분되는 두 source 방식을 지원한다. 두 방식 모두
`training/configs/datasets/` 아래의 tracked YAML registration으로 끝나며, 등록
후에는 같은 caption audit, immutable 16-D/GOP15 준비, decode validation을
사용한다. Byte나 source revision이 다르면 기존 registration 이름을 재사용하지
않는다.

### Workflow A: robot에서 Hugging Face에 올리고 server에서 download

Robot-side Hub upload는 이 guide의 범위 밖이다. Dataset이 Hub에 존재한 뒤 server
workspace root에서 다음을 실행한다. Public dataset은 login이 필요 없다. Private
dataset은 token prompt 또는 `HF_TOKEN`을 사용하고 token을 YAML에 저장하지 않는다.

```bash
hf auth login
hf auth whoami
training/grootctl dataset register-hf <dataset_name> --repo-id User/Repo
```

`register-hf`는 `main`을 정확한 immutable 40자리 Hub commit으로 resolve한다.
`--revision`으로 branch, tag, commit을 직접 줄 수도 있다. 등록 뒤에는 움직이는
branch를 신뢰하지 않는다.

### Workflow B: robot에서 변환 완료된 LeRobot v3.0을 직접 복사

Robot에서 Cyclo conversion이 끝나면 완성된 LeRobot directory를 새로운 server
directory로 복사한다. 일반 `rsync`면 충분하다. Hugging Face upload, checksum
manifest, source commit, UTC handoff ID는 필요하지 않다.

예를 들어 robot에서 다음을 실행한다.

```bash
rsync -a --partial -e 'ssh -i /home/robotis/Robotis2_key -p 34301' \
  /home/robotis/cyclo_intelligence/docker/workspace/lerobot/<dataset_name>/ \
  k_humanoid_3@59.150.32.1:/NHNHOME/WORKSPACE/26motir001_D/intern_team/training/artifacts/datasets/<new_server_dataset_directory>/
```

새 dataset을 이전 directory에 섞지 말고 새로운 destination directory를 사용한다.
`rsync`가 끝나면 Robotis2에서 등록한다.

```bash
training/grootctl dataset register-local <dataset_name> \
  --source-root /NHNHOME/WORKSPACE/26motir001_D/intern_team/training/artifacts/datasets/<new_server_dataset_directory>
```

이것이 필수 registration command의 전부다. Directory와 LeRobot
`meta/info.json`이 존재하는지 확인하고 관측된 version/episode/frame 수와 현재
robot/task default를 기록한 뒤
`training/configs/datasets/<dataset_name>.yaml`을 만든다. Producer checksum을
계산하거나 비교하지 않는다.

필요하면 naming convention 없이 일반 provenance note를 선택적으로 추가할 수 있다.

```bash
training/grootctl dataset register-local <dataset_name> \
  --source-root <server_dataset_directory> \
  --source-revision <optional_label_or_commit> \
  --handoff <optional_existing_notes_file>
```

이전 Task 700008 sample source는
`/home/robotis/cyclo_intelligence/docker/workspace/lerobot/Task_700008_OMX_Insert_F2_Intern_MCAP_lerobot_v30`이었다.
그 historical transfer는 더 엄격한 partial-directory/checksum 절차를 사용했지만
현재 routine direct registration에는 그 단계가 필요하지 않다.

### 공통 registration option

두 workflow 전에 현재 F2 robot/task default를 확인한다.

```bash
training/grootctl dataset defaults
```

Default와 evidence는 `training/configs/base/f2_omx_insert_contract.yaml`에 있다. 같은
robot/task dataset이면 `--robot-revision`과 `--task`를 생략한다. Dataset이 다른
contract를 사용할 때 override한다. `--task`는 GR00T에 전달하는 정확한
full-episode instruction이다.

Hub registration에서 `--repo-id`는 dataset repository이고 `--revision`은 Hub
branch/tag/commit이다. Direct registration의 `--source-revision`과 `--handoff`는
선택적인 note이며 필수가 아니고 copied byte와 대조하지 않는다.

두 명령 모두 source를 처리하지 않고
`training/configs/datasets/<dataset_name>.yaml`을 만든다. 계속하기 전에 검토한다.

```bash
training/grootctl dataset show <dataset_name>
```

## 2. Source 획득 또는 재검사, caption audit, prepare, validate

일반적으로 다음 한 명령을 사용한다.

```bash
training/grootctl dataset pipeline <dataset_name>
```

Source 획득, dataset preflight, task-specific caption audit, preparation,
prepared-data validation을 순서대로 실행하며 하나라도 실패하면 즉시 중단한다.
Registration이 가리키는 local source를 재사용하고 prepared derivative는 receipt가
registration과 일치할 때만 재사용하므로 동일 명령을 안전하게 다시 실행할 수 있다.

### `dataset download`

```bash
training/grootctl dataset download <dataset_name>
```

Hub registration이면 `repo_id`와 pinned commit을 읽어 정확한 snapshot을
`training/artifacts/datasets/<dataset_name>_source`에 받고 다음을 만든다.

- `training/artifacts/manifests/`의 파일별 SHA-256 manifest
- source 안의 `.groot_registry_download.json`: repo, commit, manifest digest,
  file count, byte count 기록

Direct registration이면 network나 checksum 작업을 하지 않는다. `register-local`이
기록한 directory인지 확인하고 그대로 재사용한다.

두 방식 모두 LeRobot payload 자체는 수정하지 않는다. 이후 preflight, caption,
preparation, decode 단계에서 이 training pipeline에 필요한 dataset structure를
계속 검증한다.

### Dataset preflight

등록된 source 또는 prepared training derivative에 detection-only check를 직접
실행한다.

```bash
training/grootctl dataset preflight <dataset_name> --stage source
training/grootctl dataset preflight <dataset_name>
```

Vocabulary, episode metadata, annotation JSON 전반에서 leading/trailing 또는
repeated caption space, tab, newline, non-breaking space, 기타 non-ASCII
whitespace를 reject한다. 또한 global frame index와 episode별 frame index의 연속성,
declared count, 각 episode metadata의 length와 global
`dataset_from_index`/`dataset_to_index` bound가 실제 data row와 일치하는지
검사한다. 첫 번째 정확한 위치를 보고하며 data를 수정하지 않는다. `dataset
pipeline`, `dataset validate`, `train check`, `train start`에서 자동 실행된다.

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

등록된 Hub 또는 direct-transfer source를 변경하지 않고 `_arms16_gop15` suffix의 별도 immutable derivative를
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
시작하지 않는다. 생성된 description, hypothesis, annotation notes의
`REPLACE ME`를 dataset-specific goal, 검증 가능한 expectation, evaluation
criteria, caveat, intended comparison으로 교체한다. 생성되는 owner는
`intern-team`이다.

`train init`은 run-level tag를 생성하지 않는다. Resolved run은 base의
`tags: [groot-n17]`를 상속한다. Tag는 resolved configuration에 보존되고 manifest
hash에 반영되는 registry metadata이며 `lerobot-train`에는 전달되지 않는다. YAML
list는 상속된 list에 추가되지 않고 전체를 교체하므로, base tag를 의도적으로
교체할 때만 run-level `tags`를 추가한다. 일반 baseline을 위해 policy/optimizer
설정을 run 파일에 복사할 필요는 없다.

기본값은 absolute action, native/GOP15 camera, photometric augmentation, batch 32,
worker 8, BF16 compute/FP32 trainable parameter, 20,000 step, team OMX W&B online tracking이다. 명확한
실험 목적이 있을 때만 override한다. 예를 들어 memory를 줄이면 다음을 추가한다.

```yaml
train:
  batch_size: 16
  num_workers: 8
```

### `train check`

Base YAML과 dataset registration을 resolve하고 정확한 camera/joint 계약을
주입하며 detection-only dataset preflight를 다시 실행한다. Step/LR/action 설정
불일치와 free disk를 검사하고 최종 `lerobot-train` command와 tmux/log/status
명령을 출력한다. Dry run이므로 model을 load하지 않고 GPU 학습이나 tmux session을
시작하지 않는다. YAML을 수정할 때마다 다시 실행한다.

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

W&B tracking은
[`robotis-intern-team-peanut-belt/gr00t-omx-insert`](https://wandb.ai/robotis-intern-team-peanut-belt/gr00t-omx-insert)
project를 기본값으로 사용한다. 첫 tracked launch 전에 LeRobot 환경에서 로그인한다.

```bash
cd /NHNHOME/WORKSPACE/26motir001_D/intern_team/lerobot
source ./activate_lerobot.sh
wandb login
```

`train check`는 W&B에 접속하지 않고 상속된 entity, project, online mode를
render한다. `train start`에는 유효한 W&B 인증과 network가 필요하다. Bounded smoke
또는 local-only run에서 의도적으로 끄려면 다음을 추가한다.

```yaml
tracking:
  backend: none
  enable: false
  mode: disabled
```

Hugging Face와 W&B 인증은 별개이며 token을 tracked config에 저장하지 않는다.

### Sequential multi-run queue

Training을 시작하지 않고 queue와 모든 resolved run을 검증한다.

```bash
training/grootctl queue Task_700009_model_0_then_model_1_chunk20 --dry-run
```

Queue runner는 foreground-only다. SSH 연결과 분리하려면 tmux 안에서 실행한다.

```bash
cd /NHNHOME/WORKSPACE/26motir001_D/intern_team
tmux new-session -s task700009-chunk-compare
# tmux가 열린 뒤 실행:
training/grootctl queue Task_700009_model_0_then_model_1_chunk20
# 중단하지 않고 detach: Ctrl-b, 다음 d
```

다른 shell에서 queue state와 active attempt를 확인한다.

```bash
cat training/runs/_queues/Task_700009_model_0_then_model_1_chunk20/queue_status.yaml
tail -f training/runs/<active_run_name>/attempt-1/train.log
```

상속된 multi-queue 기본값은 run마다 한 번만 시도하고, free disk 100 GiB를
요구하며, queue budget은 24시간이고 첫 run 실패 시 중단한다. 현재 queue
runner에는 전용 stop command가 없다. Trainer가 실행 중일 수 있으므로 tmux
session을 임의로 kill하지 않는다.

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
