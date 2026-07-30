#!/usr/bin/env bash
set -uo pipefail

TRAINING_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd -- "${TRAINING_DIR}/.." && pwd)"
PROJECT_DIR="$(cd -- "${TASK_DIR}/.." && pwd)"
LEROBOT_DIR="${PROJECT_DIR}/lerobot"
DATASET_ROOT="${TASK_DIR}/datasets/Task_700007_OMX_Insert_MCAP_lerobot_v30_arms16_head-letterbox640x480_taskfix"
QUEUE_FILE="${TRAINING_DIR}/weekend_queue.tsv"
QUEUE_ROOT="${TASK_DIR}/runs/2026-07-24_weekend_queue"
STATUS_FILE="${QUEUE_ROOT}/queue_status.tsv"
LOCK_FILE="${QUEUE_ROOT}/queue.lock"
STEPS=80000
BATCH_SIZE=8
MIN_FREE_GB=100

mkdir -p "${QUEUE_ROOT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "ERROR: another weekend queue process holds ${LOCK_FILE}" >&2
  exit 1
fi

if ! declare -F conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi
source "${LEROBOT_DIR}/activate_lerobot.sh" >/dev/null

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: prepared dataset is missing: ${DATASET_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${QUEUE_FILE}" ]]; then
  echo "ERROR: queue definition is missing: ${QUEUE_FILE}" >&2
  exit 1
fi
if ! wandb status >/dev/null 2>&1; then
  echo "ERROR: W&B authentication check failed" >&2
  exit 1
fi

if [[ ! -f "${STATUS_FILE}" ]]; then
  printf 'utc_time\trun_name\tattempt\tstatus\texit_code\toutput_dir\tlog_file\n' >"${STATUS_FILE}"
fi

record_status() {
  local run_name="$1" attempt="$2" status="$3" exit_code="$4" output_dir="$5" log_file="$6"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${run_name}" "${attempt}" "${status}" \
    "${exit_code}" "${output_dir}" "${log_file}" >>"${STATUS_FILE}"
}

free_gb() {
  df -Pk "${QUEUE_ROOT}" | awk 'NR==2 {printf "%d", $4/1024/1024}'
}

next_attempt_dir() {
  local run_name="$1" attempt=1 candidate
  while true; do
    candidate="${QUEUE_ROOT}/${run_name}_attempt${attempt}"
    if [[ ! -e "${candidate}" ]]; then
      printf '%s\t%s\n' "${attempt}" "${candidate}"
      return
    fi
    if [[ -f "${candidate}/QUEUE_SUCCESS" ]]; then
      attempt=$((attempt + 1))
      continue
    fi
    attempt=$((attempt + 1))
  done
}

already_succeeded() {
  local run_name="$1"
  compgen -G "${QUEUE_ROOT}/${run_name}_attempt*/QUEUE_SUCCESS" >/dev/null
}

run_one() {
  local run_name="$1" action_mode="$2" seed="$3" learning_rate="$4"
  local augmentation="$5" save_frequency="$6"
  local attempt_and_dir attempt output_dir log_file exit_code

  if already_succeeded "${run_name}"; then
    echo "SKIP successful run: ${run_name}"
    return 0
  fi

  if (( $(free_gb) < MIN_FREE_GB )); then
    echo "STOP: less than ${MIN_FREE_GB} GiB free before ${run_name}" >&2
    record_status "${run_name}" "-" "blocked_low_disk" "75" "-" "-"
    return 75
  fi

  attempt_and_dir="$(next_attempt_dir "${run_name}")"
  attempt="${attempt_and_dir%%$'\t'*}"
  output_dir="${attempt_and_dir#*$'\t'}"
  log_file="${QUEUE_ROOT}/${run_name}_attempt${attempt}.log"

  local -a command=(
    lerobot-train
    --dataset.repo_id=local/omx_insert_arms16_letterbox
    --dataset.root="${DATASET_ROOT}"
    --dataset.video_backend=torchcodec
    --dataset.eval_split=0.1
    --dataset.image_transforms.enable=false
    --policy.type=groot
    --policy.device=cuda
    --policy.base_model_path=nvidia/GR00T-N1.7-3B
    --policy.embodiment_tag=new_embodiment
    --policy.chunk_size=40
    --policy.n_action_steps=8
    --policy.use_bf16=true
    --policy.max_steps="${STEPS}"
    --policy.optimizer_lr="${learning_rate}"
    --policy.push_to_hub=false
    --policy.repo_id="local/${run_name}"
    --seed="${seed}"
    --batch_size="${BATCH_SIZE}"
    --num_workers=4
    --prefetch_factor=2
    --persistent_workers=true
    --steps="${STEPS}"
    --save_checkpoint=true
    --save_freq="${save_frequency}"
    --use_policy_training_preset=true
    --env_eval_freq=0
    --eval_steps=5000
    --max_eval_samples=1000
    --log_freq=20
    --output_dir="${output_dir}"
    --job_name="${run_name}"
    --wandb.enable=true
    --wandb.project=gr00t-omx-insert
    --wandb.disable_artifact=true
    --wandb.notes="Weekend queue; LeRobot N1.7; OMX Insert prepared derivative; batch 8; horizon 40; ${action_mode}; seed ${seed}; lr ${learning_rate}; augmentation ${augmentation}."
  )

  if [[ "${action_mode}" == "relative" ]]; then
    command+=(--policy.use_relative_actions=true --policy.relative_exclude_joints='["gripper"]')
  else
    command+=(--policy.use_relative_actions=false)
  fi

  if [[ "${augmentation}" == "photometric" ]]; then
    command+=(
      --dataset.image_transforms.enable=true
      --dataset.image_transforms.max_num_transforms=2
      --dataset.image_transforms.tfs.affine.weight=0
      --dataset.image_transforms.tfs.hue.weight=0
      --dataset.image_transforms.tfs.sharpness.weight=0
      --dataset.image_transforms.tfs.saturation.kwargs.saturation='[0.8,1.2]'
    )
  fi

  echo "START ${run_name} attempt=${attempt} free_gb=$(free_gb)"
  record_status "${run_name}" "${attempt}" "started" "-" "${output_dir}" "${log_file}"
  printf '%q ' "${command[@]}" >"${QUEUE_ROOT}/${run_name}_attempt${attempt}.command.txt"
  printf '\n' >>"${QUEUE_ROOT}/${run_name}_attempt${attempt}.command.txt"

  "${command[@]}" >"${log_file}" 2>&1
  exit_code=$?

  if (( exit_code == 0 )); then
    touch "${output_dir}/QUEUE_SUCCESS"
    record_status "${run_name}" "${attempt}" "completed" "0" "${output_dir}" "${log_file}"
    echo "DONE ${run_name} free_gb=$(free_gb)"
  else
    if rg -qi 'CUDA out of memory|outofmemoryerror|CUBLAS_STATUS_ALLOC_FAILED' "${log_file}"; then
      record_status "${run_name}" "${attempt}" "failed_oom_continuing" "${exit_code}" "${output_dir}" "${log_file}"
      echo "OOM ${run_name}; continuing to the next queued run" >&2
    else
      record_status "${run_name}" "${attempt}" "failed_continuing" "${exit_code}" "${output_dir}" "${log_file}"
      echo "FAILED ${run_name} exit=${exit_code}; continuing" >&2
    fi
  fi

  sleep 20
  return 0
}

echo "Queue started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while IFS=$'\t' read -r run_name action_mode seed learning_rate augmentation save_frequency; do
  [[ -z "${run_name}" || "${run_name}" == \#* ]] && continue
  run_one "${run_name}" "${action_mode}" "${seed}" "${learning_rate}" "${augmentation}" "${save_frequency}"
  status=$?
  if (( status == 75 )); then
    break
  fi
done <"${QUEUE_FILE}"
echo "Queue ended at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
