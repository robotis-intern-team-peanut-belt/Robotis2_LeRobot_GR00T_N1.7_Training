#!/usr/bin/env bash
set -euo pipefail

TRAINING_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd -- "${TRAINING_DIR}/.." && pwd)"
QUEUE_ROOT="${TASK_DIR}/runs/2026-07-24_weekend_queue"
PID_FILE="${QUEUE_ROOT}/queue.pid"
SUPERVISOR_LOG="${QUEUE_ROOT}/queue_supervisor.log"

mkdir -p "${QUEUE_ROOT}"
if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "ERROR: queue already running with PID $(cat "${PID_FILE}")" >&2
  exit 1
fi

nohup setsid "${TRAINING_DIR}/run_weekend_queue.sh" >>"${SUPERVISOR_LOG}" 2>&1 &
QUEUE_PID=$!
printf '%s\n' "${QUEUE_PID}" >"${PID_FILE}"
echo "Started queue PID ${QUEUE_PID}"
echo "Supervisor log: ${SUPERVISOR_LOG}"
echo "Status: ${QUEUE_ROOT}/queue_status.tsv"
