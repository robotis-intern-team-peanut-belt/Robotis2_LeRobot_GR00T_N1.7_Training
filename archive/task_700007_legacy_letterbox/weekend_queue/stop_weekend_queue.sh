#!/usr/bin/env bash
set -euo pipefail

TRAINING_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd -- "${TRAINING_DIR}/.." && pwd)"
PID_FILE="${TASK_DIR}/runs/2026-07-24_weekend_queue/queue.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "No queue PID file found."
  exit 0
fi
QUEUE_PID="$(cat "${PID_FILE}")"
if kill -0 "${QUEUE_PID}" 2>/dev/null; then
  # The queue is started with setsid; stop its process group, including training.
  kill -TERM -- "-${QUEUE_PID}"
  echo "Sent TERM to queue process group ${QUEUE_PID}"
else
  echo "Queue PID ${QUEUE_PID} is no longer running."
fi
