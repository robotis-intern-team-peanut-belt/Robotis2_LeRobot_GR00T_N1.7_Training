#!/usr/bin/env bash
set -euo pipefail

TRAINING_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if ! declare -F conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi
source "${TRAINING_ROOT}/activate_lerobot.sh" >/dev/null
cd "${TRAINING_ROOT}"
exec python -m data_pipeline.prepare_omx_insert_native "$@"
