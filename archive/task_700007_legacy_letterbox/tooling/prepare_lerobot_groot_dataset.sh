#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE="$(cd -- "${TRAINING_ROOT}/.." && pwd)"

if ! declare -F conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi
source "${WORKSPACE}/lerobot/activate_lerobot.sh"
exec python "${SCRIPT_DIR}/prepare_lerobot_groot_dataset.py" "$@"
