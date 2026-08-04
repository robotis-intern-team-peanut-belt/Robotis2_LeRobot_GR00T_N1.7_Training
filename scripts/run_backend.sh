#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:?backend is required}"
shift

TRAINING_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd -- "${TRAINING_ROOT}/.." && pwd)"

unset VIRTUAL_ENV

RUNTIME_CONDA_BASE="${GROOT_RUNTIME_CONDA_BASE:-$(dirname -- "${WORKSPACE}")/FastWAM_Workspace/miniforge3}"
if [[ ! -f "${RUNTIME_CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "ERROR: runtime Conda base is missing: ${RUNTIME_CONDA_BASE}" >&2
  exit 2
fi
source "${RUNTIME_CONDA_BASE}/etc/profile.d/conda.sh"
unset RUNTIME_CONDA_BASE

case "${BACKEND}" in
  lerobot|lerobot_groot|lerobot_act)
    source "${TRAINING_ROOT}/activate_lerobot.sh" >/dev/null
    ;;
  isaac_groot)
    source "${TRAINING_ROOT}/activate_groot.sh" >/dev/null
    ;;
  *)
    echo "ERROR: unknown backend: ${BACKEND}" >&2
    exit 2
    ;;
esac

if [[ "$#" -eq 0 ]]; then
  echo "ERROR: backend command is required" >&2
  exit 2
fi

exec "$@"
