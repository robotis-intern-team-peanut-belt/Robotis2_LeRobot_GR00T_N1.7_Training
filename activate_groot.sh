#!/usr/bin/env bash
# Source from the workspace or training repository:
#   source training/activate_groot.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced so it can update the current shell."
    echo "Run: source training/activate_groot.sh"
    exit 1
fi

GROOT_TRAINING_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GROOT_WORKSPACE="$(cd -- "${GROOT_TRAINING_ROOT}/.." && pwd)"
GROOT_REPO_DIR="${GROOT_WORKSPACE}/Isaac-GR00T"
GROOT_EXPECTED_VENV="${GROOT_REPO_DIR}/.venv"
GROOT_CONDA_ENV="${GROOT_CONDA_ENV:-groot-runtime}"
GROOT_CONDA_BASE="${GROOT_RUNTIME_CONDA_BASE:-$(dirname -- "${GROOT_WORKSPACE}")/FastWAM_Workspace/miniforge3}"

source "${GROOT_TRAINING_ROOT}/load_wandb_env.sh" || return 1

if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" != "${GROOT_EXPECTED_VENV}" ]]; then
    echo "ERROR: another virtual environment is active: ${VIRTUAL_ENV}"
    echo "Open a fresh shell before switching ML repositories."
    return 1
fi

if [[ -f "${GROOT_CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${GROOT_CONDA_BASE}/etc/profile.d/conda.sh"
elif ! declare -F conda >/dev/null 2>&1; then
    echo "ERROR: Conda initialization is unavailable: ${GROOT_CONDA_BASE}"
    return 1
fi
conda activate "${GROOT_CONDA_ENV}" || return 1
GROOT_FFMPEG_LIB_DIR="${CONDA_PREFIX}/lib"

if [[ ! -x "${CONDA_PREFIX}/bin/ffmpeg" ]]; then
    echo "ERROR: FFmpeg is missing from Conda environment '${GROOT_CONDA_ENV}'."
    echo "Install Conda ffmpeg<8 in that environment."
    return 1
fi

if [[ ! -f "${GROOT_EXPECTED_VENV}/bin/activate" ]]; then
    echo "ERROR: ${GROOT_EXPECTED_VENV} does not exist."
    echo "From Isaac-GR00T, run uv sync with the groot-runtime Python 3.12."
    return 1
fi

export LD_LIBRARY_PATH="${GROOT_FFMPEG_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
source "${GROOT_EXPECTED_VENV}/bin/activate"
cd "${GROOT_REPO_DIR}" || return 1

echo "Isaac-GR00T environment ready."
echo "  Repository: ${GROOT_REPO_DIR}"
echo "  Python:     $(command -v python)"
echo "  FFmpeg:     ${CONDA_PREFIX}/bin/ffmpeg"
echo "  W&B key:    $([[ -n "${WANDB_API_KEY:-}" ]] && echo 'loaded locally' || echo 'not configured')"

unset GROOT_TRAINING_ROOT GROOT_WORKSPACE GROOT_REPO_DIR GROOT_EXPECTED_VENV
unset GROOT_CONDA_BASE GROOT_FFMPEG_LIB_DIR
