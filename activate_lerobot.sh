#!/usr/bin/env bash
# Source from the workspace or training repository:
#   source training/activate_lerobot.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced so it can update the current shell."
    echo "Run: source training/activate_lerobot.sh"
    exit 1
fi

LEROBOT_TRAINING_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_WORKSPACE="$(cd -- "${LEROBOT_TRAINING_ROOT}/.." && pwd)"
LEROBOT_REPO_DIR="${LEROBOT_WORKSPACE}/lerobot"
LEROBOT_EXPECTED_VENV="${LEROBOT_REPO_DIR}/.venv"
LEROBOT_CONDA_ENV="${LEROBOT_CONDA_ENV:-groot-runtime}"
LEROBOT_CONDA_BASE="${GROOT_RUNTIME_CONDA_BASE:-$(dirname -- "${LEROBOT_WORKSPACE}")/FastWAM_Workspace/miniforge3}"

if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" != "${LEROBOT_EXPECTED_VENV}" ]]; then
    echo "ERROR: another virtual environment is active: ${VIRTUAL_ENV}"
    echo "Open a fresh shell before switching ML repositories."
    return 1
fi

if [[ -f "${LEROBOT_CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${LEROBOT_CONDA_BASE}/etc/profile.d/conda.sh"
elif ! declare -F conda >/dev/null 2>&1; then
    echo "ERROR: Conda initialization is unavailable: ${LEROBOT_CONDA_BASE}"
    return 1
fi
conda activate "${LEROBOT_CONDA_ENV}" || return 1
LEROBOT_FFMPEG_LIB_DIR="${CONDA_PREFIX}/lib"

if [[ ! -x "${CONDA_PREFIX}/bin/ffmpeg" ]]; then
    echo "ERROR: FFmpeg is missing from Conda environment '${LEROBOT_CONDA_ENV}'."
    echo "Install Conda ffmpeg<8 in that environment."
    return 1
fi

if [[ ! -f "${LEROBOT_EXPECTED_VENV}/bin/activate" ]]; then
    echo "ERROR: ${LEROBOT_EXPECTED_VENV} does not exist."
    echo "From lerobot, run uv sync --locked with the required workflow extras."
    return 1
fi

export LD_LIBRARY_PATH="${LEROBOT_FFMPEG_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
source "${LEROBOT_EXPECTED_VENV}/bin/activate"
cd "${LEROBOT_REPO_DIR}" || return 1

echo "LeRobot environment ready."
echo "  Repository: ${LEROBOT_REPO_DIR}"
echo "  Python:     $(command -v python)"
echo "  FFmpeg:     ${CONDA_PREFIX}/bin/ffmpeg"

unset LEROBOT_TRAINING_ROOT LEROBOT_WORKSPACE LEROBOT_REPO_DIR LEROBOT_EXPECTED_VENV
unset LEROBOT_CONDA_BASE LEROBOT_FFMPEG_LIB_DIR
