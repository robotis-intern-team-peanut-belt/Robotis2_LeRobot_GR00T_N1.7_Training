#!/usr/bin/env bash
# Load the workspace-local W&B API key without evaluating the env file as shell.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: load_wandb_env.sh must be sourced." >&2
    exit 1
fi

_training_load_wandb_env() {
    local env_file line value line_number=0
    env_file="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/.env.secret"

    [[ -f "${env_file}" ]] || return 0
    if [[ ! -r "${env_file}" ]]; then
        echo "ERROR: W&B secret file is not readable: ${env_file}" >&2
        return 1
    fi

    while IFS= read -r line || [[ -n "${line}" ]]; do
        ((line_number += 1))
        line="${line%$'\r'}"
        line="${line#"${line%%[![:space:]]*}"}"
        case "${line}" in
            ""|\#*)
                continue
                ;;
        esac

        if [[ "${line}" =~ ^(export[[:space:]]+)?WANDB_API_KEY[[:space:]]*=[[:space:]]*(.*)$ ]]; then
            value="${BASH_REMATCH[2]}"
            value="${value%"${value##*[![:space:]]}"}"
            if (( ${#value} >= 2 )); then
                if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]] ||
                   [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
                    value="${value:1:${#value}-2}"
                fi
            fi
            if [[ -n "${value}" ]]; then
                export WANDB_API_KEY="${value}"
            fi
            continue
        fi

        echo "ERROR: unsupported entry in ${env_file}:${line_number}; only WANDB_API_KEY is allowed." >&2
        return 1
    done < "${env_file}"
}

if ! _training_load_wandb_env; then
    unset -f _training_load_wandb_env
    return 1
fi
unset -f _training_load_wandb_env
