#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 OWNER/DATASET_REPO [REVISION]" >&2
  echo "Example: $0 RobotisSW/Task_700007_OMX_Insert_MCAP_lerobot_v30" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

repo_id="$1"
revision="${2:-main}"

if [[ ! "$repo_id" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid Hugging Face dataset repo: $repo_id" >&2
  exit 2
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face 'hf' CLI is not available." >&2
  echo "Activate the LeRobot environment first: source ../lerobot/activate_lerobot.sh" >&2
  exit 1
fi

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
training_root="$(cd "${script_root}/../.." && pwd)"
dataset_name="${repo_id##*/}"
destination="${training_root}/artifacts/datasets/${dataset_name}"

mkdir -p "${training_root}/artifacts/datasets"

echo "Repository : $repo_id"
echo "Revision   : $revision"
echo "Destination: $destination"

hf download "$repo_id" \
  --repo-type dataset \
  --revision "$revision" \
  --local-dir "$destination"

{
  echo "repo_id=$repo_id"
  echo "requested_revision=$revision"
  echo "downloaded_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$destination/DOWNLOAD_RECEIPT.txt"

echo "Dataset download complete: $destination"
