#!/usr/bin/env python3
"""Fail-closed, detection-only preflight for LeRobot caption and episode metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


class DatasetPreflightError(RuntimeError):
    """The dataset can fail or silently misindex during training."""


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def whitespace_problem(value: str) -> str | None:
    """Describe whitespace that should not appear in a single-line caption."""
    if value != value.strip():
        return "leading or trailing whitespace"
    if "  " in value:
        return "repeated ASCII spaces"
    for char in value:
        if char.isspace() and char != " ":
            return f"non-ASCII-space whitespace U+{ord(char):04X}"
    return None


def check_caption(value: Any, location: str) -> None:
    if not isinstance(value, str):
        raise DatasetPreflightError(f"{location}: caption must be a string")
    if not value:
        raise DatasetPreflightError(f"{location}: caption must be nonempty")
    problem = whitespace_problem(value)
    if problem:
        raise DatasetPreflightError(
            f"{location}: suspicious caption whitespace ({problem}): {value!r}"
        )


def check_caption_list(value: Any, location: str) -> None:
    if not isinstance(value, list):
        raise DatasetPreflightError(f"{location}: expected a caption list")
    for index, caption in enumerate(value):
        check_caption(caption, f"{location}[{index}]")


def parquet_captions(root: Path) -> int:
    checked = 0
    vocabularies = (
        (root / "meta" / "tasks.parquet", "task_index", "task"),
        (root / "meta" / "subtasks.parquet", "subtask_index", "subtask"),
    )
    for path, index_column, text_column in vocabularies:
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        if index_column not in frame.columns:
            raise DatasetPreflightError(f"{path}: missing {index_column!r}")
        if text_column in frame.columns:
            captions = frame[text_column].tolist()
        elif frame.index.name == text_column:
            captions = frame.index.tolist()
        else:
            raise DatasetPreflightError(f"{path}: missing {text_column!r}")
        indices = [int(value) for value in frame[index_column].tolist()]
        if indices != list(range(len(indices))):
            raise DatasetPreflightError(
                f"{path}: indices must be unique, contiguous, and row-aligned"
            )
        for row_index, caption in zip(indices, captions, strict=True):
            check_caption(caption, f"{path.relative_to(root)}[{row_index}].{text_column}")
            checked += 1
    return checked


def episode_metadata(root: Path) -> tuple[dict[int, dict[str, Any]], int]:
    episodes: dict[int, dict[str, Any]] = {}
    captions_checked = 0
    paths = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not paths:
        raise DatasetPreflightError("no meta/episodes Parquet files found")
    for path in paths:
        for row_number, row in enumerate(pq.read_table(path).to_pylist()):
            if "episode_index" not in row:
                raise DatasetPreflightError(f"{path}: missing 'episode_index'")
            episode = int(row["episode_index"])
            if episode in episodes:
                raise DatasetPreflightError(f"duplicate episode metadata for episode {episode}")
            for field in ("tasks", "subtask_instructions"):
                if field in row and row[field] is not None:
                    check_caption_list(
                        row[field],
                        f"{path.relative_to(root)} row {row_number} episode {episode}.{field}",
                    )
                    captions_checked += len(row[field])
            episodes[episode] = row
    return episodes, captions_checked


def annotation_captions(root: Path) -> int:
    checked = 0
    for path in sorted((root / "annotations").glob("**/*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DatasetPreflightError(f"{path}: invalid JSON: {exc}") from exc
        entries = value.get("sub_task_annotation", [])
        if not isinstance(entries, list):
            raise DatasetPreflightError(
                f"{path.relative_to(root)}.sub_task_annotation: expected a list"
            )
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or "sub_task_instruction" not in entry:
                raise DatasetPreflightError(
                    f"{path.relative_to(root)}.sub_task_annotation[{index}]: "
                    "missing sub_task_instruction"
                )
            check_caption(
                entry["sub_task_instruction"],
                f"{path.relative_to(root)}.sub_task_annotation[{index}].sub_task_instruction",
            )
            checked += 1
    return checked


def data_rows(root: Path) -> tuple[dict[int, dict[str, int]], int]:
    """Validate global indices and summarize the actual bounds of every episode."""
    summaries: dict[int, dict[str, int]] = {}
    expected_global_index = 0
    paths = sorted((root / "data").glob("**/*.parquet"))
    if not paths:
        raise DatasetPreflightError("no data Parquet files found")
    for path in paths:
        schema = set(pq.read_schema(path).names)
        required = {"index", "episode_index", "frame_index"}
        missing = required - schema
        if missing:
            raise DatasetPreflightError(f"{path}: missing columns {sorted(missing)}")
        table = pq.read_table(path, columns=sorted(required))
        for row_number, row in enumerate(table.to_pylist()):
            global_index = int(row["index"])
            episode = int(row["episode_index"])
            frame = int(row["frame_index"])
            if global_index != expected_global_index:
                raise DatasetPreflightError(
                    f"{path.relative_to(root)} row {row_number}: global index "
                    f"{global_index}, expected {expected_global_index}; indices must be "
                    "unique and contiguous across every data file"
                )
            summary = summaries.setdefault(
                episode,
                {
                    "first_index": global_index,
                    "last_index": global_index,
                    "frames": 0,
                    "next_frame_index": 0,
                },
            )
            if frame != summary["next_frame_index"]:
                raise DatasetPreflightError(
                    f"episode {episode}: frame_index {frame}, expected "
                    f"{summary['next_frame_index']}"
                )
            if global_index != summary["last_index"] and summary["frames"] > 0:
                raise DatasetPreflightError(
                    f"episode {episode}: rows are not globally contiguous"
                )
            summary["last_index"] = global_index + 1
            summary["frames"] += 1
            summary["next_frame_index"] += 1
            expected_global_index += 1
    return summaries, expected_global_index


def require_int(row: dict[str, Any], field: str, episode: int) -> int:
    if field not in row or row[field] is None:
        raise DatasetPreflightError(f"episode {episode}: metadata missing {field!r}")
    try:
        return int(row[field])
    except (TypeError, ValueError) as exc:
        raise DatasetPreflightError(
            f"episode {episode}: metadata {field!r} must be an integer"
        ) from exc


def validate_episode_bounds(
    metadata: dict[int, dict[str, Any]],
    observed: dict[int, dict[str, int]],
) -> None:
    expected_episodes = list(range(len(observed)))
    if sorted(observed) != expected_episodes:
        raise DatasetPreflightError("data episode indices must be contiguous from zero")
    if sorted(metadata) != expected_episodes:
        raise DatasetPreflightError(
            "episode metadata indices must exactly match data episode indices"
        )
    for episode in expected_episodes:
        meta = metadata[episode]
        actual = observed[episode]
        checks = {
            "length": actual["frames"],
            "dataset_from_index": actual["first_index"],
            "dataset_to_index": actual["last_index"],
        }
        mismatches = []
        for field, expected in checks.items():
            observed_value = require_int(meta, field, episode)
            if observed_value != expected:
                mismatches.append(f"{field}={observed_value}, expected {expected}")
        if mismatches:
            raise DatasetPreflightError(
                f"episode {episode}: " + "; ".join(mismatches)
                + ". LeRobot temporal-window lookup requires global dataset indices; "
                "per-file-reset offsets can silently mix episodes and crash held-out evaluation."
            )


def audit(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise DatasetPreflightError(f"missing {info_path}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetPreflightError(f"{info_path}: invalid JSON: {exc}") from exc

    vocabulary_count = parquet_captions(root)
    metadata, episode_caption_count = episode_metadata(root)
    annotation_count = annotation_captions(root)
    observed, total_frames = data_rows(root)
    validate_episode_bounds(metadata, observed)

    declared_episodes = int(info.get("total_episodes", -1))
    declared_frames = int(info.get("total_frames", -1))
    if declared_episodes != len(observed):
        raise DatasetPreflightError(
            f"meta/info.json total_episodes={declared_episodes}, observed {len(observed)}"
        )
    if declared_frames != total_frames:
        raise DatasetPreflightError(
            f"meta/info.json total_frames={declared_frames}, observed {total_frames}"
        )

    return {
        "dataset": str(root),
        "ready": True,
        "episodes": len(observed),
        "frames": total_frames,
        "captions_checked": vocabulary_count + episode_caption_count + annotation_count,
        "checks": {
            "caption_whitespace": "passed",
            "global_frame_indices": "passed",
            "episode_frame_indices": "passed",
            "episode_metadata_global_bounds": "passed",
            "declared_counts": "passed",
        },
    }


def main() -> None:
    args = cli()
    report = audit(args.dataset)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    try:
        main()
    except (DatasetPreflightError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
