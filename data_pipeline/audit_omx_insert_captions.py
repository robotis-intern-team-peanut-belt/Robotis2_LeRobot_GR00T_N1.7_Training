#!/usr/bin/env python3
"""Audit the overall OMX prompt and five ordered subtask annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


class CaptionAuditError(RuntimeError):
    """The converted dataset does not satisfy the language contract."""


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-subtasks-per-episode", type=int, default=5)
    parser.add_argument(
        "--language-mode",
        choices=("full_episode", "per_subtask"),
        default="full_episode",
    )
    parser.add_argument("--expected-overall-task")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def indexed_strings(path: Path, index_column: str, text_column: str) -> dict[int, str]:
    if not path.is_file():
        return {}
    frame = pd.read_parquet(path)
    if index_column not in frame.columns or text_column not in frame.columns:
        if frame.index.name == text_column and index_column in frame.columns:
            strings = frame.index.astype(str).tolist()
        else:
            raise CaptionAuditError(
                f"{path}: expected {index_column!r} and {text_column!r}"
            )
    else:
        strings = frame[text_column].astype(str).tolist()
    indices = [int(value) for value in frame[index_column].tolist()]
    if indices != list(range(len(indices))):
        raise CaptionAuditError(
            f"{path}: indices must be unique, contiguous, and row-aligned for LeRobot iloc lookup"
        )
    if any(not value.strip() for value in strings):
        raise CaptionAuditError(f"{path}: captions must be nonempty")
    return dict(zip(indices, strings, strict=True))


def runs(values: list[int]) -> list[int]:
    result: list[int] = []
    for value in values:
        if not result or result[-1] != value:
            result.append(value)
    return result


def run_segments(values: list[int]) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            segments.append((values[start], start, index))
            start = index
    return segments


def episode_metadata(root: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        for row in pq.read_table(path).to_pylist():
            result[int(row["episode_index"])] = {
                key: row.get(key)
                for key in ("length", "tasks", "subtask_instructions")
            }
    return result


def validate_annotation(
    root: Path,
    episode: int,
    frame_count: int,
    observed_segments: list[tuple[int, int, int]],
    subtasks: dict[int, str],
) -> str:
    matches = sorted((root / "annotations").glob(f"**/episode_{episode:06d}.json"))
    if len(matches) != 1:
        raise CaptionAuditError(
            f"episode {episode}: expected one annotation JSON, found {len(matches)}"
        )
    path = matches[0]
    value = json.loads(path.read_text(encoding="utf-8"))
    metadata = value.get("meta_data", {})
    if int(metadata.get("task_duration", -1)) != frame_count:
        raise CaptionAuditError(f"episode {episode}: annotation task_duration mismatch")
    if metadata.get("valid_duration") != [0, frame_count]:
        raise CaptionAuditError(f"episode {episode}: annotation valid_duration mismatch")
    entries = value.get("sub_task_annotation")
    if not isinstance(entries, list) or len(entries) != len(observed_segments):
        raise CaptionAuditError(f"episode {episode}: annotation segment count mismatch")
    for entry, (subtask_index, start, end) in zip(entries, observed_segments, strict=True):
        if int(entry.get("sub_task_idx", -1)) != subtask_index:
            raise CaptionAuditError(f"episode {episode}: annotation subtask order mismatch")
        if entry.get("sub_task_instruction") != subtasks[subtask_index]:
            raise CaptionAuditError(f"episode {episode}: annotation caption mismatch")
        if entry.get("frame_duration") != [start, end]:
            raise CaptionAuditError(f"episode {episode}: annotation frame range mismatch")
    return str(path.relative_to(root))


def audit(
    root: Path,
    expected_episodes: int | None = None,
    expected_subtasks: int = 5,
    language_mode: str = "full_episode",
    expected_overall_task: str | None = None,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise CaptionAuditError(f"missing {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    tasks = indexed_strings(root / "meta" / "tasks.parquet", "task_index", "task")
    subtasks = indexed_strings(
        root / "meta" / "subtasks.parquet", "subtask_index", "subtask"
    )
    if not tasks:
        raise CaptionAuditError("tasks.parquet contains no trainable task captions")
    if expected_overall_task is not None and tasks != {0: expected_overall_task}:
        raise CaptionAuditError("overall task does not match the approved instruction")

    rows: list[dict[str, int]] = []
    for path in sorted((root / "data").glob("**/*.parquet")):
        schema_names = pq.read_schema(path).names
        required = {"episode_index", "frame_index", "task_index"}
        missing = required - set(schema_names)
        if missing:
            raise CaptionAuditError(f"{path}: missing {sorted(missing)}")
        columns = ["episode_index", "frame_index", "task_index"]
        if "subtask_index" in schema_names:
            columns.append("subtask_index")
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    if not rows:
        raise CaptionAuditError("no data parquet rows found")

    by_episode: dict[int, list[dict[str, int]]] = {}
    for row in rows:
        by_episode.setdefault(int(row["episode_index"]), []).append(row)
    observed_episodes = sorted(by_episode)
    declared_episodes = int(info.get("total_episodes", -1))
    if observed_episodes != list(range(len(observed_episodes))):
        raise CaptionAuditError("episode indices must be contiguous from zero")
    if declared_episodes != len(observed_episodes):
        raise CaptionAuditError(
            f"metadata declares {declared_episodes} episodes; parquet has {len(observed_episodes)}"
        )
    if expected_episodes is not None and len(observed_episodes) != expected_episodes:
        raise CaptionAuditError(
            f"expected {expected_episodes} episodes, found {len(observed_episodes)}"
        )

    meta_episodes = episode_metadata(root)
    summaries: list[dict[str, Any]] = []
    modes: set[str] = set()
    for episode in observed_episodes:
        episode_rows = sorted(by_episode[episode], key=lambda row: int(row["frame_index"]))
        frame_indices = [int(row["frame_index"]) for row in episode_rows]
        if frame_indices != list(range(len(frame_indices))):
            raise CaptionAuditError(f"episode {episode}: frame indices are not contiguous from zero")
        task_runs = runs([int(row["task_index"]) for row in episode_rows])
        if any(index not in tasks for index in task_runs):
            raise CaptionAuditError(f"episode {episode}: task_index does not resolve to text")

        has_subtask_index = all("subtask_index" in row for row in episode_rows)
        subtask_runs = (
            runs([int(row["subtask_index"]) for row in episode_rows])
            if has_subtask_index
            else []
        )
        if subtask_runs and any(index not in subtasks for index in subtask_runs):
            raise CaptionAuditError(f"episode {episode}: subtask_index does not resolve to text")
        expected_order = list(range(expected_subtasks))
        if subtask_runs != expected_order:
            raise CaptionAuditError(
                f"episode {episode}: expected ordered subtasks {expected_order}, got {subtask_runs}"
            )

        training_captions = [tasks[index] for index in task_runs]
        stored_subtask_captions = [subtasks[index] for index in subtask_runs]
        metadata = meta_episodes.get(episode)
        if not metadata:
            raise CaptionAuditError(f"episode {episode}: missing episode metadata")
        if int(metadata.get("length", -1)) != len(episode_rows):
            raise CaptionAuditError(f"episode {episode}: metadata length mismatch")
        if metadata.get("subtask_instructions") != stored_subtask_captions:
            raise CaptionAuditError(f"episode {episode}: metadata subtask captions mismatch")
        observed_segments = run_segments(
            [int(row["subtask_index"]) for row in episode_rows]
        )
        annotation_path = validate_annotation(
            root, episode, len(episode_rows), observed_segments, subtasks
        )

        if language_mode == "full_episode":
            if len(tasks) != 1 or task_runs != [0]:
                mode = "invalid_full_episode_task_mapping"
            elif metadata.get("tasks") != [tasks[0]]:
                mode = "invalid_full_episode_metadata"
            else:
                mode = "full_episode_prompt_with_subtask_annotations"
        else:
            captions_match = training_captions == stored_subtask_captions
            if len(task_runs) == expected_subtasks and captions_match:
                mode = "per_frame_subtasks_visible_to_training"
            else:
                mode = "invalid_per_subtask_task_mapping"
        modes.add(mode)
        summaries.append(
            {
                "episode_index": episode,
                "frames": len(episode_rows),
                "mode": mode,
                "task_run_indices": task_runs,
                "training_captions": training_captions,
                "subtask_run_indices": subtask_runs,
                "stored_subtask_captions": stored_subtask_captions,
                "subtask_segments": [
                    {
                        "subtask_index": index,
                        "start_frame": start,
                        "end_frame_exclusive": end,
                    }
                    for index, start, end in observed_segments
                ],
                "episode_metadata": metadata,
                "annotation": annotation_path,
            }
        )

    expected_mode = (
        "full_episode_prompt_with_subtask_annotations"
        if language_mode == "full_episode"
        else "per_frame_subtasks_visible_to_training"
    )
    ready = modes == {expected_mode}
    report = {
        "dataset": str(root),
        "episodes": len(observed_episodes),
        "language_mode": language_mode,
        "expected_subtasks_per_episode": expected_subtasks,
        "task_vocabulary": tasks,
        "subtask_vocabulary": subtasks,
        "modes": sorted(modes),
        "language_contract_ready": ready,
        "episode_summaries": summaries,
    }
    if not ready:
        raise CaptionAuditError(
            json.dumps(report, indent=2, ensure_ascii=False)
            + "\nThe standard LeRobot/GR00T task channel does not match the "
            "selected language mode. Stop before rewriting captions."
        )
    return report


def main() -> None:
    args = cli()
    report = audit(
        args.dataset,
        expected_episodes=args.expected_episodes,
        expected_subtasks=args.expected_subtasks_per_episode,
        language_mode=args.language_mode,
        expected_overall_task=args.expected_overall_task,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    try:
        main()
    except (CaptionAuditError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
