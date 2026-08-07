#!/usr/bin/env python3
"""Audit OMX full-episode, direct-subtask, and mapped-subtask language."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


class CaptionAuditError(RuntimeError):
    """The converted dataset does not satisfy the selected language contract."""


LANGUAGE_MODES = ("full_episode", "per_subtask", "mapped_subtask")


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-subtasks-per-episode", type=int, default=5)
    parser.add_argument("--language-mode", choices=LANGUAGE_MODES, default="full_episode")
    parser.add_argument("--expected-overall-task")
    parser.add_argument(
        "--expected-task",
        action="append",
        dest="expected_task_vocabulary",
        help="Expected task text in task_index order; repeat once per task.",
    )
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
    result: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            result.append((values[start], start, index))
            start = index
    return result


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


def _validate_requested_vocabulary(
    tasks: dict[int, str], expected_task_vocabulary: list[str] | None
) -> None:
    if expected_task_vocabulary is None:
        return
    if any(not isinstance(item, str) or not item.strip() for item in expected_task_vocabulary):
        raise CaptionAuditError("expected task vocabulary entries must be nonempty strings")
    expected = dict(enumerate(expected_task_vocabulary))
    if tasks != expected:
        raise CaptionAuditError(
            f"task vocabulary mismatch: expected {expected!r}, observed {tasks!r}"
        )


def audit(
    root: Path,
    expected_episodes: int | None = None,
    expected_subtasks: int = 5,
    language_mode: str = "full_episode",
    expected_overall_task: str | None = None,
    expected_task_vocabulary: list[str] | None = None,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if language_mode not in LANGUAGE_MODES:
        raise CaptionAuditError(f"unsupported language mode: {language_mode}")
    if language_mode == "full_episode" and not expected_overall_task:
        raise CaptionAuditError("full_episode requires expected_overall_task")
    if language_mode == "mapped_subtask" and expected_task_vocabulary is None:
        raise CaptionAuditError("mapped_subtask requires expected_task_vocabulary")

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
    _validate_requested_vocabulary(tasks, expected_task_vocabulary)

    rows: list[dict[str, int]] = []
    for path in sorted((root / "data").glob("**/*.parquet")):
        schema_names = pq.read_schema(path).names
        required = {"episode_index", "frame_index", "task_index", "subtask_index"}
        missing = required - set(schema_names)
        if missing:
            raise CaptionAuditError(f"{path}: missing {sorted(missing)}")
        rows.extend(pq.read_table(path, columns=sorted(required)).to_pylist())
    if not rows:
        raise CaptionAuditError("no data parquet rows found")

    by_episode: dict[int, list[dict[str, int]]] = {}
    for row in rows:
        by_episode.setdefault(int(row["episode_index"]), []).append(row)
    observed_episodes = sorted(by_episode)
    if observed_episodes != list(range(len(observed_episodes))):
        raise CaptionAuditError("episode indices must be contiguous from zero")
    if int(info.get("total_episodes", -1)) != len(observed_episodes):
        raise CaptionAuditError("metadata and parquet episode counts differ")
    if expected_episodes is not None and len(observed_episodes) != expected_episodes:
        raise CaptionAuditError(
            f"expected {expected_episodes} episodes, found {len(observed_episodes)}"
        )

    metadata_by_episode = episode_metadata(root)
    expected_order = list(range(expected_subtasks))
    summaries: list[dict[str, Any]] = []
    modes: set[str] = set()
    for episode in observed_episodes:
        episode_rows = sorted(by_episode[episode], key=lambda row: int(row["frame_index"]))
        frame_indices = [int(row["frame_index"]) for row in episode_rows]
        if frame_indices != list(range(len(frame_indices))):
            raise CaptionAuditError(f"episode {episode}: frame indices are not contiguous")
        task_runs = runs([int(row["task_index"]) for row in episode_rows])
        subtask_values = [int(row["subtask_index"]) for row in episode_rows]
        subtask_runs = runs(subtask_values)
        if any(index not in tasks for index in task_runs):
            raise CaptionAuditError(f"episode {episode}: unresolved task_index")
        if any(index not in subtasks for index in subtask_runs):
            raise CaptionAuditError(f"episode {episode}: unresolved subtask_index")
        if subtask_runs != expected_order:
            raise CaptionAuditError(
                f"episode {episode}: expected ordered subtasks {expected_order}, got {subtask_runs}"
            )

        metadata = metadata_by_episode.get(episode)
        if not metadata:
            raise CaptionAuditError(f"episode {episode}: missing episode metadata")
        rich_captions = [subtasks[index] for index in subtask_runs]
        training_captions = [tasks[index] for index in task_runs]
        if int(metadata.get("length", -1)) != len(episode_rows):
            raise CaptionAuditError(f"episode {episode}: metadata length mismatch")
        if metadata.get("subtask_instructions") != rich_captions:
            raise CaptionAuditError(f"episode {episode}: rich subtask metadata mismatch")
        annotation = validate_annotation(
            root,
            episode,
            len(episode_rows),
            run_segments(subtask_values),
            subtasks,
        )

        if language_mode == "full_episode":
            valid = task_runs == [0] and metadata.get("tasks") == [tasks[0]]
            mode = (
                "full_episode_prompt_with_subtask_annotations"
                if valid
                else "invalid_full_episode_task_mapping"
            )
        elif language_mode == "per_subtask":
            valid = (
                task_runs == expected_order
                and training_captions == rich_captions
                and metadata.get("tasks") == training_captions
            )
            mode = (
                "per_frame_subtasks_visible_to_training"
                if valid
                else "invalid_per_subtask_task_mapping"
            )
        else:
            valid = (
                task_runs == subtask_runs == expected_order
                and metadata.get("tasks") == training_captions
            )
            mode = (
                "mapped_subtasks_visible_with_rich_annotations"
                if valid
                else "invalid_mapped_subtask_task_mapping"
            )
        modes.add(mode)
        summaries.append(
            {
                "episode_index": episode,
                "frames": len(episode_rows),
                "mode": mode,
                "task_run_indices": task_runs,
                "training_captions": training_captions,
                "subtask_run_indices": subtask_runs,
                "stored_subtask_captions": rich_captions,
                "annotation": annotation,
            }
        )

    expected_mode = {
        "full_episode": "full_episode_prompt_with_subtask_annotations",
        "per_subtask": "per_frame_subtasks_visible_to_training",
        "mapped_subtask": "mapped_subtasks_visible_with_rich_annotations",
    }[language_mode]
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
            + "\nThe task channel does not match the selected language mode."
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
        expected_task_vocabulary=args.expected_task_vocabulary,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    try:
        main()
    except (CaptionAuditError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=__import__("sys").stderr)
        raise SystemExit(1)
