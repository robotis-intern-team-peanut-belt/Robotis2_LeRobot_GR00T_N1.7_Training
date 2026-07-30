import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_pipeline.audit_omx_insert_captions import CaptionAuditError, audit


def fixture(root: Path, language_mode: str) -> None:
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "annotations/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"total_episodes": 1}))
    captions = [f"step {index}" for index in range(5)]
    tasks = ["overall task"] if language_mode == "full_episode" else captions
    pd.DataFrame({"task_index": range(len(tasks)), "task": tasks}).to_parquet(
        root / "meta/tasks.parquet"
    )
    pd.DataFrame(
        {"subtask_index": range(5), "subtask": captions}
    ).to_parquet(root / "meta/subtasks.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [{"episode_index": 0, "length": 10, "tasks": [tasks[0]], "subtask_instructions": captions}]
        ),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    rows = []
    for frame in range(10):
        subtask = frame // 2
        rows.append(
            {
                "episode_index": 0,
                "frame_index": frame,
                "task_index": subtask if language_mode == "per_subtask" else 0,
                "subtask_index": subtask,
            }
        )
    pq.write_table(
        pa.Table.from_pylist(rows), root / "data/chunk-000/file-000.parquet"
    )
    (root / "annotations/chunk-000/episode_000000.json").write_text(
        json.dumps(
            {
                "meta_data": {"task_duration": 10, "valid_duration": [0, 10]},
                "sub_task_annotation": [
                    {
                        "sub_task_idx": index,
                        "sub_task_instruction": caption,
                        "frame_duration": [index * 2, (index + 1) * 2],
                    }
                    for index, caption in enumerate(captions)
                ],
            }
        )
    )


def test_five_task_runs_are_visible_to_training(tmp_path: Path) -> None:
    fixture(tmp_path, language_mode="per_subtask")
    report = audit(tmp_path, expected_episodes=1, language_mode="per_subtask")
    assert report["language_contract_ready"] is True


def test_full_episode_prompt_with_annotations_is_accepted(tmp_path: Path) -> None:
    fixture(tmp_path, language_mode="full_episode")
    report = audit(
        tmp_path,
        expected_episodes=1,
        language_mode="full_episode",
        expected_overall_task="overall task",
    )
    assert report["language_contract_ready"] is True


def test_different_training_and_stored_caption_vocabularies_are_ambiguous(
    tmp_path: Path,
) -> None:
    fixture(tmp_path, language_mode="per_subtask")
    tasks = pd.read_parquet(tmp_path / "meta/tasks.parquet")
    tasks["task"] = [f"different {index}" for index in range(5)]
    tasks.to_parquet(tmp_path / "meta/tasks.parquet")
    with pytest.raises(CaptionAuditError, match="invalid_per_subtask"):
        audit(tmp_path, expected_episodes=1, language_mode="per_subtask")


def test_annotation_range_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture(tmp_path, language_mode="full_episode")
    path = tmp_path / "annotations/chunk-000/episode_000000.json"
    value = json.loads(path.read_text())
    value["sub_task_annotation"][1]["frame_duration"] = [3, 4]
    path.write_text(json.dumps(value))
    with pytest.raises(CaptionAuditError, match="annotation frame range mismatch"):
        audit(tmp_path, expected_episodes=1)
