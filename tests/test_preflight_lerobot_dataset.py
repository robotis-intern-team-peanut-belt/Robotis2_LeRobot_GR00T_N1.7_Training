import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_pipeline.preflight_lerobot_dataset import DatasetPreflightError, audit


def fixture(root: Path) -> None:
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "annotations/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps({"total_episodes": 2, "total_frames": 4})
    )
    pd.DataFrame({"task_index": [0], "task": ["overall task"]}).to_parquet(
        root / "meta/tasks.parquet"
    )
    pd.DataFrame(
        {"subtask_index": [0], "subtask": ["pick component"]}
    ).to_parquet(root / "meta/subtasks.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": episode,
                    "length": 2,
                    "dataset_from_index": episode * 2,
                    "dataset_to_index": episode * 2 + 2,
                    "tasks": ["overall task"],
                    "subtask_instructions": ["pick component"],
                }
                for episode in range(2)
            ]
        ),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"index": index, "episode_index": index // 2, "frame_index": index % 2}
                for index in range(4)
            ]
        ),
        root / "data/chunk-000/file-000.parquet",
    )
    for episode in range(2):
        (root / f"annotations/chunk-000/episode_{episode:06d}.json").write_text(
            json.dumps(
                {
                    "sub_task_annotation": [
                        {
                            "sub_task_idx": 0,
                            "sub_task_instruction": "pick component",
                            "frame_duration": [0, 2],
                        }
                    ]
                }
            )
        )


def test_valid_dataset_passes(tmp_path: Path) -> None:
    fixture(tmp_path)
    report = audit(tmp_path)
    assert report["ready"] is True
    assert report["episodes"] == 2
    assert report["frames"] == 4


@pytest.mark.parametrize(
    ("caption", "problem"),
    [
        ("pick  component", "repeated ASCII spaces"),
        (" pick component", "leading or trailing whitespace"),
        ("pick component ", "leading or trailing whitespace"),
        ("pick\tcomponent", r"U\+0009"),
        ("pick\ncomponent", r"U\+000A"),
        ("pick\u00a0component", r"U\+00A0"),
    ],
)
def test_suspicious_caption_whitespace_is_rejected(
    tmp_path: Path, caption: str, problem: str
) -> None:
    fixture(tmp_path)
    tasks = pd.read_parquet(tmp_path / "meta/tasks.parquet")
    tasks.loc[0, "task"] = caption
    tasks.to_parquet(tmp_path / "meta/tasks.parquet")
    with pytest.raises(DatasetPreflightError, match=problem):
        audit(tmp_path)


def test_second_file_local_episode_offsets_are_rejected(tmp_path: Path) -> None:
    """Reproduce the Task 700009 metadata defect that failed eval at step 5,000."""
    fixture(tmp_path)
    path = tmp_path / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[1]["dataset_from_index"] = 0
    rows[1]["dataset_to_index"] = 2
    pq.write_table(pa.Table.from_pylist(rows), path)

    with pytest.raises(
        DatasetPreflightError,
        match=r"episode 1: dataset_from_index=0, expected 2.*per-file-reset",
    ):
        audit(tmp_path)


def test_global_index_gap_is_rejected(tmp_path: Path) -> None:
    fixture(tmp_path)
    path = tmp_path / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[3]["index"] = 4
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(DatasetPreflightError, match="global index 4, expected 3"):
        audit(tmp_path)
