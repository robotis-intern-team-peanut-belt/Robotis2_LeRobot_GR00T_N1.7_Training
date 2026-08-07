import hashlib
import json
import os
from pathlib import Path
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

import registry
from data_pipeline.audit_omx_insert_captions import CaptionAuditError, audit
from data_pipeline.create_language_variants import (
    LANGUAGE_RECEIPT,
    SHORT_FULL_PROMPT,
    WORD_PROMPTS,
    create_variants,
)


RICH = [
    "Use left gripper to pick servo and insert into its cutout hole in foam bed.",
    "Use left gripper to pick electronic and insert into its cutout hole in foam bed.",
    "Use left gripper to pick wrapped cable and insert into its cutout hole in foam bed.",
    "Use right gripper to pick OMX-F3 and insert into its cutout hole in foam bed.",
    "Use right gripper to pick OMX-Base and insert into its cutout hole in foam bed.",
]
OVERALL = "Insert all five OMX components."


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v3_parent(root: Path) -> None:
    # One 45-frame episode with five nine-frame subtask runs gives six valid H40
    # starts, all of which cross the first subtask boundary.
    root.mkdir(parents=True)
    (root / "meta").mkdir()
    subtasks = [index for index in range(5) for _ in range(9)]
    table = pa.table(
        {
            "timestamp": pa.array([index / 15 for index in range(45)], pa.float32()),
            "frame_index": pa.array(range(45), pa.int64()),
            "episode_index": pa.array([0] * 45, pa.int64()),
            "index": pa.array(range(45), pa.int64()),
            "task_index": pa.array([0] * 45, pa.int64()),
            "subtask_index": pa.array(subtasks, pa.int64()),
        }
    )
    path = root / "data/chunk-000/file-000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(table, path)
    pd.DataFrame(
        {"task_index": [0], "task_name": ["OMX_Insert"], "task": [OVERALL]}
    ).set_index("task", drop=False).to_parquet(root / "meta/tasks.parquet")
    pd.DataFrame(
        {"subtask_index": list(range(5)), "subtask": RICH}
    ).to_parquet(root / "meta/subtasks.parquet")
    episode = {
        "episode_index": 0,
        "length": 45,
        "tasks": [OVERALL],
        "subtask_instructions": RICH,
        "stats/task_index/mean": [0.0],
        "stats/task_index/std": [0.0],
        "stats/task_index/min": [0],
        "stats/task_index/max": [0],
        "stats/task_index/count": [45],
    }
    episode_path = root / "meta/episodes/chunk-000/file-000.parquet"
    episode_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([episode]), episode_path)
    _write_json(
        root / "meta/info.json",
        {"codebase_version": "v3.0", "total_episodes": 1, "total_tasks": 1},
    )
    _write_json(
        root / "meta/stats.json",
        {"task_index": {"min": [0], "max": [0], "mean": [0], "std": [0]}},
    )
    annotation = {
        "meta_data": {"task_duration": 45, "valid_duration": [0, 45]},
        "sub_task_annotation": [
            {
                "sub_task_idx": index,
                "sub_task_instruction": caption,
                "frame_duration": [index * 9, (index + 1) * 9],
            }
            for index, caption in enumerate(RICH)
        ],
    }
    _write_json(root / "annotations/chunk-000/episode_000000.json", annotation)
    video = root / "videos/observation.images.rgb.cam_head/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"synthetic-mp4-payload")
    _write_json(
        root / ".groot_registry_download.json",
        {
            "repo_id": "local/canonical",
            "revision": "canonical-r1",
            "source_kind": "direct_local_directory",
        },
    )


def _v2_parent(v3: Path, root: Path) -> None:
    shutil.copytree(v3, root)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["codebase_version"] = "v2.1"
    _write_json(info_path, info)
    for path in (root / "meta/tasks.parquet", root / "meta/subtasks.parquet"):
        path.unlink()
    shutil.rmtree(root / "meta/episodes")
    data_path = root / "data/chunk-000/file-000.parquet"
    data = pq.read_table(data_path)
    data = data.append_column(
        "annotation.human.task_description", pa.array([0] * 45, pa.int64())
    )
    data_path.unlink()
    native_path = root / "data/chunk-000/episode_000000.parquet"
    pq.write_table(data, native_path)
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": OVERALL}])
    _write_jsonl(
        root / "meta/episodes.jsonl",
        [
            {
                "episode_index": 0,
                "length": 45,
                "tasks": [OVERALL],
                "subtask_instructions": RICH,
            }
        ],
    )
    _write_jsonl(
        root / "meta/episodes_stats.jsonl",
        [
            {
                "episode_index": 0,
                "stats": {
                    "task_index": {
                        "min": [0],
                        "max": [0],
                        "mean": [0],
                        "std": [0],
                        "count": [45],
                    }
                },
            }
        ],
    )
    _write_json(
        root / "ISAAC_DERIVATION_RECEIPT.json",
        {
            "schema_version": 1,
            "dataset": "canonical",
            "source_root": str(v3),
            "source_revision": "canonical-r1",
            "source_metadata_sha256": "0" * 64,
            "converter_commit": "converter-test-commit",
            "modality_json_sha256": "1" * 64,
            "modality_config_sha256": "2" * 64,
            "action_representation": "absolute",
            "action_horizon": 40,
        },
    )


@pytest.fixture
def language_parents(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "canonical-source"
    prepared = tmp_path / "canonical-prepared"
    isaac = tmp_path / "canonical-isaac"
    _v3_parent(source)
    shutil.copytree(source, prepared)
    _write_json(
        prepared / "DERIVATION_RECEIPT.json",
        {
            "schema_version": 1,
            "source": str(source),
            "source_revision": "canonical-r1",
            "source_manifest_sha256": "3" * 64,
        },
    )
    _v2_parent(prepared, isaac)
    registration = tmp_path / "canonical.yaml"
    registration.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "canonical",
                "repo_id": "local/canonical",
                "revision": "canonical-r1",
                "download": {"root": str(source)},
                "prepared": {"root": str(prepared), "immutable": True},
                "contract": {"language": {"mode": "full_episode"}},
                "validation": {},
                "training": {"repo_id": "local/canonical-prepared"},
                "adapters": {"isaac_groot": {"root": str(isaac)}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return source, prepared, isaac, registration


def test_create_all_language_views_without_duplicating_media(
    language_parents: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    source, prepared, isaac, parent_registration = language_parents
    output = tmp_path / "datasets"
    registrations = tmp_path / "configs/datasets"
    results = create_variants(
        source_parent=source,
        prepared_parent=prepared,
        isaac_parent=isaac,
        parent_registration=parent_registration,
        output_root=output,
        registrations_dir=registrations,
        name_prefix="Task_700013_test",
    )

    assert [result["language_mode"] for result in results] == [
        "full_episode",
        "per_subtask",
        "mapped_subtask",
    ]
    parent_video = next((source / "videos").rglob("*.mp4"))
    parent_data = next((source / "data").rglob("*.parquet"))
    original_annotation = source / "annotations/chunk-000/episode_000000.json"
    for result in results:
        for kind, parent in (("source", source), ("prepared", prepared), ("isaac", isaac)):
            root = Path(result["paths"][kind])
            video = next((root / "videos").rglob("*.mp4"))
            assert (video.stat().st_dev, video.stat().st_ino) == (
                next((parent / "videos").rglob("*.mp4")).stat().st_dev,
                next((parent / "videos").rglob("*.mp4")).stat().st_ino,
            )
            receipt = json.loads((root / LANGUAGE_RECEIPT).read_text(encoding="utf-8"))
            assert receipt["media"]["hardlinked_mp4_count"] == 1
            assert receipt["h40"]["valid_h40_start_count"] == 6
            assert receipt["h40"]["cross_boundary_start_count"] == 6
        variant_data = next((Path(result["paths"]["source"]) / "data").rglob("*.parquet"))
        assert variant_data.stat().st_ino != parent_data.stat().st_ino
        annotation = Path(result["paths"]["source"]) / original_annotation.relative_to(source)
        assert _sha256(annotation) == _sha256(original_annotation)

    assert parent_video.stat().st_nlink >= 4

    short, rich, words = results
    short_root = Path(short["paths"]["source"])
    short_report = audit(
        short_root,
        expected_episodes=1,
        language_mode="full_episode",
        expected_overall_task=SHORT_FULL_PROMPT,
        expected_task_vocabulary=[SHORT_FULL_PROMPT],
    )
    assert short_report["language_contract_ready"] is True

    rich_report = audit(
        Path(rich["paths"]["source"]),
        expected_episodes=1,
        language_mode="per_subtask",
        expected_task_vocabulary=RICH,
    )
    assert rich_report["episode_summaries"][0]["training_captions"] == RICH

    words_root = Path(words["paths"]["source"])
    words_report = audit(
        words_root,
        expected_episodes=1,
        language_mode="mapped_subtask",
        expected_task_vocabulary=WORD_PROMPTS,
    )
    assert words_report["episode_summaries"][0]["training_captions"] == WORD_PROMPTS
    assert words_report["episode_summaries"][0]["stored_subtask_captions"] == RICH
    assert pd.read_parquet(words_root / "meta/subtasks.parquet")["subtask"].tolist() == RICH
    assert _sha256(
        words_root / "annotations/chunk-000/episode_000000.json"
    ) == _sha256(original_annotation)
    word_data = pq.read_table(next((words_root / "data").rglob("*.parquet")))
    assert word_data["task_index"].to_pylist() == word_data["subtask_index"].to_pylist()

    native = Path(words["paths"]["isaac"])
    native_data = pq.read_table(next((native / "data").rglob("*.parquet")))
    assert native_data["task_index"].to_pylist() == native_data[
        "annotation.human.task_description"
    ].to_pylist()
    native_receipt = json.loads(
        (native / "ISAAC_DERIVATION_RECEIPT.json").read_text(encoding="utf-8")
    )
    assert native_receipt["source_root"] == words["paths"]["prepared"]
    assert native_receipt["action_horizon"] == 40

    rich_registration = yaml.safe_load(Path(rich["registration"]).read_text())
    assert rich_registration["contract"]["language"]["convention"].startswith(
        "Frame-aligned task_index selects one of five ordered rich subtask prompts"
    )

    word_registration = yaml.safe_load(Path(words["registration"]).read_text())
    caption_audit = word_registration["validation"]["caption_audit"]
    assert caption_audit["language_mode"] == "mapped_subtask"
    assert caption_audit["expected_task_vocabulary"] == WORD_PROMPTS
    assert "expected_overall_task" not in caption_audit
    assert "preparation" not in word_registration
    assert word_registration["contract"]["language"]["convention"].startswith(
        "Frame-aligned task_index selects one of five object-token prompts"
    )
    registry_report = registry.audit_dataset_captions(word_registration)
    assert registry_report is not None
    assert registry_report["language_mode"] == "mapped_subtask"
    assert registry_report["language_contract_ready"] is True


def test_audit_requires_overall_task_only_for_full_episode(
    language_parents: tuple[Path, Path, Path, Path]
) -> None:
    source, _, _, _ = language_parents
    with pytest.raises(CaptionAuditError, match="requires expected_overall_task"):
        audit(source, language_mode="full_episode")
    with pytest.raises(CaptionAuditError, match="requires expected_task_vocabulary"):
        audit(source, language_mode="mapped_subtask")
    # The per-subtask mode gets past argument validation without an overall prompt;
    # this canonical fixture then fails only because its task mapping is full-episode.
    with pytest.raises(CaptionAuditError, match="task channel does not match"):
        audit(source, language_mode="per_subtask")
