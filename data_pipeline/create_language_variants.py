#!/usr/bin/env python3
"""Create receipt-bound Task 700013 language views without duplicating MP4 data.

The command consumes an already completed canonical source, prepared LeRobot v3,
and native Isaac v2.1 derivative.  It creates three immutable views:

* one short prompt on every frame;
* the five rich subtask sentences selected by ``subtask_index``; and
* five object-word prompts selected by ``subtask_index`` while preserving the
  rich annotation vocabulary independently.

Only MP4 files are hardlinked.  Every other file is copied before task-bearing
Parquet/JSON metadata is atomically replaced.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


SHORT_FULL_PROMPT = "servo, electronic, cable, OMX-F3, OMX-Base"
MAINLANG_SERVO_FIRST_PROMPT = (
    "Insert 5 OMX kit components (servo, electronic, wrapped cable, light grey "
    "OMX-F3, light grey OMX-Base) into their cutout holes in ivory foam bed."
)
WORD_PROMPTS = ["servo", "electronic", "cable", "OMX-F3", "OMX-Base"]
HORIZON = 40
LANGUAGE_RECEIPT = "LANGUAGE_VARIANT_RECEIPT.json"


class LanguageVariantError(RuntimeError):
    """A parent or generated language view violates the immutable contract."""


@dataclass(frozen=True)
class Variant:
    suffix: str
    language_mode: str
    vocabulary: list[str]
    use_subtask_index: bool


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


def atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(path, yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True))


def atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(table, temporary, compression="snappy", use_dictionary=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in (root / "meta").rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def payload_digest(root: Path) -> tuple[str, int, int]:
    """Return a stable payload digest excluding self-referential receipts/markers."""
    excluded = {
        LANGUAGE_RECEIPT,
        "DERIVATION_RECEIPT.json",
        "ISAAC_DERIVATION_RECEIPT.json",
        ".groot_registry_download.json",
    }
    lines: list[str] = []
    count = size = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in excluded:
            continue
        relative = path.relative_to(root).as_posix()
        length = path.stat().st_size
        lines.append(f"{sha256_file(path)}  {relative}")
        count += 1
        size += length
    rendered = "\n".join(lines) + ("\n" if lines else "")
    return hashlib.sha256(rendered.encode()).hexdigest(), count, size


def require_parent(root: Path, version: str, receipt: str | None = None) -> Path:
    root = root.expanduser().resolve()
    info = root / "meta" / "info.json"
    if not root.is_dir() or not info.is_file():
        raise LanguageVariantError(f"missing dataset parent: {root}")
    observed = json.loads(info.read_text(encoding="utf-8")).get("codebase_version")
    if observed != version:
        raise LanguageVariantError(f"{root}: expected {version}, observed {observed!r}")
    if receipt is not None and not (root / receipt).is_file():
        raise LanguageVariantError(f"{root}: missing parent {receipt}")
    return root


def clone_tree(parent: Path, stage: Path) -> dict[str, Any]:
    """Copy a dataset tree while hardlinking regular MP4 payloads only."""
    if stage.exists():
        raise LanguageVariantError(f"staging path already exists: {stage}")
    stage.mkdir(parents=True)
    media: list[dict[str, Any]] = []
    copied = 0
    for source in sorted(parent.rglob("*")):
        relative = source.relative_to(parent)
        destination = stage / relative
        if source.is_symlink():
            raise LanguageVariantError(f"symlinks are not accepted in dataset parents: {source}")
        if source.is_dir():
            destination.mkdir(exist_ok=True)
            continue
        if not source.is_file():
            raise LanguageVariantError(f"unsupported parent entry: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative.parts and relative.parts[0] == "videos" and source.suffix.lower() == ".mp4":
            if source.stat().st_dev != stage.stat().st_dev:
                raise LanguageVariantError("parent and destination must share a filesystem for MP4 hardlinks")
            os.link(source, destination)
            if (source.stat().st_dev, source.stat().st_ino) != (
                destination.stat().st_dev,
                destination.stat().st_ino,
            ):
                raise LanguageVariantError(f"hardlink verification failed: {destination}")
            media.append(
                {
                    "path": relative.as_posix(),
                    "bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                }
            )
        else:
            shutil.copy2(source, destination)
            copied += 1
            if source.stat().st_ino == destination.stat().st_ino:
                raise LanguageVariantError(f"non-media file was unexpectedly hardlinked: {destination}")
    return {
        "hardlinked_mp4_count": len(media),
        "hardlinked_mp4_bytes": sum(item["bytes"] for item in media),
        "hardlinked_mp4": media,
        "copied_non_media_files": copied,
    }


def rich_subtasks(root: Path) -> list[str]:
    path = root / "meta" / "subtasks.parquet"
    frame = pd.read_parquet(path)
    if "subtask_index" not in frame.columns or "subtask" not in frame.columns:
        raise LanguageVariantError(f"{path}: expected subtask_index and subtask columns")
    indices = [int(value) for value in frame["subtask_index"].tolist()]
    values = frame["subtask"].astype(str).tolist()
    if indices != list(range(5)) or len(values) != 5 or any(not item.strip() for item in values):
        raise LanguageVariantError(f"{path}: expected five contiguous rich subtasks")
    return values


def task_indices(table: pa.Table, variant: Variant) -> pa.Array:
    if not variant.use_subtask_index:
        return pa.array([0] * len(table), type=pa.int64())
    if "subtask_index" not in table.column_names:
        raise LanguageVariantError("per-subtask language requires subtask_index")
    values = table["subtask_index"].to_pylist()
    if any(int(value) not in range(len(variant.vocabulary)) for value in values):
        raise LanguageVariantError("subtask_index is outside the task vocabulary")
    return pa.array([int(value) for value in values], type=pa.int64())


def replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    if name not in table.column_names:
        return table.append_column(name, values)
    field = table.schema.field(name)
    return table.set_column(table.schema.get_field_index(name), field, values.cast(field.type))


def numeric_stats(values: Iterable[int], include_count: bool = False) -> dict[str, list[Any]]:
    items = [int(value) for value in values]
    if not items:
        raise LanguageVariantError("cannot compute task_index stats for no rows")
    mean = sum(items) / len(items)
    variance = sum((value - mean) ** 2 for value in items) / len(items)
    result: dict[str, list[Any]] = {
        "mean": [mean],
        "std": [math.sqrt(variance)],
        "min": [float(min(items))],
        "max": [float(max(items))],
    }
    if include_count:
        result["count"] = [len(items)]
    return result


def rewrite_tasks_parquet(root: Path, vocabulary: list[str]) -> None:
    path = root / "meta" / "tasks.parquet"
    old = pd.read_parquet(path)
    task_name = None
    if "task_name" in old.columns and len(old):
        task_name = str(old["task_name"].iloc[0])
    data: dict[str, Any] = {"task_index": list(range(len(vocabulary)))}
    if task_name is not None:
        data["task_name"] = [task_name] * len(vocabulary)
    if old.index.name == "task" and "task" not in old.columns:
        frame = pd.DataFrame(data, index=pd.Index(vocabulary, name="task"))
    else:
        data = {"task_index": list(range(len(vocabulary))), "task": vocabulary, **(
            {"task_name": [task_name] * len(vocabulary)} if task_name is not None else {}
        )}
        frame = pd.DataFrame(data)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def rewrite_v3_data(root: Path, variant: Variant) -> dict[int, list[int]]:
    by_episode: dict[int, list[tuple[int, int]]] = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        values = task_indices(table, variant)
        table = replace_column(table, "task_index", values)
        episodes = table["episode_index"].to_pylist()
        frames = table["frame_index"].to_pylist()
        for episode, frame, task in zip(episodes, frames, values.to_pylist(), strict=True):
            by_episode.setdefault(int(episode), []).append((int(frame), int(task)))
        atomic_parquet(path, table)
    ordered: dict[int, list[int]] = {}
    for episode, values in by_episode.items():
        values.sort()
        frames = [frame for frame, _ in values]
        if frames != list(range(len(frames))):
            raise LanguageVariantError(f"episode {episode}: non-contiguous frame_index")
        ordered[episode] = [task for _, task in values]
    return ordered


def update_info_and_global_stats(root: Path, values: list[int], total_tasks: int) -> None:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_tasks"] = total_tasks
    atomic_json(info_path, info)
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["task_index"] = numeric_stats(values)
    atomic_json(stats_path, stats)


def rewrite_v3_episodes(
    root: Path, by_episode: dict[int, list[int]], vocabulary: list[str]
) -> None:
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        original = pq.read_table(path)
        rows = original.to_pylist()
        for row in rows:
            episode = int(row["episode_index"])
            values = by_episode[episode]
            seen: list[int] = []
            for value in values:
                if not seen or seen[-1] != value:
                    seen.append(value)
            row["tasks"] = [vocabulary[index] for index in seen]
            stats = numeric_stats(values, include_count=True)
            for key, item in stats.items():
                row[f"stats/task_index/{key}"] = item
        atomic_parquet(path, pa.Table.from_pylist(rows, schema=original.schema))


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows))


def rewrite_v2(root: Path, variant: Variant) -> dict[int, list[int]]:
    by_episode: dict[int, list[tuple[int, int]]] = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        values = task_indices(table, variant)
        table = replace_column(table, "task_index", values)
        annotation = "annotation.human.task_description"
        if annotation not in table.column_names:
            raise LanguageVariantError(f"{path}: missing {annotation}")
        table = replace_column(table, annotation, values)
        for episode, frame, task in zip(
            table["episode_index"].to_pylist(),
            table["frame_index"].to_pylist(),
            values.to_pylist(),
            strict=True,
        ):
            by_episode.setdefault(int(episode), []).append((int(frame), int(task)))
        atomic_parquet(path, table)

    ordered: dict[int, list[int]] = {}
    for episode, values in by_episode.items():
        values.sort()
        frames = [frame for frame, _ in values]
        if frames != list(range(len(frames))):
            raise LanguageVariantError(f"episode {episode}: non-contiguous frame_index")
        ordered[episode] = [task for _, task in values]
    atomic_jsonl(
        root / "meta" / "tasks.jsonl",
        ({"task_index": index, "task": task} for index, task in enumerate(variant.vocabulary)),
    )
    episodes_path = root / "meta" / "episodes.jsonl"
    episodes = jsonl(episodes_path)
    for row in episodes:
        values = ordered[int(row["episode_index"])]
        seen: list[int] = []
        for value in values:
            if not seen or seen[-1] != value:
                seen.append(value)
        row["tasks"] = [variant.vocabulary[index] for index in seen]
    atomic_jsonl(episodes_path, episodes)

    stats_path = root / "meta" / "episodes_stats.jsonl"
    episode_stats = jsonl(stats_path)
    for row in episode_stats:
        row.setdefault("stats", {})["task_index"] = numeric_stats(
            ordered[int(row["episode_index"])], include_count=True
        )
    atomic_jsonl(stats_path, episode_stats)
    all_values = [value for episode in sorted(ordered) for value in ordered[episode]]
    update_info_and_global_stats(root, all_values, len(variant.vocabulary))
    return ordered


def boundary_manifest(parent: Path, horizon: int = HORIZON) -> tuple[pa.Table, dict[str, Any]]:
    by_episode: dict[int, list[tuple[int, int, int]]] = {}
    for path in sorted((parent / "data").rglob("*.parquet")):
        table = pq.read_table(
            path, columns=["index", "episode_index", "frame_index", "subtask_index"]
        )
        columns = table.to_pydict()
        for index, episode, frame, subtask in zip(
            columns["index"],
            columns["episode_index"],
            columns["frame_index"],
            columns["subtask_index"],
            strict=True,
        ):
            by_episode.setdefault(int(episode), []).append(
                (int(frame), int(subtask), int(index))
            )

    rows: list[dict[str, int]] = []
    valid = boundaries = 0
    segment_lengths: dict[int, list[int]] = {}
    for episode, values in sorted(by_episode.items()):
        values.sort()
        subtasks = [value[1] for value in values]
        starts = [0]
        for index in range(1, len(subtasks)):
            if subtasks[index] != subtasks[index - 1]:
                starts.append(index)
        starts.append(len(subtasks))
        boundaries += max(0, len(starts) - 2)
        for left, right in zip(starts[:-1], starts[1:], strict=True):
            segment_lengths.setdefault(subtasks[left], []).append(right - left)
        for start in range(max(0, len(values) - horizon + 1)):
            valid += 1
            first = subtasks[start]
            offset = next(
                (delta for delta in range(1, horizon) if subtasks[start + delta] != first),
                None,
            )
            if offset is None:
                continue
            rows.append(
                {
                    "episode_index": episode,
                    "frame_index": values[start][0],
                    "index": values[start][2],
                    "start_subtask_index": first,
                    "next_subtask_index": subtasks[start + offset],
                    "boundary_frame_index": values[start + offset][0],
                    "first_next_offset": offset,
                }
            )
    schema = pa.schema(
        [
            ("episode_index", pa.int64()),
            ("frame_index", pa.int64()),
            ("index", pa.int64()),
            ("start_subtask_index", pa.int64()),
            ("next_subtask_index", pa.int64()),
            ("boundary_frame_index", pa.int64()),
            ("first_next_offset", pa.int64()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    stats = {
        "action_horizon": horizon,
        "episode_count": len(by_episode),
        "subtask_boundary_count": boundaries,
        "valid_h40_start_count": valid,
        "cross_boundary_start_count": len(rows),
        "clean_start_count": valid - len(rows),
        "cross_boundary_fraction": (len(rows) / valid if valid else 0.0),
        "segment_lengths": {
            str(key): {"count": len(items), "min": min(items), "max": max(items)}
            for key, items in sorted(segment_lengths.items())
        },
        "sampling_change_applied": False,
    }
    return table, stats


def write_boundary_manifest(root: Path, table: pa.Table) -> tuple[str, str]:
    relative = "meta/language_h40_boundary_starts.parquet"
    path = root / relative
    atomic_parquet(path, table)
    return relative, sha256_file(path)


def language_receipt(
    *,
    variant: Variant,
    parent: Path,
    destination: Path,
    tree_kind: str,
    media: dict[str, Any],
    boundary_stats: dict[str, Any],
    boundary_path: str,
    boundary_sha256: str,
) -> dict[str, Any]:
    parent_receipts = [
        name
        for name in (
            ".groot_registry_download.json",
            "DERIVATION_RECEIPT.json",
            "ISAAC_DERIVATION_RECEIPT.json",
        )
        if (parent / name).is_file()
    ]
    return {
        "schema_version": 1,
        "kind": "task700013_language_variant",
        "tree_kind": tree_kind,
        "parent_root": str(parent.resolve()),
        "destination_root": str(destination.resolve()),
        "parent_receipts": {
            name: sha256_file(parent / name) for name in parent_receipts
        },
        "language_mode": variant.language_mode,
        "task_vocabulary": variant.vocabulary,
        "task_index_source": "subtask_index" if variant.use_subtask_index else "constant_zero",
        "rich_subtask_annotations_preserved": True,
        "media": media,
        "h40_boundary_manifest": boundary_path,
        "h40_boundary_manifest_sha256": boundary_sha256,
        "h40": {
            **boundary_stats,
            "language_conflict_start_count": (
                boundary_stats["cross_boundary_start_count"]
                if variant.use_subtask_index
                else 0
            ),
            "limitation": (
                "Current-frame language supervises an H40 action chunk; boundary starts mix two subtasks."
                if variant.use_subtask_index
                else "The one full-episode prompt covers both sides of each subtask boundary."
            ),
        },
    }


def promote(stage: Path, destination: Path) -> None:
    if destination.exists():
        raise LanguageVariantError(f"destination already exists: {destination}")
    os.replace(stage, destination)


def create_one(
    *,
    variant: Variant,
    name: str,
    revision: str,
    source_parent: Path,
    prepared_parent: Path,
    isaac_parent: Path,
    output_root: Path,
) -> dict[str, Any]:
    destinations = {
        "source": output_root / f"{name}_source_v30",
        "prepared": output_root / f"{name}_prepared_v30",
        "isaac": output_root / f"{name}_isaac_v21",
    }
    if any(path.exists() for path in destinations.values()):
        existing = [str(path) for path in destinations.values() if path.exists()]
        raise LanguageVariantError(f"variant destination exists: {existing}")
    boundary_table, boundary_stats = boundary_manifest(source_parent)
    stages: dict[str, Path] = {}
    try:
        for kind, parent in (
            ("source", source_parent),
            ("prepared", prepared_parent),
            ("isaac", isaac_parent),
        ):
            destination = destinations[kind]
            stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=output_root))
            stage.rmdir()
            stages[kind] = stage
            media = clone_tree(parent, stage)
            validation = stage / "meta" / "isaac_validation.json"
            if kind == "isaac" and validation.exists():
                validation.unlink()
            if kind in {"source", "prepared"}:
                rewrite_tasks_parquet(stage, variant.vocabulary)
                by_episode = rewrite_v3_data(stage, variant)
                rewrite_v3_episodes(stage, by_episode, variant.vocabulary)
                all_values = [
                    value for episode in sorted(by_episode) for value in by_episode[episode]
                ]
                update_info_and_global_stats(stage, all_values, len(variant.vocabulary))
            else:
                rewrite_v2(stage, variant)
            boundary_path, boundary_sha = write_boundary_manifest(stage, boundary_table)
            receipt = language_receipt(
                variant=variant,
                parent=parent,
                destination=destination,
                tree_kind=kind,
                media=media,
                boundary_stats=boundary_stats,
                boundary_path=boundary_path,
                boundary_sha256=boundary_sha,
            )
            atomic_json(stage / LANGUAGE_RECEIPT, receipt)

        source_digest, source_files, source_bytes = payload_digest(stages["source"])
        atomic_json(
            stages["source"] / ".groot_registry_download.json",
            {
                "repo_id": f"local/{name}",
                "revision": revision,
                "source_kind": "direct_local_directory",
                "payload_manifest_sha256": source_digest,
                "payload_file_count": source_files,
                "payload_bytes": source_bytes,
            },
        )
        parent_prepared_receipt = json.loads(
            (prepared_parent / "DERIVATION_RECEIPT.json").read_text(encoding="utf-8")
        )
        atomic_json(
            stages["prepared"] / "DERIVATION_RECEIPT.json",
            {
                **parent_prepared_receipt,
                "source": str(destinations["source"].resolve()),
                "source_revision": revision,
                "source_manifest_sha256": source_digest,
                "language_variant_receipt": str(
                    (destinations["prepared"] / LANGUAGE_RECEIPT).resolve()
                ),
                "language_mode": variant.language_mode,
                "task_vocabulary": variant.vocabulary,
            },
        )

        parent_isaac_receipt = json.loads(
            (isaac_parent / "ISAAC_DERIVATION_RECEIPT.json").read_text(encoding="utf-8")
        )
        required = {
            "converter_commit",
            "modality_json_sha256",
            "modality_config_sha256",
            "action_representation",
            "action_horizon",
        }
        if missing := required - set(parent_isaac_receipt):
            raise LanguageVariantError(f"Isaac parent receipt lacks {sorted(missing)}")
        atomic_json(
            stages["isaac"] / "ISAAC_DERIVATION_RECEIPT.json",
            {
                "schema_version": 1,
                "dataset": name,
                "source_root": str(destinations["prepared"].resolve()),
                "source_revision": revision,
                "source_metadata_sha256": metadata_digest(stages["prepared"]),
                "converter_commit": parent_isaac_receipt["converter_commit"],
                "modality_json_sha256": parent_isaac_receipt["modality_json_sha256"],
                "modality_config_sha256": parent_isaac_receipt["modality_config_sha256"],
                "action_representation": parent_isaac_receipt["action_representation"],
                "action_horizon": parent_isaac_receipt["action_horizon"],
                "native_root": str(destinations["isaac"].resolve()),
                "validation": {"status": "pending_variant_decode_validation"},
                "language_variant_receipt": str(
                    (destinations["isaac"] / LANGUAGE_RECEIPT).resolve()
                ),
            },
        )
        for kind in ("source", "prepared", "isaac"):
            promote(stages.pop(kind), destinations[kind])
        return {
            "name": name,
            "revision": revision,
            "paths": {key: str(value.resolve()) for key, value in destinations.items()},
            "source_manifest_sha256": source_digest,
            "language_mode": variant.language_mode,
            "task_vocabulary": variant.vocabulary,
            "h40": boundary_stats,
        }
    finally:
        for stage in stages.values():
            if stage.exists():
                shutil.rmtree(stage)


def registration(
    parent: Mapping[str, Any], result: Mapping[str, Any], registrations_dir: Path
) -> Path:
    value = json.loads(json.dumps(parent))
    name = str(result["name"])
    paths = result["paths"]
    mode = str(result["language_mode"])
    vocabulary = list(result["task_vocabulary"])
    value.update(
        {
            "name": name,
            "repo_id": f"local/{name}",
            "revision": result["revision"],
            "source_manifest_sha256": result["source_manifest_sha256"],
            "download": {"root": paths["source"]},
            "prepared": {"root": paths["prepared"], "immutable": True},
        }
    )
    value.pop("preparation", None)
    value.setdefault("registration", {}).update(
        {
            "source_kind": "direct_local_directory",
            "language_variant": True,
            "language_mode": mode,
            "language_variant_receipt": str(
                (Path(paths["source"]) / LANGUAGE_RECEIPT).resolve()
            ),
        }
    )
    audit: dict[str, Any] = {
        "enabled": True,
        "profile": "omx_insert",
        "language_mode": mode,
        "expected_subtasks_per_episode": 5,
        "expected_task_vocabulary": vocabulary,
        "output": str((registrations_dir.parent.parent / "artifacts/reports" / f"{name}_caption_audit.json").resolve()),
    }
    if mode == "full_episode":
        audit["expected_overall_task"] = vocabulary[0]
    value.setdefault("validation", {})["caption_audit"] = audit
    value.setdefault("training", {})["repo_id"] = f"local/{name}_prepared_v30"
    value.setdefault("adapters", {}).setdefault("isaac_groot", {})["root"] = paths["isaac"]
    language = value.setdefault("contract", {}).setdefault("language", {})
    conventions = {
        "full_episode": (
            "One full-episode task via task_index; five ordered subtasks are annotations only"
        ),
        "per_subtask": (
            "Frame-aligned task_index selects one of five ordered rich subtask prompts; "
            "the full sequence remains in the episode annotations"
        ),
        "mapped_subtask": (
            "Frame-aligned task_index selects one of five object-token prompts mapped to "
            "the five ordered rich subtask annotations"
        ),
    }
    language.update(
        {
            "convention": conventions[mode],
            "mode": mode,
            "task_vocabulary": vocabulary,
            "instruction": (
                vocabulary[0]
                if mode == "full_episode"
                else f"{len(vocabulary)} prompts selected by task_index ({mode})."
            ),
        }
    )
    destination = registrations_dir / f"{name}.yaml"
    if destination.exists():
        raise LanguageVariantError(f"registration already exists: {destination}")
    atomic_yaml(destination, value)
    return destination


def create_variants(
    *,
    source_parent: Path,
    prepared_parent: Path,
    isaac_parent: Path,
    parent_registration: Path,
    output_root: Path,
    registrations_dir: Path,
    name_prefix: str = "Task_700013",
    variant_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    source_parent = require_parent(source_parent, "v3.0")
    prepared_parent = require_parent(prepared_parent, "v3.0", "DERIVATION_RECEIPT.json")
    isaac_parent = require_parent(isaac_parent, "v2.1", "ISAAC_DERIVATION_RECEIPT.json")
    source_subtasks = rich_subtasks(source_parent)
    if rich_subtasks(prepared_parent) != source_subtasks:
        raise LanguageVariantError("source and prepared rich subtask vocabularies differ")
    parent_config = yaml.safe_load(parent_registration.read_text(encoding="utf-8"))
    if not isinstance(parent_config, dict):
        raise LanguageVariantError("parent registration must be a YAML mapping")
    output_root.mkdir(parents=True, exist_ok=True)
    registrations_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        Variant(
            "mainlang_servo_first_v1",
            "full_episode",
            [MAINLANG_SERVO_FIRST_PROMPT],
            False,
        ),
        Variant("short_full_v1", "full_episode", [SHORT_FULL_PROMPT], False),
        Variant("subtask_full_v1", "per_subtask", source_subtasks, True),
        Variant("subtask_words_v1", "mapped_subtask", WORD_PROMPTS, True),
    ]
    if variant_names is None:
        # Preserve the historical default set for callers that do not opt in to
        # the canonical servo-first view.
        variant_names = ["short_full_v1", "subtask_full_v1", "subtask_words_v1"]
    requested = set(variant_names)
    known = {variant.suffix for variant in variants}
    if unknown := requested - known:
        raise LanguageVariantError(f"unknown variants: {sorted(unknown)}")
    variants = [variant for variant in variants if variant.suffix in requested]
    results: list[dict[str, Any]] = []
    parent_revision = str(parent_config.get("revision", "canonical"))
    for variant in variants:
        name = f"{name_prefix}_{variant.suffix}"
        revision = f"{parent_revision}:{variant.suffix}"
        result = create_one(
            variant=variant,
            name=name,
            revision=revision,
            source_parent=source_parent,
            prepared_parent=prepared_parent,
            isaac_parent=isaac_parent,
            output_root=output_root,
        )
        result["registration"] = str(
            registration(parent_config, result, registrations_dir).resolve()
        )
        results.append(result)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parent", type=Path, required=True)
    parser.add_argument("--prepared-parent", type=Path, required=True)
    parser.add_argument("--isaac-parent", type=Path, required=True)
    parser.add_argument("--parent-registration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--registrations-dir", type=Path, required=True)
    parser.add_argument("--name-prefix", default="Task_700013")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=(
            "mainlang_servo_first_v1",
            "short_full_v1",
            "subtask_full_v1",
            "subtask_words_v1",
        ),
        default=None,
        help="Generate only the selected language views (historical three-view default).",
    )
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = create_variants(
        source_parent=args.source_parent,
        prepared_parent=args.prepared_parent,
        isaac_parent=args.isaac_parent,
        parent_registration=args.parent_registration,
        output_root=args.output_root,
        registrations_dir=args.registrations_dir,
        name_prefix=args.name_prefix,
        variant_names=args.variants,
    )
    rendered = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.summary:
        atomic_text(args.summary, rendered)
    print(rendered, end="")


if __name__ == "__main__":
    try:
        main()
    except (LanguageVariantError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=__import__("sys").stderr)
        raise SystemExit(1)
