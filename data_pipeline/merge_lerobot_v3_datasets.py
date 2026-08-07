#!/usr/bin/env python3
"""Merge compatible Cyclo LeRobot v3 datasets without losing custom metadata.

LeRobot's upstream aggregator handles data, episode metadata, task indices, and
video shard offsets.  Cyclo datasets also carry a rich subtask vocabulary,
per-episode JSON annotations, frame-reuse evidence, and ``full_episode_index``.
This wrapper preserves and reindexes those fields, hardlinks the copied MP4
payloads back to their immutable sources, and emits a lineage receipt.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.compute_stats import aggregate_feature_stats


RECEIPT = "MERGE_RECEIPT.json"


class MergeError(RuntimeError):
    """The requested merge would lose or misindex dataset information."""


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_digest(root: Path) -> dict[str, Any]:
    lines: list[str] = []
    count = size = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == RECEIPT:
            continue
        relative = path.relative_to(root).as_posix()
        length = path.stat().st_size
        lines.append(f"{sha256_file(path)}  {relative}")
        count += 1
        size += length
    rendered = "\n".join(lines) + ("\n" if lines else "")
    return {
        "payload_manifest_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "payload_file_count": count,
        "payload_bytes": size,
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(path: Path, table: pa.Table) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(table, temporary, compression="snappy", use_dictionary=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def require_source(path: Path) -> Path:
    root = path.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not root.is_dir() or not info_path.is_file():
        raise MergeError(f"missing LeRobot dataset: {root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise MergeError(f"{root}: expected codebase_version v3.0")
    required = [
        root / "meta" / "tasks.parquet",
        root / "meta" / "subtasks.parquet",
        root / "meta" / "frame_reuse.parquet",
        root / "meta" / "episodes",
        root / "data",
        root / "videos",
        root / "annotations",
    ]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise MergeError(f"{root}: missing Cyclo payloads: {missing}")
    return root


def aggregate_feature_view(info: Mapping[str, Any]) -> dict[str, Any]:
    """Return the schema representation used for strict upstream aggregation.

    Cyclo v3 converters have emitted equivalent feature metadata in two forms:
    one carries per-tensor ``fps`` and PyAV encoder-option placeholders while the
    other carries only the values that describe the actual recorded video. Those
    fields do not change tensors, MP4 payloads, or runtime camera geometry, but
    LeRobot's aggregator compares them literally. Keep only contract-bearing
    fields, so a real camera/schema mismatch still fails equality below.
    """
    normalized = json.loads(json.dumps(dict(info)))
    features = normalized.get("features", {})
    if not isinstance(features, dict):
        raise MergeError("meta/info.json: features must be an object")
    video_info_keys = {
        "video.fps",
        "video.height",
        "video.width",
        "video.channels",
        "video.codec",
        "video.pix_fmt",
        "video.is_depth_map",
        "has_audio",
    }
    for feature in features.values():
        if not isinstance(feature, dict):
            raise MergeError("meta/info.json: feature entry must be an object")
        # Dataset-level fps is authoritative; per-tensor fps is a redundant
        # serializer difference in these Cyclo v3 trees.
        feature.pop("fps", None)
        if feature.get("dtype") == "video":
            details = feature.get("info")
            if not isinstance(details, dict):
                raise MergeError("meta/info.json: video feature has no info object")
            missing = video_info_keys - set(details)
            if missing:
                raise MergeError(
                    "meta/info.json: video feature misses contract fields "
                    f"{sorted(missing)}"
                )
            feature["info"] = {
                key: details[key] for key in sorted(video_info_keys)
            }
    return normalized


def source_episode_mapping(source: Path) -> dict[int, int]:
    """Map retained source episode IDs onto the contiguous v3 range."""
    data_episodes: set[int] = set()
    for path in sorted((source / "data").rglob("*.parquet")):
        data_episodes.update(
            int(value)
            for value in pq.read_table(path, columns=["episode_index"])
            .column("episode_index")
            .to_pylist()
        )
    metadata_episodes: set[int] = set()
    for path in sorted((source / "meta" / "episodes").rglob("*.parquet")):
        metadata_episodes.update(
            int(value)
            for value in pq.read_table(path, columns=["episode_index"])
            .column("episode_index")
            .to_pylist()
        )
    annotation_episodes = {
        int(path.stem.rsplit("_", 1)[-1])
        for path in (source / "annotations").rglob("episode_*.json")
    }
    if not data_episodes or data_episodes != metadata_episodes:
        raise MergeError(f"{source}: data and episode metadata IDs differ")
    if data_episodes != annotation_episodes:
        raise MergeError(f"{source}: data and annotation episode IDs differ")
    expected = int(json.loads((source / "meta" / "info.json").read_text())["total_episodes"])
    if len(data_episodes) != expected:
        raise MergeError(
            f"{source}: meta total_episodes={expected}, retained IDs={len(data_episodes)}"
        )
    return {old: new for new, old in enumerate(sorted(data_episodes))}


def replace_integer_column(table: pa.Table, column: str, mapping: Mapping[int, int]) -> pa.Table:
    if column not in table.column_names:
        return table
    field = table.schema.field(column)
    values = [mapping[int(value)] for value in table.column(column).to_pylist()]
    return table.set_column(
        table.schema.get_field_index(column), field, pa.array(values, type=field.type)
    )


def compact_view_episode_indices(view: Path, mapping: Mapping[int, int]) -> bool:
    """Rewrite only temporary Parquet data/metadata when source IDs have gaps."""
    if all(old == new for old, new in mapping.items()):
        return False
    for path in sorted((view / "data").rglob("*.parquet")):
        atomic_parquet(path, replace_integer_column(pq.read_table(path), "episode_index", mapping))
    for path in sorted((view / "meta" / "episodes").rglob("*.parquet")):
        table = replace_integer_column(pq.read_table(path), "episode_index", mapping)
        table = replace_integer_column(table, "full_episode_index", mapping)
        for stat in ("min", "max", "mean"):
            column = f"stats/episode_index/{stat}"
            if column not in table.column_names:
                continue
            field = table.schema.field(column)
            values = [
                [mapping[int(item)] for item in value]
                for value in table.column(column).to_pylist()
            ]
            table = table.set_column(
                table.schema.get_field_index(column), field, pa.array(values, type=field.type)
            )
        atomic_parquet(path, table)
    return True


def normalized_aggregate_views(sources: list[Path], parent: Path) -> tuple[Path, list[Path]]:
    """Create short-lived metadata-normalized source views for aggregation only."""
    views_root = Path(tempfile.mkdtemp(prefix=".merge-feature-view-", dir=parent))
    views: list[Path] = []
    try:
        canonical_infos = [
            aggregate_feature_view(json.loads((source / "meta" / "info.json").read_text()))
            for source in sources
        ]
        reference_features = canonical_infos[0]["features"]
        for source, info in zip(sources, canonical_infos, strict=True):
            if info["features"] != reference_features:
                raise MergeError(
                    "sources differ in contract-bearing feature metadata after "
                    f"normalization: {source}"
                )
            view = views_root / source.name
            view.mkdir()
            shutil.copytree(source / "meta", view / "meta")
            atomic_json(view / "meta" / "info.json", info)
            mapping = source_episode_mapping(source)
            if all(old == new for old, new in mapping.items()):
                (view / "data").symlink_to(source / "data", target_is_directory=True)
            else:
                shutil.copytree(source / "data", view / "data")
                compact_view_episode_indices(view, mapping)
            for name in ("videos", "annotations"):
                (view / name).symlink_to(source / name, target_is_directory=True)
            views.append(view)
        return views_root, views
    except Exception:
        shutil.rmtree(views_root, ignore_errors=True)
        raise


def rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def cyclo_episode_stats(sources: list[Path]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate trustworthy per-episode stats instead of malformed Cyclo globals.

    Some Cyclo v3 datasets serialize global video means/stds with the episode
    dimension retained (for example ``3 x episodes x 1``). Their per-episode
    metadata has the intended ``3 x 1 x 1`` values and sample counts, so it is
    the lossless source for weighted aggregation.
    """
    by_feature: dict[str, list[dict[str, np.ndarray]]] = {}
    for source in sources:
        for path in sorted((source / "meta" / "episodes").rglob("*.parquet")):
            for row in pq.read_table(path).to_pylist():
                feature_names = {
                    key.removeprefix("stats/").removesuffix("/mean")
                    for key in row
                    if key.startswith("stats/") and key.endswith("/mean")
                }
                for feature in feature_names:
                    values: dict[str, np.ndarray] = {}
                    for stat in ("mean", "std", "min", "max", "count"):
                        key = f"stats/{feature}/{stat}"
                        if key not in row or row[key] is None:
                            raise MergeError(f"{path}: episode stats missing {key}")
                        values[stat] = np.asarray(row[key])
                    by_feature.setdefault(feature, []).append(values)
    if not by_feature:
        raise MergeError("no per-episode statistics found")
    return {
        feature: aggregate_feature_stats(feature_stats)
        for feature, feature_stats in by_feature.items()
    }


def indexed_vocabulary(
    source: Path, filename: str, index_column: str, text_column: str
) -> list[dict[str, Any]]:
    path = source / "meta" / filename
    values = rows(path)
    indices = [int(row[index_column]) for row in values]
    if indices != list(range(len(values))):
        raise MergeError(f"{path}: {index_column} must be contiguous and row-aligned")
    for row in values:
        if not isinstance(row.get(text_column), str) or not row[text_column].strip():
            raise MergeError(f"{path}: invalid {text_column}: {row.get(text_column)!r}")
    return values


def union_vocabulary(
    sources: list[Path], filename: str, index_column: str, text_column: str
) -> tuple[list[dict[str, Any]], list[dict[int, int]]]:
    merged: list[dict[str, Any]] = []
    by_text: dict[str, int] = {}
    mappings: list[dict[int, int]] = []
    for source in sources:
        mapping: dict[int, int] = {}
        for row in indexed_vocabulary(source, filename, index_column, text_column):
            text = row[text_column]
            if text not in by_text:
                by_text[text] = len(merged)
                merged.append(dict(row))
            destination = by_text[text]
            existing = merged[destination]
            for key in set(existing) & set(row) - {index_column}:
                if existing[key] != row[key]:
                    raise MergeError(
                        f"{filename}: conflicting {key!r} for shared {text_column} {text!r}"
                    )
            mapping[int(row[index_column])] = destination
        mappings.append(mapping)
    for index, row in enumerate(merged):
        row[index_column] = index
    return merged, mappings


def write_vocabulary(
    output: Path, filename: str, entries: list[dict[str, Any]], columns: list[str]
) -> None:
    frame = pd.DataFrame(entries, columns=columns)
    path = output / "meta" / filename
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def numeric_stats(values: Iterable[int], include_count: bool = False) -> dict[str, list[Any]]:
    items = [int(value) for value in values]
    if not items:
        raise MergeError("cannot calculate statistics for an empty column")
    mean = sum(items) / len(items)
    variance = sum((item - mean) ** 2 for item in items) / len(items)
    result: dict[str, list[Any]] = {
        "mean": [mean],
        "std": [math.sqrt(variance)],
        "min": [float(min(items))],
        "max": [float(max(items))],
    }
    if include_count:
        result["count"] = [len(items)]
    return result


def source_for_episode(episode: int, episode_offsets: list[int]) -> int:
    source_index = bisect_right(episode_offsets, episode) - 1
    if source_index < 0 or source_index >= len(episode_offsets) - 1:
        raise MergeError(f"episode {episode} is outside source ranges")
    return source_index


def rewrite_data_subtasks(
    output: Path,
    sources: list[Path],
    episode_offsets: list[int],
    subtask_mappings: list[dict[int, int]],
) -> dict[int, dict[str, list[int]]]:
    by_episode: dict[int, dict[str, list[int]]] = {}
    expected_index = 0
    output_paths = sorted((output / "data").rglob("*.parquet"))
    source_schemas = [
        pq.read_schema(path)
        for source in sources
        for path in sorted((source / "data").rglob("*.parquet"))
    ]
    if len(output_paths) != len(source_schemas):
        raise MergeError(
            f"aggregate emitted {len(output_paths)} data files for "
            f"{len(source_schemas)} source files"
        )
    for path, source_schema in zip(output_paths, source_schemas, strict=True):
        table = pq.read_table(path)
        required = {"index", "episode_index", "frame_index", "task_index", "subtask_index"}
        missing = required - set(table.column_names)
        if missing:
            raise MergeError(f"{path}: missing columns {sorted(missing)}")
        data = table.to_pydict()
        remapped: list[int] = []
        for index, episode, frame, task, subtask in zip(
            data["index"],
            data["episode_index"],
            data["frame_index"],
            data["task_index"],
            data["subtask_index"],
            strict=True,
        ):
            index = int(index)
            episode = int(episode)
            frame = int(frame)
            if index != expected_index:
                raise MergeError(f"{path}: global index {index}, expected {expected_index}")
            source_index = source_for_episode(episode, episode_offsets)
            old_subtask = int(subtask)
            if old_subtask not in subtask_mappings[source_index]:
                raise MergeError(
                    f"episode {episode}: unknown source subtask_index {old_subtask}"
                )
            new_subtask = subtask_mappings[source_index][old_subtask]
            remapped.append(new_subtask)
            summary = by_episode.setdefault(
                episode,
                {
                    "index": [],
                    "episode_index": [],
                    "frame_index": [],
                    "task_index": [],
                    "subtask_index": [],
                },
            )
            summary["index"].append(index)
            summary["episode_index"].append(episode)
            summary["frame_index"].append(frame)
            summary["task_index"].append(int(task))
            summary["subtask_index"].append(new_subtask)
            expected_index += 1
        field = table.schema.field("subtask_index")
        replacement = pa.array(remapped, type=field.type)
        table = table.set_column(table.schema.get_field_index("subtask_index"), field, replacement)
        table = table.cast(source_schema)
        atomic_parquet(path, table)
    return by_episode


def rewrite_episode_metadata(
    output: Path,
    by_episode: dict[int, dict[str, list[int]]],
    task_vocabulary: list[dict[str, Any]],
) -> None:
    task_text = {int(row["task_index"]): row["task"] for row in task_vocabulary}
    for path in sorted((output / "meta" / "episodes").rglob("*.parquet")):
        original = pq.read_table(path)
        values = original.to_pylist()
        for row in values:
            episode = int(row["episode_index"])
            observed = by_episode[episode]
            frames = observed["frame_index"]
            if frames != list(range(len(frames))):
                raise MergeError(f"episode {episode}: frame_index is not contiguous")
            row["full_episode_index"] = episode
            seen_tasks = list(dict.fromkeys(observed["task_index"]))
            row["tasks"] = [task_text[index] for index in seen_tasks]
            for field in ("index", "episode_index", "task_index"):
                stats = numeric_stats(observed[field], include_count=True)
                for stat, item in stats.items():
                    column = f"stats/{field}/{stat}"
                    if column in row:
                        row[column] = item
        atomic_parquet(path, pa.Table.from_pylist(values, schema=original.schema))


def rewrite_global_stats(
    output: Path, by_episode: dict[int, dict[str, list[int]]]
) -> None:
    stats_path = output / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for field in ("index", "episode_index", "frame_index", "task_index", "subtask_index"):
        values = [
            value
            for episode in sorted(by_episode)
            for value in by_episode[episode][field]
        ]
        stats[field] = numeric_stats(values)
    atomic_json(stats_path, stats)


def copy_annotations(sources: list[Path], output: Path, episode_offsets: list[int]) -> int:
    count = 0
    destination_root = output / "annotations" / "chunk-000"
    destination_root.mkdir(parents=True, exist_ok=True)
    for source_index, source in enumerate(sources):
        paths = sorted((source / "annotations").rglob("episode_*.json"))
        expected = episode_offsets[source_index + 1] - episode_offsets[source_index]
        if len(paths) != expected:
            raise MergeError(f"{source}: expected {expected} annotations, found {len(paths)}")
        for local_episode, path in enumerate(paths):
            destination = destination_root / (
                f"episode_{episode_offsets[source_index] + local_episode:06d}.json"
            )
            shutil.copy2(path, destination)
            count += 1
    return count


def merge_frame_reuse(
    sources: list[Path], output: Path, episode_offsets: list[int]
) -> int:
    frames: list[pd.DataFrame] = []
    for source_index, source in enumerate(sources):
        frame = pd.read_parquet(source / "meta" / "frame_reuse.parquet")
        frame["episode_index"] = frame["episode_index"] + episode_offsets[source_index]
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    path = output / "meta" / "frame_reuse.parquet"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        merged.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return len(merged)


def relink_videos(sources: list[Path], output: Path) -> dict[str, Any]:
    video_keys = sorted(
        path.name for path in (sources[0] / "videos").iterdir() if path.is_dir()
    )
    linked = size = 0
    for key in video_keys:
        source_files = [
            path
            for source in sources
            for path in sorted((source / "videos" / key).rglob("*.mp4"))
        ]
        destinations = sorted((output / "videos" / key).rglob("*.mp4"))
        if len(source_files) != len(destinations):
            raise MergeError(
                f"{key}: aggregate emitted {len(destinations)} videos for {len(source_files)} inputs"
            )
        for source, destination in zip(source_files, destinations, strict=True):
            if sha256_file(source) != sha256_file(destination):
                raise MergeError(f"aggregate changed MP4 payload: {source} -> {destination}")
            destination.unlink()
            os.link(source, destination)
            if (source.stat().st_dev, source.stat().st_ino) != (
                destination.stat().st_dev,
                destination.stat().st_ino,
            ):
                raise MergeError(f"MP4 hardlink verification failed: {destination}")
            linked += 1
            size += source.stat().st_size
    return {"hardlinked_mp4_count": linked, "hardlinked_mp4_bytes": size}


def merge(sources: list[Path], output: Path, repo_id: str) -> dict[str, Any]:
    if len(sources) < 2:
        raise MergeError("at least two --source values are required")
    sources = [require_source(source) for source in sources]
    if len(set(sources)) != len(sources):
        raise MergeError("source roots must be unique")
    output = output.expanduser().resolve()
    if output.exists():
        raise MergeError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    source_info = [
        json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
        for source in sources
    ]
    annotation_paths = {info.get("annotation_path") for info in source_info}
    if None in annotation_paths or len(annotation_paths) != 1:
        raise MergeError(
            f"sources must have one shared nonempty annotation_path: {annotation_paths}"
        )
    annotation_path = annotation_paths.pop()
    episode_counts = [int(info["total_episodes"]) for info in source_info]
    frame_counts = [int(info["total_frames"]) for info in source_info]
    episode_offsets = [0]
    for count in episode_counts:
        episode_offsets.append(episode_offsets[-1] + count)

    tasks, _ = union_vocabulary(sources, "tasks.parquet", "task_index", "task")
    subtasks, subtask_mappings = union_vocabulary(
        sources, "subtasks.parquet", "subtask_index", "subtask"
    )
    source_receipts = [
        {"root": str(source), **payload_digest(source)} for source in sources
    ]

    stage = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if stage.exists():
        raise MergeError(f"staging path already exists: {stage}")
    views_root: Path | None = None
    try:
        views_root, aggregate_sources = normalized_aggregate_views(sources, output.parent)
        aggregate_module = importlib.import_module("lerobot.datasets.aggregate")
        original_aggregate_stats = aggregate_module.aggregate_stats
        aggregate_module.aggregate_stats = lambda _: cyclo_episode_stats(sources)
        try:
            aggregate_datasets(
                repo_ids=[f"local/{source.name}" for source in aggregate_sources],
                roots=aggregate_sources,
                aggr_repo_id=repo_id,
                aggr_root=stage,
                concatenate_videos=False,
                concatenate_data=False,
            )
        finally:
            aggregate_module.aggregate_stats = original_aggregate_stats
        merged_info_path = stage / "meta" / "info.json"
        merged_info = json.loads(merged_info_path.read_text(encoding="utf-8"))
        merged_info["annotation_path"] = annotation_path
        atomic_json(merged_info_path, merged_info)
        write_vocabulary(
            stage, "tasks.parquet", tasks, ["task_index", "task", "task_name"]
        )
        write_vocabulary(
            stage, "subtasks.parquet", subtasks, ["subtask_index", "subtask"]
        )
        by_episode = rewrite_data_subtasks(stage, sources, episode_offsets, subtask_mappings)
        rewrite_episode_metadata(stage, by_episode, tasks)
        rewrite_global_stats(stage, by_episode)
        annotation_count = copy_annotations(sources, stage, episode_offsets)
        frame_reuse_count = merge_frame_reuse(sources, stage, episode_offsets)
        media = relink_videos(sources, stage)
        atomic_json(
            stage / "info.json",
            {
                "merge_config": {
                    "output_dataset_name": output.name,
                    "repo_id": repo_id,
                    "source_datasets": [str(source) for source in sources],
                    "episode_order": "sources_in_cli_order_then_source_episode_order",
                    "mp4_storage": "hardlinks_to_source_payloads",
                }
            },
        )
        output_payload = payload_digest(stage)
        receipt = {
            "schema_version": 1,
            "kind": "cyclo_lerobot_v3_merge",
            "repo_id": repo_id,
            "output_root": str(output),
            "sources": source_receipts,
            "source_episode_offsets": episode_offsets,
            "total_episodes": sum(episode_counts),
            "total_frames": sum(frame_counts),
            "task_vocabulary": [row["task"] for row in tasks],
            "subtask_vocabulary": [row["subtask"] for row in subtasks],
            "subtask_index_mappings": [
                {str(key): value for key, value in sorted(mapping.items())}
                for mapping in subtask_mappings
            ],
            "annotation_count": annotation_count,
            "frame_reuse_count": frame_reuse_count,
            "media": media,
            "output_payload": output_payload,
        }
        atomic_json(stage / RECEIPT, receipt)
        os.replace(stage, output)
        return receipt
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    finally:
        if views_root is not None:
            shutil.rmtree(views_root, ignore_errors=True)


def main() -> None:
    args = cli()
    receipt = merge(args.source, args.output, args.repo_id)
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (MergeError, OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
