#!/usr/bin/env python3
"""Non-destructively convert one local LeRobot v3 dataset to Isaac GR00T v2.1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--modality-json", type=Path, required=True)
    return parser.parse_args()


def add_language_annotation(root: Path) -> None:
    key = "annotation.human.task_description"
    parquet_paths = sorted((root / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise RuntimeError(f"no episode parquet files under {root / 'data'}")
    for path in parquet_paths:
        table = pq.read_table(path)
        if "task_index" not in table.column_names:
            raise RuntimeError(f"{path}: task_index is missing")
        task_index = table["task_index"].cast(pa.int64())
        if key in table.column_names:
            if not table[key].equals(task_index):
                raise RuntimeError(f"{path}: existing {key} differs from task_index")
            continue
        table = table.append_column(key, task_index)
        temporary = path.with_name(f".{path.name}.annotation.tmp")
        pq.write_table(table, temporary)
        os.replace(temporary, path)

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info.setdefault("features", {})[key] = {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    }
    info_path.write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def convert(source: Path, destination: Path, modality_json: Path) -> None:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    modality_json = modality_json.expanduser().resolve()
    if not source.is_dir():
        raise RuntimeError(f"source dataset does not exist: {source}")
    if destination.exists():
        raise RuntimeError(f"destination already exists: {destination}")
    if source == destination or source in destination.parents:
        raise RuntimeError(
            "destination must be a separate sibling tree, not inside the source"
        )
    if not modality_json.is_file():
        raise RuntimeError(f"modality template is missing: {modality_json}")

    isaac_root = Path(os.environ.get("ISAAC_GROOT_ROOT", "")).expanduser().resolve()
    converter_dir = isaac_root / "scripts" / "lerobot_conversion"
    if not (converter_dir / "convert_v3_to_v2.py").is_file():
        raise RuntimeError(
            "ISAAC_GROOT_ROOT does not identify the Isaac-GR00T repository"
        )
    sys.path.insert(0, str(converter_dir))

    import convert_v3_to_v2 as native

    native.validate_local_dataset_version(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.convert-", dir=destination.parent)
    )
    try:
        records = native.load_episode_records(source)
        metadata = native.load_info(source)
        video_keys = [
            key
            for key, feature in metadata["features"].items()
            if feature.get("dtype") == "video"
        ]
        chunks_size = metadata.get("chunks_size", native.DEFAULT_CHUNK_SIZE)

        native.convert_info(source, stage, records, video_keys)
        native.copy_global_stats(source, stage)
        native.convert_tasks(source, stage)
        native.convert_data(source, stage, records, chunks_size)
        native.convert_videos(source, stage, records, video_keys, chunks_size)
        native.convert_episodes_metadata(stage, records)
        native.copy_ancillary_directories(source, stage)

        add_language_annotation(stage)
        shutil.copy2(modality_json, stage / "meta" / "modality.json")
        os.replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


if __name__ == "__main__":
    args = parse_args()
    convert(args.source, args.destination, args.modality_json)
