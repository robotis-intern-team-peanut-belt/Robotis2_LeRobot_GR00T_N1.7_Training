#!/usr/bin/env python3
"""Create an immutable Task 700013 v3 derivative with corrected arm captions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq


RECEIPT = "INSTRUCTION_CORRECTION_RECEIPT.json"
REPLACEMENTS = {
    "Use right gripper to pick OMX-F3 and insert into its cutout hole in foam bed.":
        "Use left gripper to pick OMX-F3 and insert into its cutout hole in foam bed.",
    "Use right gripper to pick OMX-Base and insert into its cutout hole in foam bed.":
        "Use left gripper to pick OMX-Base and insert into its cutout hole in foam bed.",
}


class CorrectionError(RuntimeError):
    """The source does not match the exact correction contract."""


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def clone_tree(source: Path, stage: Path) -> dict[str, int]:
    stage.mkdir(parents=True)
    linked = linked_bytes = copied = 0
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        destination = stage / relative
        if item.is_symlink():
            raise CorrectionError(f"source symlink is not accepted: {item}")
        if item.is_dir():
            destination.mkdir(exist_ok=True)
        elif item.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if relative.parts[0] == "videos" and item.suffix.lower() == ".mp4":
                os.link(item, destination)
                linked += 1
                linked_bytes += item.stat().st_size
            else:
                shutil.copy2(item, destination)
                copied += 1
        else:
            raise CorrectionError(f"unsupported source entry: {item}")
    return {
        "hardlinked_mp4_count": linked,
        "hardlinked_mp4_bytes": linked_bytes,
        "copied_non_mp4_count": copied,
    }


def replace_text(value: str, counts: dict[str, int]) -> str:
    if value in REPLACEMENTS:
        counts[value] += 1
        return REPLACEMENTS[value]
    return value


def rewrite_subtasks(stage: Path, counts: dict[str, int]) -> None:
    path = stage / "meta" / "subtasks.parquet"
    original = pq.read_table(path)
    values = original.to_pylist()
    if [int(row["subtask_index"]) for row in values] != list(range(5)):
        raise CorrectionError("expected exactly five contiguous subtask indices")
    for row in values:
        row["subtask"] = replace_text(str(row["subtask"]), counts)
    atomic_parquet(path, pa.Table.from_pylist(values, schema=original.schema))


def rewrite_episodes(stage: Path, counts: dict[str, int]) -> int:
    episodes = 0
    for path in sorted((stage / "meta" / "episodes").rglob("*.parquet")):
        original = pq.read_table(path)
        values = original.to_pylist()
        for row in values:
            instructions = row.get("subtask_instructions")
            if not isinstance(instructions, list) or len(instructions) != 5:
                raise CorrectionError(
                    f"episode {row.get('episode_index')}: expected five subtask instructions"
                )
            row["subtask_instructions"] = [
                replace_text(str(instruction), counts) for instruction in instructions
            ]
            episodes += 1
        atomic_parquet(path, pa.Table.from_pylist(values, schema=original.schema))
    return episodes


def rewrite_annotations(stage: Path, counts: dict[str, int]) -> int:
    annotations = 0
    for path in sorted((stage / "annotations").rglob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        entries = value.get("sub_task_annotation")
        if not isinstance(entries, list) or len(entries) != 5:
            raise CorrectionError(f"{path}: expected five sub_task_annotation entries")
        for entry in entries:
            entry["sub_task_instruction"] = replace_text(
                str(entry["sub_task_instruction"]), counts
            )
        atomic_json(path, value)
        annotations += 1
    return annotations


def correct(source: Path, output: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not (source / "meta" / "info.json").is_file():
        raise CorrectionError(f"missing LeRobot v3 source: {source}")
    info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise CorrectionError("source must be LeRobot v3.0")
    if output.exists():
        raise CorrectionError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_payload = payload_digest(source)
    stage = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if stage.exists():
        raise CorrectionError(f"staging path already exists: {stage}")
    try:
        media = clone_tree(source, stage)
        counts = {old: 0 for old in REPLACEMENTS}
        rewrite_subtasks(stage, counts)
        episodes = rewrite_episodes(stage, counts)
        annotations = rewrite_annotations(stage, counts)
        expected_each = 1 + episodes + annotations
        if episodes != int(info["total_episodes"]) or annotations != episodes:
            raise CorrectionError(
                f"expected matching episode/annotation counts, got {episodes}/{annotations}"
            )
        if any(count != expected_each for count in counts.values()):
            raise CorrectionError(
                f"expected {expected_each} replacements per instruction, observed {counts}"
            )
        output_payload = payload_digest(stage)
        receipt = {
            "schema_version": 1,
            "kind": "task700013_instruction_correction",
            "source_root": str(source),
            "output_root": str(output),
            "source_payload": source_payload,
            "output_payload": output_payload,
            "replacements": REPLACEMENTS,
            "replacement_counts": counts,
            "episodes": episodes,
            "annotations": annotations,
            "unchanged_fields": [
                "data parquet values and indices",
                "task vocabulary and task indices",
                "subtask indices and frame boundaries",
                "videos and frame-reuse evidence",
            ],
            "media": media,
            "authority_note": (
                "Owner corrected the recording fact: OMX-F3 and OMX-Base were "
                "picked with the left gripper in all Task 700013 demonstrations."
            ),
        }
        atomic_json(stage / RECEIPT, receipt)
        os.replace(stage, output)
        return receipt
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def main() -> None:
    args = cli()
    print(
        json.dumps(
            correct(args.source, args.output),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (CorrectionError, OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
