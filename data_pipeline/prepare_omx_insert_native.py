#!/usr/bin/env python3
"""Prepare native-640x480 F2 OMX Insert data for LeRobot GR00T N1.7.

The source is copied to a new immutable derivative. State and action are reduced
to the exact named 16-D dual-arm/gripper contract, task metadata is normalized
for current LeRobot, and all three camera streams are verified as native
640x480 at 15 fps. Video geometry is never changed. Streams whose keyframe
interval exceeds the configured default are re-encoded to H.264 GOP15.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HEAD = "observation.images.rgb.cam_head"
WRISTS = (
    "observation.images.rgb.cam_left_wrist",
    "observation.images.rgb.cam_right_wrist",
)
CAMERAS = (HEAD, *WRISTS)
CONTROLS = [
    *[f"arm_l_joint{i}" for i in range(1, 8)],
    "gripper_l_joint1",
    *[f"arm_r_joint{i}" for i in range(1, 8)],
    "gripper_r_joint1",
]
EXPECTED_FPS = 15.0
EXPECTED_IMAGE_SHAPE = [3, 480, 640]


class PreparationError(RuntimeError):
    """The source does not satisfy the declared native-camera contract."""


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fps-tolerance",
        type=float,
        default=0.05,
        help="Allowed absolute difference from 15 fps for each encoded stream.",
    )
    parser.add_argument("--gop-size", type=int, default=15)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--source-revision")
    parser.add_argument("--source-manifest-sha256")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def inspect_source(info: dict[str, Any]) -> tuple[list[int], list[int]]:
    """Validate metadata and return the explicit state/action selection."""
    if info.get("codebase_version") != "v3.0":
        raise PreparationError("expected LeRobot v3.0 source data")
    if float(info.get("fps", -1)) != EXPECTED_FPS:
        raise PreparationError(f"expected dataset fps={EXPECTED_FPS:g}")

    features = info.get("features", {})
    for camera in CAMERAS:
        if features.get(camera, {}).get("shape") != EXPECTED_IMAGE_SHAPE:
            raise PreparationError(
                f"{camera} must already be native 640x480; no resize or letterbox is allowed"
            )

    selections: list[list[int]] = []
    for key in ("observation.state", "action"):
        feature = features.get(key, {})
        shape = feature.get("shape")
        names = feature.get("names")
        if not isinstance(shape, list) or len(shape) != 1:
            raise PreparationError(f"{key} must be a one-dimensional feature")
        if not isinstance(names, list) or len(names) != shape[0]:
            raise PreparationError(f"{key} must have one unique name per dimension")
        if len(names) != len(set(names)):
            raise PreparationError(f"{key} contains duplicate names")
        if names[: len(CONTROLS)] != CONTROLS:
            raise PreparationError(
                f"{key} does not begin with the exact ordered 16-D arm/gripper contract"
            )
        selections.append(list(range(len(CONTROLS))))
    return selections[0], selections[1]


def sliced_column(table: pa.Table, key: str, indices: list[int]) -> pa.Array:
    column = table[key].combine_chunks()
    values = np.asarray(column.values).reshape(len(column), column.type.list_size)
    selected = values[:, indices]
    flat = pa.array(
        np.ascontiguousarray(selected, dtype=np.float32).reshape(-1),
        type=pa.float32(),
    )
    return pa.FixedSizeListArray.from_arrays(flat, len(indices))


def rewrite_data(root: Path, state_indices: list[int], action_indices: list[int]) -> None:
    paths = sorted((root / "data").glob("**/*.parquet"))
    if not paths:
        raise PreparationError("no data parquet files found")
    for path in paths:
        table = pq.read_table(path)
        for key, indices in (
            ("observation.state", state_indices),
            ("action", action_indices),
        ):
            table = table.set_column(
                table.schema.get_field_index(key),
                key,
                sliced_column(table, key, indices),
            )
        metadata = dict(table.schema.metadata or {})
        if b"huggingface" in metadata:
            embedded = json.loads(metadata[b"huggingface"])
            for key in ("observation.state", "action"):
                feature = embedded.get("info", {}).get("features", {}).get(key)
                if feature:
                    feature.update(
                        {
                            "feature": {"dtype": "float32", "_type": "Value"},
                            "length": len(CONTROLS),
                            "_type": "Sequence",
                        }
                    )
            metadata[b"huggingface"] = json.dumps(embedded).encode()
            table = table.replace_schema_metadata(metadata)
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="snappy", use_dictionary=True)
        os.replace(temporary, path)


def rewrite_stats(
    root: Path, state_indices: list[int], action_indices: list[int]
) -> None:
    """Slice robot statistics only; native camera statistics stay byte-for-byte."""
    path = root / "meta" / "stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    for key, indices in (
        ("observation.state", state_indices),
        ("action", action_indices),
    ):
        for statistic in ("mean", "std", "min", "max"):
            stats[key][statistic] = np.asarray(stats[key][statistic])[indices].tolist()
    write_json(path, stats)

    for episode_path in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        rows = pq.read_table(episode_path).to_pylist()
        for row in rows:
            for key, indices in (
                ("observation.state", state_indices),
                ("action", action_indices),
            ):
                for statistic in ("mean", "std", "min", "max"):
                    column = f"stats/{key}/{statistic}"
                    if row.get(column) is not None:
                        row[column] = np.asarray(row[column])[indices].tolist()
        temporary = episode_path.with_suffix(".parquet.tmp")
        pq.write_table(
            pa.Table.from_pylist(rows),
            temporary,
            compression="snappy",
            use_dictionary=True,
        )
        os.replace(temporary, episode_path)


def normalize_tasks(root: Path) -> list[str]:
    path = root / "meta" / "tasks.parquet"
    tasks = pd.read_parquet(path)
    if "task" in tasks.columns:
        strings = tasks["task"].astype(str).tolist()
        if not strings or any(not item.strip() for item in strings):
            raise PreparationError("task strings must be nonempty")
        if len(strings) != len(set(strings)):
            raise PreparationError("task strings must be unique")
        tasks = tasks.drop(columns=["task"])
        tasks.index = pd.Index(strings, name="task")
        tasks.to_parquet(path)
        return strings
    if tasks.index.name == "task":
        strings = [str(item) for item in tasks.index.tolist()]
        if strings and all(item.strip() for item in strings):
            return strings
    raise PreparationError("tasks.parquet lacks a valid task column or task index")


def probe_gop_size(path: Path) -> tuple[int, int]:
    """Return (maximum keyframe interval, decoded frame count)."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=key_frame",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    flags = [
        int(line.split(",", 1)[0])
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    if not flags or flags[0] != 1:
        raise PreparationError(f"{path}: video is empty or does not start on a keyframe")
    keyframes = [index for index, flag in enumerate(flags) if flag == 1]
    intervals = [right - left for left, right in zip(keyframes, keyframes[1:])]
    intervals.append(len(flags) - keyframes[-1])
    return max(intervals), len(flags)


def probe_video(path: Path) -> tuple[int, int, float, str, int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,codec_name",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise PreparationError(f"expected one video stream: {path}")
    stream = streams[0]
    try:
        fps = float(Fraction(stream["avg_frame_rate"]))
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise PreparationError(f"invalid frame rate in {path}") from exc
    max_gop, frames = probe_gop_size(path)
    return (
        int(stream["width"]),
        int(stream["height"]),
        fps,
        str(stream.get("codec_name", "")),
        max_gop,
        frames,
    )


def validate_videos(
    root: Path, fps_tolerance: float, required_max_gop: int | None = None
) -> dict[str, dict[str, int]]:
    inventory: dict[str, dict[str, int]] = {}
    for camera in CAMERAS:
        paths = sorted((root / "videos" / camera).glob("**/*.mp4"))
        if not paths:
            raise PreparationError(f"no videos found for {camera}")
        max_observed_gop = 0
        total_frames = 0
        for path in paths:
            width, height, fps, codec, max_gop, frames = probe_video(path)
            if (width, height) != (640, 480):
                raise PreparationError(f"{path}: expected native 640x480, got {width}x{height}")
            if abs(fps - EXPECTED_FPS) > fps_tolerance:
                raise PreparationError(f"{path}: expected 15 fps, got {fps:g}")
            if codec != "h264":
                raise PreparationError(f"{path}: expected H.264, got {codec or 'unknown'}")
            if required_max_gop is not None and max_gop > required_max_gop:
                raise PreparationError(
                    f"{path}: maximum GOP {max_gop} exceeds {required_max_gop}"
                )
            max_observed_gop = max(max_observed_gop, max_gop)
            total_frames += frames
        inventory[camera] = {
            "video_files": len(paths),
            "decoded_frames": total_frames,
            "max_gop": max_observed_gop,
        }
    return inventory


def transcode_long_gop_videos(
    root: Path,
    fps_tolerance: float,
    gop_size: int,
    crf: int,
    preset: str,
) -> dict[str, dict[str, int]]:
    if gop_size <= 0:
        raise PreparationError("gop size must be positive")
    if not 0 <= crf <= 51:
        raise PreparationError("CRF must be between 0 and 51")
    operations: dict[str, dict[str, int]] = {}
    for camera in CAMERAS:
        copied = 0
        transcoded = 0
        for path in sorted((root / "videos" / camera).glob("**/*.mp4")):
            width, height, fps, codec, max_gop, _ = probe_video(path)
            if (width, height) != (640, 480) or abs(fps - EXPECTED_FPS) > fps_tolerance:
                raise PreparationError(f"{path}: geometry/fps changed before GOP preparation")
            if codec == "h264" and max_gop <= gop_size:
                copied += 1
                continue
            temporary = path.with_name(f".{path.stem}.gop{gop_size}.tmp.mp4")
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(path),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    preset,
                    "-crf",
                    str(crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-g",
                    str(gop_size),
                    "-keyint_min",
                    str(gop_size),
                    "-sc_threshold",
                    "0",
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ],
                check=True,
            )
            os.replace(temporary, path)
            transcoded += 1
        operations[camera] = {"copied": copied, "transcoded": transcoded}
    return operations


def update_info(root: Path, info: dict[str, Any]) -> None:
    for key in ("observation.state", "action"):
        info["features"][key].update({"shape": [len(CONTROLS)], "names": CONTROLS})
    video_mb = sum(path.stat().st_size for path in (root / "videos").glob("**/*.mp4")) / 1e6
    data_mb = sum(path.stat().st_size for path in (root / "data").glob("**/*.parquet")) / 1e6
    if "total_videos_size_in_mb" in info:
        info["total_videos_size_in_mb"] = video_mb
    if "total_size_in_mb" in info:
        info["total_size_in_mb"] = video_mb + data_mb
    write_json(root / "meta" / "info.json", info)
    if (root / "info.json").exists():
        write_json(root / "info.json", info)


def validate_derivative(
    root: Path, expected_frames: int, fps_tolerance: float, gop_size: int
) -> dict[str, dict[str, int]]:
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    if tasks.index.name != "task" or not isinstance(tasks.index[0], str):
        raise PreparationError("natural-language task index was not normalized")
    frames = 0
    for path in sorted((root / "data").glob("**/*.parquet")):
        table = pq.read_table(path, columns=["observation.state", "action"])
        if any(
            table.schema.field(key).type.list_size != len(CONTROLS)
            for key in ("observation.state", "action")
        ):
            raise PreparationError(f"non-16-D data in {path}")
        frames += len(table)
    if frames != expected_frames:
        raise PreparationError(f"frame count changed: {expected_frames} -> {frames}")
    return validate_videos(root, fps_tolerance, required_max_gop=gop_size)


def source_revision(source: Path) -> str | None:
    marker = source / ".groot_registry_download.json"
    if marker.is_file():
        return json.loads(marker.read_text(encoding="utf-8")).get("revision")
    receipt = source / "DOWNLOAD_RECEIPT.txt"
    if receipt.is_file():
        for line in receipt.read_text(encoding="utf-8").splitlines():
            if line.startswith("resolved_hub_commit="):
                return line.split("=", 1)[1]
    return None


def prepare(
    source: Path,
    output: Path,
    fps_tolerance: float = 0.05,
    gop_size: int = 15,
    crf: int = 18,
    preset: str = "medium",
    source_revision_value: str | None = None,
    source_manifest_sha256: str | None = None,
) -> Path:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise PreparationError(f"missing source: {source}")
    if (source / "DERIVATION_RECEIPT.json").exists():
        raise PreparationError("source is already a derived dataset; use native Cyclo output")
    if output.exists():
        raise PreparationError(f"output exists and will not be overwritten: {output}")
    if output == source or source in output.parents:
        raise PreparationError("output must be a separate sibling, not inside the source")
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        raise PreparationError("ffprobe and ffmpeg are required")

    info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
    state_indices, action_indices = inspect_source(info)
    source_inventory = validate_videos(source, fps_tolerance)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        shutil.copytree(
            source,
            temporary,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".cache", "*.tmp"),
        )
        rewrite_data(temporary, state_indices, action_indices)
        rewrite_stats(temporary, state_indices, action_indices)
        tasks = normalize_tasks(temporary)
        video_operations = transcode_long_gop_videos(
            temporary, fps_tolerance, gop_size, crf, preset
        )
        update_info(temporary, info)
        output_inventory = validate_derivative(
            temporary, int(info["total_frames"]), fps_tolerance, gop_size
        )
        for camera in CAMERAS:
            for key in ("video_files", "decoded_frames"):
                if output_inventory[camera][key] != source_inventory[camera][key]:
                    raise PreparationError(
                        f"{camera}: {key} changed during preparation"
                    )
        write_json(
            temporary / "DERIVATION_RECEIPT.json",
            {
                "source": str(source),
                "source_revision": source_revision_value or source_revision(source),
                "source_manifest_sha256": source_manifest_sha256,
                "state": {
                    "source_dimension": int(
                        json.loads(
                            (source / "meta" / "info.json").read_text(encoding="utf-8")
                        )["features"]["observation.state"]["shape"][0]
                    ),
                    "output_dimension": len(CONTROLS),
                    "names": CONTROLS,
                },
                "action": {
                    "source_dimension": int(
                        json.loads(
                            (source / "meta" / "info.json").read_text(encoding="utf-8")
                        )["features"]["action"]["shape"][0]
                    ),
                    "output_dimension": len(CONTROLS),
                    "names": CONTROLS,
                },
                "cameras": {
                    camera: {
                        "capture": "native_640x480_15fps",
                        "geometry_transform": "identity",
                        "source": source_inventory[camera],
                        "output": output_inventory[camera],
                        "operation": video_operations[camera],
                    }
                    for camera in CAMERAS
                },
                "camera_file_operation": (
                    f"native geometry retained; H.264 max GOP {gop_size}; "
                    f"long-GOP streams re-encoded with CRF {crf}, preset {preset}; "
                    "no resize, crop, pad, or letterbox"
                ),
                "image_statistics": (
                    "copied from source; training uses ImageNet statistics by default"
                ),
                "tasks": tasks,
            },
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> None:
    args = cli()
    created = prepare(
        args.source,
        args.output,
        args.fps_tolerance,
        args.gop_size,
        args.crf,
        args.preset,
        args.source_revision,
        args.source_manifest_sha256,
    )
    print(f"Created: {created}")
    print("Source was not modified; camera geometry stayed native with no letterbox.")


if __name__ == "__main__":
    try:
        main()
    except (PreparationError, OSError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
