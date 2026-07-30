#!/usr/bin/env python3
"""Create a LeRobot GR00T derivative while preserving the downloaded source."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SOURCE = "Task_700007_OMX_Insert_MCAP_lerobot_v30"
SUFFIX = "arms16_head-letterbox640x480_taskfix"
HEAD = "observation.images.rgb.cam_head"
WRISTS = ("observation.images.rgb.cam_left_wrist", "observation.images.rgb.cam_right_wrist")
CONTROLS = [
    *[f"arm_l_joint{i}" for i in range(1, 8)], "gripper_l_joint1",
    *[f"arm_r_joint{i}" for i in range(1, 8)], "gripper_r_joint1",
]


def cli():
    base = Path(__file__).resolve().parent / "datasets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=base / SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    return parser.parse_args()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def inspect(info):
    f = info["features"]
    if info.get("codebase_version") != "v3.0":
        raise RuntimeError("Expected LeRobot v3.0")
    if f["observation.state"]["shape"] != [22] or f["action"]["shape"] != [19]:
        raise RuntimeError("Expected source state/action dimensions 22/19")
    if f[HEAD]["shape"] != [3, 720, 1280]:
        raise RuntimeError("Expected a 1280x720 head camera")
    if any(f[key]["shape"] != [3, 480, 640] for key in WRISTS):
        raise RuntimeError("Expected both wrist cameras at 640x480")
    result = []
    for key in ("observation.state", "action"):
        names = f[key].get("names")
        if not names or any(name not in names for name in CONTROLS):
            raise RuntimeError(f"{key} lacks required named controls")
        selected = [names.index(name) for name in CONTROLS]
        if selected != list(range(16)):
            raise RuntimeError(f"Unsafe/unexpected {key} order: {selected}")
        result.append(selected)
    return result


def sliced_column(table, key, indices):
    col = table[key].combine_chunks()
    values = np.asarray(col.values).reshape(len(col), col.type.list_size)[:, indices]
    flat = pa.array(np.ascontiguousarray(values, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, 16)


def rewrite_data(root, state_ix, action_ix):
    paths = sorted((root / "data").glob("**/*.parquet"))
    if not paths:
        raise RuntimeError("No data parquet found")
    for path in paths:
        table = pq.read_table(path)
        for key, indices in (("observation.state", state_ix), ("action", action_ix)):
            table = table.set_column(table.schema.get_field_index(key), key, sliced_column(table, key, indices))
        metadata = dict(table.schema.metadata or {})
        if b"huggingface" in metadata:
            embedded = json.loads(metadata[b"huggingface"])
            for key in ("observation.state", "action"):
                feature = embedded.get("info", {}).get("features", {}).get(key)
                if feature:
                    feature.update({"feature": {"dtype": "float32", "_type": "Value"},
                                    "length": 16, "_type": "Sequence"})
            metadata[b"huggingface"] = json.dumps(embedded).encode()
            table = table.replace_schema_metadata(metadata)
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp, compression="snappy", use_dictionary=True)
        os.replace(tmp, path)


def adjust_head(stats):
    """Add the 25% black letterbox area to stored image moments."""
    if not stats or "mean" not in stats or "std" not in stats:
        return
    mean, std = np.asarray(stats["mean"]), np.asarray(stats["std"])
    new_mean = 0.75 * mean
    stats["mean"] = new_mean.tolist()
    stats["std"] = np.sqrt(np.maximum(0.75 * (std**2 + mean**2) - new_mean**2, 0)).tolist()


def rewrite_stats(root, state_ix, action_ix):
    path = root / "meta/stats.json"
    stats = json.loads(path.read_text())
    for key, indices in (("observation.state", state_ix), ("action", action_ix)):
        for stat in ("mean", "std", "min", "max"):
            stats[key][stat] = np.asarray(stats[key][stat])[indices].tolist()
    adjust_head(stats.get(HEAD))
    write_json(path, stats)

    for path in sorted((root / "meta/episodes").glob("**/*.parquet")):
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            for key, indices in (("observation.state", state_ix), ("action", action_ix)):
                for stat in ("mean", "std", "min", "max"):
                    column = f"stats/{key}/{stat}"
                    if row.get(column) is not None:
                        row[column] = np.asarray(row[column])[indices].tolist()
            head_stats = {stat: row.get(f"stats/{HEAD}/{stat}") for stat in ("mean", "std")}
            adjust_head(head_stats)
            for stat, value in head_stats.items():
                if value is not None:
                    row[f"stats/{HEAD}/{stat}"] = value
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(rows), tmp, compression="snappy", use_dictionary=True)
        os.replace(tmp, path)


def repair_tasks(root):
    path = root / "meta/tasks.parquet"
    tasks = pd.read_parquet(path)
    if "task" not in tasks or "task_index" not in tasks:
        raise RuntimeError("tasks.parquet lacks task/task_index")
    strings = tasks["task"].astype(str).tolist()
    if not strings or any(not x.strip() for x in strings) or len(strings) != len(set(strings)):
        raise RuntimeError("Task strings must be nonempty and unique")
    fixed = tasks.drop(columns=["task"])
    fixed.index = pd.Index(strings, name="task")
    fixed.to_parquet(path)
    return strings


def transcode(root, crf, preset):
    videos = sorted((root / "videos" / HEAD).glob("**/*.mp4"))
    if not videos:
        raise RuntimeError("No head videos found")
    for index, path in enumerate(videos, 1):
        print(f"[head video {index}/{len(videos)}] {path.relative_to(root)}", flush=True)
        tmp = path.with_name(path.stem + ".tmp.mp4")
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
            "-vf", "scale=640:360:flags=lanczos,pad=640:480:0:60:color=black",
            "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-movflags", "+faststart", str(tmp),
        ], check=True)
        os.replace(tmp, path)


def update_info(root, info):
    for key in ("observation.state", "action"):
        info["features"][key].update({"shape": [16], "names": CONTROLS})
    info["features"][HEAD]["shape"] = [3, 480, 640]
    info["features"][HEAD]["info"].update({"video.height": 480, "video.width": 640})
    video_mb = sum(p.stat().st_size for p in (root / "videos").glob("**/*.mp4")) / 1e6
    data_mb = sum(p.stat().st_size for p in (root / "data").glob("**/*.parquet")) / 1e6
    if "total_videos_size_in_mb" in info:
        info["total_videos_size_in_mb"] = video_mb
    if "total_size_in_mb" in info:
        info["total_size_in_mb"] = video_mb + data_mb
    write_json(root / "meta/info.json", info)
    if (root / "info.json").exists():
        write_json(root / "info.json", info)


def validate(root, expected_frames):
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    if not isinstance(tasks.index[0], str):
        raise RuntimeError("Natural-language task index was not repaired")
    frames = 0
    for path in sorted((root / "data").glob("**/*.parquet")):
        table = pq.read_table(path, columns=["observation.state", "action"])
        if any(table.schema.field(k).type.list_size != 16 for k in ("observation.state", "action")):
            raise RuntimeError(f"Non-16-D data in {path}")
        frames += len(table)
    if frames != expected_frames:
        raise RuntimeError(f"Frame count changed: {expected_frames} -> {frames}")
    for path in sorted((root / "videos").glob("**/*.mp4")):
        size = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path),
        ], check=True, capture_output=True, text=True).stdout.strip()
        if size != "640x480":
            raise RuntimeError(f"Unexpected {size} video: {path}")


def main():
    args = cli()
    source = args.source.resolve()
    output = args.output.resolve() if args.output else source.with_name(f"{source.name}_{SUFFIX}")
    if not source.is_dir():
        raise RuntimeError(f"Missing source: {source}")
    if output.exists():
        raise RuntimeError(f"Output exists and will not be overwritten: {output}")
    if output == source or source in output.parents:
        raise RuntimeError("Output must be a separate sibling")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg/ffprobe are required")
    info = json.loads((source / "meta/info.json").read_text())
    state_ix, action_ix = inspect(info)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        shutil.copytree(source, tmp, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".cache", "*.tmp"))
        rewrite_data(tmp, state_ix, action_ix)
        rewrite_stats(tmp, state_ix, action_ix)
        tasks = repair_tasks(tmp)
        transcode(tmp, args.crf, args.preset)
        update_info(tmp, info)
        commit = None
        receipt = source / "DOWNLOAD_RECEIPT.txt"
        if receipt.exists():
            for line in receipt.read_text().splitlines():
                if line.startswith("resolved_hub_commit="):
                    commit = line.split("=", 1)[1]
        write_json(tmp / "DERIVATION_RECEIPT.json", {
            "source": str(source), "source_hub_commit": commit, "suffix": SUFFIX,
            "state": {"from": 22, "to": 16, "names": CONTROLS},
            "action": {"from": 19, "to": 16, "names": CONTROLS},
            "head": "1280x720 -> 640x360 Lanczos -> centered 640x480 black letterbox",
            "wrists": "copied unchanged at 640x480", "tasks": tasks,
            "statistics": "state/action sliced; head moments adjusted analytically for 25% black padding",
        })
        validate(tmp, int(info["total_frames"]))
        os.replace(tmp, output)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(f"Created: {output}\nSource was not modified.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError, AssertionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
