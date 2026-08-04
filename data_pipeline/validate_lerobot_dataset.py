#!/usr/bin/env python3
"""Decode one canonical dataset sample inside the pinned LeRobot environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset,
        video_backend=args.video_backend,
    )
    sample = dataset[0]
    state = np.asarray(sample["observation.state"])
    action = np.asarray(sample["action"])
    images = {
        key: list(np.asarray(value).shape)
        for key, value in sample.items()
        if key.startswith("observation.images.")
    }
    result = {
        "sample": 0,
        "state_shape": list(state.shape),
        "action_shape": list(action.shape),
        "finite": bool(np.isfinite(state).all() and np.isfinite(action).all()),
        "images": images,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
