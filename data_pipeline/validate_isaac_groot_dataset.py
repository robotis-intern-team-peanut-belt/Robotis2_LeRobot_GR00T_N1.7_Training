#!/usr/bin/env python3
"""Validate one real sample from an Isaac-GR00T LeRobot v2.1 derivative."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionRepresentation


def load_module(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import modality config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate(dataset: Path, modality_config: Path) -> dict:
    load_module(modality_config.resolve())
    configs = MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]
    action_configs = configs["action"].action_configs or []
    if not action_configs or any(
        item.rep != ActionRepresentation.ABSOLUTE for item in action_configs
    ):
        raise RuntimeError("F2 native modality config is not all-absolute")
    if configs["action"].delta_indices != list(range(40)):
        raise RuntimeError("F2 native modality config does not use horizon 40")

    loader = LeRobotEpisodeLoader(dataset.resolve(), configs)
    if len(loader) < 1:
        raise RuntimeError("native dataset has no episodes")
    episode_id = int(loader.episodes_metadata[0]["episode_index"])
    frame = loader._load_parquet_data(episode_id)
    if frame.empty:
        raise RuntimeError("first native episode is empty")

    state_columns = [f"state.{key}" for key in configs["state"].modality_keys]
    action_columns = [f"action.{key}" for key in configs["action"].modality_keys]
    state = np.concatenate([np.asarray(frame[key].iloc[0]) for key in state_columns])
    action = np.concatenate([np.asarray(frame[key].iloc[0]) for key in action_columns])
    if state.shape != (16,) or action.shape != (16,):
        raise RuntimeError(
            f"expected finite 16-D state/action, got {state.shape}/{action.shape}"
        )
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise RuntimeError("non-finite state/action in native sample")

    language_key = f"language.{configs['language'].modality_keys[0]}"
    language = frame[language_key].iloc[0]
    if not isinstance(language, str) or not language.strip():
        raise RuntimeError("native language sample is empty")

    videos = loader._load_video_data(episode_id, np.asarray([0]))
    if list(videos) != configs["video"].modality_keys:
        raise RuntimeError(f"native camera order mismatch: {list(videos)}")
    shapes = {key: list(np.asarray(value[0]).shape) for key, value in videos.items()}
    if any(shape[:2] != [480, 640] or shape[-1] != 3 for shape in shapes.values()):
        raise RuntimeError(f"native camera shape mismatch: {shapes}")

    return {
        "dataset": str(dataset.resolve()),
        "codebase_version": loader.info_meta.get("codebase_version"),
        "episodes": len(loader),
        "state_shape": list(state.shape),
        "action_shape": list(action.shape),
        "camera_shapes": shapes,
        "language": language,
        "action_representation": "absolute",
        "action_horizon": 40,
    }


if __name__ == "__main__":
    args = parse_args()
    report = validate(args.dataset, args.modality_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
