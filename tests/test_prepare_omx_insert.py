import json
from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.prepare_omx_insert_native import (
    CAMERAS,
    CONTROLS,
    EXPECTED_IMAGE_SHAPE,
    PreparationError,
    inspect_source,
    normalize_tasks,
)


def source_info(head_shape: list[int] | None = None, fps: int = 15) -> dict:
    features = {
        "observation.state": {
            "shape": [22],
            "names": [*CONTROLS, "head_pan", "head_tilt", "lift", "x", "y", "yaw"],
        },
        "action": {
            "shape": [19],
            "names": [*CONTROLS, "x", "y", "yaw"],
        },
    }
    features.update(
        {
            camera: {
                "shape": head_shape if camera == CAMERAS[0] and head_shape else EXPECTED_IMAGE_SHAPE
            }
            for camera in CAMERAS
        }
    )
    return {"codebase_version": "v3.0", "fps": fps, "features": features}


def test_native_640x480_contract_is_accepted() -> None:
    state_indices, action_indices = inspect_source(source_info())
    assert state_indices == list(range(16))
    assert action_indices == list(range(16))


def test_legacy_1280x720_head_is_rejected_instead_of_letterboxed() -> None:
    with pytest.raises(PreparationError, match="native 640x480"):
        inspect_source(source_info([3, 720, 1280]))


def test_non_15_fps_source_is_rejected() -> None:
    with pytest.raises(PreparationError, match="fps=15"):
        inspect_source(source_info(fps=30))


def test_control_order_is_not_silently_rearranged() -> None:
    info = source_info()
    names = info["features"]["action"]["names"]
    names[0], names[1] = names[1], names[0]
    with pytest.raises(PreparationError, match="exact ordered 16-D"):
        inspect_source(info)


def test_task_column_is_normalized_to_le_robot_index(tmp_path: Path) -> None:
    metadata = tmp_path / "meta"
    metadata.mkdir()
    path = metadata / "tasks.parquet"
    pd.DataFrame(
        [{"task_index": 0, "task": "Insert the component.", "task_name": "OMX"}]
    ).to_parquet(path)

    assert normalize_tasks(tmp_path) == ["Insert the component."]
    normalized = pd.read_parquet(path)
    assert normalized.index.name == "task"
    assert normalized.index.tolist() == ["Insert the component."]
    assert "task" not in normalized.columns
