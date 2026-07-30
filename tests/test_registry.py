from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import registry


JOINTS = [
    *[f"arm_l_joint{i}" for i in range(1, 8)],
    "gripper_l_joint1",
    *[f"arm_r_joint{i}" for i in range(1, 8)],
    "gripper_r_joint1",
]
CAMERAS = [
    "observation.images.rgb.cam_head",
    "observation.images.rgb.cam_left_wrist",
    "observation.images.rgb.cam_right_wrist",
]


def write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    features = {
        "observation.state": {"dtype": "float32", "shape": [16], "names": JOINTS},
        "action": {"dtype": "float32", "shape": [16], "names": JOINTS},
    }
    features.update({key: {"dtype": "video", "shape": [3, 480, 640]} for key in CAMERAS})
    (root / "meta/info.json").write_text(json.dumps({
        "codebase_version": "v3.0",
        "total_episodes": 2,
        "total_frames": 20,
        "fps": 15,
        "features": features,
    }))
    return root


def contract() -> dict:
    return {
        "fps": 15,
        "cameras": [{"key": key, "shape": [3, 480, 640]} for key in CAMERAS],
        "state": {"dim": 16, "names": JOINTS},
        "action": {"dim": 16, "names": JOINTS},
    }


def config(root: Path, name: str = "test-run") -> dict:
    return {
        "schema_version": 1,
        "backend": "lerobot",
        "name": name,
        "description": "",
        "hypothesis": "",
        "tags": [],
        "dataset": {"repo_id": "local/test", "root": str(root), "eval_split": 0.1},
        "contract": contract(),
        "policy": {
            "type": "groot",
            "device": "cuda",
            "max_steps": 20,
            "optimizer_lr": 1e-4,
            "use_relative_actions": True,
            "relative_exclude_joints": ["gripper"],
        },
        "train": {
            "steps": 20,
            "batch_size": 1,
            "num_workers": 1,
            "use_policy_training_preset": True,
        },
        "optimizer": {"type": "adamw", "lr": 1e-4},
        "scheduler": {"type": "cosine"},
        "tracking": {"backend": "none", "mode": "disabled"},
        "resources": {"min_free_disk_gib": 0, "timeout_hours": 1},
        "artifacts": {},
        "annotation": {},
    }


def dataset_registration(root: Path) -> dict:
    return {
        "schema_version": 1,
        "name": "test_dataset",
        "repo_id": "org/source",
        "revision": "a" * 40,
        "download": {"root": str(root)},
        "prepared": {"root": str(root), "immutable": True},
        "training": {"repo_id": "local/test", "video_backend": "torchcodec", "eval_split": 0.1},
        "contract": contract(),
    }


def test_deep_merge_and_list_replacement() -> None:
    assert registry.merge({"a": {"b": 1, "c": [1, 2]}}, {"a": {"c": [3]}}) == {
        "a": {"b": 1, "c": [3]}
    }


def test_recursive_base_and_substitution(tmp_path: Path) -> None:
    root = dataset(tmp_path)
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    write_yaml(base, config(root, "base"))
    write_yaml(child, {"_base": "base.yaml", "name": "child", "policy": {"max_steps": 30},
                       "train": {"steps": 30}})
    item = registry.experiment(child, ["train.batch_size=2"])
    assert item.name == "child"
    assert item.data["policy"]["max_steps"] == 30
    assert item.data["train"]["batch_size"] == 2


def test_dataset_reference_injects_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = dataset(tmp_path)
    registrations = tmp_path / "registrations"
    write_yaml(registrations / "test_dataset.yaml", dataset_registration(root))
    monkeypatch.setattr(registry, "DATASETS", registrations)
    value = config(root)
    value.pop("dataset")
    value.pop("contract")
    value["dataset_ref"] = "test_dataset"
    source = tmp_path / "run.yaml"
    write_yaml(source, value)
    item = registry.experiment(source)
    assert item.data["dataset"]["registry_ref"] == "test_dataset"
    assert item.data["dataset"]["root"] == str(root)
    assert item.data["contract"]["action"]["names"] == JOINTS


def test_circular_base_detection(tmp_path: Path) -> None:
    write_yaml(tmp_path / "a.yaml", {"_base": "b.yaml"})
    write_yaml(tmp_path / "b.yaml", {"_base": "a.yaml"})
    with pytest.raises(registry.RegistryError, match="circular"):
        registry.layered(tmp_path / "a.yaml")


@pytest.mark.parametrize("name", ["../escape", ".hidden", "space name", "a/b"])
def test_unsafe_names_rejected(tmp_path: Path, name: str) -> None:
    value = config(dataset(tmp_path), name)
    with pytest.raises(registry.RegistryError, match="unsafe"):
        registry.validate(value, Path("test.yaml"), True, False)


def test_unknown_top_level_rejected(tmp_path: Path) -> None:
    value = config(dataset(tmp_path))
    value["typo"] = True
    with pytest.raises(registry.RegistryError, match="unknown"):
        registry.validate(value, Path("test.yaml"), True, False)


def test_date_like_tag_must_be_quoted(tmp_path: Path) -> None:
    value = config(dataset(tmp_path))
    value["tags"] = [yaml.safe_load("2026-07-28")]
    with pytest.raises(registry.RegistryError, match="quote date-like YAML tags"):
        registry.validate(value, Path("test.yaml"), True, False)


def test_policy_and_train_steps_must_match(tmp_path: Path) -> None:
    value = config(dataset(tmp_path))
    value["train"]["steps"] = 21
    with pytest.raises(registry.RegistryError, match="max_steps"):
        registry.validate(value, Path("test.yaml"), True, False)


def test_contract_camera_order_is_checked(tmp_path: Path) -> None:
    value = config(dataset(tmp_path))
    value["contract"]["cameras"].reverse()
    with pytest.raises(registry.RegistryError, match="camera"):
        registry.validate(value, Path("test.yaml"), True, False)


def test_command_is_dry_and_has_output_guard_path(tmp_path: Path) -> None:
    value = config(dataset(tmp_path))
    argv = registry.command(value, tmp_path / "output")
    assert argv[0] == "lerobot-train"
    assert "--output_dir=" + str(tmp_path / "output") in argv
    assert "--wandb.enable=false" in argv
    assert not (tmp_path / "output").exists()


def test_optimizer_forwarded_only_without_policy_preset(tmp_path: Path) -> None:
    value = config(dataset(tmp_path))
    assert not any(arg.startswith("--optimizer.") for arg in registry.command(value))
    value["train"]["use_policy_training_preset"] = False
    argv = registry.command(value)
    assert "--optimizer.type=adamw" in argv
    assert "--optimizer.lr=0.0001" in argv


def test_duplicate_output_rejected_before_launch(tmp_path: Path) -> None:
    root = dataset(tmp_path)
    source = tmp_path / "run.yaml"
    write_yaml(source, config(root))
    item = registry.experiment(source)
    run_dir = tmp_path / "runs/test-run"
    (run_dir / "output").mkdir(parents=True)
    with pytest.raises(registry.RegistryError, match="existing output"):
        registry.execute(item, dry_run=True, run_dir=run_dir)
