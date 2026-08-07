from __future__ import annotations

from pathlib import Path

import pytest

import backends
import registry
from test_registry import contract, dataset, dataset_registration, write_yaml


def resolved_policy(name: str, root: Path) -> dict:
    value = registry.layered(registry.ROOT / "configs" / "policies" / name)
    return registry.merge(
        value,
        {
            "name": "backend-test",
            "dataset": {
                "repo_id": "local/backend-test",
                "root": str(root),
                "eval_split": 0.1,
                "video_backend": "torchcodec",
            },
            "contract": contract(),
            "tracking": {"backend": "none", "mode": "disabled"},
            "resources": {"min_free_disk_gib": 0, "timeout_hours": 1},
        },
    )


def test_lerobot_groot_command_is_explicitly_all_absolute(tmp_path: Path) -> None:
    value = resolved_policy("lerobot_groot_n17.yaml", dataset(tmp_path))
    registry.validate(value, Path("test.yaml"), True, False)
    command = registry.command(value, tmp_path / "output")
    assert command[:3] == [
        str(registry.ROOT / "scripts/run_backend.sh"),
        "lerobot",
        "lerobot-train",
    ]
    assert "--policy.use_relative_actions=false" in command


def test_act_uses_visual_proprioception_without_language_flags(tmp_path: Path) -> None:
    value = resolved_policy("lerobot_act.yaml", dataset(tmp_path))
    registry.validate(value, Path("test.yaml"), True, False)
    command = registry.command(value, tmp_path / "output")
    assert value["policy"]["type"] == "act"
    assert value["policy"]["chunk_size"] == 100
    assert not any("language" in argument or "task" in argument for argument in command)
    assert "--policy.chunk_size=100" in command


def test_native_dataset_defaults_and_command_are_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = dataset(tmp_path)
    registrations = tmp_path / "registrations"
    registration = dataset_registration(canonical)
    registration["adapters"] = {
        "isaac_groot": {
            "root": str(tmp_path / "native"),
            "modality_config_path": str(
                registry.ROOT / "configs/modalities/f2_arms16_absolute_h40.py"
            ),
            "modality_json_path": str(
                registry.ROOT / "configs/modalities/f2_arms16_modality.json"
            ),
            "embodiment_tag": "NEW_EMBODIMENT",
        }
    }
    write_yaml(registrations / "test_dataset.yaml", registration)
    monkeypatch.setattr(registry, "DATASETS", registrations)

    value = registry.layered(registry.ROOT / "configs/policies/isaac_groot_n17.yaml")
    value = registry.merge(
        value,
        registry.dataset_defaults("test_dataset", "isaac-groot"),
    )
    value = registry.merge(
        value,
        {
            "name": "native-test",
            "tracking": {"backend": "none", "mode": "disabled"},
            "resources": {"min_free_disk_gib": 0, "timeout_hours": 1},
        },
    )
    registry.validate(value, Path("test.yaml"), False, False)
    command = registry.command(value, tmp_path / "output")
    assert command[:3] == [
        str(registry.ROOT / "scripts/run_backend.sh"),
        "isaac_groot",
        "python",
    ]
    assert "--use-relative-action" not in command
    assert "--no-use-relative-action" not in command
    assert value["policy"]["action_representation"] == "absolute"
    assert "--seed" not in command
    assert value["dataset"]["root"] == str((tmp_path / "native").resolve())
    invalid_seed = registry.merge(value, {"train": {"seed": 7}})
    with pytest.raises(registry.RegistryError, match="requires seed 42"):
        registry.validate(invalid_seed, Path("test.yaml"), False, False)


def test_native_wandb_mode_and_team_entity_are_forwarded(tmp_path: Path) -> None:
    value = resolved_policy("isaac_groot_n17.yaml", dataset(tmp_path))
    value["dataset"]["modality_config_path"] = str(
        tmp_path / "f2_arms16_absolute_h40.py"
    )
    value = registry.merge(
        value,
        {
            "tracking": {
                "backend": "wandb",
                "enable": True,
                "entity": "robotis-intern-team-peanut-belt",
                "project": "isaac-groot-omx-insert",
                "mode": "online",
            }
        },
    )
    command = registry.command(value, tmp_path / "output")
    assert command[:6] == [
        str(registry.ROOT / "scripts/run_backend.sh"),
        "isaac_groot",
        "env",
        "WANDB_MODE=online",
        "WANDB_ENTITY=robotis-intern-team-peanut-belt",
        "python",
    ]


@pytest.mark.parametrize(
    ("policy", "base"),
    [
        ("lerobot-groot", "../policies/lerobot_groot_n17.yaml"),
        ("isaac-groot", "../policies/isaac_groot_n17.yaml"),
        ("lerobot-act", "../policies/lerobot_act.yaml"),
    ],
)
def test_run_init_selects_policy_recipe(
    policy: str,
    base: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registrations = tmp_path / "registrations"
    write_yaml(
        registrations / "test_dataset.yaml",
        dataset_registration(dataset(tmp_path / "data")),
    )
    monkeypatch.setattr(registry, "DATASETS", registrations)
    monkeypatch.setattr(registry, "CONFIGS", tmp_path / policy)
    created = registry.init_config("comparison", "test_dataset", policy)
    assert registry.read_yaml(created)["_base"] == base


def test_backend_aliases_are_stable() -> None:
    assert backends.canonical_backend("groot") == "lerobot_groot"
    assert backends.canonical_backend("act") == "lerobot_act"
    assert backends.canonical_backend("isaac-groot") == "isaac_groot"
