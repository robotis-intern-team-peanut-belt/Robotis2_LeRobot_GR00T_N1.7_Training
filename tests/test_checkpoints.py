from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import checkpoints


def write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def completed_run(tmp_path: Path) -> Path:
    run = tmp_path / "runs" / "act_smoke"
    model = run / "output/checkpoints/last/pretrained_model"
    model.mkdir(parents=True)
    for name in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "train_config.json",
    ):
        (model / name).write_text(f"{name}\n", encoding="utf-8")
    config = run / "resolved_config.yaml"
    write_yaml(
        config,
        {
            "backend": "lerobot_act",
            "artifacts": {
                "required_checkpoint_files": [
                    "config.json",
                    "model.safetensors",
                    "policy_preprocessor.json",
                    "policy_postprocessor.json",
                    "train_config.json",
                ]
            },
        },
    )
    result = run / "result.yaml"
    write_yaml(
        result,
        {
            "run_name": "act_smoke",
            "status": {"state": "completed"},
            "artifacts": {
                "resolved_config": str(config),
                "pretrained_model": str(model),
                "manifest": str(run / "manifest.json"),
            },
        },
    )
    return result


def test_package_verify_and_transfer_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = completed_run(tmp_path)
    monkeypatch.setattr(checkpoints, "PACKAGES", tmp_path / "packages")
    packaged = checkpoints.package_checkpoint(result, "f2_act_candidate")
    package = Path(packaged["package"])
    assert packaged["backend"] == "lerobot_act"
    assert (package / "model.safetensors").is_file()
    assert checkpoints.verify_package(package)["state"] == "verified"

    plan = checkpoints.transfer_package(
        package,
        target="robotis@f2",
        dry_run=True,
    )
    assert plan["state"] == "dry_run"
    assert plan["robot_workspace"] == {
        "selected": "/home/robotis/cyclo_intelligence/docker/workspace",
        "home_alias": "/home/robotis/cyclo_intelligence/docker/workspace",
        "ssd_storage": "/mnt/ssd/cyclo_intelligence/workspace",
        "state": "planned",
        "device_inode": None,
    }
    assert "mountpoint -q" in plan["commands"][0]
    assert "stat -Lc" in plan["commands"][0]
    assert plan["remote_host_path"].endswith("/model/lerobot/f2_act_candidate")
    assert plan["remote_container_path"] == "/workspace/model/lerobot/f2_act_candidate"
    assert any(command.startswith("rsync ") for command in plan["commands"])


def test_package_refuses_checksum_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = completed_run(tmp_path)
    monkeypatch.setattr(checkpoints, "PACKAGES", tmp_path / "packages")
    package = Path(checkpoints.package_checkpoint(result)["package"])
    (package / "config.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(checkpoints.CheckpointError, match="checksum mismatch"):
        checkpoints.verify_package(package)


def test_native_transfer_uses_cyclo_groot_model_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = completed_run(tmp_path)
    config = Path(yaml.safe_load(result.read_text())["artifacts"]["resolved_config"])
    value = yaml.safe_load(config.read_text())
    value["backend"] = "isaac_groot"
    write_yaml(config, value)
    monkeypatch.setattr(checkpoints, "PACKAGES", tmp_path / "packages")
    package = checkpoints.package_checkpoint(result, "f2_native_candidate")["package"]
    plan = checkpoints.transfer_package(
        package,
        target="f2",
        dry_run=True,
    )
    assert plan["remote_host_path"].endswith("/model/groot/f2_native_candidate")
    assert plan["robot_workspace"]["selected"] == str(
        checkpoints.DEFAULT_ROBOT_WORKSPACE
    )
    assert plan["remote_container_path"] == "/workspace/model/groot/f2_native_candidate"


def test_transfer_requires_explicit_safe_live_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = completed_run(tmp_path)
    monkeypatch.setattr(checkpoints, "PACKAGES", tmp_path / "packages")
    package = checkpoints.package_checkpoint(result)["package"]
    with pytest.raises(checkpoints.CheckpointError, match="absolute path"):
        checkpoints.transfer_package(
            package, target="f2", robot_workspace="relative/path", dry_run=True
        )


def test_transfer_records_verified_workspace_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = completed_run(tmp_path)
    monkeypatch.setattr(checkpoints, "PACKAGES", tmp_path / "packages")
    monkeypatch.setattr(checkpoints, "RECEIPTS", tmp_path / "receipts")
    package = checkpoints.package_checkpoint(result)["package"]

    def fake_run(command, **kwargs):
        stdout = "8:42\n" if kwargs.get("capture_output") else None
        return checkpoints.subprocess.CompletedProcess(command, 0, stdout=stdout)

    monkeypatch.setattr(checkpoints.subprocess, "run", fake_run)
    report = checkpoints.transfer_package(package, target="f2")

    assert report["robot_workspace"]["state"] == "verified_same_directory"
    assert report["robot_workspace"]["device_inode"] == "8:42"
    receipt = Path(report["receipt"])
    recorded = json.loads(receipt.read_text(encoding="utf-8"))
    assert recorded["robot_workspace"] == report["robot_workspace"]


def test_transfer_fails_closed_when_home_bind_mount_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = completed_run(tmp_path)
    monkeypatch.setattr(checkpoints, "PACKAGES", tmp_path / "packages")
    monkeypatch.setattr(checkpoints, "RECEIPTS", tmp_path / "receipts")
    package = checkpoints.package_checkpoint(result)["package"]

    def reject_alias(command, **kwargs):
        raise checkpoints.subprocess.CalledProcessError(
            40,
            command,
            stderr="Cyclo home repository is not an active bind mount",
        )

    monkeypatch.setattr(checkpoints.subprocess, "run", reject_alias)
    with pytest.raises(checkpoints.CheckpointError, match="bind-mount validation"):
        checkpoints.transfer_package(package, target="f2")

    receipts = list((tmp_path / "receipts").glob("*_transfer.json"))
    assert len(receipts) == 1
    recorded = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert recorded["state"] == "failed_workspace_alias_check"
