from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import registry
from test_registry import dataset, dataset_registration, write_yaml


def test_empty_list_flag_is_omitted() -> None:
    assert registry.flags("policy", {"relative_exclude_joints": []}) == []


def test_training_root_relative_reference_is_resolved() -> None:
    expected = registry.ROOT / "configs/base/multi_queue.yaml"
    assert (
        registry.find("configs/base/multi_queue.yaml", registry.QUEUES)
        == expected.resolve()
    )


def test_image_transform_map_is_one_structured_flag() -> None:
    values = {"tfs": {"brightness": {"weight": 1.0}}}
    assert registry.flags("dataset.image_transforms", values) == [
        '--dataset.image_transforms.tfs={"brightness":{"weight":1.0}}'
    ]


def test_queue_base_inheritance(tmp_path: Path) -> None:
    write_yaml(
        tmp_path / "base.yaml",
        {
            "schema_version": 1,
            "continue_on_failure": False,
            "min_free_disk_gib": 10,
            "max_attempts": 1,
            "time_budget_hours": 4,
        },
    )
    write_yaml(
        tmp_path / "queue.yaml",
        {
            "_base": "base.yaml",
            "name": "queue",
            "description": "test",
            "runs": [{"config": "run.yaml", "overrides": []}],
        },
    )
    value = registry.layered(tmp_path / "queue.yaml")
    assert value["max_attempts"] == 1
    assert value["runs"][0]["config"] == "run.yaml"


def test_config_init_creates_small_editable_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = dataset(tmp_path)
    configs = tmp_path / "configs"
    registrations = tmp_path / "registrations"
    write_yaml(registrations / "test_dataset.yaml", dataset_registration(root))
    monkeypatch.setattr(registry, "CONFIGS", configs)
    monkeypatch.setattr(registry, "DATASETS", registrations)
    created = registry.init_config("new_run", "test_dataset")
    value = yaml.safe_load(created.read_text())
    assert value["_base"] == "../policies/lerobot_groot_n17.yaml"
    assert value["dataset_ref"] == "test_dataset"
    assert "dataset" not in value
    assert "tags" not in value
    assert value["description"].startswith("REPLACE ME:")
    assert value["hypothesis"].startswith("REPLACE ME:")
    assert value["annotation"]["owner"] == "intern-team"
    assert value["annotation"]["notes"].startswith("REPLACE ME:")
    with pytest.raises(registry.RegistryError, match="already exists"):
        registry.init_config("new_run", "test_dataset")


def test_ready_dataset_needs_no_prepare_script(tmp_path: Path) -> None:
    root = dataset(tmp_path)
    value = dataset_registration(root)
    assert registry.prepare_dataset(value) == root


def test_result_file_uses_latest_numbered_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry, "RUNS", tmp_path)
    for attempt in (2, 10):
        path = tmp_path / "run" / f"attempt-{attempt}" / "result.yaml"
        write_yaml(path, {"run_name": "run", "status": {"state": "completed"}})
    assert registry.result_file("run").parent.name == "attempt-10"


class FakeCuda:
    def __init__(self, total_gib: float, available: bool = True) -> None:
        self.total_gib = total_gib
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def current_device(self) -> int:
        return 0

    def get_device_properties(self, _device: int) -> SimpleNamespace:
        return SimpleNamespace(total_memory=self.total_gib * 1024**3)

    def get_device_name(self, _device: int) -> str:
        return "test-gpu"


def test_gpu_memory_guard_accepts_sufficient_device() -> None:
    data = {"resources": {"memory_guard_gib": 80}}
    observed = registry.validate_gpu_resources(
        data, SimpleNamespace(cuda=FakeCuda(89.0))
    )
    assert observed == {"device": "test-gpu", "total_memory_gib": 89.0}


def test_gpu_memory_guard_rejects_small_device() -> None:
    data = {"resources": {"memory_guard_gib": 80}}
    with pytest.raises(registry.RegistryError, match="memory guard"):
        registry.validate_gpu_resources(data, SimpleNamespace(cuda=FakeCuda(40.0)))


def test_run_tmux_parser_matches_run_options() -> None:
    args = registry.parser().parse_args(
        [
            "run-tmux",
            "campaign/run",
            "--set",
            "train.steps=10",
            "--session",
            "groot-test",
            "--run-dir",
            "training/runs/test",
            "--resume",
            "--dry-run",
        ]
    )
    assert args.action == "run-tmux"
    assert args.reference == "campaign/run"
    assert args.set == ["train.steps=10"]
    assert args.session == "groot-test"
    assert args.run_dir == Path("training/runs/test")
    assert args.resume is True
    assert args.dry_run is True


def test_tmux_report_prints_monitoring_and_stop_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "state": "launched",
        "run_name": "test-run",
        "tmux_session": "train-test-run",
        "run_dir": "/tmp/test-run",
        "launch_log": "/tmp/test-run.launch.log",
        "preflight": "config, dataset, disk, and GPU checks passed",
        "commands": {
            "trainer": "lerobot-train --dataset.repo_id=local/test",
            "attach": "tmux attach -t =groot-test-run",
            "status": "training/trainctl result show test-run",
            "train_log": "tail -f /tmp/test-run/train.log",
            "launch_log": "tail -f /tmp/test-run.launch.log",
            "list_sessions": "tmux list-sessions",
            "stop": "tmux kill-session -t =groot-test-run",
        },
    }
    registry.print_tmux_report(report)
    output = capsys.readouterr().out
    assert "continues if this terminal or SSH connection closes" in output
    assert "lerobot-train --dataset.repo_id=local/test" in output
    assert "Ctrl-b, then d" in output
    assert "training/trainctl result show test-run" in output
    assert "STOP TRAINING" in output


def test_launch_tmux_starts_detached_session_and_preserves_startup_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = registry.Experiment(tmp_path / "run.yaml", {"name": "test-run"})
    monkeypatch.setattr(registry, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(registry, "CLI", registry.ROOT / "trainctl")
    monkeypatch.setattr(registry.shutil, "which", lambda name: f"/usr/bin/{name}")
    session_checks = iter([False, True])
    monkeypatch.setattr(registry, "tmux_has_session", lambda name: next(session_checks))
    monkeypatch.setattr(
        registry,
        "execute",
        lambda item, dry_run, resume, run_dir: {
            "output_dir": str(run_dir / "output"),
            "command": ["lerobot-train", "--dataset.repo_id=local/test"],
        },
    )
    monkeypatch.setattr(
        registry,
        "validate_gpu_resources",
        lambda data: {"device": "test-gpu", "total_memory_gib": 89.0},
    )
    calls = []
    monkeypatch.setattr(
        registry.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(registry.time, "sleep", lambda seconds: None)

    report = registry.launch_tmux(
        item, sets=["train.steps=20"], run_dir=tmp_path / "run"
    )

    assert report["state"] == "launched"
    assert report["tmux_session"] == "train-test-run"
    assert report["preflight"].endswith("(test-gpu, 89.0 GiB visible)")
    assert calls[0][:4] == ["tmux", "new-session", "-d", "-s"]
    shell_command = calls[0][-1]
    assert "training/trainctl" in shell_command
    assert "--set" in shell_command
    assert "train.steps=20" in shell_command
    assert "tee" in shell_command
    assert "test-run.tmux.log" in shell_command


def test_metadata_derived_contract_keeps_human_fields(tmp_path: Path) -> None:
    root = dataset(tmp_path)
    value = dataset_registration(root)
    value["contract"] = {
        "auto_from_metadata": True,
        "robot_revision": "f2-test",
        "units": {"state": "radians", "action": "joint_targets"},
    }
    resolved = registry.resolved_contract(value)
    assert resolved["fps"] == 15
    assert resolved["state"]["names"][0] == "arm_l_joint1"
    assert [camera["key"] for camera in resolved["cameras"]] == [
        "observation.images.rgb.cam_head",
        "observation.images.rgb.cam_left_wrist",
        "observation.images.rgb.cam_right_wrist",
    ]
    assert resolved["robot_revision"] == "f2-test"


def test_dataset_init_creates_minimal_auto_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = dataset(tmp_path)
    registrations = tmp_path / "registrations"
    monkeypatch.setattr(registry, "DATASETS", registrations)
    created = registry.init_dataset(
        "new_dataset",
        "org/new_dataset",
        "a" * 40,
        root,
        "f2-test",
        None,
        0.1,
        "radians_and_normalized_gripper",
        "joint_position_targets",
        "English imperative task instruction",
        "new_embodiment",
    )
    value = yaml.safe_load(created.read_text())
    assert value["prepared"]["root"] == str(root)
    assert value["contract"]["auto_from_metadata"] is True
    assert value["expected"] == {
        "codebase_version": "v3.0",
        "total_episodes": 2,
        "total_frames": 20,
    }
    with pytest.raises(registry.RegistryError, match="already exists"):
        registry.init_dataset(
            "new_dataset",
            "org/new_dataset",
            "a" * 40,
            root,
            "f2-test",
            None,
            0.1,
            "radians",
            "joint_targets",
            "English imperative",
            "new_embodiment",
        )


def test_register_hf_omx_dataset_pins_current_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registrations = tmp_path / "registrations"
    monkeypatch.setattr(registry, "DATASETS", registrations)
    created = registry.register_hf_omx_dataset(
        "campaign",
        "User/Repo",
        "a" * 40,
        "f2-commit",
        "Insert all five parts.",
        tmp_path / "source",
        tmp_path / "prepared",
    )
    value = yaml.safe_load(created.read_text())
    assert value["revision"] == "a" * 40
    assert value["preparation"]["inject_source_provenance"] is True
    assert value["contract"]["units"]["action"] == "absolute_joint_position_targets"
    assert value["contract"]["cameras"][0]["shape"] == [3, 480, 640]
    assert value["contract"]["action"]["dim"] == 16
    assert value["registration"]["robot_revision_source"] == "command_line"
    assert value["registration"]["task_source"] == "command_line"
    assert value["validation"]["caption_audit"]["expected_overall_task"] == (
        "Insert all five parts."
    )


def test_register_hf_parser_needs_only_name_and_repo() -> None:
    args = registry.parser().parse_args(
        [
            "dataset",
            "register-hf",
            "campaign",
            "--repo-id",
            "User/Repo",
        ]
    )
    assert args.verb == "register-hf"
    assert args.revision == "main"
    assert args.robot_revision is None
    assert args.task is None


def test_register_local_needs_only_name_and_source_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = dataset(tmp_path / "payload")
    registrations = tmp_path / "registrations"
    monkeypatch.setattr(registry, "DATASETS", registrations)
    monkeypatch.setattr(registry, "ROOT", tmp_path / "training")

    created = registry.register_local_omx_dataset(
        "campaign_local",
        source,
        None,
        None,
        tmp_path / "prepared",
    )
    value = yaml.safe_load(created.read_text())
    marker = yaml.safe_load((source / ".groot_registry_download.json").read_text())
    assert value["repo_id"] == "local/campaign_local"
    assert value["revision"] == "local"
    assert value["registration"]["source_kind"] == "direct_local_directory"
    assert "source_manifest_sha256" not in value
    assert value["expected"]["codebase_version"] == "v3.0"
    assert marker == {
        "repo_id": "local/campaign_local",
        "revision": "local",
        "source_kind": "direct_local_directory",
    }
    assert registry.download_dataset(registry.dataset_spec(created)) == source

    (source / "meta/info.json").write_text("{}")
    assert registry.download_dataset(registry.dataset_spec(created)) == source


def test_register_local_accepts_optional_free_form_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = dataset(tmp_path / "payload")
    handoff = tmp_path / "notes.md"
    handoff.write_text("optional notes\n")
    monkeypatch.setattr(registry, "DATASETS", tmp_path / "registrations")
    monkeypatch.setattr(registry, "ROOT", tmp_path / "training")

    created = registry.register_local_omx_dataset(
        "campaign_local",
        source,
        None,
        None,
        source_revision="robot-copy-july",
        handoff=handoff,
    )
    value = yaml.safe_load(created.read_text())
    assert value["revision"] == "robot-copy-july"
    assert value["registration"]["handoff"] == str(handoff.resolve())


def test_register_local_parser_needs_only_name_and_source_root() -> None:
    args = registry.parser().parse_args(
        [
            "dataset",
            "register-local",
            "campaign",
            "--source-root",
            "/srv/dataset",
        ]
    )
    assert args.verb == "register-local"
    assert args.source_revision is None
    assert args.handoff is None
    assert args.robot_revision is None
    assert args.task is None


def test_register_hf_uses_visible_project_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = tmp_path / "f2_defaults.yaml"
    write_yaml(
        profile,
        {
            "schema_version": 1,
            "name": "f2_omx_insert",
            "robot_revision": "f2-profile-commit",
            "task": "Insert all five parts.",
            "evidence": "/handoffs/example.md",
        },
    )
    registrations = tmp_path / "registrations"
    monkeypatch.setattr(registry, "F2_OMX_PROFILE", profile)
    monkeypatch.setattr(registry, "DATASETS", registrations)

    created = registry.register_hf_omx_dataset(
        "campaign",
        "User/Repo",
        "a" * 40,
        None,
        None,
        tmp_path / "source",
        tmp_path / "prepared",
    )
    value = yaml.safe_load(created.read_text())
    assert value["contract"]["robot_revision"] == "f2-profile-commit"
    assert value["contract"]["language"]["instruction"] == "Insert all five parts."
    assert value["registration"]["robot_revision_source"] == "project_default"
    assert value["registration"]["task_source"] == "project_default"
    report = registry.hf_registration_report(created)
    assert report["hub"]["pinned_commit"] == "a" * 40
    assert report["robot_revision"]["source"] == "project_default"


def test_dataset_defaults_parser() -> None:
    args = registry.parser().parse_args(["dataset", "defaults"])
    assert args.action == "dataset"
    assert args.verb == "defaults"


def test_download_registration_upgrades_to_payload_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.bin").write_bytes(b"payload")
    (source / ".groot_registry_download.json").write_text(
        '{"repo_id":"User/Repo","revision":"' + "a" * 40 + '"}'
    )
    manifest = tmp_path / "manifest.txt"
    value = {
        "name": "campaign",
        "repo_id": "User/Repo",
        "revision": "a" * 40,
        "download": {"root": str(source), "manifest": str(manifest)},
    }
    assert registry.download_dataset(value) == source
    marker = registry.download_provenance(value)
    assert marker["payload_file_count"] == 1
    assert marker["payload_bytes"] == 7
    assert len(marker["payload_manifest_sha256"]) == 64
    assert "data.bin" in manifest.read_text()


def test_existing_prepared_derivative_requires_matching_receipt(tmp_path: Path) -> None:
    source = tmp_path / "source"
    prepared = tmp_path / "prepared"
    source.mkdir()
    prepared.mkdir()
    script = tmp_path / "prepare.sh"
    script.write_text("#!/bin/sh\nexit 1\n")
    receipt = {
        "source": str(source.resolve()),
        "source_revision": "a" * 40,
        "source_manifest_sha256": "b" * 64,
    }
    (prepared / "DERIVATION_RECEIPT.json").write_text(__import__("json").dumps(receipt))
    value = {
        "revision": "a" * 40,
        "source_manifest_sha256": "b" * 64,
        "download": {"root": str(source)},
        "prepared": {"root": str(prepared)},
        "preparation": {"script": str(script)},
    }
    assert registry.prepare_dataset(value) == prepared


def test_system_tmux_environment_drops_conda_lib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONDA_PREFIX", "/opt/groot-runtime")
    monkeypatch.setenv(
        "LD_LIBRARY_PATH", "/opt/groot-runtime/lib:/usr/local/cuda/lib64"
    )
    environment = registry.system_tmux_environment()
    assert environment["LD_LIBRARY_PATH"] == "/usr/local/cuda/lib64"


def test_dataset_pipeline_parser() -> None:
    args = registry.parser().parse_args(["dataset", "pipeline", "campaign"])
    assert args.action == "dataset"
    assert args.verb == "pipeline"


def test_simple_train_lifecycle_parser() -> None:
    initialized = registry.parser().parse_args(
        [
            "train",
            "init",
            "baseline",
            "--dataset",
            "campaign",
        ]
    )
    checked = registry.parser().parse_args(
        [
            "train",
            "check",
            "baseline",
            "--set",
            "train.batch_size=16",
        ]
    )
    started = registry.parser().parse_args(["train", "start", "baseline"])
    status = registry.parser().parse_args(["train", "status", "baseline"])
    assert initialized.dataset == "campaign"
    assert checked.set == ["train.batch_size=16"]
    assert started.verb == "start"
    assert started.resume is False
    assert status.verb == "status"


def test_final_training_metrics_reads_last_logged_step() -> None:
    text = "\n".join(
        [
            "step:10 smpl:40 loss:1.511 grdn:2.573 smp/s:3 mem_gb:35.94",
            "step:20 smpl:80 loss:1.293 grdn:1.519 smp/s:13 mem_gb:35.94",
        ]
    )
    assert registry.final_training_metrics(text) == {
        "step": 20,
        "loss": 1.293,
        "gradient_norm": 1.519,
        "samples_per_second": 13.0,
        "cuda_memory_gib": 35.94,
    }


def test_dataset_pipeline_runs_fail_closed_stages_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        registry,
        "download_dataset",
        lambda data: calls.append("download") or tmp_path / "source",
    )
    monkeypatch.setattr(
        registry,
        "preflight_dataset_root",
        lambda root: calls.append(("preflight", root.name)) or {"ready": True},
    )
    monkeypatch.setattr(
        registry,
        "audit_dataset_captions",
        lambda data: calls.append("audit") or {"ready": True},
    )
    monkeypatch.setattr(
        registry,
        "prepare_dataset",
        lambda data: calls.append("prepare") or tmp_path / "prepared",
    )
    monkeypatch.setattr(
        registry,
        "validate_dataset",
        lambda data, decode: calls.append(("validate", decode)) or {"ok": True},
    )
    report = registry.dataset_pipeline({"name": "campaign", "revision": "a" * 40})
    assert calls == [
        "download",
        ("preflight", "source"),
        "audit",
        "prepare",
        ("validate", True),
    ]
    assert report["validation"] == {"ok": True}


def test_dataset_preflight_parser_selects_source_stage() -> None:
    args = registry.parser().parse_args(
        ["dataset", "preflight", "campaign", "--stage", "source"]
    )
    assert args.verb == "preflight"
    assert args.stage == "source"


def test_final_training_metrics_reads_native_summary() -> None:
    text = (
        "{'train_runtime': 37.7, 'train_samples_per_second': 0.026, "
        "'train_steps_per_second': 0.026, 'train_loss': 1.8579963445663452}"
    )
    assert registry.final_training_metrics(text) == {
        "step": None,
        "loss": 1.8579963445663452,
        "gradient_norm": None,
        "samples_per_second": 0.026,
        "cuda_memory_gib": None,
    }


def test_checkpoint_transfer_parser_defaults_to_cyclo_home_workspace() -> None:
    args = registry.parser().parse_args(
        ["checkpoint", "transfer", "candidate", "--target", "robotis@f2"]
    )
    assert args.robot_workspace == ("/home/robotis/cyclo_intelligence/docker/workspace")
