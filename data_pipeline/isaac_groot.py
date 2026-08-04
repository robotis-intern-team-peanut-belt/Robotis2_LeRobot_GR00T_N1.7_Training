"""Prepare and validate native Isaac-GR00T data from canonical LeRobot v3."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid
from typing import Any, Mapping


TRAINING_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = TRAINING_ROOT.parent
ISAAC_ROOT = WORKSPACE / "Isaac-GR00T"
CONVERTER_PYTHON = TRAINING_ROOT / ".venv-isaac-converter" / "bin" / "python"
SETUP = TRAINING_ROOT / "scripts" / "setup_isaac_converter.sh"
RUN_BACKEND = TRAINING_ROOT / "scripts" / "run_backend.sh"
CONVERT = TRAINING_ROOT / "data_pipeline" / "convert_lerobot_v3_to_isaac_v2.py"
VALIDATE = TRAINING_ROOT / "data_pipeline" / "validate_isaac_groot_dataset.py"
MODALITY_JSON = TRAINING_ROOT / "configs" / "modalities" / "f2_arms16_modality.json"
MODALITY_CONFIG = TRAINING_ROOT / "configs" / "modalities" / "f2_arms16_absolute_h40.py"
RECEIPT = "ISAAC_DERIVATION_RECEIPT.json"
VALIDATION = "meta/isaac_validation.json"


class IsaacDatasetError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_digest(root: Path) -> str:
    digest = hashlib.sha256()
    meta = root / "meta"
    for path in sorted(item for item in meta.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_sha(root: Path) -> str | None:
    done = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return done.stdout.strip() if done.returncode == 0 else None


def adapter_spec(data: Mapping[str, Any]) -> dict[str, str]:
    configured = data.get("adapters", {}).get("isaac_groot", {})
    root = configured.get(
        "root",
        TRAINING_ROOT / "artifacts" / "datasets" / f"{data['name']}_isaac_v21",
    )
    return {
        "root": str(Path(str(root)).expanduser().resolve()),
        "modality_config_path": str(
            Path(str(configured.get("modality_config_path", MODALITY_CONFIG)))
            .expanduser()
            .resolve()
        ),
        "modality_json_path": str(
            Path(str(configured.get("modality_json_path", MODALITY_JSON)))
            .expanduser()
            .resolve()
        ),
        "embodiment_tag": str(configured.get("embodiment_tag", "NEW_EMBODIMENT")),
    }


def expected_receipt(data: Mapping[str, Any], source: Path) -> dict[str, Any]:
    spec = adapter_spec(data)
    return {
        "schema_version": 1,
        "dataset": data["name"],
        "source_root": str(source.resolve()),
        "source_revision": str(data["revision"]),
        "source_metadata_sha256": metadata_digest(source),
        "converter_commit": git_sha(ISAAC_ROOT),
        "modality_json_sha256": sha256(Path(spec["modality_json_path"])),
        "modality_config_sha256": sha256(Path(spec["modality_config_path"])),
        "action_representation": "absolute",
        "action_horizon": 40,
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_native(root: Path, modality_config: Path) -> dict[str, Any]:
    output = root / VALIDATION
    subprocess.run(
        [
            str(RUN_BACKEND),
            "isaac_groot",
            "python",
            str(VALIDATE),
            "--dataset",
            str(root),
            "--modality-config",
            str(modality_config),
            "--output",
            str(output),
        ],
        check=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def prepare_isaac_dataset(
    data: Mapping[str, Any],
    *,
    bootstrap_converter: bool = True,
) -> dict[str, Any]:
    """Serialize conversion/promotion for one deterministic destination."""
    destination = Path(adapter_spec(data)["root"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.prepare.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _prepare_isaac_dataset_unlocked(
            data, bootstrap_converter=bootstrap_converter
        )


def _prepare_isaac_dataset_unlocked(
    data: Mapping[str, Any],
    *,
    bootstrap_converter: bool = True,
) -> dict[str, Any]:
    source = Path(str(data["prepared"]["root"])).expanduser().resolve()
    if not source.is_dir():
        raise IsaacDatasetError(f"canonical prepared dataset is missing: {source}")
    if (
        json.loads((source / "meta" / "info.json").read_text()).get("codebase_version")
        != "v3.0"
    ):
        raise IsaacDatasetError(f"canonical source is not LeRobot v3.0: {source}")

    spec = adapter_spec(data)
    destination = Path(spec["root"])
    modality_config = Path(spec["modality_config_path"])
    expected = expected_receipt(data, source)
    receipt_path = destination / RECEIPT
    if destination.exists():
        if not receipt_path.is_file():
            raise IsaacDatasetError(
                f"native destination exists without {RECEIPT}; move it aside: {destination}"
            )
        observed = json.loads(receipt_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if observed.get(key) != value:
                raise IsaacDatasetError(
                    f"native derivative receipt mismatch for {key}; use a new destination"
                )
        return {
            "reused": True,
            "root": str(destination),
            "validation": validate_native(destination, modality_config),
        }

    if not CONVERTER_PYTHON.is_file():
        if not bootstrap_converter:
            raise IsaacDatasetError(f"converter environment is missing; run: {SETUP}")
        subprocess.run([str(SETUP)], check=True)
    if not CONVERTER_PYTHON.is_file():
        raise IsaacDatasetError("converter setup completed without a Python executable")

    stage = destination.parent / f".{destination.name}.prepare-{uuid.uuid4().hex}"
    environment = os.environ.copy()
    environment["ISAAC_GROOT_ROOT"] = str(ISAAC_ROOT)
    try:
        subprocess.run(
            [
                str(CONVERTER_PYTHON),
                str(CONVERT),
                "--source",
                str(source),
                "--destination",
                str(stage),
                "--modality-json",
                spec["modality_json_path"],
            ],
            check=True,
            env=environment,
        )
        subprocess.run(
            [
                str(RUN_BACKEND),
                "isaac_groot",
                "python",
                str(ISAAC_ROOT / "gr00t/data/stats.py"),
                "--dataset-path",
                str(stage),
                "--embodiment-tag",
                spec["embodiment_tag"],
                "--modality-config-path",
                str(modality_config),
            ],
            check=True,
        )
        validation = validate_native(stage, modality_config)
        receipt = {
            **expected,
            "native_root": str(destination),
            "validation": validation,
        }
        write_json(stage / RECEIPT, receipt)
        os.replace(stage, destination)
        return {"reused": False, "root": str(destination), "validation": validation}
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
