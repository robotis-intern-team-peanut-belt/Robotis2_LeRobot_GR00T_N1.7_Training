"""Freeze, verify, and safely transfer trained policy packages."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import yaml

import backends


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
PACKAGES = ROOT / "artifacts" / "models"
RECEIPTS = ROOT / "artifacts" / "transfer_receipts"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
TARGET_RE = re.compile(r"^(?:[A-Za-z0-9._-]+@)?(?:[A-Za-z0-9._-]+|\[[0-9A-Fa-f:]+\])$")
DEFAULT_ROBOT_REPOSITORY = PurePosixPath("/home/robotis/cyclo_intelligence")
DEFAULT_ROBOT_WORKSPACE = DEFAULT_ROBOT_REPOSITORY / "docker" / "workspace"
ROBOT_STORAGE_WORKSPACE = PurePosixPath("/mnt/ssd/cyclo_intelligence/workspace")


class CheckpointError(RuntimeError):
    """A checkpoint package or transfer is unsafe or incomplete."""


def _now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _result_path(reference: str | Path) -> Path:
    direct = Path(reference)
    if direct.is_file():
        return direct.resolve()
    run = RUNS / str(reference)
    candidates = [run / "result.yaml", *sorted(run.glob("attempt-*/result.yaml"))]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise CheckpointError(f"completed result not found for {reference}")
    return existing[-1].resolve()


def _load_result(reference: str | Path) -> tuple[Path, dict[str, Any]]:
    path = _result_path(reference)
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if value.get("status", {}).get("state") != "completed":
        raise CheckpointError(f"run is not completed: {path}")
    return path, value


def _resolved_config(
    result_path: Path, result: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    configured = result.get("artifacts", {}).get("resolved_config")
    path = (
        Path(str(configured))
        if configured
        else result_path.parent / "resolved_config.yaml"
    )
    if not path.is_file():
        raise CheckpointError(f"resolved training config is missing: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return path.resolve(), value


def _model_path(result: Mapping[str, Any]) -> Path:
    value = result.get("artifacts", {}).get("pretrained_model")
    if not value:
        raise CheckpointError("result does not identify a pretrained model")
    path = Path(str(value)).expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise CheckpointError(f"model directory is incomplete: {path}")
    return path


def _manifest_text(root: Path) -> tuple[str, int, int]:
    lines: list[str] = []
    size = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "SHA256SUMS":
            continue
        lines.append(f"{_sha256(path)}  {relative}")
        size += path.stat().st_size
    return "\n".join(lines) + "\n", len(lines), size


def verify_package(reference: str | Path) -> dict[str, Any]:
    root = Path(reference)
    if not root.is_dir():
        candidate = PACKAGES / str(reference)
        root = candidate if candidate.is_dir() else root
    root = root.expanduser().resolve()
    manifest_path = root / "SHA256SUMS"
    package_manifest = root / "deployment_manifest.json"
    if not manifest_path.is_file() or not package_manifest.is_file():
        raise CheckpointError(f"package manifest is missing: {root}")
    checked = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise CheckpointError(f"invalid SHA256SUMS line: {line!r}") from exc
        path = root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise CheckpointError(f"checksum mismatch: {relative}")
        checked += 1
    metadata = json.loads(package_manifest.read_text(encoding="utf-8"))
    if metadata.get("package_name") != root.name:
        raise CheckpointError(
            "deployment manifest package name does not match directory"
        )
    return {
        "state": "verified",
        "package": str(root),
        "backend": metadata.get("backend"),
        "files": checked,
        "bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
    }


def package_checkpoint(
    reference: str | Path, name: str | None = None
) -> dict[str, Any]:
    result_path, result = _load_result(reference)
    config_path, config = _resolved_config(result_path, result)
    model = _model_path(result)
    package_name = name or str(result.get("run_name") or result_path.parent.name)
    if not NAME_RE.fullmatch(package_name):
        raise CheckpointError(f"unsafe package name {package_name!r}")
    destination = PACKAGES / package_name
    if destination.exists():
        raise CheckpointError(f"package already exists: {destination}")
    backend = backends.canonical_backend(config.get("backend"))
    required = list(config.get("artifacts", {}).get("required_checkpoint_files", []))
    missing = [relative for relative in required if not (model / relative).exists()]
    if missing:
        raise CheckpointError(
            "checkpoint is missing required files: " + ", ".join(missing)
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{package_name}.package-{os.getpid()}"
    if stage.exists():
        raise CheckpointError(f"temporary package path already exists: {stage}")
    try:
        shutil.copytree(model, stage, symlinks=False)
        package_metadata = {
            "schema_version": 1,
            "created_utc": _now(),
            "package_name": package_name,
            "backend": backend,
            "model_family": backends.model_family(backend),
            "source_run": result.get("run_name"),
            "source_result": str(result_path),
            "source_resolved_config": str(config_path),
            "source_model": str(model),
            "source_manifest": result.get("artifacts", {}).get("manifest"),
            "action_representation": "absolute",
            "load_path": ".",
            "consumer_status": "not_acknowledged",
        }
        _atomic_json(stage / "deployment_manifest.json", package_metadata)
        checksum_text, file_count, byte_count = _manifest_text(stage)
        (stage / "SHA256SUMS").write_text(checksum_text, encoding="utf-8")
        os.replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    verified = verify_package(destination)
    return {
        **verified,
        "created": True,
        "model_family": backends.model_family(backend),
        "model_files": file_count,
        "model_bytes": byte_count,
    }


def _ssh_base(target: str, port: int, identity_file: Path | None) -> list[str]:
    command = ["ssh", "-p", str(port)]
    if identity_file:
        command.extend(["-i", str(identity_file.expanduser().resolve())])
    command.extend(["--", target])
    return command


def _rsync_ssh(port: int, identity_file: Path | None) -> str:
    values = ["ssh", "-p", str(port)]
    if identity_file:
        values.extend(["-i", str(identity_file.expanduser().resolve())])
    return shlex.join(values)


def _workspace_alias_check(workspace: PurePosixPath) -> str:
    """Return a fail-closed remote check for the Cyclo home/SSD bind mount."""
    home_repository = shlex.quote(str(DEFAULT_ROBOT_REPOSITORY))
    home_workspace = shlex.quote(str(DEFAULT_ROBOT_WORKSPACE))
    storage_workspace = shlex.quote(str(ROBOT_STORAGE_WORKSPACE))
    selected_workspace = shlex.quote(str(workspace))
    return (
        "set -eu; "
        f"if ! mountpoint -q {home_repository}; then "
        f"echo 'Cyclo home repository is not an active bind mount: {home_repository}' >&2; "
        "exit 40; fi; "
        f"test -d {home_workspace}; test -d {storage_workspace}; "
        f"test -d {selected_workspace}; "
        f"home_id=$(stat -Lc '%d:%i' {home_workspace}); "
        f"storage_id=$(stat -Lc '%d:%i' {storage_workspace}); "
        f"selected_id=$(stat -Lc '%d:%i' {selected_workspace}); "
        'if [ "$home_id" != "$storage_id" ] || '
        '[ "$home_id" != "$selected_id" ]; then '
        "echo 'Cyclo workspace aliases do not identify the same directory' >&2; "
        "exit 41; fi; "
        "printf '%s\\n' \"$home_id\""
    )


def transfer_package(
    reference: str | Path,
    *,
    target: str,
    robot_workspace: str = str(DEFAULT_ROBOT_WORKSPACE),
    port: int = 22,
    identity_file: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    verified = verify_package(reference)
    package = Path(verified["package"])
    metadata = json.loads((package / "deployment_manifest.json").read_text())
    if not TARGET_RE.fullmatch(target):
        raise CheckpointError(f"unsafe SSH target {target!r}")
    if not 1 <= port <= 65535:
        raise CheckpointError("SSH port must be between 1 and 65535")
    if identity_file is not None and not identity_file.expanduser().is_file():
        raise CheckpointError(f"SSH identity file not found: {identity_file}")
    for executable in ("ssh", "rsync"):
        if shutil.which(executable) is None:
            raise CheckpointError(
                f"required transfer tool is unavailable: {executable}"
            )
    workspace = PurePosixPath(robot_workspace)
    if not workspace.is_absolute() or ".." in workspace.parts:
        raise CheckpointError("robot workspace must be an absolute path without '..'")
    family = str(metadata["model_family"])
    remote_parent = workspace / "model" / family
    remote_final = remote_parent / package.name
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    remote_stage = remote_parent / f".{package.name}.partial-{stamp}"
    container_path = PurePosixPath("/workspace/model") / family / package.name
    ssh = _ssh_base(target, port, identity_file)
    alias_check = _workspace_alias_check(workspace)
    preflight = (
        f"test -d {shlex.quote(str(remote_parent))} && "
        f"test ! -e {shlex.quote(str(remote_final))} && "
        f"test ! -e {shlex.quote(str(remote_stage))} && "
        f"mkdir {shlex.quote(str(remote_stage))}"
    )
    rsync = [
        "rsync",
        "-a",
        "--partial",
        "--checksum",
        "--protect-args",
        "-e",
        _rsync_ssh(port, identity_file),
        f"{package}/",
        f"{target}:{remote_stage}/",
    ]
    promote = (
        f"cd {shlex.quote(str(remote_stage))} && sha256sum -c SHA256SUMS && "
        f"cd {shlex.quote(str(remote_parent))} && "
        f"mv {shlex.quote(remote_stage.name)} {shlex.quote(package.name)}"
    )
    report = {
        "state": "dry_run" if dry_run else "transferred_checksum_verified",
        "package": str(package),
        "backend": metadata["backend"],
        "target": target,
        "robot_workspace": {
            "selected": str(workspace),
            "home_alias": str(DEFAULT_ROBOT_WORKSPACE),
            "ssd_storage": str(ROBOT_STORAGE_WORKSPACE),
            "state": "planned" if dry_run else "pending_verification",
            "device_inode": None,
        },
        "remote_host_path": str(remote_final),
        "remote_container_path": str(container_path),
        "commands": [
            shlex.join([*ssh, alias_check]),
            shlex.join([*ssh, preflight]),
            shlex.join(rsync),
            shlex.join([*ssh, promote]),
        ],
        "consumer_status": "not_acknowledged",
    }
    if dry_run:
        return report
    try:
        alias_result = subprocess.run(
            [*ssh, alias_check],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        report["state"] = "failed_workspace_alias_check"
        report["error"] = (exc.stderr or str(exc)).strip()
        receipt = RECEIPTS / f"{stamp}_{package.name}_transfer.json"
        _atomic_json(receipt, report)
        raise CheckpointError(
            "robot workspace bind-mount validation failed: " + report["error"]
        ) from exc
    identity = alias_result.stdout.strip()
    if not re.fullmatch(r"[0-9]+:[0-9]+", identity):
        report["state"] = "failed_workspace_alias_check"
        report["error"] = f"unexpected workspace identity output: {identity!r}"
        receipt = RECEIPTS / f"{stamp}_{package.name}_transfer.json"
        _atomic_json(receipt, report)
        raise CheckpointError(report["error"])
    report["robot_workspace"]["state"] = "verified_same_directory"
    report["robot_workspace"]["device_inode"] = identity
    subprocess.run([*ssh, preflight], check=True)
    try:
        subprocess.run(rsync, check=True)
        subprocess.run([*ssh, promote], check=True)
    except BaseException as exc:
        report["state"] = "failed_partial_staging_retained"
        report["error"] = str(exc)
        _atomic_json(RECEIPTS / f"{stamp}_{package.name}_transfer.json", report)
        raise
    receipt = RECEIPTS / f"{stamp}_{package.name}_transfer.json"
    _atomic_json(receipt, report)
    report["receipt"] = str(receipt)
    return report


def acknowledge_package(
    reference: str | Path, consumer_receipt: Path
) -> dict[str, Any]:
    """Record acknowledgment without mutating the already transferred package."""
    verified = verify_package(reference)
    receipt = consumer_receipt.expanduser().resolve()
    if not receipt.is_file():
        raise CheckpointError(f"consumer receipt not found: {receipt}")
    package = Path(verified["package"])
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = RECEIPTS / f"{stamp}_{package.name}_acknowledgment.json"
    value = {
        "schema_version": 1,
        "acknowledged_utc": _now(),
        "package": str(package),
        "package_sha256s_sha256": _sha256(package / "SHA256SUMS"),
        "consumer_receipt": str(receipt),
        "consumer_receipt_sha256": _sha256(receipt),
        "consumer_status": "acknowledged",
    }
    _atomic_json(record, value)
    return {**verified, **value, "acknowledgment_record": str(record)}
