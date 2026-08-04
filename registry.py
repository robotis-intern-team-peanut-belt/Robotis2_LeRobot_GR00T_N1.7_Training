#!/usr/bin/env python3
"""Workstation-native LeRobot GR00T experiment registry."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

import backends
import checkpoints

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
CONFIGS = ROOT / "configs" / "runs"
QUEUES = ROOT / "configs" / "queues"
DATASETS = ROOT / "configs" / "datasets"
F2_OMX_PROFILE = ROOT / "configs" / "contracts" / "f2_omx_insert.yaml"
RUNS = ROOT / "runs"
CLI = Path(os.environ.get("TRAINCTL_EXECUTABLE", ROOT / "grootctl")).resolve()
LEROBOT = (
    Path(os.environ.get("LEROBOT_ROOT", WORKSPACE / "lerobot")).expanduser().resolve()
)
ISAAC_GROOT = (
    Path(os.environ.get("ISAAC_GROOT_ROOT", WORKSPACE / "Isaac-GR00T"))
    .expanduser()
    .resolve()
)
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
VAR_RE = re.compile(r"\$\{([a-z_]+)\}")
OOM_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED", re.I
)
TRAIN_METRIC_RE = re.compile(
    r"\bstep:(?P<step>\d+).*?\bloss:(?P<loss>[-+0-9.eE]+).*?"
    r"\bgrdn:(?P<gradient>[-+0-9.eE]+).*?\bsmp/s:(?P<rate>[-+0-9.eE]+).*?"
    r"\bmem_gb:(?P<memory>[-+0-9.eE]+)"
)
TOP_KEYS = {
    "schema_version",
    "backend",
    "name",
    "description",
    "hypothesis",
    "tags",
    "dataset_ref",
    "dataset",
    "contract",
    "policy",
    "train",
    "optimizer",
    "scheduler",
    "tracking",
    "resources",
    "artifacts",
    "annotation",
    "adapters",
}
DATASET_FLAGS = {
    "repo_id",
    "root",
    "episodes",
    "image_transforms",
    "revision",
    "use_imagenet_stats",
    "video_backend",
    "return_uint8",
    "depth_output_unit",
    "streaming",
    "eval_split",
}
OMX_INSERT_CAMERAS = (
    "observation.images.rgb.cam_head",
    "observation.images.rgb.cam_left_wrist",
    "observation.images.rgb.cam_right_wrist",
)
OMX_INSERT_JOINTS = (
    *(f"arm_l_joint{i}" for i in range(1, 8)),
    "gripper_l_joint1",
    *(f"arm_r_joint{i}" for i in range(1, 8)),
    "gripper_r_joint1",
)


class RegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class Experiment:
    source: Path
    data: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.data["name"])


def now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RegistryError(f"{path}: cannot read YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise RegistryError(f"{path}: top level must be a mapping")
    return value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(path, yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_manifest(root: Path) -> tuple[str, str, int, int]:
    """Hash source payload files while excluding local cache/registry metadata."""
    lines: list[str] = []
    count = size = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if (
            relative.parts[0] == ".cache"
            or relative.name == ".groot_registry_download.json"
        ):
            continue
        length = path.stat().st_size
        lines.append(f"{sha256_file(path)}  {relative.as_posix()}")
        count += 1
        size += length
    text = "\n".join(lines) + ("\n" if lines else "")
    return text, hashlib.sha256(text.encode()).hexdigest(), count, size


def merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def layered(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise RegistryError(
            "circular _base chain: " + " -> ".join(map(str, (*seen, path)))
        )
    raw = read_yaml(path)
    base = raw.pop("_base", None)
    if base is None:
        return raw
    if not isinstance(base, str):
        raise RegistryError(f"{path}: _base must be a string")
    base_path = Path(base)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    if not base_path.is_file():
        raise RegistryError(f"{path}: missing _base {base}")
    return merge(layered(base_path, (*seen, path)), raw)


def apply_sets(data: dict[str, Any], values: Sequence[str]) -> dict[str, Any]:
    result = copy.deepcopy(data)
    for item in values:
        if "=" not in item:
            raise RegistryError(f"--set expects dotted.key=value: {item!r}")
        dotted, raw = item.split("=", 1)
        parts = dotted.split(".")
        if not all(parts):
            raise RegistryError(f"invalid dotted key: {dotted!r}")
        node = result
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            if not isinstance(node[part], dict):
                raise RegistryError(f"{dotted}: {part!r} is not a mapping")
            node = node[part]
        node[parts[-1]] = yaml.safe_load(raw)
    return result


def substitute(value: Any, variables: Mapping[str, str]) -> Any:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in variables:
                raise RegistryError(f"unknown variable ${{{key}}}")
            return variables[key]

        return VAR_RE.sub(replace, value)
    if isinstance(value, list):
        return [substitute(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: substitute(item, variables) for key, item in value.items()}
    return value


def find(reference: str | Path, root: Path, relative: Path | None = None) -> Path:
    ref = Path(reference)
    candidates = (
        [ref]
        if ref.is_absolute()
        else [
            *(([relative / ref] if relative else [])),
            Path.cwd() / ref,
            Path(__file__).resolve().parent / ref,
            root / ref,
            *(([root / f"{ref}.yaml", root / f"{ref}.yml"] if not ref.suffix else [])),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RegistryError(f"file not found: {reference}")


def info(root: Path) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(
            f"invalid or missing dataset metadata {path}: {exc}"
        ) from exc
    return value


def resolved_contract(data: Mapping[str, Any]) -> dict[str, Any]:
    """Expand an opt-in metadata-derived contract from the prepared LeRobot tree."""
    contract = copy.deepcopy(dict(data["contract"]))
    auto = bool(contract.pop("auto_from_metadata", False))
    if not auto:
        return contract
    root = Path(str(data["prepared"]["root"]))
    metadata = info(root)
    features = metadata.get("features", {})
    state = features.get("observation.state", {})
    action = features.get("action", {})
    cameras = [
        {"key": key, "shape": value.get("shape")}
        for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") in {"video", "image"}
    ]
    if not cameras or not state.get("names") or not action.get("names"):
        raise RegistryError(
            f"{root}: metadata cannot derive camera/state/action contract"
        )
    derived = {
        "fps": metadata.get("fps"),
        "cameras": cameras,
        "state": {"dim": state.get("shape", [None])[0], "names": state.get("names")},
        "action": {"dim": action.get("shape", [None])[0], "names": action.get("names")},
    }
    return merge(derived, contract)


def validate_contract(
    config: Mapping[str, Any], metadata: Mapping[str, Any], source: Path
) -> None:
    contract = config["contract"]
    features = metadata.get("features", {})
    for section, key in (("state", "observation.state"), ("action", "action")):
        expected = contract.get(section, {})
        observed = features.get(key, {})
        if observed.get("shape") != [expected.get("dim")]:
            raise RegistryError(f"{source}: {key} dimension mismatch")
        if observed.get("names") != expected.get("names"):
            raise RegistryError(f"{source}: {key} name/order mismatch")
    expected_keys = []
    for camera in contract.get("cameras", []):
        key = camera["key"]
        expected_keys.append(key)
        if features.get(key, {}).get("shape") != camera.get("shape"):
            raise RegistryError(f"{source}: camera contract mismatch for {key}")
    observed_keys = [
        key
        for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") in {"video", "image"}
    ]
    if observed_keys != expected_keys:
        raise RegistryError(f"{source}: camera key/order mismatch")
    if metadata.get("fps") != contract.get("fps"):
        raise RegistryError(f"{source}: FPS mismatch")


def validate(
    data: Mapping[str, Any], source: Path, require_dataset: bool, launch: bool
) -> None:
    unknown = set(data) - TOP_KEYS
    if unknown:
        raise RegistryError(
            f"{source}: unknown top-level keys: {', '.join(sorted(unknown))}"
        )
    if data.get("schema_version") != 1:
        raise RegistryError(f"{source}: requires schema_version: 1")
    try:
        backend = backends.canonical_backend(str(data.get("backend")))
    except ValueError as exc:
        raise RegistryError(f"{source}: {exc}") from exc
    if not NAME_RE.fullmatch(str(data.get("name", ""))):
        raise RegistryError(f"{source}: unsafe name {data.get('name')!r}")
    tags = data.get("tags")
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise RegistryError(
            f"{source}: tags must be a list of strings; quote date-like YAML tags"
        )
    for key in (
        "dataset",
        "contract",
        "policy",
        "train",
        "optimizer",
        "scheduler",
        "tracking",
        "resources",
        "artifacts",
        "annotation",
    ):
        if not isinstance(data.get(key), dict):
            raise RegistryError(f"{source}: {key} must be a mapping")
    dataset, policy, train = data["dataset"], data["policy"], data["train"]
    if not dataset.get("repo_id") or not dataset.get("root"):
        raise RegistryError(f"{source}: dataset.repo_id and dataset.root are required")
    expected_type = "act" if backend == "lerobot_act" else "groot"
    if policy.get("type") != expected_type or policy.get("device") not in {
        "cuda",
        "cpu",
        "mps",
    }:
        raise RegistryError(f"{source}: invalid {backend} policy type/device")
    if backend == "lerobot_groot" and policy.get("max_steps") != train.get("steps"):
        raise RegistryError(f"{source}: policy.max_steps must equal train.steps")
    if "optimizer_lr" in policy and not math.isclose(
        float(data["optimizer"]["lr"]), float(policy["optimizer_lr"])
    ):
        raise RegistryError(f"{source}: optimizer.lr must equal policy.optimizer_lr")
    for key in ("batch_size", "steps"):
        if (
            not isinstance(train.get(key), int)
            or isinstance(train.get(key), bool)
            or train[key] <= 0
        ):
            raise RegistryError(f"{source}: train.{key} must be a positive integer")
    workers = train.get("num_workers")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
        raise RegistryError(
            f"{source}: train.num_workers must be a non-negative integer"
        )
    if backend == "isaac_groot":
        if policy.get("action_representation") != "absolute":
            raise RegistryError(
                f"{source}: native Isaac policy must declare absolute actions"
            )
        if not dataset.get("modality_config_path"):
            raise RegistryError(
                f"{source}: native Isaac dataset needs modality_config_path"
            )
        if policy.get("chunk_size") != 40:
            raise RegistryError(
                f"{source}: native Isaac modality contract requires horizon 40"
            )
        if train.get("seed", 42) != 42:
            raise RegistryError(
                f"{source}: unmodified native Isaac launcher currently requires seed 42"
            )
    relative = policy.get("use_relative_actions")
    excluded = policy.get("relative_exclude_joints")
    if relative and (not isinstance(excluded, list) or not excluded):
        raise RegistryError(f"{source}: relative actions require exclusions")
    if not relative and excluded:
        raise RegistryError(
            f"{source}: exclusions set while relative actions are disabled"
        )
    names = data["contract"].get("action", {}).get("names", [])
    for term in excluded or []:
        if not any(str(term).lower() in name.lower() for name in names):
            raise RegistryError(f"{source}: exclusion {term!r} matches no action")
    tracking = data["tracking"]
    if tracking.get("backend") not in {"wandb", "none"}:
        raise RegistryError(f"{source}: tracking.backend must be wandb or none")
    if tracking.get("backend") == "wandb" and not tracking.get("project"):
        raise RegistryError(f"{source}: W&B project is required")
    if tracking.get("mode") not in {None, "online", "offline", "disabled"}:
        raise RegistryError(f"{source}: invalid W&B mode")
    if float(data["resources"].get("timeout_hours", 0)) <= 0:
        raise RegistryError(f"{source}: timeout_hours must be positive")
    memory_guard = data["resources"].get("memory_guard_gib")
    if memory_guard is not None and float(memory_guard) <= 0:
        raise RegistryError(f"{source}: memory_guard_gib must be positive")
    if require_dataset:
        dataset_root = Path(str(dataset["root"]))
        validate_contract(data, info(dataset_root), source)
        if backend == "isaac_groot":
            receipt = dataset_root / "ISAAC_DERIVATION_RECEIPT.json"
            if not receipt.is_file():
                raise RegistryError(
                    f"{source}: native Isaac derivative is not prepared; run "
                    f"trainctl dataset prepare-isaac {dataset.get('registry_ref')}"
                )
        else:
            preflight_dataset_root(dataset_root)


def dataset_defaults(reference: str, backend: str | None = None) -> dict[str, Any]:
    """Resolve one registered dataset into backend-specific fields plus its contract."""
    spec = dataset_spec(reference)
    training = spec.get("training", {})
    selected = backends.canonical_backend(backend or "lerobot_groot")
    if selected == "isaac_groot":
        from data_pipeline.isaac_groot import adapter_spec

        adapter = adapter_spec(spec)
        return {
            "dataset": {
                "registry_ref": spec["name"],
                "source_revision": spec["revision"],
                "repo_id": f"local/{spec['name']}_isaac_v21",
                "root": adapter["root"],
                "modality_config_path": adapter["modality_config_path"],
                "embodiment_tag": adapter["embodiment_tag"],
                "eval_split": 0.0,
            },
            "contract": resolved_contract(spec),
        }
    return {
        "dataset": {
            "registry_ref": spec["name"],
            "source_revision": spec["revision"],
            "repo_id": training.get("repo_id", spec["repo_id"]),
            "root": spec["prepared"]["root"],
            "video_backend": training.get("video_backend", "torchcodec"),
            "eval_split": training.get("eval_split", 0.0),
            "image_transforms": {"enable": False},
        },
        "contract": resolved_contract(spec),
    }


def experiment(
    reference: str | Path,
    sets: Sequence[str] = (),
    relative: Path | None = None,
    require_dataset: bool = True,
    launch: bool = False,
) -> Experiment:
    path = find(reference, CONFIGS, relative)
    data = apply_sets(layered(path), sets)
    dataset_reference = data.pop("dataset_ref", None)
    if dataset_reference is not None:
        if not isinstance(dataset_reference, str) or not dataset_reference:
            raise RegistryError(f"{path}: dataset_ref must be a non-empty string")
        data = merge(dataset_defaults(dataset_reference, data.get("backend")), data)
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise RegistryError(f"{path}: name is required")
    data = substitute(
        data,
        {
            "workspace": str(WORKSPACE),
            "training_root": str(ROOT),
            "name": name,
        },
    )
    validate(data, path, require_dataset, launch)
    return Experiment(path, data)


def render(value: Any) -> str:
    return backends.render(value)


def flags(
    prefix: str, values: Mapping[str, Any], skip: set[str] | None = None
) -> list[str]:
    return backends.dotted_flags(prefix, values, skip)


def output_dir(data: Mapping[str, Any], run_dir: Path | None = None) -> Path:
    explicit = data["artifacts"].get("output_dir")
    return (
        Path(str(explicit))
        if explicit
        else (run_dir or RUNS / str(data["name"])) / "output"
    )


def command(
    data: Mapping[str, Any], out: Path | None = None, resume_config: Path | None = None
) -> list[str]:
    try:
        return backends.build_command(data, out or output_dir(data), resume_config)
    except ValueError as exc:
        raise RegistryError(str(exc)) from exc


def fingerprint_dataset(root: Path) -> dict[str, Any]:
    metadata, inventory = hashlib.sha256(), hashlib.sha256()
    count = size = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative, length = path.relative_to(root).as_posix(), path.stat().st_size
        inventory.update(f"{relative}\0{length}\n".encode())
        count, size = count + 1, size + length
        if relative.startswith("meta/"):
            metadata.update(relative.encode() + b"\0" + path.read_bytes())
    return {
        "root": str(root),
        "metadata_sha256": metadata.hexdigest(),
        "inventory_sha256": inventory.hexdigest(),
        "file_count": count,
        "bytes": size,
    }


def git_identity(path: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        done = subprocess.run(
            ["git", "-C", str(path), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return done.stdout.strip() if done.returncode == 0 else None

    sha = git("rev-parse", "HEAD")
    return {
        "root": str(path),
        "sha": sha,
        "dirty": bool(git("status", "--porcelain")) if sha else None,
    }


def manifest(item: Experiment, argv: Sequence[str], run_dir: Path) -> dict[str, Any]:
    blob = json.dumps(item.data, sort_keys=True, separators=(",", ":")).encode()
    backend = backends.canonical_backend(item.data.get("backend"))
    code_root = ISAAC_GROOT if backend == "isaac_groot" else LEROBOT
    return {
        "schema_version": 1,
        "created_utc": now(),
        "run_name": item.name,
        "backend": backend,
        "source_config": str(item.source),
        "config_sha256": hashlib.sha256(blob).hexdigest(),
        "command_sha256": hashlib.sha256(
            (shlex.join(argv) + "\n").encode()
        ).hexdigest(),
        "dataset": fingerprint_dataset(Path(str(item.data["dataset"]["root"]))),
        "code": {
            backend: git_identity(code_root),
            "training": git_identity(ROOT),
            "registry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "paths": {
            "run_dir": str(run_dir),
            "output_dir": str(output_dir(item.data, run_dir)),
        },
        "safety": {
            "robot_hardware_invoked": False,
            "allowed_backend": backend,
        },
    }


def blank_result(name: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_name": name,
        "status": {"state": "not_started", "exit_code": None, "failure_category": None},
        "timing": {"started_utc": None, "finished_utc": None, "runtime_seconds": None},
        "wandb": {"entity": None, "project": None, "run_id": None, "url": None},
        "metrics": {
            "best_eval_loss": None,
            "best_eval_step": None,
            "final_eval_loss": None,
            "final_train_loss": None,
            "samples_per_second": None,
            "cuda_memory_gib": None,
        },
        "best_checkpoint": {"step": None, "path": None, "selection_reason": None},
        "artifacts": {
            "resolved_config": None,
            "manifest": None,
            "train_log": None,
            "pretrained_model": None,
        },
        "summary": None,
        "interpretation": None,
        "anomalies": [],
        "next_action": None,
        "evaluation": {
            "offline": {"status": "not_evaluated", "notes": None},
            "shadow": {"status": "not_evaluated", "notes": None},
            "guarded_hardware": {"status": "not_evaluated", "notes": None},
        },
    }


def stream(
    argv: Sequence[str], log: Path, timeout: float, append: bool
) -> tuple[int, bool]:
    with log.open("a" if append else "w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline, timed_out = time.monotonic() + timeout, False
        while process.poll() is None:
            if time.monotonic() >= deadline:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            for key, _ in selector.select(1):
                line = key.fileobj.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    handle.write(line)
                    handle.flush()
        tail = process.stdout.read()
        if tail:
            sys.stdout.write(tail)
            handle.write(tail)
        selector.close()
        code = process.wait()
    return (code if code >= 0 else 128 - code), timed_out


def final_training_metrics(text: str) -> dict[str, float | int | None] | None:
    matches = list(TRAIN_METRIC_RE.finditer(text))
    if matches:
        values = matches[-1].groupdict()
        return {
            "step": int(values["step"]),
            "loss": float(values["loss"]),
            "gradient_norm": float(values["gradient"]),
            "samples_per_second": float(values["rate"]),
            "cuda_memory_gib": float(values["memory"]),
        }
    native_losses = re.findall(r"'train_loss':\s*([-+0-9.eE]+)", text)
    if not native_losses:
        return None
    native_rates = re.findall(r"'train_samples_per_second':\s*([-+0-9.eE]+)", text)
    return {
        "step": None,
        "loss": float(native_losses[-1]),
        "gradient_norm": None,
        "samples_per_second": float(native_rates[-1]) if native_rates else None,
        "cuda_memory_gib": None,
    }


def validate_gpu_resources(
    data: Mapping[str, Any], torch_module: Any | None = None
) -> dict[str, Any] | None:
    """Require enough visible CUDA memory in the selected backend environment."""
    required = data["resources"].get("memory_guard_gib")
    if required is None:
        return None
    if torch_module is None:
        probe = (
            "import json, torch; "
            "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
            "d=torch.cuda.current_device(); p=torch.cuda.get_device_properties(d); "
            "print(json.dumps({'device': torch.cuda.get_device_name(d), "
            "'total_memory_gib': round(p.total_memory/1024**3, 3)}))"
        )
        backend = backends.canonical_backend(data.get("backend"))
        selected = "isaac_groot" if backend == "isaac_groot" else "lerobot"
        done = subprocess.run(
            [str(ROOT / "scripts/run_backend.sh"), selected, "python", "-c", probe],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if done.returncode != 0:
            detail = (done.stderr or done.stdout).strip()
            raise RegistryError(f"GPU memory guard failed in {backend}: {detail}")
        try:
            observed = json.loads(done.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise RegistryError(
                f"GPU memory guard returned invalid output: {done.stdout}"
            ) from exc
        total = float(observed["total_memory_gib"])
        device_name = str(observed["device"])
    else:
        if not torch_module.cuda.is_available():
            raise RegistryError(
                "CUDA is unavailable; refusing non-dry GPU training launch"
            )
        device = torch_module.cuda.current_device()
        total = (
            float(torch_module.cuda.get_device_properties(device).total_memory)
            / 1024**3
        )
        device_name = torch_module.cuda.get_device_name(device)
    required = float(required)
    if total < required:
        raise RegistryError(
            f"GPU has {total:.1f} GiB; memory guard requires {required:.1f} GiB"
        )
    return {"device": device_name, "total_memory_gib": round(total, 3)}


def execute(
    item: Experiment, dry_run: bool, resume: bool = False, run_dir: Path | None = None
) -> dict[str, Any]:
    validate(item.data, item.source, True, True)
    run_dir = run_dir or RUNS / item.name
    out = output_dir(item.data, run_dir)
    free = (
        shutil.disk_usage(
            run_dir.parent if run_dir.parent.exists() else RUNS.parent
        ).free
        / 1024**3
    )
    required = float(item.data["resources"].get("min_free_disk_gib", 0))
    if free < required:
        raise RegistryError(f"{free:.1f} GiB free; {required:.1f} GiB required")
    backend = backends.canonical_backend(item.data.get("backend"))
    resume_config = None
    if resume:
        if backend == "isaac_groot":
            raise RegistryError("native Isaac resume is not yet supported by trainctl")
        resume_config = out / "checkpoints/last/pretrained_model/train_config.json"
        if not resume_config.is_file():
            raise RegistryError(f"resume config missing: {resume_config}")
    elif out.exists():
        raise RegistryError(f"existing output directory: {out}")
    argv = command(item.data, out, resume_config)
    if dry_run:
        return {
            "run_name": item.name,
            "run_dir": str(run_dir),
            "output_dir": str(out),
            "command": argv,
        }
    validate_gpu_resources(item.data)
    executable = Path(argv[0])
    if not executable.is_file() and shutil.which(argv[0]) is None:
        raise RegistryError(f"backend launcher is unavailable: {argv[0]}")
    if run_dir.exists() and not resume:
        raise RegistryError(f"existing run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=resume)
    if not resume:
        atomic_text(
            run_dir / "source_config.yaml", item.source.read_text(encoding="utf-8")
        )
        atomic_yaml(run_dir / "resolved_config.yaml", item.data)
        atomic_text(run_dir / "command.txt", shlex.join(argv) + "\n")
        atomic_json(run_dir / "manifest.json", manifest(item, argv, run_dir))
    result = blank_result(item.name)
    result["status"]["state"] = "running"
    result["timing"]["started_utc"] = now()
    result["artifacts"].update(
        {
            "resolved_config": str(run_dir / "resolved_config.yaml"),
            "manifest": str(run_dir / "manifest.json"),
            "train_log": str(run_dir / "train.log"),
        }
    )
    atomic_yaml(run_dir / "result.yaml", result)
    started = time.monotonic()
    code, timed_out = stream(
        argv,
        run_dir / "train.log",
        float(item.data["resources"]["timeout_hours"]) * 3600,
        resume,
    )
    text = (run_dir / "train.log").read_text(errors="replace")[-2_000_000:]
    category = (
        None
        if code == 0
        else (
            "timeout"
            if timed_out
            else (
                "oom"
                if OOM_RE.search(text)
                else "signal" if code >= 128 else "trainer_error"
            )
        )
    )
    result["status"] = {
        "state": "completed" if code == 0 else "failed",
        "exit_code": code,
        "failure_category": category,
    }
    result["timing"].update(
        {"finished_utc": now(), "runtime_seconds": round(time.monotonic() - started, 3)}
    )
    observed = final_training_metrics(text)
    if observed:
        result["metrics"].update(
            {
                "final_train_loss": observed["loss"],
                "samples_per_second": observed["samples_per_second"],
                "cuda_memory_gib": observed["cuda_memory_gib"],
            }
        )
    backend = backends.canonical_backend(item.data.get("backend"))
    if backend == "isaac_groot":
        native_root = out / item.name
        checkpoints_found = sorted(
            (path for path in native_root.glob("checkpoint-*") if path.is_dir()),
            key=lambda path: int(path.name.removeprefix("checkpoint-")),
        )
        model = (
            native_root
            if (native_root / "config.json").is_file()
            else (checkpoints_found[-1] if checkpoints_found else native_root)
        )
    else:
        model = out / "checkpoints/last/pretrained_model"
    if model.exists() and (model / "config.json").is_file():
        result["artifacts"]["pretrained_model"] = str(model.resolve())
        result["best_checkpoint"] = {
            "step": observed["step"] if observed else None,
            "path": str(model.resolve()),
            "selection_reason": "Last saved endpoint; not quality-selected.",
        }
    if code == 0:
        result["summary"] = (
            "Trainer exited successfully. Review evaluation before selecting a checkpoint."
        )
    atomic_yaml(run_dir / "result.yaml", result)
    return result


def system_tmux_environment() -> dict[str, str]:
    """Keep Conda's FFmpeg libraries from overriding host tmux/ncurses."""
    environment = os.environ.copy()
    prefixes = [
        Path(value).resolve()
        for key, value in environment.items()
        if key.startswith("CONDA_PREFIX") and value
    ]
    entries = environment.get("LD_LIBRARY_PATH", "").split(":")
    kept = []
    for entry in entries:
        if not entry:
            continue
        resolved = Path(entry).resolve()
        if any(resolved == prefix / "lib" for prefix in prefixes):
            continue
        kept.append(entry)
    if kept:
        environment["LD_LIBRARY_PATH"] = ":".join(kept)
    else:
        environment.pop("LD_LIBRARY_PATH", None)
    return environment


def tmux_has_session(name: str) -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", f"={name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=system_tmux_environment(),
            check=False,
        ).returncode
        == 0
    )


def launch_tmux(
    item: Experiment,
    sets: Sequence[str] = (),
    resume: bool = False,
    run_dir: Path | None = None,
    session: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if shutil.which("tmux") is None:
        raise RegistryError(
            "tmux is unavailable; use trainctl run in a persistent terminal"
        )
    session = session or f"train-{item.name}"
    if not NAME_RE.fullmatch(session):
        raise RegistryError(f"unsafe tmux session name {session!r}")
    if tmux_has_session(session):
        raise RegistryError(
            f"tmux session already exists: {session}; attach with: tmux attach -t ={session}"
        )

    effective_run_dir = run_dir or RUNS / item.name
    if not effective_run_dir.is_absolute():
        effective_run_dir = (WORKSPACE / effective_run_dir).resolve()
    plan = execute(item, True, resume, effective_run_dir)
    launch_log = RUNS / "_launch_logs" / f"{item.name}.tmux.log"
    train_log = effective_run_dir / "train.log"
    if effective_run_dir == RUNS / item.name:
        status_command = "{} result show {}".format(CLI, item.name)
    else:
        status_command = "sed -n 1,220p {}".format(
            shlex.quote(str(effective_run_dir / "result.yaml"))
        )

    foreground = "exec" if CLI.name == "trainctl" else "run"
    child = [str(CLI), foreground, str(item.source)]
    for value in sets:
        child.extend(["--set", value])
    if resume:
        child.append("--resume")
    child.extend(["--run-dir", str(effective_run_dir)])
    tee = ["tee", "-a", str(launch_log)] if resume else ["tee", str(launch_log)]
    pipeline = f"set -o pipefail; {shlex.join(child)} 2>&1 | {shlex.join(tee)}"
    shell_command = shlex.join(["/bin/bash", "-lc", pipeline])

    report = {
        "state": "dry_run" if dry_run else "launched",
        "run_name": item.name,
        "tmux_session": session,
        "run_dir": str(effective_run_dir),
        "output_dir": plan["output_dir"],
        "launch_log": str(launch_log),
        "train_log": str(train_log),
        "commands": {
            "trainer": shlex.join(plan["command"]),
            "attach": f"tmux attach -t ={session}",
            "status": status_command,
            "train_log": f"tail -f {shlex.quote(str(train_log))}",
            "launch_log": f"tail -f {shlex.quote(str(launch_log))}",
            "list_sessions": "tmux list-sessions",
            "stop": f"tmux kill-session -t ={session}",
        },
    }
    if dry_run:
        report["preflight"] = (
            "config, dataset, and disk checks passed; GPU launch check skipped"
        )
        report["tmux_command"] = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session,
            "-c",
            str(WORKSPACE),
            shell_command,
        ]
        return report

    gpu = validate_gpu_resources(item.data)
    report["preflight"] = (
        "config, dataset, disk, and GPU checks passed ({}, {:.1f} GiB visible)".format(
            gpu["device"], gpu["total_memory_gib"]
        )
    )
    launch_log.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                session,
                "-c",
                str(WORKSPACE),
                shell_command,
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=system_tmux_environment(),
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RegistryError(f"tmux launch failed: {detail or exc}") from exc
    time.sleep(0.5)
    if not tmux_has_session(session):
        detail = (
            launch_log.read_text(errors="replace")[-4000:]
            if launch_log.is_file()
            else ""
        )
        suffix = f"\nStartup output:\n{detail}" if detail else ""
        raise RegistryError(f"tmux session exited during startup: {session}{suffix}")
    return report


def print_tmux_report(report: Mapping[str, Any]) -> None:
    commands = report["commands"]
    dry = report["state"] == "dry_run"
    print(
        "DRY RUN: no tmux session was started."
        if dry
        else "STARTED: training is running in a detached tmux session."
    )
    if dry:
        print("A real launch will continue if its terminal or SSH connection closes.")
    else:
        print("Training continues if this terminal or SSH connection closes.")
    print("Preflight: {}".format(report["preflight"]))
    print()
    print("Run:           {}".format(report["run_name"]))
    print("tmux session:  {}".format(report["tmux_session"]))
    print("Run directory: {}".format(report["run_dir"]))
    print("Startup log:   {}".format(report["launch_log"]))
    print()
    print("Resolved trainer command:")
    print("  {}".format(commands["trainer"]))
    print()
    print("Watch the live terminal:")
    print("  {}".format(commands["attach"]))
    print("  Detach without stopping training: Ctrl-b, then d")
    print()
    print("Check progress without attaching:")
    print("  {}".format(commands["train_log"]))
    print("  {}".format(commands["status"]))
    print(
        "  {}  # includes failures before train.log exists".format(
            commands["launch_log"]
        )
    )
    print()
    print("List training sessions:")
    print("  {}".format(commands["list_sessions"]))
    print()
    print("STOP TRAINING (only if intended):")
    print("  {}".format(commands["stop"]))


def resolve_queue(
    reference: str | Path,
) -> tuple[Path, dict[str, Any], list[Experiment]]:
    path = find(reference, QUEUES)
    data = layered(path)
    allowed = {
        "schema_version",
        "name",
        "description",
        "continue_on_failure",
        "min_free_disk_gib",
        "max_attempts",
        "time_budget_hours",
        "runs",
    }
    if set(data) - allowed or data.get("schema_version") != 1:
        raise RegistryError(f"{path}: invalid queue schema")
    if not NAME_RE.fullmatch(str(data.get("name", ""))):
        raise RegistryError(f"{path}: unsafe queue name")
    if not isinstance(data.get("runs"), list) or not data["runs"]:
        raise RegistryError(f"{path}: runs must be non-empty")
    if (
        int(data.get("max_attempts", 0)) <= 0
        or float(data.get("time_budget_hours", 0)) <= 0
    ):
        raise RegistryError(f"{path}: max_attempts/time_budget_hours must be positive")
    items = []
    for index, entry in enumerate(data["runs"]):
        if not isinstance(entry, dict) or not isinstance(entry.get("config"), str):
            raise RegistryError(f"{path}: runs[{index}] needs config")
        sets = entry.get("overrides", [])
        if not isinstance(sets, list) or not all(
            isinstance(value, str) for value in sets
        ):
            raise RegistryError(f"{path}: overrides must be key=value strings")
        items.append(experiment(entry["config"], sets, path.parent, True, True))
    names = [item.name for item in items]
    duplicate = sorted({name for name in names if names.count(name) > 1})
    if duplicate:
        raise RegistryError(f"{path}: duplicate names: {', '.join(duplicate)}")
    return path, data, items


def run_queue(reference: str | Path, dry_run: bool) -> dict[str, Any]:
    path, queue, items = resolve_queue(reference)
    if dry_run:
        return {
            "queue": str(path),
            "runs": [
                execute(item, True, run_dir=RUNS / item.name / "attempt-1")
                for item in items
            ],
        }
    state_dir = RUNS / "_queues" / queue["name"]
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / "queue.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RegistryError(f"queue lock held: {state_dir / 'queue.lock'}") from exc
    status = {
        "schema_version": 1,
        "queue": queue["name"],
        "source": str(path),
        "state": "running",
        "started_utc": now(),
        "finished_utc": None,
        "runs": [],
    }
    status_path, started = state_dir / "queue_status.yaml", time.monotonic()
    atomic_yaml(status_path, status)
    try:
        for item in items:
            if time.monotonic() - started >= float(queue["time_budget_hours"]) * 3600:
                status["state"] = "blocked_time_budget"
                break
            free = shutil.disk_usage(RUNS).free / 1024**3
            if free < float(queue["min_free_disk_gib"]):
                status["state"] = "blocked_low_disk"
                break
            succeeded = False
            for attempt in range(1, int(queue["max_attempts"]) + 1):
                event = {
                    "run_name": item.name,
                    "attempt": attempt,
                    "state": "running",
                    "started_utc": now(),
                }
                status["runs"].append(event)
                atomic_yaml(status_path, status)
                try:
                    result = execute(
                        item, False, run_dir=RUNS / item.name / f"attempt-{attempt}"
                    )
                    event.update(result["status"])
                except RegistryError as exc:
                    event.update(
                        {
                            "state": "blocked",
                            "failure_category": "registry_guard",
                            "error": str(exc),
                        }
                    )
                event["finished_utc"] = now()
                atomic_yaml(status_path, status)
                if event["state"] == "completed":
                    succeeded = True
                    break
            if not succeeded and not queue.get("continue_on_failure"):
                status["state"] = "failed"
                break
        else:
            status["state"] = "completed"
    finally:
        status["finished_utc"] = now()
        atomic_yaml(status_path, status)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return status


def dataset_spec(reference: str | Path) -> dict[str, Any]:
    path = find(reference, DATASETS)
    data = substitute(
        read_yaml(path), {"workspace": str(WORKSPACE), "training_root": str(ROOT)}
    )
    for key in ("name", "repo_id", "revision", "download", "prepared", "contract"):
        if key not in data:
            raise RegistryError(f"{path}: missing {key}")
    if data.get("schema_version") != 1:
        raise RegistryError(f"{path}: schema must be 1")
    revision = str(data["revision"])
    source_kind = data.get("registration", {}).get("source_kind")
    if source_kind == "direct_local_directory":
        if not revision:
            raise RegistryError(f"{path}: local source label must be non-empty")
    elif not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RegistryError(f"{path}: revision must be a pinned SHA")
    return data


def validate_dataset(data: Mapping[str, Any], decode: bool) -> dict[str, Any]:
    root = Path(str(data["prepared"]["root"]))
    metadata = info(root)
    expected = data.get("expected", {})
    for key in ("codebase_version", "total_episodes", "total_frames"):
        if key in expected and metadata.get(key) != expected[key]:
            raise RegistryError(f"{root}: {key} mismatch")
    contract = resolved_contract(data)
    validate_contract({"contract": contract}, metadata, root)
    preflight = preflight_dataset_root(root)
    decoded = None
    if decode:
        with tempfile.TemporaryDirectory(
            prefix="trainctl-lerobot-validate-"
        ) as temporary:
            output = Path(temporary) / "decoded.json"
            subprocess.run(
                [
                    str(ROOT / "scripts/run_backend.sh"),
                    "lerobot",
                    "python",
                    str(ROOT / "data_pipeline/validate_lerobot_dataset.py"),
                    "--dataset",
                    str(root),
                    "--repo-id",
                    str(data.get("training", {}).get("repo_id", data["repo_id"])),
                    "--video-backend",
                    str(data.get("training", {}).get("video_backend", "torchcodec")),
                    "--output",
                    str(output),
                ],
                check=True,
            )
            decoded = json.loads(output.read_text(encoding="utf-8"))
        if (
            decoded["state_shape"][-1] != contract["state"]["dim"]
            or decoded["action_shape"][-1] != contract["action"]["dim"]
        ):
            raise RegistryError("decoded state/action dimension mismatch")
        if not decoded["finite"]:
            raise RegistryError("decoded state/action contains NaN/Inf")
        expected_images = {
            camera["key"]: camera["shape"] for camera in contract.get("cameras", [])
        }
        if decoded.get("images") != expected_images:
            raise RegistryError("decoded image keys/shapes do not match the contract")
    return {
        "root": str(root),
        "metadata": {
            key: metadata.get(key)
            for key in ("codebase_version", "total_episodes", "total_frames", "fps")
        },
        "preflight": preflight,
        "fingerprint": fingerprint_dataset(root),
        "decoded": decoded,
    }


def preflight_dataset_root(root: Path) -> dict[str, Any]:
    """Run the lightweight caption and episode-index guard used before training."""
    try:
        from data_pipeline.preflight_lerobot_dataset import DatasetPreflightError, audit
    except ImportError as exc:
        raise RegistryError(
            "LeRobot dataset preflight dependencies are unavailable"
        ) from exc
    try:
        return audit(root)
    except DatasetPreflightError as exc:
        raise RegistryError(f"dataset preflight failed: {exc}") from exc


def register_verified_legacy_download(
    data: Mapping[str, Any], destination: Path
) -> bool:
    """Recover a missing marker only from an exact registered SHA-256 manifest."""
    expected = data.get("source_manifest_sha256")
    if not expected:
        return False
    configured = data.get("download", {}).get("manifest")
    manifest = (
        Path(str(configured))
        if configured
        else destination.with_name(f"{destination.name}_SHA256SUMS.txt")
    )
    if not manifest.is_file() or sha256_file(manifest) != expected:
        raise RegistryError(
            f"legacy source marker is missing and manifest identity failed: {manifest}"
        )
    checked = subprocess.run(
        ["sha256sum", "-c", str(manifest)],
        cwd=destination,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if checked.returncode != 0:
        raise RegistryError(
            f"legacy source checksum verification failed: {checked.stdout}"
        )
    atomic_json(
        destination / ".groot_registry_download.json",
        {
            "repo_id": data["repo_id"],
            "revision": data["revision"],
            "payload_manifest": str(manifest.resolve()),
            "payload_manifest_sha256": expected,
            "source_kind": "verified_legacy_manifest",
        },
    )
    return True


def download_dataset(data: Mapping[str, Any]) -> Path:
    destination = Path(str(data["download"]["root"]))
    marker = destination / ".groot_registry_download.json"
    expected = {"repo_id": data["repo_id"], "revision": data["revision"]}
    if destination.exists() and any(destination.iterdir()):
        if not marker.is_file() and register_verified_legacy_download(
            data, destination
        ):
            return destination
        try:
            recorded = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(
                f"refusing non-empty unregistered destination: {destination}"
            ) from exc
        if any(recorded.get(key) != value for key, value in expected.items()):
            raise RegistryError(
                f"refusing non-empty or mismatched destination: {destination}"
            )
        if recorded.get("source_kind") == "direct_local_directory":
            return destination
        if recorded.get("payload_manifest_sha256"):
            return destination
    else:
        from huggingface_hub import snapshot_download

        destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=str(data["repo_id"]),
            repo_type="dataset",
            revision=str(data["revision"]),
            local_dir=destination,
        )

    manifest_text, digest, file_count, byte_count = payload_manifest(destination)
    manifest_path = Path(
        str(
            data["download"].get(
                "manifest",
                ROOT
                / "artifacts/manifests"
                / f"{data['name']}_{str(data['revision'])[:12]}_SHA256SUMS.txt",
            )
        )
    )
    atomic_text(manifest_path, manifest_text)
    atomic_json(
        marker,
        {
            **expected,
            "payload_manifest": str(manifest_path.resolve()),
            "payload_manifest_sha256": digest,
            "payload_file_count": file_count,
            "payload_bytes": byte_count,
        },
    )
    return destination


def download_provenance(data: Mapping[str, Any]) -> dict[str, Any]:
    marker = Path(str(data["download"]["root"])) / ".groot_registry_download.json"
    if not marker.is_file():
        return {}
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"invalid download provenance marker: {marker}") from exc
    return value if isinstance(value, dict) else {}


def prepare_dataset(data: Mapping[str, Any]) -> Path:
    source = Path(str(data["download"]["root"]))
    destination = Path(str(data["prepared"]["root"]))
    preparation = data.get("preparation")
    if preparation is None:
        if source == destination and destination.is_dir():
            return destination
        if destination.is_dir():
            return destination
        raise RegistryError(
            "no preparation command is configured and prepared.root does not exist"
        )
    script = Path(str(preparation["script"]))
    if not source.is_dir() or not script.is_file():
        raise RegistryError("downloaded source or preparation script is missing")
    if source == destination:
        raise RegistryError(
            "preparation must write a separate derivative, not mutate the download"
        )

    provenance = download_provenance(data)
    expected_manifest = data.get("source_manifest_sha256") or provenance.get(
        "payload_manifest_sha256"
    )
    if destination.exists():
        receipt_path = destination / "DERIVATION_RECEIPT.json"
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(
                f"prepared destination exists without a valid receipt: {destination}"
            ) from exc
        if Path(str(receipt.get("source", ""))).resolve() != source.resolve():
            raise RegistryError(f"prepared receipt source mismatch: {receipt_path}")
        if receipt.get("source_revision") != data["revision"]:
            raise RegistryError(f"prepared receipt revision mismatch: {receipt_path}")
        if (
            expected_manifest
            and receipt.get("source_manifest_sha256") != expected_manifest
        ):
            raise RegistryError(f"prepared receipt manifest mismatch: {receipt_path}")
        return destination

    args = list(map(str, preparation.get("args", [])))
    if preparation.get("inject_source_provenance"):
        if "--source-revision" not in args:
            args.extend(["--source-revision", str(data["revision"])])
        if expected_manifest and "--source-manifest-sha256" not in args:
            args.extend(["--source-manifest-sha256", str(expected_manifest)])
    subprocess.run(
        [str(script), "--source", str(source), "--output", str(destination), *args],
        check=True,
    )
    return destination


def audit_dataset_captions(data: Mapping[str, Any]) -> dict[str, Any] | None:
    settings = data.get("validation", {}).get("caption_audit")
    if not settings or not settings.get("enabled", True):
        return None
    if settings.get("profile") != "omx_insert":
        raise RegistryError("unsupported caption audit profile")
    expected_task = settings.get("expected_overall_task")
    if not expected_task:
        raise RegistryError("caption audit requires expected_overall_task")
    try:
        from data_pipeline.audit_omx_insert_captions import CaptionAuditError, audit
    except ImportError as exc:
        raise RegistryError("OMX caption audit dependencies are unavailable") from exc
    try:
        report = audit(
            Path(str(data["download"]["root"])),
            expected_episodes=settings.get("expected_episodes"),
            expected_subtasks=int(settings.get("expected_subtasks_per_episode", 5)),
            language_mode=str(settings.get("language_mode", "full_episode")),
            expected_overall_task=str(expected_task),
        )
    except CaptionAuditError as exc:
        raise RegistryError(f"caption audit failed: {exc}") from exc
    output = Path(
        str(
            settings.get(
                "output",
                ROOT / "artifacts/reports" / f"{data['name']}_caption_audit.json",
            )
        )
    )
    atomic_json(output, report)
    return {
        "output": str(output),
        "language_contract_ready": report["language_contract_ready"],
    }


def dataset_pipeline(
    data: Mapping[str, Any], target: str = "lerobot"
) -> dict[str, Any]:
    """Prepare canonical LeRobot v3 and optionally its native Isaac derivative."""
    if target not in {"all", "lerobot", "isaac-groot"}:
        raise RegistryError(f"unknown dataset pipeline target: {target}")
    source = download_dataset(data)
    source_preflight = preflight_dataset_root(source)
    caption = audit_dataset_captions(data)
    prepared = prepare_dataset(data)
    checked = validate_dataset(data, decode=True)
    result = {
        "dataset": data["name"],
        "revision": data["revision"],
        "downloaded": str(source),
        "source_preflight": source_preflight,
        "caption_audit": caption,
        "prepared": str(prepared),
        "validation": checked,
    }
    if target in {"all", "isaac-groot"}:
        try:
            from data_pipeline.isaac_groot import (
                IsaacDatasetError,
                prepare_isaac_dataset,
            )

            result["isaac_groot"] = prepare_isaac_dataset(data)
        except IsaacDatasetError as exc:
            raise RegistryError(str(exc)) from exc
    return result


def import_wandb(item: Experiment, run_id: str | None, output: Path | None) -> Path:
    tracking, annotation = item.data["tracking"], item.data["annotation"]
    run_id = run_id or annotation.get("wandb_run_id")
    if not tracking.get("entity") or not tracking.get("project") or not run_id:
        raise RegistryError("W&B entity, project, and run ID are required")
    import wandb

    run = wandb.Api().run(f"{tracking['entity']}/{tracking['project']}/{run_id}")
    summary = dict(run.summary)
    if output is not None:
        path = output
    elif annotation.get("result_path"):
        path = Path(str(annotation["result_path"]))
    else:
        try:
            path = result_file(item.name)
        except RegistryError:
            path = RUNS / item.name / "result.yaml"
    result = read_yaml(path) if path.is_file() else blank_result(item.name)
    result["wandb"] = {
        "entity": tracking["entity"],
        "project": tracking["project"],
        "run_id": run_id,
        "url": run.url,
    }
    aliases = {
        "final_eval_loss": ("eval_loss", "eval/loss"),
        "final_train_loss": ("train_loss", "train/loss", "loss"),
        "samples_per_second": ("samples_per_second", "train/samples_per_second"),
        "cuda_memory_gib": ("cuda_memory_gib", "train/cuda_memory_gib"),
    }
    for destination, names in aliases.items():
        result["metrics"][destination] = next(
            (summary[name] for name in names if name in summary), None
        )
    atomic_yaml(path, result)
    return path


def portable_path(path: Path) -> str:
    """Prefer a workspace-relative variable while keeping external paths valid."""
    resolved = path.expanduser().resolve()
    try:
        return "${workspace}/" + resolved.relative_to(WORKSPACE).as_posix()
    except ValueError:
        return str(resolved)


def resolve_hf_dataset_revision(repo_id: str, revision: str) -> str:
    """Turn a Hub branch/tag into the immutable dataset commit it currently names."""
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    try:
        from huggingface_hub import HfApi

        resolved = HfApi().dataset_info(repo_id=repo_id, revision=revision).sha
    except Exception as exc:
        raise RegistryError(
            f"cannot resolve Hugging Face dataset revision {repo_id}@{revision}: {exc}"
        ) from exc
    if not resolved or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise RegistryError(
            f"Hub returned an invalid dataset commit for {repo_id}@{revision}"
        )
    return resolved


def f2_omx_registration_defaults() -> dict[str, Any]:
    """Load the human-reviewable defaults used by common Hub registration."""
    data = substitute(
        read_yaml(F2_OMX_PROFILE),
        {"workspace": str(WORKSPACE), "training_root": str(ROOT)},
    )
    if data.get("schema_version") != 1:
        raise RegistryError(f"{F2_OMX_PROFILE}: requires schema_version: 1")
    for key in ("name", "robot_revision", "task", "evidence"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise RegistryError(f"{F2_OMX_PROFILE}: {key} must be a non-empty string")
    return {
        "profile": str(F2_OMX_PROFILE),
        "hub_revision": "main",
        "robot_revision": data["robot_revision"],
        "task": data["task"],
        "evidence": data["evidence"],
    }


def omx_insert_contract(robot_revision: str, task: str) -> dict[str, Any]:
    return {
        "robot_revision": robot_revision,
        "embodiment": "new_embodiment",
        "fps": 15,
        "units": {
            "state": "radians_and_normalized_gripper",
            "action": "absolute_joint_position_targets",
        },
        "language": {
            "convention": (
                "One full-episode task via task_index; five ordered subtasks are annotations only"
            ),
            "instruction": task,
        },
        "source_camera_transform": {
            "head": "native 640x480 at 15 fps; identity",
            "left_wrist": "native 640x480 at 15 fps; identity",
            "right_wrist": "native 640x480 at 15 fps; identity",
        },
        "video_encoding": {
            "geometry": "identity",
            "codec": "h264",
            "gop_size": 15,
            "crf": 18,
            "preset": "medium",
        },
        "cameras": [{"key": key, "shape": [3, 480, 640]} for key in OMX_INSERT_CAMERAS],
        "state": {"dim": 16, "names": list(OMX_INSERT_JOINTS)},
        "action": {"dim": 16, "names": list(OMX_INSERT_JOINTS)},
    }


def isaac_adapter_defaults(name: str) -> dict[str, Any]:
    """Make the generated native derivative explicit in each registration."""
    return {
        "isaac_groot": {
            "root": portable_path(
                ROOT / "artifacts" / "datasets" / f"{name}_isaac_v21"
            ),
            "modality_config_path": (
                "${training_root}/configs/modalities/f2_arms16_absolute_h40.py"
            ),
            "modality_json_path": (
                "${training_root}/configs/modalities/f2_arms16_modality.json"
            ),
            "embodiment_tag": "NEW_EMBODIMENT",
        }
    }


def register_hf_omx_dataset(
    name: str,
    repo_id: str,
    revision: str,
    robot_revision: str | None,
    task: str | None,
    source_root: Path | None = None,
    prepared_root: Path | None = None,
    training_repo_id: str | None = None,
    eval_split: float = 0.1,
) -> Path:
    """Register the current F2 OMX Insert Hub-to-GR00T pipeline."""
    defaults = f2_omx_registration_defaults()
    robot_revision_source = "command_line" if robot_revision else "project_default"
    task_source = "command_line" if task else "project_default"
    robot_revision = robot_revision or str(defaults["robot_revision"])
    task = task or str(defaults["task"])
    if not NAME_RE.fullmatch(name):
        raise RegistryError(f"unsafe name {name!r}")
    if "/" not in repo_id:
        raise RegistryError("--repo-id must look like User/Repo")
    if not robot_revision.strip() or not task.strip():
        raise RegistryError("--robot-revision and --task must be non-empty")
    if not 0 <= eval_split < 1:
        raise RegistryError("--eval-split must be at least 0 and less than 1")
    destination = DATASETS / f"{name}.yaml"
    if destination.exists():
        raise RegistryError(f"dataset registration already exists: {destination}")

    pinned = resolve_hf_dataset_revision(repo_id, revision)
    source_root = source_root or ROOT / "artifacts/datasets" / f"{name}_source"
    prepared_root = (
        prepared_root or ROOT / "artifacts/datasets" / f"{name}_arms16_gop15"
    )
    manifest_path = (
        ROOT / "artifacts/manifests" / f"{name}_{pinned[:12]}_SHA256SUMS.txt"
    )
    data = {
        "schema_version": 1,
        "name": name,
        "repo_id": repo_id,
        "revision": pinned,
        "registration": {
            "contract_profile": portable_path(F2_OMX_PROFILE),
            "requested_hub_revision": revision,
            "robot_revision_source": robot_revision_source,
            "task_source": task_source,
        },
        "download": {
            "root": portable_path(source_root),
            "manifest": portable_path(manifest_path),
        },
        "prepared": {"root": portable_path(prepared_root), "immutable": True},
        "preparation": {
            "script": "${training_root}/scripts/prepare_omx_insert_native.sh",
            "inject_source_provenance": True,
            "args": ["--gop-size", "15", "--crf", "18", "--preset", "medium"],
        },
        "validation": {
            "caption_audit": {
                "enabled": True,
                "profile": "omx_insert",
                "language_mode": "full_episode",
                "expected_subtasks_per_episode": 5,
                "expected_overall_task": task,
                "output": portable_path(
                    ROOT / "artifacts/reports" / f"{name}_caption_audit.json"
                ),
            }
        },
        "training": {
            "repo_id": training_repo_id or f"local/{name}_arms16_gop15",
            "video_backend": "torchcodec",
            "eval_split": eval_split,
        },
        "adapters": isaac_adapter_defaults(name),
        "contract": omx_insert_contract(robot_revision, task),
    }
    atomic_yaml(destination, data)
    return destination


def register_local_omx_dataset(
    name: str,
    source_root: Path,
    robot_revision: str | None,
    task: str | None,
    prepared_root: Path | None = None,
    training_repo_id: str | None = None,
    eval_split: float = 0.1,
    source_revision: str | None = None,
    handoff: Path | None = None,
) -> Path:
    """Register an already copied local F2 OMX Insert LeRobot directory."""
    defaults = f2_omx_registration_defaults()
    robot_revision_source = "command_line" if robot_revision else "project_default"
    task_source = "command_line" if task else "project_default"
    robot_revision = robot_revision or str(defaults["robot_revision"])
    task = task or str(defaults["task"])
    if not NAME_RE.fullmatch(name):
        raise RegistryError(f"unsafe name {name!r}")
    if not robot_revision.strip() or not task.strip():
        raise RegistryError("--robot-revision and --task must be non-empty")
    if not 0 <= eval_split < 1:
        raise RegistryError("--eval-split must be at least 0 and less than 1")

    source_root = source_root.expanduser().resolve()
    if not source_root.is_dir() or not any(source_root.iterdir()):
        raise RegistryError(
            f"--source-root must be a non-empty directory: {source_root}"
        )
    revision = (source_revision or "local").strip()
    if not revision:
        raise RegistryError("--source-revision must be non-empty when provided")
    resolved_handoff = handoff.expanduser().resolve() if handoff else None
    if resolved_handoff is not None and not resolved_handoff.is_file():
        raise RegistryError(f"handoff not found: {resolved_handoff}")

    destination = DATASETS / f"{name}.yaml"
    if destination.exists():
        raise RegistryError(f"dataset registration already exists: {destination}")

    repo_id = f"local/{name}"
    prepared_root = (
        prepared_root or ROOT / "artifacts/datasets" / f"{name}_arms16_gop15"
    )
    metadata = info(source_root)
    registration: dict[str, Any] = {
        "source_kind": "direct_local_directory",
        "contract_profile": portable_path(F2_OMX_PROFILE),
        "robot_revision_source": robot_revision_source,
        "task_source": task_source,
    }
    if source_revision:
        registration["source_revision_source"] = "command_line"
    if resolved_handoff is not None:
        registration["handoff"] = portable_path(resolved_handoff)
    data: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "repo_id": repo_id,
        "revision": revision,
        "registration": registration,
        "download": {"root": portable_path(source_root)},
        "prepared": {"root": portable_path(prepared_root), "immutable": True},
        "preparation": {
            "script": "${training_root}/scripts/prepare_omx_insert_native.sh",
            "inject_source_provenance": True,
            "args": ["--gop-size", "15", "--crf", "18", "--preset", "medium"],
        },
        "validation": {
            "caption_audit": {
                "enabled": True,
                "profile": "omx_insert",
                "language_mode": "full_episode",
                "expected_subtasks_per_episode": 5,
                "expected_overall_task": task,
                "output": portable_path(
                    ROOT / "artifacts/reports" / f"{name}_caption_audit.json"
                ),
            }
        },
        "training": {
            "repo_id": training_repo_id or f"local/{name}_arms16_gop15",
            "video_backend": "torchcodec",
            "eval_split": eval_split,
        },
        "expected": {
            key: metadata[key]
            for key in ("codebase_version", "total_episodes", "total_frames")
            if key in metadata
        },
        "adapters": isaac_adapter_defaults(name),
        "contract": omx_insert_contract(robot_revision, task),
    }
    atomic_json(
        source_root / ".groot_registry_download.json",
        {
            "repo_id": repo_id,
            "revision": revision,
            "source_kind": "direct_local_directory",
        },
    )
    atomic_yaml(destination, data)
    return destination


def hf_registration_report(path: Path) -> dict[str, Any]:
    data = dataset_spec(path)
    registration = data.get("registration", {})
    return {
        "registered": str(path),
        "dataset": data["name"],
        "hub": {
            "repo_id": data["repo_id"],
            "requested_revision": registration.get("requested_hub_revision"),
            "pinned_commit": data["revision"],
        },
        "robot_revision": {
            "value": data["contract"]["robot_revision"],
            "source": registration.get("robot_revision_source", "registration"),
        },
        "task": {
            "value": data["contract"]["language"]["instruction"],
            "source": registration.get("task_source", "registration"),
        },
        "next": [
            f"{ROOT / 'trainctl'} dataset show {data['name']}",
            f"{ROOT / 'trainctl'} dataset pipeline {data['name']}",
        ],
    }


def local_registration_report(path: Path) -> dict[str, Any]:
    data = dataset_spec(path)
    registration = data.get("registration", {})
    return {
        "registered": str(path),
        "dataset": data["name"],
        "source": {
            "kind": registration.get("source_kind"),
            "root": data["download"]["root"],
            "label": data["revision"],
            "handoff": registration.get("handoff"),
        },
        "robot_revision": {
            "value": data["contract"]["robot_revision"],
            "source": registration.get("robot_revision_source", "registration"),
        },
        "task": {
            "value": data["contract"]["language"]["instruction"],
            "source": registration.get("task_source", "registration"),
        },
        "next": [
            f"{ROOT / 'trainctl'} dataset show {data['name']}",
            f"{ROOT / 'trainctl'} dataset pipeline {data['name']}",
        ],
    }


def init_dataset(
    name: str,
    repo_id: str,
    revision: str,
    root: Path,
    robot_revision: str,
    training_repo_id: str | None,
    eval_split: float,
    state_units: str,
    action_units: str,
    language_convention: str,
    embodiment: str,
) -> Path:
    """Create the small dataset identity file used by training configs."""
    if not NAME_RE.fullmatch(name):
        raise RegistryError(f"unsafe name {name!r}")
    if not repo_id or "/" not in repo_id:
        raise RegistryError("--repo-id must look like organization/dataset")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RegistryError("--revision must be an exact 40-character commit SHA")
    if not 0 <= eval_split < 1:
        raise RegistryError("--eval-split must be at least 0 and less than 1")
    destination = DATASETS / f"{name}.yaml"
    if destination.exists():
        raise RegistryError(f"dataset registration already exists: {destination}")
    stored_root = portable_path(root)
    data: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "repo_id": repo_id,
        "revision": revision,
        "download": {"root": stored_root},
        "prepared": {"root": stored_root, "immutable": True},
        "training": {
            "repo_id": training_repo_id or f"local/{name}",
            "video_backend": "torchcodec",
            "eval_split": eval_split,
        },
        "contract": {
            "auto_from_metadata": True,
            "robot_revision": robot_revision,
            "embodiment": embodiment,
            "units": {"state": state_units, "action": action_units},
            "language": {"convention": language_convention},
        },
    }
    metadata_path = root.expanduser().resolve() / "meta" / "info.json"
    if metadata_path.is_file():
        metadata = info(root.expanduser().resolve())
        data["expected"] = {
            key: metadata[key]
            for key in ("codebase_version", "total_episodes", "total_frames")
            if key in metadata
        }
        check = substitute(
            data,
            {
                "workspace": str(WORKSPACE),
                "training_root": str(ROOT),
            },
        )
        resolved_contract(check)
    atomic_yaml(destination, data)
    return destination


def init_config(
    name: str,
    dataset_reference: str,
    policy: str = "lerobot-groot",
) -> Path:
    """Create a small editable run config without overwriting existing work."""
    if not NAME_RE.fullmatch(name):
        raise RegistryError(f"unsafe name {name!r}")
    dataset_spec(dataset_reference)
    try:
        backend = backends.canonical_backend(policy)
    except ValueError as exc:
        raise RegistryError(str(exc)) from exc
    destination = CONFIGS / f"{name}.yaml"
    if destination.exists():
        raise RegistryError(f"config already exists: {destination}")
    atomic_yaml(
        destination,
        {
            "_base": backends.POLICY_BASES[backend],
            "name": name,
            "description": "REPLACE ME: describe this run's purpose and deliberate changes.",
            "hypothesis": "REPLACE ME: state a falsifiable expectation and reason.",
            "dataset_ref": dataset_reference,
            "annotation": {
                "owner": "intern-team",
                "notes": (
                    "REPLACE ME: add dataset-specific evaluation criteria, caveats, "
                    "and intended comparisons."
                ),
            },
        },
    )
    return destination


def result_file(reference: str | Path, attempt: int | None = None) -> Path:
    direct = Path(reference)
    if direct.is_file():
        return direct.resolve()
    run_dir = RUNS / str(reference)
    if attempt is not None:
        candidate = run_dir / f"attempt-{attempt}" / "result.yaml"
        if candidate.is_file():
            return candidate.resolve()
        raise RegistryError(f"result not found: {candidate}")
    attempts = sorted(
        run_dir.glob("attempt-*/result.yaml"),
        key=lambda path: int(path.parent.name.removeprefix("attempt-")),
    )
    candidates = [run_dir / "result.yaml", *attempts]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise RegistryError(f"no result found for {reference}")
    return existing[-1].resolve()


def list_results() -> list[dict[str, Any]]:
    summaries = []
    if not RUNS.is_dir():
        return summaries
    for run_dir in sorted(
        path
        for path in RUNS.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    ):
        try:
            path = result_file(run_dir.name)
        except RegistryError:
            continue
        value = read_yaml(path)
        summaries.append(
            {
                "run_name": value.get("run_name", run_dir.name),
                "state": value.get("status", {}).get("state"),
                "final_eval_loss": value.get("metrics", {}).get("final_eval_loss"),
                "wandb_url": value.get("wandb", {}).get("url"),
                "result": str(path),
            }
        )
    return summaries


def parser() -> argparse.ArgumentParser:
    program = os.environ.get("TRAINCTL_PROGRAM", "grootctl")
    root = argparse.ArgumentParser(prog=program)
    subs = root.add_subparsers(dest="action", required=True)
    lifecycle_name = "run" if program == "trainctl" else "train"
    train_parent = subs.add_parser(
        lifecycle_name, help="simple dataset-configured training lifecycle"
    )
    train_parent.set_defaults(action="train")
    train = train_parent.add_subparsers(dest="verb", required=True)
    train_init = train.add_parser(
        "init", help="create a run YAML from current defaults"
    )
    train_init.add_argument("name")
    train_init.add_argument("--dataset", required=True)
    train_init.add_argument(
        "--policy",
        default="lerobot-groot",
        choices=("lerobot-groot", "isaac-groot", "lerobot-act"),
    )
    train_help = {
        "check": "validate and print the detached launch plan without training",
        "start": "validate and launch training in detached tmux",
    }
    for verb in ("check", "start"):
        child = train.add_parser(verb, help=train_help[verb])
        child.add_argument("reference")
        child.add_argument("--set", action="append", default=[])
        child.add_argument("--run-dir", type=Path)
        child.add_argument("--session")
        if verb == "start":
            child.add_argument("--resume", action="store_true")
    train_status = train.add_parser("status", help="show the saved run result")
    train_status.add_argument("reference")
    train_status.add_argument("--attempt", type=int)
    config = subs.add_parser("config").add_subparsers(dest="verb", required=True)
    for verb in ("resolve", "validate", "command"):
        child = config.add_parser(verb)
        child.add_argument("reference")
        child.add_argument("--set", action="append", default=[])
        child.add_argument("--no-dataset", action="store_true")
    initialize = config.add_parser("init")
    initialize.add_argument("name")
    initialize.add_argument("--dataset", required=True)
    initialize.add_argument(
        "--policy",
        default="lerobot-groot",
        choices=("lerobot-groot", "isaac-groot", "lerobot-act"),
    )
    foreground_name = "exec" if program == "trainctl" else "run"
    run = subs.add_parser(foreground_name)
    run.set_defaults(action="run")
    run.add_argument("reference")
    run.add_argument("--set", action="append", default=[])
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--run-dir", type=Path)
    run_tmux = subs.add_parser(
        "run-tmux",
        help="launch a run in an SSH-safe detached tmux session",
        description=(
            "Preflight a normal run, launch it in detached tmux, preserve startup "
            "output, and print attach/log/status/stop commands."
        ),
    )
    run_tmux.add_argument("reference")
    run_tmux.add_argument("--set", action="append", default=[])
    run_tmux.add_argument(
        "--dry-run",
        action="store_true",
        help="print the launch and monitoring plan without starting tmux",
    )
    run_tmux.add_argument("--resume", action="store_true")
    run_tmux.add_argument("--run-dir", type=Path)
    run_tmux.add_argument(
        "--session", help="override the default groot-<run-name> session"
    )
    queue = subs.add_parser("queue")
    queue.add_argument("reference")
    queue.add_argument("--dry-run", action="store_true")
    dataset = subs.add_parser(
        "dataset", help="register, fetch, prepare, and validate datasets"
    ).add_subparsers(dest="verb", required=True)
    dataset.add_parser(
        "defaults",
        help="show the current robot revision and task used when options are omitted",
    )
    dataset_register = dataset.add_parser(
        "register-hf",
        help="register a Hub dataset using current F2 OMX defaults",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Create a small tracked registration for a Hugging Face LeRobot dataset.\n"
            "Typical use needs only a local name and User/Repo:\n\n"
            "  trainctl dataset register-hf my_data --repo-id User/Repo\n\n"
            "The default Hub revision is main and is automatically pinned to its exact\n"
            "commit. Robot revision and task come from the visible project defaults;\n"
            "override them only when the producer handoff says they changed."
        ),
    )
    dataset_register.add_argument("name")
    dataset_register.add_argument(
        "--repo-id", required=True, help="Hugging Face User/Repo"
    )
    dataset_register.add_argument(
        "--revision",
        default="main",
        help="Hub branch/tag/commit; usually omit (default: main, pinned automatically)",
    )
    dataset_register.add_argument(
        "--robot-revision",
        help="producer robot source/config; usually omit to use dataset defaults",
    )
    dataset_register.add_argument(
        "--task",
        help="full-episode instruction; usually omit to use dataset defaults",
    )
    dataset_register.add_argument("--source-root", type=Path)
    dataset_register.add_argument("--prepared-root", type=Path)
    dataset_register.add_argument("--training-repo-id")
    dataset_register.add_argument("--eval-split", type=float, default=0.1)
    dataset_register_local = dataset.add_parser(
        "register-local",
        help="register a LeRobot directory already copied onto this server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Create a tracked registration for a local LeRobot directory.\n"
            "Typical use needs only a local name and --source-root:\n\n"
            "  trainctl dataset register-local my_data --source-root /path/to/dataset"
        ),
    )
    dataset_register_local.add_argument("name")
    dataset_register_local.add_argument("--source-root", type=Path, required=True)
    dataset_register_local.add_argument(
        "--source-revision",
        help="optional free-form source label or commit (default: local)",
    )
    dataset_register_local.add_argument(
        "--handoff",
        type=Path,
        help="optional provenance note; no naming convention is required",
    )
    dataset_register_local.add_argument(
        "--robot-revision",
        help="producer robot source/config; usually omit to use dataset defaults",
    )
    dataset_register_local.add_argument(
        "--task",
        help="full-episode instruction; usually omit to use dataset defaults",
    )
    dataset_register_local.add_argument("--prepared-root", type=Path)
    dataset_register_local.add_argument("--training-repo-id")
    dataset_register_local.add_argument("--eval-split", type=float, default=0.1)
    dataset_init = dataset.add_parser("init")
    dataset_init.add_argument("name")
    dataset_init.add_argument("--repo-id", required=True)
    dataset_init.add_argument("--revision", required=True)
    dataset_init.add_argument("--root", type=Path, required=True)
    dataset_init.add_argument("--robot-revision", required=True)
    dataset_init.add_argument("--training-repo-id")
    dataset_init.add_argument("--eval-split", type=float, default=0.1)
    dataset_init.add_argument("--state-units", default="radians_and_normalized_gripper")
    dataset_init.add_argument(
        "--action-units", default="absolute_joint_position_targets"
    )
    dataset_init.add_argument(
        "--language-convention", default="English imperative task instruction"
    )
    dataset_init.add_argument("--embodiment", default="new_embodiment")
    dataset_help = {
        "show": "print the resolved registration",
        "download": "download a Hub source or reuse a registered local directory",
        "preflight": "detect caption-whitespace and episode-index defects",
        "prepare": "create or verify the immutable training derivative",
        "prepare-isaac": "create or verify the native Isaac-GR00T v2.1 derivative",
        "validate": "check the prepared dataset contract",
        "pipeline": "acquire/recheck, audit captions, prepare, and decode-validate",
    }
    for verb in (
        "show",
        "download",
        "preflight",
        "prepare",
        "prepare-isaac",
        "validate",
        "pipeline",
    ):
        child = dataset.add_parser(verb, help=dataset_help[verb])
        child.add_argument("reference")
        if verb == "validate":
            child.add_argument("--decode", action="store_true")
        if verb == "preflight":
            child.add_argument(
                "--stage",
                choices=("source", "prepared"),
                default="prepared",
                help="registration path to inspect (default: prepared)",
            )
        if verb == "pipeline":
            child.add_argument(
                "--for",
                dest="for_backend",
                choices=("all", "lerobot", "isaac-groot"),
                default="all",
                help="prepare all backends by default",
            )
    checkpoint = subs.add_parser(
        "checkpoint", help="freeze, verify, and deliver complete model packages"
    ).add_subparsers(dest="verb", required=True)
    package = checkpoint.add_parser("package", help="freeze a completed run")
    package.add_argument("reference")
    package.add_argument("--name")
    verify = checkpoint.add_parser("verify", help="verify every package checksum")
    verify.add_argument("reference")
    transfer = checkpoint.add_parser(
        "transfer", help="stage, verify, and atomically promote on a robot"
    )
    transfer.add_argument("reference")
    transfer.add_argument(
        "--target", required=True, help="SSH target, for example robot@f2"
    )
    transfer.add_argument(
        "--robot-workspace",
        default=str(checkpoints.DEFAULT_ROBOT_WORKSPACE),
        help=(
            "Cyclo host workspace alias; defaults to "
            "/home/robotis/cyclo_intelligence/docker/workspace"
        ),
    )
    transfer.add_argument("--port", type=int, default=22)
    transfer.add_argument("--identity-file", type=Path)
    transfer.add_argument("--dry-run", action="store_true")
    acknowledge = checkpoint.add_parser(
        "acknowledge", help="record a robot-generated consumer receipt"
    )
    acknowledge.add_argument("reference")
    acknowledge.add_argument("--consumer-receipt", type=Path, required=True)
    result = subs.add_parser("result").add_subparsers(dest="verb", required=True)
    wandb = result.add_parser("import-wandb")
    wandb.add_argument("reference")
    wandb.add_argument("--run-id")
    wandb.add_argument("--output", type=Path)
    show = result.add_parser("show")
    show.add_argument("reference")
    show.add_argument("--attempt", type=int)
    result.add_parser("list")
    return root


def dump(value: Any) -> None:
    sys.stdout.write(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.action == "train":
            if args.verb == "init":
                print(init_config(args.name, args.dataset, args.policy))
                return 0
            if args.verb == "status":
                dump(read_yaml(result_file(args.reference, args.attempt)))
                return 0
            item = experiment(
                args.reference, args.set, require_dataset=True, launch=True
            )
            report = launch_tmux(
                item,
                sets=args.set,
                resume=getattr(args, "resume", False),
                run_dir=args.run_dir,
                session=args.session,
                dry_run=args.verb == "check",
            )
            print_tmux_report(report)
            return 0
        if args.action == "config":
            if args.verb == "init":
                print(init_config(args.name, args.dataset, args.policy))
                return 0
            item = experiment(
                args.reference, args.set, require_dataset=not args.no_dataset
            )
            if args.verb == "resolve":
                dump(item.data)
            elif args.verb == "validate":
                print(f"OK {item.name}: {item.source}")
            else:
                print(shlex.join(command(item.data)))
            return 0
        if args.action == "run":
            item = experiment(
                args.reference, args.set, require_dataset=True, launch=True
            )
            result = execute(item, args.dry_run, args.resume, args.run_dir)
            dump(result)
            return 0 if args.dry_run or result["status"]["state"] == "completed" else 1
        if args.action == "run-tmux":
            item = experiment(
                args.reference, args.set, require_dataset=True, launch=True
            )
            report = launch_tmux(
                item, args.set, args.resume, args.run_dir, args.session, args.dry_run
            )
            print_tmux_report(report)
            return 0
        if args.action == "queue":
            result = run_queue(args.reference, args.dry_run)
            dump(result)
            return 0 if args.dry_run or result["state"] == "completed" else 1
        if args.action == "dataset":
            if args.verb == "defaults":
                dump(f2_omx_registration_defaults())
                return 0
            if args.verb == "register-hf":
                path = register_hf_omx_dataset(
                    args.name,
                    args.repo_id,
                    args.revision,
                    args.robot_revision,
                    args.task,
                    args.source_root,
                    args.prepared_root,
                    args.training_repo_id,
                    args.eval_split,
                )
                dump(hf_registration_report(path))
                return 0
            if args.verb == "register-local":
                path = register_local_omx_dataset(
                    name=args.name,
                    source_root=args.source_root,
                    robot_revision=args.robot_revision,
                    task=args.task,
                    prepared_root=args.prepared_root,
                    training_repo_id=args.training_repo_id,
                    eval_split=args.eval_split,
                    source_revision=args.source_revision,
                    handoff=args.handoff,
                )
                dump(local_registration_report(path))
                return 0
            if args.verb == "init":
                print(
                    init_dataset(
                        args.name,
                        args.repo_id,
                        args.revision,
                        args.root,
                        args.robot_revision,
                        args.training_repo_id,
                        args.eval_split,
                        args.state_units,
                        args.action_units,
                        args.language_convention,
                        args.embodiment,
                    )
                )
                return 0
            data = dataset_spec(args.reference)
            if args.verb == "show":
                dump(data)
            elif args.verb == "download":
                print(download_dataset(data))
            elif args.verb == "preflight":
                key = "download" if args.stage == "source" else "prepared"
                dump(preflight_dataset_root(Path(str(data[key]["root"]))))
            elif args.verb == "prepare":
                print(prepare_dataset(data))
            elif args.verb == "prepare-isaac":
                from data_pipeline.isaac_groot import prepare_isaac_dataset

                dump(prepare_isaac_dataset(data))
            elif args.verb == "pipeline":
                dump(dataset_pipeline(data, args.for_backend))
            else:
                dump(validate_dataset(data, args.decode))
            return 0
        if args.action == "checkpoint":
            if args.verb == "package":
                dump(checkpoints.package_checkpoint(args.reference, args.name))
            elif args.verb == "verify":
                dump(checkpoints.verify_package(args.reference))
            elif args.verb == "transfer":
                dump(
                    checkpoints.transfer_package(
                        args.reference,
                        target=args.target,
                        robot_workspace=args.robot_workspace,
                        port=args.port,
                        identity_file=args.identity_file,
                        dry_run=args.dry_run,
                    )
                )
            else:
                dump(
                    checkpoints.acknowledge_package(
                        args.reference, args.consumer_receipt
                    )
                )
            return 0
        if args.action == "result":
            if args.verb == "list":
                dump(list_results())
            elif args.verb == "show":
                dump(read_yaml(result_file(args.reference, args.attempt)))
            else:
                print(
                    import_wandb(
                        experiment(args.reference, require_dataset=False),
                        args.run_id,
                        args.output,
                    )
                )
            return 0
        raise RegistryError(f"unknown action: {args.action}")
    except (
        RegistryError,
        checkpoints.CheckpointError,
        subprocess.CalledProcessError,
        ImportError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
