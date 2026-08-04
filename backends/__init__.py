"""Backend-specific command rendering for the training registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
RUN_BACKEND = ROOT / "scripts" / "run_backend.sh"
ISAAC_GROOT = WORKSPACE / "Isaac-GR00T"

ALIASES = {
    "lerobot": "lerobot_groot",
    "lerobot-groot": "lerobot_groot",
    "lerobot_groot": "lerobot_groot",
    "groot": "lerobot_groot",
    "lerobot-act": "lerobot_act",
    "lerobot_act": "lerobot_act",
    "act": "lerobot_act",
    "isaac-groot": "isaac_groot",
    "isaac_groot": "isaac_groot",
    "isaac": "isaac_groot",
}

POLICY_BASES = {
    "lerobot_groot": "../policies/lerobot_groot_n17.yaml",
    "lerobot_act": "../policies/lerobot_act.yaml",
    "isaac_groot": "../policies/isaac_groot_n17.yaml",
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


def canonical_backend(value: str | None) -> str:
    try:
        return ALIASES[str(value or "")]
    except KeyError as exc:
        known = ", ".join(sorted(POLICY_BASES))
        raise ValueError(f"unknown backend {value!r}; choose one of: {known}") from exc


def is_lerobot(value: str | None) -> bool:
    return canonical_backend(value).startswith("lerobot_")


def render(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def dotted_flags(
    prefix: str,
    values: Mapping[str, Any],
    skip: set[str] | None = None,
) -> list[str]:
    result: list[str] = []
    for key, value in values.items():
        if value is None or value == [] or key in (skip or set()):
            continue
        dotted = f"{prefix}.{key}" if prefix else key
        if dotted == "dataset.image_transforms.tfs" and isinstance(value, dict):
            result.append(f"--{dotted}={render(value)}")
        elif isinstance(value, dict):
            result.extend(dotted_flags(dotted, value))
        else:
            result.append(f"--{dotted}={render(value)}")
    return result


def _tyro_bool(name: str, value: bool) -> str:
    return f"--{name}" if value else f"--no-{name}"


def _lerobot_command(
    data: Mapping[str, Any],
    out: Path,
    resume_config: Path | None,
) -> list[str]:
    if resume_config:
        return [
            str(RUN_BACKEND),
            "lerobot",
            "lerobot-train",
            f"--config_path={resume_config}",
            "--resume=true",
        ]
    dataset = {
        key: value for key, value in data["dataset"].items() if key in DATASET_FLAGS
    }
    train = {
        key: value
        for key, value in data["train"].items()
        if key
        not in {"timeout_seconds", "gradient_accumulation_steps", "save_total_limit"}
    }
    result = [
        str(RUN_BACKEND),
        "lerobot",
        "lerobot-train",
        *dotted_flags("dataset", dataset),
        *dotted_flags("policy", data["policy"]),
        *dotted_flags("", train),
    ]
    if not train.get("use_policy_training_preset", True):
        result += dotted_flags("optimizer", data["optimizer"])
        scheduler = data.get("scheduler")
        if scheduler:
            result += dotted_flags("scheduler", scheduler)
    result += [f"--output_dir={out}", f"--job_name={data['name']}"]
    tracking = data["tracking"]
    if tracking["backend"] == "wandb":
        result += dotted_flags("wandb", tracking, {"backend", "url"})
    else:
        result.append("--wandb.enable=false")
    return result


def _isaac_command(data: Mapping[str, Any], out: Path) -> list[str]:
    policy = data["policy"]
    train = data["train"]
    tracking = data["tracking"]
    dataset = data["dataset"]
    result = [
        str(RUN_BACKEND),
        "isaac_groot",
        "python",
        str(ISAAC_GROOT / "gr00t/experiment/launch_finetune.py"),
        "--base-model-path",
        str(policy["base_model_path"]),
        "--dataset-path",
        str(dataset["root"]),
        "--embodiment-tag",
        str(policy.get("embodiment_tag", "NEW_EMBODIMENT")),
        "--modality-config-path",
        str(dataset["modality_config_path"]),
        "--output-dir",
        str(out),
        "--experiment-name",
        str(data["name"]),
        "--num-gpus",
        str(train.get("num_gpus", 1)),
        "--max-steps",
        str(train["steps"]),
        "--global-batch-size",
        str(train["batch_size"]),
        "--gradient-accumulation-steps",
        str(train.get("gradient_accumulation_steps", 1)),
        "--dataloader-num-workers",
        str(train["num_workers"]),
        "--save-steps",
        str(train["save_freq"]),
        "--save-total-limit",
        str(train.get("save_total_limit", 5)),
        "--learning-rate",
        str(policy["optimizer_lr"]),
        "--weight-decay",
        str(policy["optimizer_weight_decay"]),
        "--warmup-ratio",
        str(policy["warmup_ratio"]),
        "--state-dropout-prob",
        str(policy.get("state_dropout_prob", 0.2)),
        "--shard-size",
        str(dataset.get("shard_size", 1024)),
        "--episode-sampling-rate",
        str(dataset.get("episode_sampling_rate", 0.1)),
        "--num-shards-per-epoch",
        str(dataset.get("num_shards_per_epoch", 100000)),
        _tyro_bool("tune-llm", bool(policy.get("tune_llm", False))),
        _tyro_bool("tune-visual", bool(policy.get("tune_visual", False))),
        _tyro_bool("tune-projector", bool(policy.get("tune_projector", True))),
        _tyro_bool(
            "tune-diffusion-model", bool(policy.get("tune_diffusion_model", True))
        ),
        _tyro_bool("use-percentiles", bool(policy.get("use_percentiles", True))),
        _tyro_bool("save-only-model", bool(train.get("save_only_model", False))),
        _tyro_bool(
            "use-wandb",
            tracking.get("backend") == "wandb" and tracking.get("enable", True),
        ),
    ]
    color = policy.get("color_jitter_params")
    if color:
        result.append("--color-jitter-params")
        for key, value in color.items():
            result.extend([str(key), str(value)])
    if tracking.get("project"):
        result.extend(["--wandb-project", str(tracking["project"])])
    return result


def build_command(
    data: Mapping[str, Any],
    out: Path,
    resume_config: Path | None = None,
) -> list[str]:
    backend = canonical_backend(str(data.get("backend")))
    if backend == "isaac_groot":
        if resume_config is not None:
            raise ValueError(
                "native Isaac resume uses output_dir plus resume_from_checkpoint"
            )
        return _isaac_command(data, out)
    return _lerobot_command(data, out, resume_config)


def model_family(value: str | None) -> str:
    return "groot" if canonical_backend(value) == "isaac_groot" else "lerobot"
