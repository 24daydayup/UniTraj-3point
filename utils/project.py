from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch

from utils.config import args as default_args
from utils.flow_matching import SingleStageTrajectoryFlow

PATCH_STAT_KEYS = (
    "avg_num_patches",
    "avg_patch_len",
    "max_patch_len",
    "avg_patch_score",
)


def to_plain_dict(value: Any) -> Any:
    """Recursively convert namespaces to plain Python containers."""

    if isinstance(value, SimpleNamespace):
        return {key: to_plain_dict(getattr(value, key)) for key in vars(value)}
    if isinstance(value, dict):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_dict(item) for item in value]
    return value


def to_namespace(value: Any) -> Any:
    """Recursively convert dictionaries to namespaces."""

    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def load_default_config() -> SimpleNamespace:
    return to_namespace(deepcopy(default_args))


def adaptive_patch_enabled(model_config: Any) -> bool:
    """Return whether adaptive patching is enabled for a model config."""

    adaptive_cfg = getattr(model_config, "adaptive_patch", None)
    if adaptive_cfg is None:
        return False
    if isinstance(adaptive_cfg, dict):
        return bool(adaptive_cfg.get("enabled", False))
    return bool(getattr(adaptive_cfg, "enabled", False))


def build_flow_model_from_config(config: SimpleNamespace) -> SingleStageTrajectoryFlow:
    """Build the current single-stage flow model from a project config."""

    return SingleStageTrajectoryFlow(
        input_dim=2,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        max_len=config.data.traj_length,
        mlp_ratio=config.model.mlp_ratio,
        use_moe=config.model.use_moe,
        num_experts=config.model.num_experts,
        top_k=config.model.top_k,
        adaptive_patch_cfg=to_plain_dict(getattr(config.model, "adaptive_patch", None)),
    )


def checkpoint_filenames(model_config: Any) -> tuple[str, str]:
    if adaptive_patch_enabled(model_config):
        return "best_flow_adaptive_patch.pt", "final_flow_adaptive_patch.pt"
    return "best_flow_model.pt", "final_flow_model.pt"


def resolve_device(device_preference: str = "auto", gpu: Optional[int] = None) -> torch.device:
    if gpu is not None:
        if torch.cuda.is_available() and gpu >= 0:
            return torch.device(f"cuda:{gpu}")
        return torch.device("cpu")

    if device_preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_preference)


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_config_snapshot(config: SimpleNamespace, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(to_plain_dict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_config_snapshot(config_path: Path) -> SimpleNamespace:
    return to_namespace(json.loads(config_path.read_text(encoding="utf-8")))


def find_config_snapshot(model_path: str | Path) -> Optional[Path]:
    """Try to locate a saved config snapshot next to a checkpoint."""

    model_path = Path(model_path).resolve()
    candidate_paths = [
        model_path.parent.parent / "config_snapshot.json",
        model_path.parent.parent / "Files" / "config_snapshot.json",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return None
