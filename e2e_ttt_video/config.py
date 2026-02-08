from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


def _as_dict(cfg: Optional[DictConfig]) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        out = OmegaConf.to_container(cfg, resolve=True)
        return dict(out) if isinstance(out, dict) else {}
    if isinstance(cfg, dict):
        return dict(cfg)
    raise TypeError(f"Unsupported cfg type: {type(cfg)}")


def _filter_to_dataclass(dc_cls, data: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {f.name for f in fields(dc_cls)}
    return {k: v for k, v in data.items() if k in allowed}


@dataclass
class InnerLoopConfig:
    # Core inner-loop knobs (train-time and test-time)
    num_gradient_steps: int = 1
    batch_size: int = 4
    num_mc_samples: int = 4
    inner_lr_init: float = 5e-5

    # Flow-matching sampling
    flow_shift: float = 5.0
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0

    # Stability
    max_inner_grad_norm: float = 1.0

    # Optional PERK-style meta-learned learning rates
    meta_learn_lr: bool = False

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "InnerLoopConfig":
        return cls(**_filter_to_dataclass(cls, _as_dict(cfg)))


@dataclass
class OuterLoopConfig:
    outer_lr: float = 1e-5
    outer_weight_decay: float = 0.01
    truncate_steps: list[int] = field(default_factory=lambda: [0])

    # Optional training-loop defaults used by `outer_loop.py`
    num_epochs: int = 1
    gradient_accumulation_steps: int = 1
    warmup_fraction: float = 0.0
    max_outer_grad_norm: float = 1.0

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "OuterLoopConfig":
        return cls(**_filter_to_dataclass(cls, _as_dict(cfg)))


@dataclass
class InferenceConfig:
    # High-level sequential generation controls
    num_clips: int = 16
    ttt_steps_per_clip: int = 1
    num_mc_samples: int = 4

    # Clip geometry / sampling (used by `SequentialVideoGenerator`)
    height: int = 480
    width: int = 832
    num_frames_per_clip: int = 81  # 4n+1
    num_ode_steps: int = 50
    flow_shift: float = 5.0
    guidance_scale: float = 5.0
    seed: int = 42

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "InferenceConfig":
        return cls(**_filter_to_dataclass(cls, _as_dict(cfg)))


@dataclass
class E2ETTTConfig:
    """
    Convenience wrapper combining inner-loop, outer-loop, and inference configs.

    The YAML schema remains the source of truth; this dataclass exists mainly for:
    - Validation / defaults in code
    - A consistent object passed into algorithm modules
    """

    enabled: bool = True
    device: str = "cuda"
    use_bf16: bool = True
    output_dir: str = "results/e2e_ttt"

    inner_loop: InnerLoopConfig = field(default_factory=InnerLoopConfig)
    outer_loop: OuterLoopConfig = field(default_factory=OuterLoopConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "E2ETTTConfig":
        data = _as_dict(cfg)
        # Nested dataclasses
        inner = InnerLoopConfig.from_cfg(cfg.get("inner_loop", OmegaConf.create()))
        outer = OuterLoopConfig.from_cfg(cfg.get("outer_loop", OmegaConf.create()))
        inference = InferenceConfig.from_cfg(cfg.get("inference", OmegaConf.create()))
        base_kwargs = _filter_to_dataclass(cls, data)
        base_kwargs.update({"inner_loop": inner, "outer_loop": outer, "inference": inference})
        return cls(**base_kwargs)

