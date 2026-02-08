from __future__ import annotations

from typing import Dict, Optional, Set

import torch
import torch.nn as nn


def get_lora_params(
    model: nn.Module,
    lora_param_names: Optional[Set[str]] = None,
) -> Dict[str, nn.Parameter]:
    """
    Return a dict of LoRA parameters (name -> Parameter).

    Prefer passing `lora_param_names` from `GenerationBase.lora_params` when
    available; otherwise we fall back to a name substring match.
    """
    if lora_param_names is not None:
        return {n: p for n, p in model.named_parameters() if n in lora_param_names}
    return {n: p for n, p in model.named_parameters() if "lora" in n}


def snapshot_lora_state(
    model: nn.Module,
    lora_param_names: Optional[Set[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Snapshot LoRA weights as plain tensors (detached clones).
    """
    return {
        n: p.detach().clone()
        for n, p in get_lora_params(model, lora_param_names).items()
    }


def restore_lora_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """
    Restore LoRA weights from a snapshot produced by `snapshot_lora_state`.
    """
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in state.items():
            if n in params:
                params[n].copy_(v)


def count_lora_params(model: nn.Module, lora_param_names: Optional[Set[str]] = None) -> int:
    """
    Total number of scalar parameters in the LoRA "memory".
    """
    return sum(p.numel() for p in get_lora_params(model, lora_param_names).values())

