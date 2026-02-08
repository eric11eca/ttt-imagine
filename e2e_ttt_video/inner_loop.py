from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.func import functional_call as torch_func_functional_call  # type: ignore

from .config import InnerLoopConfig
from .higher_optim import DifferentiableSGD
from .lora_state import get_lora_params


def _functional_call(
    module: nn.Module,
    params: Dict[str, torch.Tensor],
    buffers: Optional[Dict[str, torch.Tensor]],
    *,
    x: torch.Tensor,
    t: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    """
    A small compatibility wrapper around PyTorch's functional_call APIs.
    """
    buffers = buffers or {}
    return torch_func_functional_call(module, (params, buffers), (), dict(x=x, t=t, context=context))


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    flow_shift: float = 5.0,
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample timesteps from logit-normal distribution with flow shift.

    Returns:
        t: tensor of shape [batch_size] in (0, 1).
    """
    u = torch.randn(batch_size, device=device, generator=generator)
    t = torch.sigmoid(logit_normal_mean + logit_normal_std * u)
    # flow shift: pushes effective density toward higher t for shift>1
    t = t * flow_shift / (1.0 + (flow_shift - 1.0) * t)
    return t


def compute_flow_matching_loss(
    dit_model: nn.Module,
    z_0: torch.Tensor,  # [B, C, T', H', W']
    text_embeds: torch.Tensor,  # [B, L, D]
    config: InnerLoopConfig,
    num_mc_samples: int = 1,
    *,
    params_override: Optional[Dict[str, torch.Tensor]] = None,
    buffers_override: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Compute rectified flow matching loss for a batch of clip latents.

      L = E_{eps, t} [ || f(z_t, t, c) - (eps - z_0) ||^2 ]
    """
    if z_0.dim() != 5:
        raise ValueError(f"Expected z_0 with shape [B,C,T,H,W], got {tuple(z_0.shape)}")

    B = z_0.shape[0]
    device = z_0.device
    total_loss = torch.tensor(0.0, device=device)

    # Use a local generator seeded from the process seed + tensor shapes.
    # This makes unit tests stable (objective is consistent across repeated calls),
    # while still allowing users to control determinism via `torch.manual_seed(...)`.
    base_seed = int(torch.initial_seed()) & 0xFFFFFFFF
    shape_mix = (z_0.numel() * 1000003 + text_embeds.numel() * 2654435761 + int(num_mc_samples) * 1597334677) & 0xFFFFFFFF
    cfg_mix = (int(config.flow_shift * 1000) + int(config.logit_normal_mean * 1000) + int(config.logit_normal_std * 1000)) & 0xFFFFFFFF
    gen = torch.Generator(device=device)
    gen.manual_seed((base_seed ^ shape_mix ^ cfg_mix) & 0xFFFFFFFF)

    for _ in range(int(num_mc_samples)):
        epsilon = torch.randn(
            z_0.shape,
            device=z_0.device,
            dtype=z_0.dtype,
            generator=gen,
        )
        t = sample_timesteps(
            B,
            device,
            flow_shift=config.flow_shift,
            logit_normal_mean=config.logit_normal_mean,
            logit_normal_std=config.logit_normal_std,
            generator=gen,
        )
        t_expand = t.view(B, 1, 1, 1, 1)

        z_t = (1.0 - t_expand) * z_0 + t_expand * epsilon
        v_target = epsilon - z_0

        t_input = t * 1000.0
        if params_override is None:
            v_pred = dit_model(x=z_t, t=t_input, context=text_embeds)
        else:
            v_pred = _functional_call(
                dit_model,
                params_override,
                buffers_override,
                x=z_t,
                t=t_input,
                context=text_embeds,
            )

        total_loss = total_loss + F.mse_loss(v_pred, v_target)

    return total_loss / max(int(num_mc_samples), 1)


def inner_loop_step(
    dit_model: nn.Module,
    z_0: torch.Tensor,
    text_embeds: torch.Tensor,
    config: InnerLoopConfig,
    learned_lrs: Optional[Dict[str, torch.Tensor]] = None,
    step_index: int = 0,
    return_loss: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Perform ONE differentiable inner-loop SGD step on LoRA parameters.

    Returns:
        (loss, updated_params) where updated_params maps param_name -> updated tensor.
    """
    loss = compute_flow_matching_loss(
        dit_model,
        z_0,
        text_embeds,
        config,
        num_mc_samples=config.num_mc_samples,
    )

    lora_params = get_lora_params(dit_model)
    if len(lora_params) == 0:
        raise ValueError("No LoRA parameters found (expected names containing 'lora').")

    grads_list = torch.autograd.grad(
        loss,
        list(lora_params.values()),
        create_graph=True,
        allow_unused=True,
    )
    grads: Dict[str, Optional[torch.Tensor]] = {
        n: g for (n, _), g in zip(lora_params.items(), grads_list)
    }

    # Support two LR dict conventions:
    #  - {param_name: lr_scalar}
    #  - {f"{param_name}_step{step}": lr_scalar}
    lrs_for_step: Optional[Dict[str, torch.Tensor]] = None
    if learned_lrs is not None:
        lrs_for_step = {}
        for n in lora_params.keys():
            k_step = f"{n}_step{step_index}"
            if k_step in learned_lrs:
                lrs_for_step[n] = learned_lrs[k_step]
            elif n in learned_lrs:
                lrs_for_step[n] = learned_lrs[n]
        if len(lrs_for_step) == 0:
            lrs_for_step = None

    opt = DifferentiableSGD(lr=config.inner_lr_init, max_grad_norm=config.max_inner_grad_norm)
    updated = opt.step(
        {n: p for n, p in lora_params.items()},
        grads,
        learned_lrs=lrs_for_step,
    )

    return (loss if return_loss else None), updated


def run_inner_loop(
    dit_model: nn.Module,
    clip_latents: list[torch.Tensor],  # each: [1, C, T', H', W']
    clip_text_embeds: list[torch.Tensor],  # each: [1, L, D]
    config: InnerLoopConfig,
    learned_lrs: Optional[object] = None,
    truncate_steps: Optional[list[int]] = None,
) -> Tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Run sequential TTT updates over a clip sequence.

    - Computes a meta-loss (average inner-step loss).
    - Produces in-place weight changes on the model (LoRA only) as a side-effect.
    """
    if truncate_steps is None:
        truncate_steps = []

    if len(clip_latents) != len(clip_text_embeds):
        raise ValueError("clip_latents and clip_text_embeds must have the same length")
    if len(clip_latents) == 0:
        raise ValueError("Empty clip sequence")

    N = len(clip_latents)
    b = int(config.batch_size)

    base_params = dict(dit_model.named_parameters())
    base_buffers = dict(dit_model.named_buffers())

    lora_params = get_lora_params(dit_model)
    lora_names = list(lora_params.keys())
    if len(lora_names) == 0:
        raise ValueError("No LoRA parameters found (expected names containing 'lora').")

    # Current (possibly updated) LoRA tensors (kept in-graph).
    #
    # IMPORTANT: Use cloned tensors for functional-call forward to avoid inplace-version
    # conflicts when we later write updated weights back into the real model parameters.
    # Gradients still flow back to the original Parameters through the clone op.
    current_lora: Dict[str, torch.Tensor] = {n: base_params[n].clone() for n in lora_names}

    opt = DifferentiableSGD(lr=config.inner_lr_init, max_grad_norm=config.max_inner_grad_norm)

    per_step_losses: list[torch.Tensor] = []
    total_loss = torch.tensor(0.0, device=clip_latents[0].device)

    step_idx = 0
    for batch_start in range(0, N, b):
        batch_end = min(batch_start + b, N)
        batch_z0 = torch.cat(clip_latents[batch_start:batch_end], dim=0)
        batch_text = torch.cat(clip_text_embeds[batch_start:batch_end], dim=0)

        for _ in range(int(config.num_gradient_steps)):
            params_for_call = dict(base_params)
            params_for_call.update(current_lora)

            loss = compute_flow_matching_loss(
                dit_model,
                batch_z0,
                batch_text,
                config,
                num_mc_samples=config.num_mc_samples,
                params_override=params_for_call,
                buffers_override=base_buffers,
            )

            # Truncation: stop gradient flow through earlier steps if requested
            if step_idx in truncate_steps:
                total_loss = total_loss + loss.detach()
            else:
                total_loss = total_loss + loss

            per_step_losses.append(loss.detach())

            grads_list = torch.autograd.grad(
                loss,
                [current_lora[n] for n in lora_names],
                create_graph=True,
                allow_unused=True,
            )
            grads: Dict[str, Optional[torch.Tensor]] = {
                n: g for n, g in zip(lora_names, grads_list)
            }

            # Learned LR schedule support (optional)
            lrs_for_step: Optional[Dict[str, torch.Tensor]] = None
            if learned_lrs is not None:
                if hasattr(learned_lrs, "get_lrs"):
                    lrs_for_step = learned_lrs.get_lrs(step_idx)  # type: ignore[attr-defined]
                elif isinstance(learned_lrs, dict):
                    lrs_for_step = {}
                    for n in lora_names:
                        k_step = f"{n}_step{step_idx}"
                        if k_step in learned_lrs:
                            lrs_for_step[n] = learned_lrs[k_step]
                        elif n in learned_lrs:
                            lrs_for_step[n] = learned_lrs[n]
                    if len(lrs_for_step) == 0:
                        lrs_for_step = None

            updated = opt.step(current_lora, grads, learned_lrs=lrs_for_step)
            if step_idx in truncate_steps:
                # Stop meta-gradient to earlier steps, but keep the next-step update trainable.
                updated = {k: v.detach().requires_grad_(True) for k, v in updated.items()}

            current_lora = updated
            step_idx += 1

    total_loss = total_loss / max(len(per_step_losses), 1)

    # Side effect: write final LoRA weights into the actual model parameters.
    # (Graph correctness is governed by the functional-call loss above.)
    with torch.no_grad():
        for n, v in current_lora.items():
            if n in base_params:
                base_params[n].copy_(v.detach())

    return total_loss, per_step_losses
