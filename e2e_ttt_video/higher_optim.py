from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _per_tensor_clip(grad: torch.Tensor, max_grad_norm: float) -> torch.Tensor:
    if max_grad_norm <= 0:
        return grad
    # Detach norm so clipping decision doesn't backprop through norm.
    gnorm = grad.detach().norm()
    if gnorm > max_grad_norm:
        grad = grad * (max_grad_norm / (gnorm + 1e-8))
    return grad


class DifferentiableSGD:
    """
    Manual differentiable SGD for inner-loop updates.

    Instead of in-place optimizer.step() (which can break graphs), we update using
    tensor operations:

        w_new = w_old - lr * grad

    This preserves gradient flow from an outer/meta loss back to the initial
    parameters.
    """

    def __init__(self, lr: float = 1e-4, max_grad_norm: float = 1.0):
        self.lr = float(lr)
        self.max_grad_norm = float(max_grad_norm)

    def step(
        self,
        params: Dict[str, torch.Tensor],
        grads: Dict[str, Optional[torch.Tensor]],
        learned_lrs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        updated: Dict[str, torch.Tensor] = {}
        for name, param in params.items():
            grad = grads.get(name)
            if grad is None:
                updated[name] = param
                continue

            # Per-parameter gradient clipping (stability)
            if self.max_grad_norm > 0:
                grad_norm = grad.detach().norm()
                if grad_norm > self.max_grad_norm:
                    grad = grad * (self.max_grad_norm / (grad_norm + 1e-8))

            lr = self.lr
            if learned_lrs is not None and name in learned_lrs:
                lr = learned_lrs[name]
            updated[name] = param - lr * grad

        return updated


class DifferentiableAdamW:
    """
    Manual differentiable AdamW (decoupled weight decay).

    This is intended for differentiating through inner-loop updates.
    It stores optimizer state (m, v, step) inside the instance.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
    ):
        self.lr = float(lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = float(max_grad_norm)

        self._step: int = 0
        self._m: Dict[str, torch.Tensor] = {}
        self._v: Dict[str, torch.Tensor] = {}

    def step(
        self,
        params: Dict[str, torch.Tensor],
        grads: Dict[str, Optional[torch.Tensor]],
        learned_lrs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        self._step += 1
        t = self._step

        updated: Dict[str, torch.Tensor] = {}
        for name, param in params.items():
            grad = grads.get(name)
            if grad is None:
                updated[name] = param
                continue

            grad = _per_tensor_clip(grad, self.max_grad_norm)

            # State init
            m = self._m.get(name)
            v = self._v.get(name)
            if m is None:
                m = torch.zeros_like(param)
            if v is None:
                v = torch.zeros_like(param)

            # Adam moments (in-graph)
            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad * grad)

            # Bias correction
            m_hat = m / (1.0 - (self.beta1**t))
            v_hat = v / (1.0 - (self.beta2**t))

            lr = self.lr
            if learned_lrs is not None and name in learned_lrs:
                lr = learned_lrs[name]

            # Decoupled weight decay (AdamW)
            if self.weight_decay != 0.0:
                param = param - lr * self.weight_decay * param

            param = param - lr * m_hat / (torch.sqrt(v_hat) + self.eps)

            self._m[name] = m
            self._v[name] = v
            updated[name] = param

        return updated


def _newton_schulz_orthogonalize(
    G: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Newton-Schulz matrix iteration used by Muon to approximately orthogonalize an update.

    Reference implementation from Keller Jordan's Muon blog post (Dec 2024):
    https://kellerjordan.github.io/posts/muon/
    """
    if G.ndim != 2:
        raise ValueError(f"Expected 2D tensor for Muon update, got shape {tuple(G.shape)}")

    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.float32)
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    for _ in range(int(steps)):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + (B @ X)

    if transposed:
        X = X.T
    return X.to(dtype=G.dtype)


class DifferentiableMuon:
    """
    Differentiable Muon optimizer for 2D parameters (hidden-layer matrices).

    Muon = SGD-momentum (optionally Nesterov) + Newton-Schulz orthogonalization of the update.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
    ):
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.ns_steps = int(ns_steps)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = float(max_grad_norm)

        self._buf: Dict[str, torch.Tensor] = {}

    def step(
        self,
        params: Dict[str, torch.Tensor],
        grads: Dict[str, Optional[torch.Tensor]],
        learned_lrs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        updated: Dict[str, torch.Tensor] = {}
        beta = self.momentum

        for name, param in params.items():
            grad = grads.get(name)
            if grad is None:
                updated[name] = param
                continue

            if grad.ndim != 2:
                raise ValueError(f"Muon only supports 2D params; got {name} with grad shape {tuple(grad.shape)}")

            grad = _per_tensor_clip(grad, self.max_grad_norm)

            buf = self._buf.get(name)
            if buf is None:
                buf = torch.zeros_like(param)

            # Momentum buffer update (in-graph)
            buf = beta * buf + (1.0 - beta) * grad

            if self.nesterov:
                # Equivalent to: (1-beta)*grad + beta*buf
                update = grad + beta * (buf - grad)
            else:
                update = buf

            update = _newton_schulz_orthogonalize(update, steps=self.ns_steps)
            scale_factor = float(max(1.0, grad.size(-2) / grad.size(-1)) ** 0.5)
            update = update * scale_factor

            lr = self.lr
            if learned_lrs is not None and name in learned_lrs:
                lr = learned_lrs[name]

            if self.weight_decay != 0.0:
                param = param - lr * self.weight_decay * param

            param = param - lr * update

            self._buf[name] = buf
            updated[name] = param

        return updated


class DifferentiableMuonClip:
    """
    Differentiable MuonClip (Moonshot / Kimi K2) inner-loop optimizer.

    - Uses Muon for 2D parameters
    - Uses AdamW for non-2D parameters
    - Optionally applies qk-clip rescaling to Q/K projection matrices after the update:

        η = min(t / max_{i,j}(q_i^T k_j), 1)
        W_q <- η^α W_q
        W_k <- η^(1-α) W_k

    Note: computing the max attention logit requires runtime model stats.
    This implementation accepts `qk_clip_max_logit` (a scalar) at `step()` time.
    """

    def __init__(
        self,
        *,
        lr_muon: float = 1e-4,
        lr_adamw: float = 1e-4,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        # qk-clip controls
        qk_clip_threshold: Optional[float] = None,
        qk_clip_alpha: float = 0.5,
        q_name_patterns: Sequence[str] = ("q_proj", "query", ".q.", ".q_"),
        k_name_patterns: Sequence[str] = ("k_proj", "key", ".k.", ".k_"),
    ):
        self.muon = DifferentiableMuon(
            lr=lr_muon,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_steps=muon_ns_steps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        self.adamw = DifferentiableAdamW(
            lr=lr_adamw,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )

        self.qk_clip_threshold = qk_clip_threshold
        self.qk_clip_alpha = float(qk_clip_alpha)
        self.q_name_patterns = tuple(q_name_patterns)
        self.k_name_patterns = tuple(k_name_patterns)

    def _is_query(self, name: str) -> bool:
        n = name.lower()
        return any(pat in n for pat in self.q_name_patterns)

    def _is_key(self, name: str) -> bool:
        n = name.lower()
        return any(pat in n for pat in self.k_name_patterns)

    def step(
        self,
        params: Dict[str, torch.Tensor],
        grads: Dict[str, Optional[torch.Tensor]],
        learned_lrs: Optional[Dict[str, torch.Tensor]] = None,
        *,
        qk_clip_max_logit: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Split params by shape
        params_2d = {n: p for n, p in params.items() if p.ndim == 2}
        params_other = {n: p for n, p in params.items() if p.ndim != 2}

        grads_2d = {n: grads.get(n) for n in params_2d.keys()}
        grads_other = {n: grads.get(n) for n in params_other.keys()}

        learned_2d = None
        learned_other = None
        if learned_lrs is not None:
            learned_2d = {n: learned_lrs[n] for n in params_2d.keys() if n in learned_lrs}
            learned_other = {n: learned_lrs[n] for n in params_other.keys() if n in learned_lrs}

        updated = {}
        updated.update(self.muon.step(params_2d, grads_2d, learned_lrs=learned_2d))
        updated.update(self.adamw.step(params_other, grads_other, learned_lrs=learned_other))

        # Optional qk-clip post-processing
        if self.qk_clip_threshold is not None and qk_clip_max_logit is not None:
            max_logit = float(qk_clip_max_logit.detach().item())
            if max_logit > 0:
                eta = min(self.qk_clip_threshold / max_logit, 1.0)
                if eta < 1.0:
                    alpha = self.qk_clip_alpha
                    scale_q = eta**alpha
                    scale_k = eta ** (1.0 - alpha)
                    for n in list(updated.keys()):
                        if self._is_query(n):
                            updated[n] = updated[n] * scale_q
                        elif self._is_key(n):
                            updated[n] = updated[n] * scale_k

        return updated


class MetaLearnedLRSchedule(nn.Module):
    """
    PERK-style meta-learned per-parameter per-step learning rates.

    Uses a softplus parameterization to ensure positivity.
    """

    def __init__(
        self,
        lora_param_names: list[str],
        num_inner_steps: int,
        init_lr: float = 5e-5,
    ):
        super().__init__()
        self.param_names = list(lora_param_names)
        self.num_steps = int(num_inner_steps)

        # raw_lrs: key -> unconstrained scalar
        self.raw_lrs = nn.ParameterDict()

        # Initialize so that softplus(raw) ~= init_lr
        init_lr_t = torch.tensor(float(init_lr))
        init_raw = torch.log(torch.exp(init_lr_t) - 1.0)

        for name in self.param_names:
            safe_name = name.replace(".", "_")
            for step in range(self.num_steps):
                key = f"{safe_name}__step{step}"
                self.raw_lrs[key] = nn.Parameter(init_raw.clone())

    def get_lrs(self, step_index: int) -> Dict[str, torch.Tensor]:
        """
        Return a mapping: param_name -> lr_scalar for the given step.
        """
        step_index = int(step_index)
        lrs: Dict[str, torch.Tensor] = {}
        for name in self.param_names:
            safe_name = name.replace(".", "_")
            key = f"{safe_name}__step{step_index}"
            raw = self.raw_lrs.get(key)
            if raw is None:
                lrs[name] = torch.tensor(5e-5, device=next(self.parameters()).device)
            else:
                lrs[name] = F.softplus(raw)
        return lrs

