from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from .config import E2ETTTConfig
from .inner_loop import compute_flow_matching_loss

logger = logging.getLogger(__name__)


class SequentialVideoGenerator:
    """
    Test-time sequential generation with E2E-TTT.

    This reference implementation is model-agnostic: it assumes the DiT model
    supports a forward call `dit(x=..., t=..., context=...) -> velocity`.
    """

    def __init__(self, dit_model, vae, text_encoder, config: E2ETTTConfig):
        self.dit = dit_model
        self.vae = vae
        self.text_encoder = text_encoder
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch.bfloat16 if config.use_bf16 else torch.float32

    @torch.no_grad()
    def _generate_single_clip(
        self,
        text_embeds: torch.Tensor,
        null_embeds: torch.Tensor,
        num_frames: int,
        seed: int,
    ) -> torch.Tensor:
        """
        Generate one clip latent z_0 via a simple Euler ODE solver.
        """
        icfg = self.config.inference
        T_latent = 1 + (num_frames - 1) // 4
        H_latent = icfg.height // 8
        W_latent = icfg.width // 8
        C_latent = 16

        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        z = torch.randn(
            1,
            C_latent,
            T_latent,
            H_latent,
            W_latent,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        timesteps = np.linspace(1.0, 0.0, int(icfg.num_ode_steps) + 1)
        timesteps = (
            icfg.flow_shift
            * timesteps
            / (1.0 + (icfg.flow_shift - 1.0) * timesteps)
        )
        timesteps = torch.tensor(timesteps, device=self.device, dtype=self.dtype)

        for i in range(int(icfg.num_ode_steps)):
            t_curr = timesteps[i]
            dt = timesteps[i + 1] - t_curr
            t_input = t_curr.unsqueeze(0) * 1000.0

            v_uncond = self.dit(x=z, t=t_input, context=null_embeds)
            v_cond = self.dit(x=z, t=t_input, context=text_embeds)
            v_guided = v_uncond + icfg.guidance_scale * (v_cond - v_uncond)
            z = z + dt * v_guided

        return z

    def _ttt_encode_clip(self, z_0: torch.Tensor, text_embeds: torch.Tensor) -> None:
        """
        Test-time TTT update: compute flow-matching loss on z_0 and update LoRA params.
        """
        icfg = self.config.inner_loop

        # Identify LoRA params (by name convention) and ensure they are trainable.
        lora_params = {
            n: p for n, p in self.dit.named_parameters() if "lora" in n and p.requires_grad
        }
        if len(lora_params) == 0:
            raise ValueError("No trainable LoRA params found for TTT.")

        for step in range(int(icfg.num_gradient_steps)):
            with torch.enable_grad():
                loss = compute_flow_matching_loss(
                    self.dit,
                    z_0,
                    text_embeds,
                    icfg,
                    num_mc_samples=icfg.num_mc_samples,
                )
                grads = torch.autograd.grad(
                    loss,
                    list(lora_params.values()),
                    allow_unused=True,
                )

            with torch.no_grad():
                for (name, param), grad in zip(lora_params.items(), grads):
                    if grad is None:
                        continue
                    grad_norm = grad.norm()
                    if icfg.max_inner_grad_norm > 0 and grad_norm > icfg.max_inner_grad_norm:
                        grad = grad * (icfg.max_inner_grad_norm / (grad_norm + 1e-8))
                    param.sub_(icfg.inner_lr_init * grad)

            logger.debug("TTT step %d loss=%.6f", step, loss.item())

    def generate_long_video(
        self,
        text_segments: list[str],
        save_dir: Optional[str] = None,
    ) -> list[torch.Tensor]:
        """
        Generate a list of clips (pixel space tensors) from a list of text segments.
        """
        icfg = self.config.inference
        N = len(text_segments)

        with torch.no_grad():
            null_embeds = self.text_encoder([""]).to(self.device, dtype=self.dtype)

        all_clips: list[torch.Tensor] = []

        logger.info("Generating %d clips sequentially with E2E-TTT", N)
        for i, text in enumerate(text_segments):
            with torch.no_grad():
                text_embeds = self.text_encoder([text]).to(self.device, dtype=self.dtype)

            z_0 = self._generate_single_clip(
                text_embeds=text_embeds,
                null_embeds=null_embeds,
                num_frames=int(icfg.num_frames_per_clip),
                seed=int(icfg.seed) + i,
            )

            with torch.no_grad():
                z_decoded = z_0 / getattr(self.vae.config, "scaling_factor", 1.0)
                video = self.vae.decode(z_decoded)
                video = ((video + 1.0) / 2.0).clamp(0, 1)
                clip_pixels = video.squeeze(0).permute(1, 0, 2, 3).cpu()
            all_clips.append(clip_pixels)

            if i < N - 1:
                self._ttt_encode_clip(z_0, text_embeds)

            if save_dir is not None:
                import os

                os.makedirs(save_dir, exist_ok=True)
                torch.save(clip_pixels, os.path.join(save_dir, f"clip_{i:03d}.pt"))

        return all_clips

