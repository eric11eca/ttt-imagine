from __future__ import annotations

import os
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig

from videotuna.flow.wanvideo import WanVideoModelFlow
from videotuna.utils.args_utils import VideoMode

from e2e_ttt_video.inner_loop import sample_timesteps


class WanVideoE2ETTTFlow(WanVideoModelFlow):
    """
    WanVideo flow extended with test-time training (TTT) between clips.

    Notes
    - Meta-training (outer loop) is implemented in `e2e_ttt_video/outer_loop.py` as a
      research reference. This Lightning flow focuses on practical sequential inference
      with in-place LoRA updates.
    - LoRA adapters are instantiated via `GenerationBase.instantiate_lora()` exactly
      like other VideoTuna LoRA finetunes.
    """

    def _set_lora_trainable_for_ttt(self) -> list[torch.nn.Parameter]:
        if not getattr(self, "use_lora", False):
            raise ValueError("E2E-TTT inference requires LoRA (`flow.params.lora_config`).")

        # Prefer explicit names recorded by `GenerationBase.instantiate_lora()`.
        lora_names = getattr(self, "lora_params", None)
        if lora_names is None:
            lora_names = set([n for n, _ in self.denoiser.named_parameters() if "lora" in n])

        trainable: list[torch.nn.Parameter] = []
        for n, p in self.denoiser.named_parameters():
            req = n in lora_names
            p.requires_grad_(req)
            if req:
                trainable.append(p)
        return trainable

    def _compute_flow_matching_loss_wan(
        self,
        z_0: torch.Tensor,  # [B, C, T', H', W']
        context: list[torch.Tensor],
        *,
        num_mc_samples: int,
        flow_shift: float,
        logit_normal_mean: float,
        logit_normal_std: float,
    ) -> torch.Tensor:
        B = z_0.shape[0]
        total = torch.tensor(0.0, device=z_0.device)
        for _ in range(int(num_mc_samples)):
            epsilon = torch.randn_like(z_0)
            t = sample_timesteps(
                B,
                z_0.device,
                flow_shift=flow_shift,
                logit_normal_mean=logit_normal_mean,
                logit_normal_std=logit_normal_std,
            )
            t_expand = t.view(B, 1, 1, 1, 1)
            z_t = (1.0 - t_expand) * z_0 + t_expand * epsilon
            v_target = epsilon - z_0

            t_input = t * 1000.0
            v_pred_list = self.denoiser(x=z_t, t=t_input, context=context, seq_len=None)
            v_pred = torch.stack(v_pred_list, dim=0)
            total = total + F.mse_loss(v_pred.float(), v_target.float())

        return total / max(int(num_mc_samples), 1)

    def inference(self, args: DictConfig):
        """
        Override inference to optionally run sequential generation + TTT.
        """
        # Validate size etc
        self._validate_args(args)

        e2e_cfg = args.get("e2e_ttt", None)
        enabled = bool(e2e_cfg.get("enabled", False)) if e2e_cfg is not None else False

        if not enabled:
            return super().inference(args)

        if args.mode != VideoMode.T2V.value:
            raise ValueError("E2E-TTT sequential inference currently supports t2v only.")

        return self.inference_t2v_e2e_ttt(args)

    def inference_t2v_e2e_ttt(self, args: DictConfig):
        """
        Sequentially generate clips for each prompt line and run TTT between clips.
        """
        rank = int(os.getenv("RANK", 0))
        if rank != 0:
            # Keep behaviour consistent with base WanVideo flow (rank0 saves outputs).
            return None

        # Load prompts (each line is treated as a "clip segment")
        prompt_list = self.load_inference_inputs(args.prompt_file, args.mode)
        if len(prompt_list) == 0:
            raise ValueError("No prompts found (empty prompt_file).")

        # E2E-TTT knobs
        e2e_cfg = args.get("e2e_ttt")
        max_clips = int(e2e_cfg.get("num_clips", len(prompt_list)))
        prompt_list = prompt_list[:max_clips]

        # Ensure LoRA params are trainable for TTT updates
        lora_params = self._set_lora_trainable_for_ttt()
        ttt_steps = int(e2e_cfg.get("ttt_steps_per_clip", 1))
        num_mc = int(e2e_cfg.get("num_mc_samples", 1))

        # Inner-loop distribution knobs (fall back to defaults)
        flow_shift = float(e2e_cfg.get("flow_shift", 5.0))
        logit_mean = float(e2e_cfg.get("logit_normal_mean", 0.0))
        logit_std = float(e2e_cfg.get("logit_normal_std", 1.0))

        # Sampling controls (re-use existing Wan sampler)
        frames = args.frames
        size = (args.width, args.height)
        sample_shift = args.time_shift
        sample_solver = args.solver
        sampling_steps = args.num_inference_steps
        guide_scale = args.unconditional_guidance_scale

        videos = []
        for i, prompt in enumerate(prompt_list):
            logger.info(f"[E2E-TTT] Clip {i+1}/{len(prompt_list)} prompt: {prompt[:120]}")

            # Step A: generate clip with current LoRA weights
            video = self.wan_t2v.generate(
                prompt,
                size=size,
                frame_num=frames,
                shift=sample_shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=int(args.seed) + i if args.seed is not None else -1,
                offload_model=self.offload_model,
            )
            videos.append(video.cpu())

            # No need to adapt after final clip
            if i == len(prompt_list) - 1:
                continue

            # Step B/C: encode generated clip and do TTT updates on LoRA weights
            with torch.no_grad():
                z0 = self.wan_t2v.vae.encode([video])[0]  # [C, T', H', W']
                z0 = z0.unsqueeze(0).to(video.device)  # [1, C, T', H', W']

                # Text conditioning (list[tensor] for WanModel)
                if not self.wan_t2v.t5_cpu:
                    self.wan_t2v.text_encoder.model.to(self.wan_t2v.device)
                    context = self.wan_t2v.text_encoder([prompt], self.wan_t2v.device)
                    if self.offload_model:
                        self.wan_t2v.text_encoder.model.cpu()
                else:
                    context = self.wan_t2v.text_encoder([prompt], torch.device("cpu"))
                    context = [t.to(self.wan_t2v.device) for t in context]

            for step in range(ttt_steps):
                # `generate()` may offload the denoiser to CPU; ensure it's on the sampling device.
                self.denoiser.to(self.wan_t2v.device)

                with torch.enable_grad():
                    loss = self._compute_flow_matching_loss_wan(
                        z0,
                        context,
                        num_mc_samples=num_mc,
                        flow_shift=flow_shift,
                        logit_normal_mean=logit_mean,
                        logit_normal_std=logit_std,
                    )
                    grads = torch.autograd.grad(loss, lora_params, allow_unused=True)

                with torch.no_grad():
                    lr = float(e2e_cfg.get("inner_lr_init", 5e-5))
                    max_gn = float(e2e_cfg.get("max_inner_grad_norm", 1.0))
                    for p, g in zip(lora_params, grads):
                        if g is None:
                            continue
                        if max_gn > 0:
                            gn = g.norm()
                            if gn > max_gn:
                                g = g * (max_gn / (gn + 1e-8))
                        p.sub_(lr * g)

                logger.info(f"[E2E-TTT]  TTT step {step+1}/{ttt_steps} loss={loss.item():.6f}")

            # Optional: offload back to CPU for VRAM savings
            if self.offload_model:
                self.denoiser.cpu()
                torch.cuda.empty_cache()

        # Save per-clip outputs
        filenames = [f"clip-{i:04d}" for i in range(len(videos))]
        self.save_videos(torch.stack(videos).unsqueeze(dim=1), args.savedir, filenames, fps=args.savefps)
        return None

