from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from .config import E2ETTTConfig
from .higher_optim import MetaLearnedLRSchedule
from .inner_loop import run_inner_loop
from .lora_state import count_lora_params, get_lora_params, restore_lora_state, snapshot_lora_state

logger = logging.getLogger(__name__)


class E2ETTTMetaTrainer:
    """
    Reference outer-loop trainer for E2E-TTT.

    This is intentionally lightweight and primarily intended for research prototyping.
    In this repository, the recommended end-user execution path is to integrate E2E-TTT
    into a VideoTuna Lightning flow (see the design doc).
    """

    def __init__(
        self,
        dit_model: nn.Module,
        vae: nn.Module,
        text_encoder: nn.Module,
        train_dataset,
        val_dataset=None,
        config: Optional[E2ETTTConfig] = None,
    ):
        self.config = config or E2ETTTConfig()
        self.device = torch.device(self.config.device)
        self.dtype = torch.bfloat16 if self.config.use_bf16 else torch.float32

        self.dit = dit_model.to(self.device)
        self.vae = vae.to(self.device).eval()
        self.text_encoder = text_encoder.to(self.device).eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        # Meta-learned LR schedule (optional)
        self.lr_schedule: Optional[MetaLearnedLRSchedule] = None
        if self.config.inner_loop.meta_learn_lr:
            lora_names = list(get_lora_params(self.dit).keys())
            # Upper bound used for allocating (param,step) lrs; can be over-allocated safely.
            max_inner_steps = 256
            self.lr_schedule = MetaLearnedLRSchedule(
                lora_param_names=lora_names,
                num_inner_steps=max_inner_steps,
                init_lr=self.config.inner_loop.inner_lr_init,
            ).to(self.device)

        # Outer optimizer: only trainable parameters (typically LoRA + optionally lr_schedule)
        outer_params = [p for p in self.dit.parameters() if p.requires_grad]
        if self.lr_schedule is not None:
            outer_params += list(self.lr_schedule.parameters())

        self.outer_optimizer = torch.optim.AdamW(
            outer_params,
            lr=self.config.outer_loop.outer_lr,
            weight_decay=self.config.outer_loop.outer_weight_decay,
            betas=(0.9, 0.999),
        )

        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
        self.val_loader = (
            DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
            if val_dataset is not None
            else None
        )

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("E2E-TTT MetaTrainer initialized")
        logger.info("  LoRA params: %s", f"{count_lora_params(self.dit):,}")

    def _encode_texts(self, texts: list[str]) -> list[torch.Tensor]:
        with torch.no_grad():
            out = []
            for t in texts:
                emb = self.text_encoder([t])
                if isinstance(emb, list):
                    # Some encoders return list[tensor] per sample; normalize to tensor.
                    emb = emb[0].unsqueeze(0)
                out.append(emb.to(self.device, dtype=self.dtype))
            return out

    def _meta_train_step(self, sample: dict) -> tuple[torch.Tensor, dict]:
        clip_latents = [z.to(self.device, dtype=self.dtype) for z in sample["clip_latents"]]
        clip_texts = sample["clip_texts"]
        clip_text_embeds = self._encode_texts(clip_texts)

        w0_snapshot = snapshot_lora_state(self.dit)

        learned_lrs = None
        if self.lr_schedule is not None:
            learned_lrs = self.lr_schedule

        meta_loss, _ = run_inner_loop(
            dit_model=self.dit,
            clip_latents=clip_latents,
            clip_text_embeds=clip_text_embeds,
            config=self.config.inner_loop,
            learned_lrs=learned_lrs,
            truncate_steps=self.config.outer_loop.truncate_steps,
        )

        return meta_loss, w0_snapshot

    def train(self) -> None:
        cfg = self.config
        logger.info("Starting meta-training for %d epochs", cfg.outer_loop.num_epochs)

        global_step = 0
        for epoch in range(int(cfg.outer_loop.num_epochs)):
            self.dit.train()
            epoch_loss = 0.0
            num_samples = 0

            for batch in self.train_loader:
                # DataLoader(batch_size=1) wraps nested lists in length-1 containers.
                sample = batch
                if isinstance(sample.get("clip_latents"), list) and len(sample["clip_latents"]) == 1 and isinstance(sample["clip_latents"][0], list):
                    sample = {
                        "clip_latents": [z.squeeze(0) for z in sample["clip_latents"][0]],
                        "clip_texts": [t for t in sample["clip_texts"][0]],
                    }

                with autocast(dtype=self.dtype, enabled=True):
                    meta_loss, w0_snapshot = self._meta_train_step(sample)
                    meta_loss = meta_loss / max(int(cfg.outer_loop.gradient_accumulation_steps), 1)

                meta_loss.backward()

                # Restore W_0 after backward so gradients are computed correctly.
                restore_lora_state(self.dit, w0_snapshot)

                num_samples += 1
                epoch_loss += meta_loss.item() * max(int(cfg.outer_loop.gradient_accumulation_steps), 1)

                if num_samples % max(int(cfg.outer_loop.gradient_accumulation_steps), 1) == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.dit.parameters() if p.requires_grad],
                        float(cfg.outer_loop.max_outer_grad_norm),
                    )
                    self.outer_optimizer.step()
                    self.outer_optimizer.zero_grad(set_to_none=True)
                    global_step += 1

            logger.info(
                "Epoch %d complete | avg meta-loss=%.6f",
                epoch + 1,
                epoch_loss / max(num_samples, 1),
            )
            self._save_checkpoint(step=global_step, epoch=epoch + 1)

    def _save_checkpoint(self, step: int, epoch: int) -> None:
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        state = {
            "lora_w0": snapshot_lora_state(self.dit),
            "outer_optimizer": self.outer_optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
        }
        if self.lr_schedule is not None:
            state["lr_schedule"] = self.lr_schedule.state_dict()

        path = ckpt_dir / f"meta_ckpt_epoch{epoch}.pt"
        torch.save(state, path)
        logger.info("Saved meta-checkpoint: %s", str(path))

