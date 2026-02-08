from __future__ import annotations

import torch
import pytest


class TinyDiTWithLoRA(torch.nn.Module):
    """
    Minimal DiT-like model with LoRA-named trainable parameters.
    """

    def __init__(self, latent_channels: int = 4, hidden: int = 32):
        super().__init__()
        self.proj_in = torch.nn.Linear(latent_channels, hidden)
        self.lora_A = torch.nn.Linear(hidden, 8, bias=False)
        self.lora_B = torch.nn.Linear(8, hidden, bias=False)
        self.proj_out = torch.nn.Linear(hidden, latent_channels)

        # Freeze non-LoRA
        for n, p in self.named_parameters():
            if "lora" in n:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

        # Make updates visible in tests
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x, t, context):
        B, C, T, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, C)
        h = self.proj_in(x_flat)
        h = h + self.lora_B(self.lora_A(h))
        out = self.proj_out(h)
        return out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)


@pytest.fixture
def tiny_dit_with_lora():
    return TinyDiTWithLoRA(latent_channels=4, hidden=32)


@pytest.fixture
def mock_clip_sequence():
    clip_latents = [torch.randn(1, 4, 2, 4, 4) for _ in range(4)]
    clip_text_embeds = [torch.randn(1, 8, 16) for _ in range(4)]
    return clip_latents, clip_text_embeds


class MockVAE(torch.nn.Module):
    class _Cfg:
        scaling_factor = 1.0

    def __init__(self):
        super().__init__()
        self.config = MockVAE._Cfg()

    def decode(self, z):
        # z: [B, C, T, H, W] -> video: [B, 3, T, H, W]
        B, C, T, H, W = z.shape
        x = z.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1, 1)
        return torch.tanh(x)


class MockTextEncoder(torch.nn.Module):
    def __init__(self, L: int = 8, D: int = 16):
        super().__init__()
        self.L = L
        self.D = D

    def forward(self, texts):
        # texts: list[str] -> [B, L, D]
        B = len(texts)
        return torch.randn(B, self.L, self.D)


@pytest.fixture
def mock_vae():
    return MockVAE()


@pytest.fixture
def mock_text_encoder():
    return MockTextEncoder(L=8, D=16)

