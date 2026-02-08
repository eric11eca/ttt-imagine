import torch
import pytest

from e2e_ttt_video.config import InnerLoopConfig
from e2e_ttt_video.inner_loop import (
    sample_timesteps,
    compute_flow_matching_loss,
    inner_loop_step,
    run_inner_loop,
)


class TinyDiT(torch.nn.Module):
    """Minimal DiT-like model for testing."""

    def __init__(self, latent_channels=4, hidden=32):
        super().__init__()
        self.proj_in = torch.nn.Linear(latent_channels, hidden)
        self.lora_A = torch.nn.Linear(hidden, 8, bias=False)
        self.lora_B = torch.nn.Linear(8, hidden, bias=False)
        self.proj_out = torch.nn.Linear(hidden, latent_channels)

        # Freeze non-LoRA
        self.proj_in.weight.requires_grad = False
        self.proj_in.bias.requires_grad = False
        self.proj_out.weight.requires_grad = False
        self.proj_out.bias.requires_grad = False

    def forward(self, x, t, context):
        B, C, T, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, C)
        h = self.proj_in(x_flat)
        h = h + self.lora_B(self.lora_A(h))
        out = self.proj_out(h)
        return out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)


@pytest.fixture
def tiny_dit():
    return TinyDiT(latent_channels=4, hidden=32)


@pytest.fixture
def config():
    return InnerLoopConfig(
        num_gradient_steps=1,
        batch_size=2,
        num_mc_samples=2,
        inner_lr_init=1e-3,
        flow_shift=1.0,
    )


def make_clip_latent(C=4, T=2, H=4, W=4):
    return torch.randn(1, C, T, H, W)


def make_text_embed(L=8, D=16):
    return torch.randn(1, L, D)


def test_sample_timesteps_shape():
    t = sample_timesteps(8, torch.device("cpu"), flow_shift=5.0)
    assert t.shape == (8,)
    assert (t > 0).all() and (t < 1).all()


def test_sample_timesteps_flow_shift():
    """Higher flow shift should push timesteps toward higher values."""
    t_low = sample_timesteps(1000, torch.device("cpu"), flow_shift=1.0)
    t_high = sample_timesteps(1000, torch.device("cpu"), flow_shift=10.0)
    assert t_high.mean() > t_low.mean()


def test_compute_flow_matching_loss(tiny_dit, config):
    z_0 = make_clip_latent()
    text = make_text_embed()
    loss = compute_flow_matching_loss(tiny_dit, z_0, text, config, num_mc_samples=2)
    assert loss.shape == ()
    assert loss.item() > 0
    assert loss.requires_grad


def test_inner_loop_step_updates_params(tiny_dit, config):
    z_0 = make_clip_latent()
    text = make_text_embed()

    orig = {n: p.data.clone() for n, p in tiny_dit.named_parameters() if "lora" in n}
    loss, updated = inner_loop_step(tiny_dit, z_0, text, config)

    for name in orig:
        assert not torch.allclose(updated[name], orig[name]), f"Parameter {name} was not updated"


def test_inner_loop_step_preserves_grad_graph(tiny_dit, config):
    z_0 = make_clip_latent()
    text = make_text_embed()
    loss, _ = inner_loop_step(tiny_dit, z_0, text, config)
    assert loss.requires_grad


def test_run_inner_loop_reduces_loss(tiny_dit, config):
    clips = [make_clip_latent() for _ in range(4)]
    texts = [make_text_embed() for _ in range(4)]
    total_loss, step_losses = run_inner_loop(tiny_dit, clips, texts, config)
    assert total_loss.item() > 0
    assert len(step_losses) >= 1


def test_run_inner_loop_truncation(tiny_dit, config):
    clips = [make_clip_latent() for _ in range(4)]
    texts = [make_text_embed() for _ in range(4)]
    total_loss, _ = run_inner_loop(tiny_dit, clips, texts, config, truncate_steps=[0])
    assert total_loss.item() > 0

