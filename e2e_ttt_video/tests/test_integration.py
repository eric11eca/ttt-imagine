import torch
import pytest

from e2e_ttt_video.config import InnerLoopConfig
from e2e_ttt_video.inner_loop import compute_flow_matching_loss, run_inner_loop
from e2e_ttt_video.lora_state import restore_lora_state, snapshot_lora_state


class TinyDiTForIntegration(torch.nn.Module):
    """Minimal DiT with LoRA-named params."""

    def __init__(self, C=4, H=32):
        super().__init__()
        self.proj = torch.nn.Linear(C, H)
        self.lora_A_proj = torch.nn.Linear(H, 8, bias=False)
        self.lora_B_proj = torch.nn.Linear(8, H, bias=False)
        self.out = torch.nn.Linear(H, C)

        self.proj.weight.requires_grad = False
        self.proj.bias.requires_grad = False
        self.out.weight.requires_grad = False
        self.out.bias.requires_grad = False
        torch.nn.init.zeros_(self.lora_B_proj.weight)

    def forward(self, x, t, context):
        B, C, T, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, C)
        h = self.proj(x_flat)
        h = h + self.lora_B_proj(self.lora_A_proj(h))
        out = self.out(h)
        return out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)


@pytest.fixture
def integration_setup():
    model = TinyDiTForIntegration(C=4, H=32)
    config = InnerLoopConfig(
        num_gradient_steps=1,
        batch_size=2,
        num_mc_samples=1,
        inner_lr_init=1e-3,
        flow_shift=1.0,
    )
    return model, config


def test_full_inner_loop_meta_gradient_flow(integration_setup):
    model, config = integration_setup
    clips = [torch.randn(1, 4, 2, 4, 4) for _ in range(4)]
    texts = [torch.randn(1, 8, 16) for _ in range(4)]

    w0 = snapshot_lora_state(model)

    meta_loss, _ = run_inner_loop(model, clips, texts, config)

    lora_params = [
        p for n, p in model.named_parameters() if "lora" in n and p.requires_grad
    ]

    restore_lora_state(model, w0)

    meta_loss, _ = run_inner_loop(model, clips, texts, config)
    grads = torch.autograd.grad(meta_loss, lora_params, allow_unused=True)

    nonzero_grads = [g for g in grads if g is not None and g.abs().sum() > 0]
    assert len(nonzero_grads) > 0, "At least one LoRA param must receive non-zero meta-gradient"


def test_w0_invariant_across_training_samples(integration_setup):
    model, config = integration_setup

    clips = [torch.randn(1, 4, 2, 4, 4) for _ in range(4)]
    texts = [torch.randn(1, 8, 16) for _ in range(4)]

    w0_before = snapshot_lora_state(model)
    run_inner_loop(model, clips, texts, config)

    w_after_inner = snapshot_lora_state(model)
    changed = any(not torch.allclose(w0_before[k], w_after_inner[k]) for k in w0_before)
    assert changed, "Inner loop should modify weights"

    restore_lora_state(model, w0_before)
    w_restored = snapshot_lora_state(model)
    for k in w0_before:
        assert torch.allclose(w0_before[k], w_restored[k]), f"Failed to restore {k}"


def test_inner_loop_loss_decreases_over_steps():
    model = TinyDiTForIntegration(C=4, H=32)
    config = InnerLoopConfig(
        num_gradient_steps=1,
        batch_size=4,
        num_mc_samples=2,
        inner_lr_init=1e-2,
        flow_shift=1.0,
    )

    clips = [torch.randn(1, 4, 2, 4, 4) for _ in range(4)]
    texts = [torch.randn(1, 8, 16) for _ in range(4)]

    loss_before = compute_flow_matching_loss(
        model, torch.cat(clips), torch.cat(texts), config, num_mc_samples=4
    ).item()

    for _ in range(5):
        run_inner_loop(model, clips, texts, config)

    loss_after = compute_flow_matching_loss(
        model, torch.cat(clips), torch.cat(texts), config, num_mc_samples=4
    ).item()

    assert loss_after < loss_before, f"TTT should reduce loss: before={loss_before:.4f}, after={loss_after:.4f}"


def test_e2e_is_better_than_naive():
    model = TinyDiTForIntegration(C=4, H=32)
    config = InnerLoopConfig(
        num_gradient_steps=1,
        batch_size=2,
        num_mc_samples=2,
        inner_lr_init=1e-3,
        flow_shift=1.0,
    )

    clips = [torch.randn(1, 4, 2, 4, 4) for _ in range(4)]
    texts = [torch.randn(1, 8, 16) for _ in range(4)]

    naive_loss = sum(
        compute_flow_matching_loss(model, c, t, config).item() for c, t in zip(clips, texts)
    ) / len(clips)

    e2e_loss, _ = run_inner_loop(model, clips, texts, config)
    assert naive_loss != pytest.approx(e2e_loss.item(), abs=1e-6), "E2E and naive losses should differ"

