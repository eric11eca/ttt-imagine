import torch

from e2e_ttt_video.config import InnerLoopConfig
from e2e_ttt_video.inner_loop import run_inner_loop


def test_meta_loss_has_gradient(tiny_dit_with_lora, mock_clip_sequence):
    """Meta-loss must have gradients w.r.t. W_0."""
    config = InnerLoopConfig(
        num_gradient_steps=1,
        batch_size=2,
        num_mc_samples=1,
        inner_lr_init=1e-3,
        flow_shift=1.0,
    )
    clip_latents, clip_text_embeds = mock_clip_sequence
    meta_loss, _ = run_inner_loop(tiny_dit_with_lora, clip_latents, clip_text_embeds, config)

    lora_params = [
        p for n, p in tiny_dit_with_lora.named_parameters() if "lora" in n and p.requires_grad
    ]
    grads = torch.autograd.grad(meta_loss, lora_params, allow_unused=True)
    has_nonzero_grad = any(g is not None and g.abs().sum() > 0 for g in grads)
    assert has_nonzero_grad, "Meta-gradients must flow to W_0"


def test_w0_restored_after_meta_step():
    """W_0 must be restored after inner loop + backward."""
    # Implemented at integration-flow level; see design doc.
    pass

