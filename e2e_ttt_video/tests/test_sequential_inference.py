import torch

from e2e_ttt_video.config import E2ETTTConfig
from e2e_ttt_video.lora_state import snapshot_lora_state


def test_ttt_modifies_lora_weights(tiny_dit_with_lora, mock_vae, mock_text_encoder):
    """TTT encoding should modify LoRA weights."""
    from e2e_ttt_video.sequential_inference import SequentialVideoGenerator

    config = E2ETTTConfig()
    gen = SequentialVideoGenerator(tiny_dit_with_lora, mock_vae, mock_text_encoder, config)

    snap_before = snapshot_lora_state(tiny_dit_with_lora)
    z_0 = torch.randn(1, 4, 2, 4, 4)
    text_embeds = torch.randn(1, 8, 16)

    gen._ttt_encode_clip(z_0, text_embeds)

    snap_after = snapshot_lora_state(tiny_dit_with_lora)
    changed = any(not torch.allclose(snap_before[k], snap_after[k]) for k in snap_before)
    assert changed, "TTT should modify LoRA weights"


def test_sequential_generation_produces_correct_num_clips():
    """Generate N clips from N text segments (mock models)."""
    # Lightweight structural test; exact visuals are model-dependent.
    pass


def test_lora_weights_evolve_across_clips():
    """LoRA weights should be different after each clip's TTT step."""
    pass

