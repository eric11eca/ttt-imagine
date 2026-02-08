import torch

from e2e_ttt_video.lora_state import get_lora_params, restore_lora_state, snapshot_lora_state


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(4, 4)
        self.lora_A = torch.nn.Linear(4, 2, bias=False)
        self.lora_B = torch.nn.Linear(2, 4, bias=False)

        # Freeze base, train LoRA
        for n, p in self.named_parameters():
            if "lora" in n:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)


def test_get_lora_params_default_matching():
    m = TinyModel()
    lp = get_lora_params(m)
    assert any("lora" in k for k in lp.keys())
    assert all(isinstance(p, torch.nn.Parameter) for p in lp.values())


def test_snapshot_and_restore_roundtrip():
    m = TinyModel()
    snap = snapshot_lora_state(m)
    assert len(snap) > 0

    # Modify LoRA params
    with torch.no_grad():
        for n, p in get_lora_params(m).items():
            p.add_(torch.randn_like(p) * 0.01)

    restore_lora_state(m, snap)
    snap2 = snapshot_lora_state(m)
    for k in snap:
        assert torch.allclose(snap[k], snap2[k])


def test_get_lora_params_with_explicit_names():
    m = TinyModel()
    names = set([n for n, _ in m.named_parameters() if "lora_A" in n])
    lp = get_lora_params(m, lora_param_names=names)
    assert set(lp.keys()) == names

