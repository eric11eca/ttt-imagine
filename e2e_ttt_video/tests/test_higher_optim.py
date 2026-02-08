import torch

from e2e_ttt_video.higher_optim import (
    DifferentiableAdamW,
    DifferentiableMuonClip,
    DifferentiableSGD,
    MetaLearnedLRSchedule,
)


def test_differentiable_sgd_updates():
    params = {"w": torch.randn(4, 4, requires_grad=True)}
    loss = (params["w"] ** 2).sum()
    grads_list = torch.autograd.grad(loss, [params["w"]], create_graph=True)
    grads = {"w": grads_list[0]}

    sgd = DifferentiableSGD(lr=0.1)
    updated = sgd.step(params, grads)

    assert "w" in updated
    assert not torch.allclose(updated["w"], params["w"])


def test_differentiable_sgd_preserves_graph():
    w = torch.randn(4, 4, requires_grad=True)
    loss = (w**2).sum()
    grad = torch.autograd.grad(loss, [w], create_graph=True)[0]

    sgd = DifferentiableSGD(lr=0.1)
    updated = sgd.step({"w": w}, {"w": grad})

    outer_loss = (updated["w"] ** 2).sum()
    outer_grad = torch.autograd.grad(outer_loss, [w])
    assert outer_grad[0] is not None


def test_meta_learned_lr_schedule():
    names = ["lora_A.weight", "lora_B.weight"]
    schedule = MetaLearnedLRSchedule(names, num_inner_steps=3, init_lr=5e-5)
    lrs = schedule.get_lrs(step_index=0)
    assert len(lrs) == 2
    for name in names:
        assert name in lrs
        assert lrs[name].item() > 0


def test_meta_learned_lr_is_trainable():
    names = ["lora_A.weight"]
    schedule = MetaLearnedLRSchedule(names, num_inner_steps=2, init_lr=1e-4)
    trainable = [p for p in schedule.parameters() if p.requires_grad]
    assert len(trainable) == 2


def test_differentiable_adamw_updates_and_preserves_graph():
    w = torch.randn(4, 4, requires_grad=True)
    loss = (w**2).sum()
    grad = torch.autograd.grad(loss, [w], create_graph=True)[0]

    opt = DifferentiableAdamW(lr=0.1, weight_decay=0.01)
    updated = opt.step({"w": w}, {"w": grad})

    assert "w" in updated
    assert not torch.allclose(updated["w"], w)

    outer_loss = (updated["w"] ** 2).sum()
    outer_grad = torch.autograd.grad(outer_loss, [w], allow_unused=True)[0]
    assert outer_grad is not None


def test_differentiable_muonclip_updates_and_qk_clip_scales():
    # Two matrices with query/key-like names
    wq = torch.randn(4, 4, requires_grad=True)
    wk = torch.randn(4, 4, requires_grad=True)
    params = {"attn.q_proj.weight": wq, "attn.k_proj.weight": wk}
    loss = (wq**2).sum() + (wk**2).sum()
    gq, gk = torch.autograd.grad(loss, [wq, wk], create_graph=True)

    opt = DifferentiableMuonClip(
        lr_muon=0.05,
        lr_adamw=0.05,
        weight_decay=0.0,
        qk_clip_threshold=1.0,
        qk_clip_alpha=0.5,
    )

    updated = opt.step(params, {"attn.q_proj.weight": gq, "attn.k_proj.weight": gk}, qk_clip_max_logit=torch.tensor(10.0))
    assert not torch.allclose(updated["attn.q_proj.weight"], wq)
    assert not torch.allclose(updated["attn.k_proj.weight"], wk)

    # Graph should still connect to originals
    outer_loss = updated["attn.q_proj.weight"].sum() + updated["attn.k_proj.weight"].sum()
    ogq, ogk = torch.autograd.grad(outer_loss, [wq, wk], allow_unused=True)
    assert ogq is not None and ogk is not None

