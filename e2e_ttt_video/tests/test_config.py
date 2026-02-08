from omegaconf import OmegaConf

from e2e_ttt_video.config import E2ETTTConfig, InferenceConfig, InnerLoopConfig, OuterLoopConfig


def test_inner_loop_from_cfg_ignores_extra_keys():
    cfg = OmegaConf.create(
        {
            "num_gradient_steps": 2,
            "batch_size": 8,
            "num_mc_samples": 3,
            "inner_lr_init": 1e-4,
            "flow_shift": 2.0,
            "enabled": True,  # extra
        }
    )
    c = InnerLoopConfig.from_cfg(cfg)
    assert c.num_gradient_steps == 2
    assert c.batch_size == 8
    assert c.flow_shift == 2.0


def test_outer_loop_from_cfg_defaults_and_keys():
    cfg = OmegaConf.create({"outer_lr": 2e-5, "truncate_steps": [0, 1]})
    c = OuterLoopConfig.from_cfg(cfg)
    assert c.outer_lr == 2e-5
    assert c.truncate_steps == [0, 1]


def test_inference_from_cfg_parses_known_fields():
    cfg = OmegaConf.create({"num_clips": 4, "height": 256, "width": 256})
    c = InferenceConfig.from_cfg(cfg)
    assert c.num_clips == 4
    assert c.height == 256
    assert c.width == 256


def test_e2e_ttt_config_from_cfg_nested():
    cfg = OmegaConf.create(
        {
            "enabled": True,
            "device": "cpu",
            "inner_loop": {"batch_size": 2, "flow_shift": 1.0},
            "outer_loop": {"outer_lr": 1e-5, "truncate_steps": [0]},
            "inference": {"num_clips": 3, "seed": 123},
            "unknown": 123,
        }
    )
    c = E2ETTTConfig.from_cfg(cfg)
    assert c.enabled is True
    assert c.device == "cpu"
    assert c.inner_loop.batch_size == 2
    assert c.inference.num_clips == 3
    assert c.inference.seed == 123

