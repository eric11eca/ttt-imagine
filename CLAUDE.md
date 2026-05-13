# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**VideoTuna** is a unified Python framework for text-to-video (T2V), image-to-video (I2V), text-to-image (T2I), and video-to-video (V2V) generation. It integrates 8+ state-of-the-art models (Wan2.1, HunyuanVideo, CogVideoX, Flux, Open-Sora, StepVideo, Mochi, VideoCrafter) under a single config-driven training and inference interface with LoRA/full fine-tuning support.

## Commands

### Setup
```bash
conda create -n videotuna python=3.10 -y && conda activate videotuna
pip install poetry
poetry install
poetry run install-deepspeed   # builds DeepSpeed with CPU Adam (needs CUDA 12.1)
poetry run install-flash-attn  # builds flash-attn 2.7.3 (needs cuda-nvcc=12.1)
```

### Development
```bash
poetry run test              # Run pytest
poetry run coverage-report   # Coverage report
poetry run format            # black + isort
poetry run format-check      # Check formatting without modifying
poetry run lint              # Ruff linting
poetry run type-check        # MyPy
```

### Running a single test
```bash
poetry run pytest tests/test_<name>.py -v
poetry run pytest tests/test_<name>.py::TestClass::test_method -v
```

### Inference
```bash
poetry run inference-wanvideo-t2v-720p
poetry run inference-hunyuan-t2v
poetry run inference-cogvideo-t2v-diffusers
# Full list of 30+ commands defined in pyproject.toml [tool.poetry.scripts]
```

### Training
```bash
poetry run train-wan2-1-t2v-lora
poetry run train-hunyuan-t2v-lora
# Full list of train-* commands in pyproject.toml
```

## Architecture

### Core Design Pattern: GenerationBase

All models inherit from `videotuna/base/generation_base.py`, which extends both `TrainBase` (PyTorch Lightning) and `InferenceBase`. Every model is composed of four components instantiated from YAML config:

```
GenerationBase
├── first_stage_model  — VAE: encodes/decodes video ↔ latent space
├── cond_stage_model   — Text encoder (T5, CLIP, etc.)
├── cond_stage_2_model — Optional secondary conditioning (e.g., image for I2V)
├── denoiser           — Core transformer/UNet doing the diffusion
└── scheduler          — DDIM or other sampler
```

Each model adds its specifics in `videotuna/flow/{model}.py` (e.g., `WanVideoModelFlow`) which subclasses `GenerationBase`. Current flow files: `wanvideo.py`, `wanvideo_e2e_ttt.py`, `hunyuanvideo.py`, `stepvideo.py`, `videocrafter.py`.

### Config-Driven Instantiation

`videotuna/utils/common_utils.py::instantiate_from_config()` resolves `target` class strings and instantiates any class from YAML:

```yaml
flow:
  target: videotuna.flow.wanvideo.WanVideoModelFlow
  params:
    denoiser_config:
      target: videotuna.models.wan.wan.modules.model.WanModel
      params: { ... }
```

Configs in `configs/{num}_{model}/` drive both inference and training. A `mapping:` key in the YAML links inference parameters to flow parameters to avoid duplication — processed in `videotuna/utils/args_utils.py::prepare_train_args()`.

### Entry Points

All `poetry run <command>` calls resolve through `scripts/__init__.py`, which calls `scripts/inference.py` or `scripts/train.py` with the appropriate `--config` YAML. The train script uses PyTorch Lightning CLI; the inference script loads the model, reads prompts from `inputs/t2v/prompts.txt` or `inputs/i2v/`, and writes output to `results/`.

### Data Loading

`videotuna/data/datasets.py::DatasetFromCSV` expects a CSV with at minimum `path` and `caption` columns:

```csv
path,caption
Dataset/ToyDataset/videos/video1.mp4,A woman walking in the park
```

The toy dataset lives in `videotuna/data/toy_videos/` and is used for quick smoke tests.

### End-to-End Test-Time Training (e2e_ttt_video/)

A separate meta-learning module with:
- **Inner loop** (`inner_loop.py`): few-step LoRA gradient updates per prompt at inference time; `compute_flow_matching_loss` + `run_inner_loop` are the core functions
- **Outer loop** (`outer_loop.py`): `E2ETTTMetaTrainer` runs MAML-style meta-parameter updates across prompts
- **Sequential inference** (`sequential_inference.py`): `SequentialVideoGenerator` generates long videos clip-by-clip with temporal consistency; applies TTT per clip via `_ttt_encode_clip`
- **Config dataclasses** (`config.py`): `InnerLoopConfig`, `OuterLoopConfig`, `InferenceConfig`, `E2ETTTConfig`
- **Differentiable optimizers** (`higher_optim.py`): `DifferentiableSGD`, `DifferentiableAdamW`, `DifferentiableMuon`, `DifferentiableMuonClip`, `MetaLearnedLRSchedule` — all support gradient-through-optimizer for meta-learning
- **LoRA state helpers** (`lora_state.py`): `get_lora_params`, `snapshot_lora_state`, `restore_lora_state`, `count_lora_params`
- **Dataset** (`dataset.py`): `NovelClipSequenceDataset` loads pre-encoded clip latents from CSV; `precompute_clip_latents` pre-encodes raw videos with the VAE
- **Metrics** (`metrics.py`): `CrossClipCoherenceMetrics` — `clip_consistency`, `feature_drift`, `boundary_smoothness` using CLIP embeddings
- **Tests** (`e2e_ttt_video/tests/`): unit tests for every sub-module; run with `poetry run pytest e2e_ttt_video/tests/ -v`
- The flow class `videotuna/flow/wanvideo_e2e_ttt.py` (`WanVideoE2ETTTFlow`) wires this into the Wan2.1 model

### Adding a New Model

Follow the existing pattern:
1. Add model implementation under `videotuna/models/{name}/`
2. Create `videotuna/flow/{name}.py` subclassing `GenerationBase`
3. Add YAML configs in `configs/{num}_{name}/`
4. Register inference/training scripts in `scripts/__init__.py` and `pyproject.toml`

## Key File Locations

| Purpose | Location |
|---|---|
| CLI entry points | `scripts/__init__.py` |
| Shell scripts (inference/train) | `shscripts/` |
| Base classes | `videotuna/base/` |
| Model flows | `videotuna/flow/` |
| YAML configs | `configs/` |
| Checkpoint loading | `videotuna/utils/load_weights.py` |
| Config arg processing | `videotuna/utils/args_utils.py` |
| Checkpointing callbacks | `videotuna/utils/callbacks.py` (`VideoTunaModelCheckpoint`, `LoraModelCheckpoint`) |
| TTT module | `e2e_ttt_video/` |
| TTT unit tests | `e2e_ttt_video/tests/` |
| Toy dataset | `videotuna/data/toy_videos/` |
| Evaluation (VBench) | `eval/vbench/` |
| Data preprocessing utilities | `data_utils/` |
| Checkpoint conversion tools | `tools/` |
