# World Model Training

This directory contains the unified world-model trainer (`train_world_model.py`) and supporting scripts for the DINO-WM architecture.

---

## Table of Contents

1. [What Changed](#what-changed)
2. [Quick Start](#quick-start)
3. [Reproducing the `trial.yaml` Baseline](#reproducing-the-trialyaml-baseline)
4. [Configuration Reference (`DinoWMConfig`)](#configuration-reference-dinowmconfig)
5. [Swappable Predictor Types](#swappable-predictor-types)
6. [State / Risk Auxiliary Supervision](#state--risk-auxiliary-supervision)
7. [Research Training Features](#research-training-features)
8. [Checkpoint Structure](#checkpoint-structure)
9. [CLI Reference](#cli-reference)
10. [Notebook Demos](#notebook-demos)

---

## What Changed

### Before: two separate trainers + Hydra

The original codebase had two independent training pipelines:

| File | Mode | Config system | GPU strategy |
|------|------|---------------|--------------|
| `train_world_model.py` | `wm` — rollout-loss `WorldModel` | argparse | manual DDP |
| `train_world_model_dinowm.py` | DINO-WM (frozen DINOv2 + ViT predictor) | Hydra + `trial.yaml` | HuggingFace `Accelerator` |

The DINO-WM trainer required Hydra for configuration composition and Accelerator for multi-GPU, making it incompatible with the rest of the codebase's DDP/argparse conventions.

### After: one unified trainer

`train_world_model.py` now runs both modes under a single entry-point:

```
python train_world_model.py --mode wm     ...   # original WorldModel
python train_world_model.py --mode dinowm ...   # DINO-WM (new default target)
```

**What was preserved from DINO-WM:**
- Frozen DINOv2 encoder with per-component learning rates (encoder, predictor, action/proprio encoders, decoder)
- Modular `WM` class (`src/world_model/backbones/wm.py`) with JEPA patch-level prediction
- I-JEPA EMA target encoder (`ema_tau`)
- Per-epoch validation, image-quality metrics (`PSNR`, `SSIM`, `LPIPS`), open-loop rollout
- Optional planning-eval integration

**What was added:**
- `DinoWMConfig` dataclass — drop-in replacement for Hydra `OmegaConf`; pass it to `DinoWMTrainer` directly from notebooks or scripts
- Accelerator replaced with manual DDP (`torch.distributed`) matching the `wm` mode
- `predictor_type` — swap the ViT predictor for an MLP or RSSM dynamics model
- `state_head` + `risk_head` — auxiliary supervision on odometry (MSE) and collision (BCE)
- AMP (`use_amp`), gradient accumulation (`accum_steps`), gradient clipping (`clip_grad_norm`), cosine LR with warmup (`lr_schedule`, `lr_warmup_epochs`)
- Configurable loss weights: `lambda_z`, `lambda_action_decode`, `lambda_state`, `lambda_risk`

### Architecture changes to `WM`

`src/world_model/backbones/wm.py` gained three capabilities:

**1. Swappable predictor** (`predictor_mode`):
- `"patch"` — existing ViT operates on `(B, T×P, D)` flattened patch tokens (unchanged)
- `"frame"` — new `_predict_frame_level()` pools patches to `(B, T, emb_dim)`, runs MLP or RSSM step-by-step, then broadcasts predictions back to patch shape

**2. State head** (`state_head: StateHead`):
- Created when `state_dim > 0`
- Input: `z_pred` pooled over patches → `(B, num_hist, emb_dim)`
- Output: pose prediction `(B, num_hist, state_dim)` supervised with MSE against odom columns

**3. Risk head** (`risk_head: RiskHead`):
- Created when `state_dim > 0`
- Input: same pooled `z_pred`
- Output: collision probability `(B, num_hist, 1)` supervised with BCE against the collision column in `state`

**`forward()` signature** (fully backward-compatible):
```python
WM.forward(obs, act, target_encoder=None, odom=None, collision=None)
```

---

## Quick Start

```bash
# Clone + install (from repo root)
pip install -e .[dev]

# DINO-WM baseline (equivalent to trial.yaml)
python scripts/world_model/train_world_model.py \
    --mode dinowm \
    --dataset lquan9/collect1 \
    --backbone_name dinov2_vits14 \
    --num_hist 3 --num_pred 1 \
    --frameskip 5 --concat_dim 1 \
    --action_emb_dim 10 --num_action_repeat 1 \
    --proprio_emb_dim 10 --num_proprio_repeat 1 \
    --encoder_lr 1e-6 --predictor_lr 5e-4 --action_encoder_lr 5e-4 \
    --epochs 1 --batch_size 2 \
    --predictor_type vit
```

Or from Python (notebook / script):

```python
import sys; sys.path.insert(0, "scripts/world_model")
from train_world_model import DinoWMTrainer, DinoWMConfig

cfg = DinoWMConfig(dataset="lquan9/collect1")
trainer = DinoWMTrainer(cfg)
trainer.run()
```

---

## Reproducing the `trial.yaml` Baseline

`configs/dino-wm/trial.yaml` was the Hydra config for the original DINO-WM trainer. The table below maps every meaningful field to its `DinoWMConfig` equivalent.

| `trial.yaml` key | `DinoWMConfig` field | Value |
|---|---|---|
| `training.epochs` | `epochs` | `1` |
| `training.batch_size` | `batch_size` | `2` |
| `training.encoder_lr` | `encoder_lr` | `1e-6` |
| `training.predictor_lr` | `predictor_lr` | `5e-4` |
| `training.action_encoder_lr` | `action_encoder_lr` | `5e-4` |
| `training.decoder_lr` | `decoder_lr` | `3e-4` |
| `training.seed` | `seed` | `0` |
| `training.save_every_x_epoch` | `save_every_x_epoch` | `1` |
| `training.num_reconstruct_samples` | `num_reconstruct_samples` | `2` |
| `img_size` | `img_size` | `224` |
| `frameskip` | `frameskip` | `5` |
| `concat_dim` | `concat_dim` | `1` |
| `normalize_action: True` | *(always on in new trainer)* | — |
| `action_emb_dim` | `action_emb_dim` | `10` |
| `num_action_repeat` | `num_action_repeat` | `1` |
| `proprio_emb_dim` | `proprio_emb_dim` | `10` |
| `num_proprio_repeat` | `num_proprio_repeat` | `1` |
| `num_hist` | `num_hist` | `3` |
| `num_pred` | `num_pred` | `1` |
| `has_predictor: True` | `has_predictor` | `True` |
| `has_decoder: False` | `has_decoder` | `False` |
| `model.train_encoder: False` | `train_encoder` | `False` |
| `model.train_predictor: True` | `train_predictor` | `True` |
| `model.train_decoder: False` | `train_decoder` | `False` |
| `debug: False` | `debug` | `False` |
| `encoder: dino` | `backbone_name` | `"dinov2_vits14"` |
| `predictor: vit` | `predictor_type` | `"vit"` |
| `plan_settings.plan_cfg_path: null` | `plan_cfg_path` | `None` |

**Exact `DinoWMConfig` equivalent of `trial.yaml`:**

```python
cfg = DinoWMConfig(
    dataset          = "lquan9/collect1",
    epochs           = 1,
    batch_size       = 2,
    img_size         = 224,
    seed             = 0,
    frameskip        = 5,
    concat_dim       = 1,
    num_hist         = 3,
    num_pred         = 1,
    has_predictor    = True,
    has_decoder      = False,
    train_encoder    = False,
    train_predictor  = True,
    train_decoder    = False,
    backbone_name    = "dinov2_vits14",
    predictor_type   = "vit",
    encoder_lr       = 1e-6,
    predictor_lr     = 5e-4,
    action_encoder_lr= 5e-4,
    decoder_lr       = 3e-4,
    action_emb_dim   = 10,
    num_action_repeat= 1,
    proprio_emb_dim  = 10,
    num_proprio_repeat = 1,
    save_every_x_epoch = 1,
    num_reconstruct_samples = 2,
    debug            = False,
    plan_cfg_path    = None,
)
```

---

## Configuration Reference (`DinoWMConfig`)

All fields and their defaults. Pass any subset to `DinoWMConfig(...)` — unset fields take the default.

### Core

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset` | `str` | `""` | HuggingFace dataset ID (required) |
| `epochs` | `int` | `20` | Training epochs |
| `batch_size` | `int` | `64` | Per-GPU batch size |
| `img_size` | `int` | `224` | Input image resolution (must be divisible by 14 for ViTs) |
| `seed` | `int` | `0` | Global RNG seed |
| `out_dir` | `str` | `"runs"` | Base directory for checkpoints and logs |
| `num_workers` | `int` | `4` | DataLoader workers |
| `device` | `str` | auto | `"cuda"` or `"cpu"` |
| `debug` | `bool` | `False` | Extra verbosity |

### Encoder

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backbone_name` | `str` | `"dinov2_vits14"` | DINOv2 variant (`dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`) |
| `train_encoder` | `bool` | `False` | Unfreeze encoder during training |
| `encoder_lr` | `float` | `1e-6` | Learning rate for encoder params |

### Temporal / Predictor

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_hist` | `int` | `3` | History frames fed to the predictor |
| `num_pred` | `int` | `1` | Prediction horizon (only 1 is supported) |
| `frameskip` | `int` | `5` | Temporal stride in the dataset |
| `concat_dim` | `int` | `1` | How action/proprio are fused: `1` = feature concat (recommended), `0` = extra patch tokens |
| `has_predictor` | `bool` | `True` | Enable predictor training |
| `train_predictor` | `bool` | `True` | Allow predictor gradients |
| `predictor_depth` | `int` | `6` | ViT predictor transformer depth |
| `predictor_heads` | `int` | `16` | ViT attention heads |
| `predictor_mlp_dim` | `int` | `2048` | ViT FFN width |
| `predictor_dropout` | `float` | `0.1` | ViT dropout |
| `predictor_lr` | `float` | `5e-4` | Predictor learning rate |

### Action / Proprio Encoders

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `action_emb_dim` | `int` | `10` | Embedding size for each action dimension |
| `proprio_emb_dim` | `int` | `10` | Embedding size for each proprio dimension |
| `num_action_repeat` | `int` | `1` | How many times action embedding is repeated |
| `num_proprio_repeat` | `int` | `1` | How many times proprio embedding is repeated |
| `action_encoder_lr` | `float` | `5e-4` | LR for action encoder, proprio encoder, action_decoder, state_head, risk_head |

### Decoder (optional)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `has_decoder` | `bool` | `False` | Enable decoder |
| `train_decoder` | `bool` | `False` | Allow decoder gradients |
| `decoder_path` | `str\|None` | `None` | Path to pretrained decoder checkpoint |
| `decoder_n_embed` | `int` | `2048` | Decoder embedding dim |
| `decoder_n_res_block` | `int` | `4` | Decoder residual blocks |
| `decoder_n_res_channel` | `int` | `128` | Decoder residual channels |
| `decoder_lr` | `float` | `3e-4` | Decoder learning rate |

### Checkpointing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `save_every_x_epoch` | `int` | `1` | Save a checkpoint every N epochs |
| `num_reconstruct_samples` | `int` | `6` | Samples to visualize in reconstruction grids |
| `experiment_name` | `str\|None` | `None` | Override auto-generated run name |
| `wandb` | `bool` | `False` | Enable W&B logging |

### Training Improvements

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_amp` | `bool` | `False` | Automatic mixed precision (FP16) |
| `accum_steps` | `int` | `1` | Gradient accumulation steps |
| `clip_grad_norm` | `float` | `1.0` | Max gradient norm (`0` = disabled) |
| `ema_tau` | `float` | `0.0` | EMA decay for I-JEPA target encoder (`0` = disabled) |
| `lr_schedule` | `str` | `"none"` | LR schedule: `"none"`, `"cosine"` |
| `lr_warmup_epochs` | `int` | `0` | Epochs for linear LR warmup before cosine |
| `lambda_z` | `float` | `1.0` | Weight on JEPA latent loss |
| `lambda_action_decode` | `float` | `1.0` | Weight on action reconstruction loss |

### Swappable Predictor

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `predictor_type` | `str` | `"vit"` | Dynamics model: `"vit"`, `"mlp"`, `"rssm"` |

### State / Risk Auxiliary Supervision

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `state_dim` | `int` | `0` | Odom columns to supervise (`0` = disabled; e.g. `6` for 6-DOF) |
| `lambda_state` | `float` | `1.0` | Weight on odometry MSE loss |
| `lambda_risk` | `float` | `0.2` | Weight on collision BCE loss |

---

## Swappable Predictor Types

The `predictor_type` field controls which dynamics model is used inside `WM`. All three share the same JEPA-style training objective.

### `"vit"` — ViT patch predictor (default)

The original DINO-WM predictor. A causal transformer that operates on the full sequence of patch tokens `(B, T×P, D)`. Captures spatial structure within each frame and temporal dependencies across frames simultaneously.

```python
cfg = DinoWMConfig(dataset="...", predictor_type="vit")
```

Loss keys: `z_loss`, `z_visual_loss`, `z_proprio_loss`, `action_decode_loss`

**When to use:** Default for best patch-level prediction quality. Heaviest compute (~100× more parameters than MLP).

### `"mlp"` — Frame-level residual MLP

A lightweight 3-layer MLP: `z_{t+1} = z_t + MLP([z_t, a_t])`. Operates on pooled frame vectors `(B, emb_dim)` rather than patch tokens. One forward pass per timestep.

```python
cfg = DinoWMConfig(dataset="...", predictor_type="mlp")
```

Loss keys: same as ViT (frame-level MSE is broadcast back to patch shape for the JEPA loss)

**When to use:** Fast ablations, compute-limited environments, or when frame-level dynamics suffice. Requires `concat_dim=1`.

### `"rssm"` — Recurrent State-Space Model (GRU)

A GRU-based dynamics model: `h_t = GRU([z_t, a_t], h_{t-1}); z_{t+1} = z_t + prior(h_t)`. Carries a hidden state across the rollout, making it suitable for partially-observable scenes.

```python
cfg = DinoWMConfig(dataset="...", predictor_type="rssm")
```

Loss keys: same as ViT

**When to use:** Sequences with memory requirements (occlusions, long-horizon planning). Comparable compute to MLP. Requires `concat_dim=1`.

### Predictor comparison

| Property | ViT | MLP | RSSM |
|---|---|---|---|
| Input shape | `(B, T×P, D)` patch tokens | `(B, emb_dim)` frame vectors | `(B, emb_dim)` + hidden state |
| Temporal modelling | attention (all-to-all) | feedforward (single step) | recurrent (GRU) |
| Memory across steps | none (context window) | none | GRU hidden state |
| Parameter count | ~20M | ~800K | ~1M |
| `concat_dim` | 0 or 1 | 1 recommended | 1 recommended |
| Best for | rich spatial prediction | fast ablations | partially-observable scenes |

---

## State / Risk Auxiliary Supervision

When `state_dim > 0`, two auxiliary heads are attached to the predicted latent and jointly trained alongside the JEPA loss.

### State head (pose MSE)

Predicts the next `state_dim` odometry dimensions from the predicted latent:

```
z_pred   (B, num_hist, num_patches, emb_dim)
  → pool patches → (B, num_hist, emb_dim)
  → StateHead → (B, num_hist, state_dim)
  → MSE vs odom[:, num_pred:, :state_dim]
```

The odom columns come from the `state` tensor returned by `collate_dinowm`. The dataset's `D_odom` attribute is used automatically to slice them — no manual configuration needed.

### Risk head (collision BCE)

Predicts collision probability from the same pooled latent:

```
z_pred   (B, num_hist, num_patches, emb_dim)
  → pool patches → (B, num_hist, emb_dim)
  → RiskHead → (B, num_hist, 1)  [sigmoid output]
  → BCE vs collision[:, num_pred:]
```

The collision label is the last column of the `state` tensor (binary, 0/1).

### Usage

```python
cfg = DinoWMConfig(
    dataset       = "lquan9/collect1",
    predictor_type = "vit",      # works with all three predictor types
    state_dim     = 6,           # supervise 6-DOF odom
    lambda_state  = 1.0,         # pose MSE weight
    lambda_risk   = 0.2,         # collision BCE weight
)
```

With state/risk enabled, the loss breakdown adds:

```
total = lambda_z * z_loss
      + lambda_action_decode * action_decode_loss
      + lambda_state         * state_loss
      + lambda_risk          * risk_loss
```

Logged keys: `train_state_loss`, `train_risk_loss`, `val_state_loss`, `val_risk_loss`

---

## Research Training Features

These features are off by default and do not affect the baseline.

### Automatic Mixed Precision (AMP)

```python
cfg = DinoWMConfig(..., use_amp=True)
```

Wraps the forward pass and loss computation in `torch.autocast("cuda")` with a `GradScaler`. Reduces GPU memory by ~40% on typical runs, speeds up training on Ampere+ GPUs.

### Gradient Accumulation

```python
cfg = DinoWMConfig(..., accum_steps=4)
```

Accumulates gradients over `accum_steps` mini-batches before stepping the optimizer. Effective batch size = `batch_size × accum_steps`. Useful for simulating large batches on memory-constrained hardware.

### Gradient Clipping

```python
cfg = DinoWMConfig(..., clip_grad_norm=1.0)   # default
cfg = DinoWMConfig(..., clip_grad_norm=0.0)   # disabled
```

Clips the global gradient norm before each optimizer step.

### I-JEPA EMA Target Encoder

```python
cfg = DinoWMConfig(..., ema_tau=0.996)
```

Maintains an exponential moving average copy of the encoder as a target. The predictor is trained to match `target_encoder(obs)` rather than `encoder(obs)` directly, which stabilises training (analogous to momentum encoder in MoCo/BYOL). The target is updated after each step:

```
target = tau * target + (1 - tau) * online
```

Set `ema_tau=0.0` to disable (uses online encoder as target — original DINO-WM behaviour).

### Cosine LR Schedule with Warmup

```python
cfg = DinoWMConfig(..., lr_schedule="cosine", lr_warmup_epochs=5)
```

Applies a linear warmup over the first `lr_warmup_epochs` epochs, then cosine annealing to 0 over the remaining epochs. Applied to all component optimizers simultaneously.

---

## Checkpoint Structure

Checkpoints are saved to `{out_dir}/{experiment_name}/`:

```
runs/
└── lquan9_collect1_dinov2_vits14_20260527_143021/
    ├── encoder.pt           # DINOv2 encoder state dict
    ├── predictor.pt         # ViT / MLP / RSSM predictor state dict
    ├── action_encoder.pt    # ProprioceptiveEmbedding (action)
    ├── proprio_encoder.pt   # ProprioceptiveEmbedding (proprio)
    ├── action_decoder.pt    # StateHead used for action decode loss
    ├── state_head.pt        # (only when state_dim > 0)
    ├── risk_head.pt         # (only when state_dim > 0)
    ├── decoder.pt           # (only when has_decoder=True)
    ├── target_encoder.pt    # (only when ema_tau > 0)
    ├── args.json            # Full DinoWMConfig serialised to JSON
    └── epoch_{N}/           # Per-epoch snapshots (every save_every_x_epoch epochs)
        ├── encoder.pt
        └── ...
```

The `args.json` file records every config field so runs are fully reproducible:

```bash
python -c "import json; print(json.load(open('runs/.../args.json')))"
```

---

## CLI Reference

All `DinoWMConfig` fields are available as CLI arguments when running `train_world_model.py --mode dinowm`.

### Predictor type

```bash
# ViT (default)
python train_world_model.py --mode dinowm --dataset lquan9/collect1 \
    --predictor_type vit

# MLP (frame-level, fast)
python train_world_model.py --mode dinowm --dataset lquan9/collect1 \
    --predictor_type mlp

# RSSM (recurrent, frame-level)
python train_world_model.py --mode dinowm --dataset lquan9/collect1 \
    --predictor_type rssm
```

### State / risk supervision

```bash
python train_world_model.py --mode dinowm --dataset lquan9/collect1 \
    --predictor_type vit \
    --state_dim 6 --lambda_state 1.0 --lambda_risk 0.2
```

### Full research stack

```bash
python train_world_model.py --mode dinowm --dataset lquan9/collect1 \
    --predictor_type rssm \
    --state_dim 6 --lambda_state 1.0 --lambda_risk 0.2 \
    --use_amp \
    --accum_steps 4 \
    --ema_tau 0.996 \
    --lr_schedule cosine --lr_warmup_epochs 2 \
    --epochs 20 --batch_size 32
```

### DDP (multi-GPU)

```bash
torchrun --nproc_per_node=4 \
    scripts/world_model/train_world_model.py \
    --mode dinowm --ddp --dataset lquan9/collect1
```

### DINO-WM CLI args (group `--help`)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--predictor_type` | `str` | `vit` | `"vit"` \| `"mlp"` \| `"rssm"` |
| `--state_dim` | `int` | `0` | Odom dims to supervise (0 = off) |
| `--lambda_state` | `float` | `1.0` | Pose MSE weight |
| `--lambda_risk` | `float` | `0.2` | Collision BCE weight |
| `--lambda_z` | `float` | `1.0` | JEPA latent loss weight |
| `--lambda_action_decode` | `float` | `1.0` | Action decode loss weight |
| `--use_amp` | flag | off | Enable mixed precision |
| `--accum_steps` | `int` | `1` | Gradient accumulation steps |
| `--clip_grad_norm` | `float` | `1.0` | Max grad norm (0 = off) |
| `--ema_tau` | `float` | `0.0` | EMA target encoder decay |
| `--lr_schedule` | `str` | `none` | `"none"` \| `"cosine"` |
| `--lr_warmup_epochs` | `int` | `0` | Linear warmup epochs |

---

## Notebook Demos

`docs/notebooks/world_model_pipeline.ipynb` contains end-to-end demos for every mode.

| Section | Topic |
|---|---|
| 5A | DINO-WM baseline (ViT, `trial.yaml` equivalent) |
| 5B | Training improvements (AMP + EMA + cosine LR) |
| 5C | Loss weight ablation (`lambda_z`, `lambda_action_decode`) |
| 5D | MLP predictor vs ViT — compute / quality trade-off |
| 5E | RSSM predictor — recurrent dynamics |
| 5F | State / risk supervision (ViT + odom MSE + collision BCE) |
| 5G | Full combo: RSSM + state/risk heads |
| 5H | Configuration cheat-sheet |

To run:

```bash
jupyter lab docs/notebooks/world_model_pipeline.ipynb
```
