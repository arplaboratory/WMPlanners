# World Model Training via JEPA

This document covers the modular world model architecture, backbone testing, and training procedures for Embodied Learning and Vision.

## Modular World Model Architecture

The world model is designed with a **modular backbone architecture**, allowing you to swap out the visual encoder while keeping the dynamics and reward heads consistent.

### Core Components
- **Backbone**: The visual encoder (e.g., CNN, DINOv2, V-JEPA2).
- **Latent Dynamics**: Predicts the next latent state $z_{t+1}$ given $z_t$ and $a_t$.
- **Fusion**: Merges visual features with proprioceptive data (odom, IMU).
- **Heads**: Decodes latent states into task-specific outputs (state reconstruction, risk/collision prediction).

### Available Backbones
| Name | Description | Source |
|---|---|---|
| `cnn` | Lightweight 4-layer CNN for toy/fast experiments | Internal |
| `dinov2` | Foundation model for robust spatial features | `external/dino` |
| `vjepa2` | Spatio-temporal features for video-based world models | `external/vjepa2` |

---

## Environment Setup

To ensure compatibility with pre-trained backbones and internal tools, we use a centralized setup script:

```bash
bash scripts/python_setup.sh
source .venv/bin/activate
```

This enforces **Python 3.11** and correctly initializes all external submodules.

---

## Backbone Demo and Standalone Testing

Before training the full world model, you can test the vision backbones using the standalone demo script. This is useful for verifying that models load correctly and datasets are being parsed properly.

### Configuration (`configs/config.yaml`)

The demo script is driven by a YAML configuration file:

```yaml
# Model configuration
model:
  name: "dinov2"  # Options: "vjepa2", "dinov2"
  model_name: "dinov2_vitb14"  # Model variant
  pretrained: true
  device: "cpu"  # "cuda" or "cpu"

# Dataset configuration
dataset:
  type: "video"  # "video" or "pair"
  batch_size: 2
  num_workers: 2

  # Video dataset specific args
  video_args:
    num_frames: 16
    sampling_stride: 4
    crop_size: 256
    random_clip: true

  # Pair dataset specific args
  pair_args:
    random_pair: true
    image_size: 224

# Training configuration (for demo purposes)
training:
  epochs: 1
  log_every: 10
```

### Running the Demo
Launch the demo using the installed CLI command:
```bash
dino --config configs/config.yaml
```

### Verification Tests
Run these scripts to verify specific components:

- **DINOv2 Loading**: `python tests/test_dino_load.py`
- **Pairwise Dataset**: `python tests/test_pairdataset.py`
- **Video Dataset**: `python tests/test_videodataset.py`
- **Full Architecture**: `python -m pytest tests/test_world_model.py`

---

## World Model Training Pipeline

The training pipeline supports two phases: self-supervised representation learning and task-specific adaptation.

### 1. Data Preparation
For initial experiments, you can generate a toy dataset:
```bash
python scripts/world_model/make_toy_dataset.py --out_dir data/toy --episodes 200
```

### 2. Training Commands

**Basic CNN Training:**
```bash
python scripts/world_model/train_world_model.py --dataset data/toy --dataset_type npz --epochs 20 --out_dir runs/cnn
```

**Training with DINOv2 Backbone (Frozen):**
```bash
python scripts/world_model/train_world_model.py \
    --data_dir data/toy \
    --dataset_type npz \
    --backbone dinov2 \
    --backbone_name dinov2_vits14 \
    --freeze_backbone \
    --epochs 10 \
    --out_dir runs/dino
```

> [!TIP]
> Use `--freeze_backbone` when using large pre-trained models (DINO/JEPA) to speed up training and prevent the encoder from collapsing on small robotics datasets.

### 3. Evaluate MPC
After training, evaluate the model's performance in a closed-loop MPC setting:
```bash
python scripts/world_model/eval_mpc.py --ckpt runs/<dino|cnn|jepa>/latest.pt --episodes 20
```

## Dependencies and Notes
- **External Submodules**: Ensure DINO and VJEPA2 submodules are initialized (`git submodule update --init --recursive`).
- **Python Version**: Required **Python 3.11** for compatibility with legacy DINOv2 dependency pins (`torch==2.0.0`).
