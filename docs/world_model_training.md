# World Model Training

This document covers the modular world model architecture, backbone options, and training procedures.

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

## Data

### HuggingFace Datasets (Recommended)

Download datasets via the pipeline client or directly:

```bash
# Via pipeline client (recommended)
python scripts/hpc/pipeline-py-client
# → Select "Download HF dataset" → enter repo name (e.g., lquan9/collect1)

# Or directly in Python
from huggingface_hub import snapshot_download
snapshot_download("lquan9/collect1", repo_type="dataset", local_dir="data/collect1")
```

### Toy Dataset (Local Testing)

```bash
python scripts/world_model/make_toy_dataset.py --out_dir data/toy --episodes 200
```

---

## Training

### Single-GPU Training (Local)

```bash
python scripts/world_model/train_world_model.py \
  --dataset data/toy --dataset_type npz \
  --backbone cnn --epochs 20 --out_dir runs/cnn
```

### Multi-GPU DDP Training (HPC)

All HPC experiments use `torchrun` with `--ddp` for DistributedDataParallel across 2 GPUs:

```bash
torchrun --standalone --nproc_per_node=2 \
  scripts/world_model/train_world_model.py --ddp \
  --dataset lquan9/collect1 \
  --backbone dinov2 \
  --backbone_name dinov2_vitb14 \
  --freeze_backbone \
  --epochs 20 --batch_size 64 \
  --lr 3e-4 --horizon 8 --img_size 224 \
  --wandb \
  --out_dir runs/dinov2
```

> [!TIP]
> Use `--freeze_backbone` when using large pre-trained models (DINO/JEPA) to speed up training and prevent the encoder from collapsing on small robotics datasets.

### Launching on HPC

Use the pipeline client to generate and submit SLURM scripts:

```bash
python scripts/hpc/pipeline-py-client
# → Select "slurm" → "Generate from template (ds_3001)"
# → Choose experiment type: train_wm
# → Confirm pre-filled defaults (partition, GPUs, dataset, backbone)
# → Submit
```

The template handles: container setup, venv activation, code refresh (`pip install -e`), and DDP launch.

### Key Training Arguments

| Argument | Description | Default |
|---|---|---|
| `--dataset` | HF repo ID or local path | *(required)* |
| `--dataset_type` | `hf` or `npz` | `hf` |
| `--backbone` | `cnn`, `dinov2`, or `vjepa2` | `cnn` |
| `--backbone_name` | Model variant (e.g., `dinov2_vitb14`) | `dinov2_vitb14` |
| `--freeze_backbone` | Freeze backbone weights | off |
| `--ddp` | Enable DistributedDataParallel | off |
| `--epochs` | Training epochs | `20` |
| `--batch_size` | Batch size per GPU | `64` |
| `--lr` | Learning rate | `3e-4` |
| `--horizon` | Multi-step rollout horizon | `8` |
| `--wandb` | Enable WANDB logging | off |
| `--out_dir` | Checkpoint output directory | `runs` |

---

## Evaluation

After training, evaluate pose prediction quality:

```bash
python scripts/world_model/eval_mpc.py --ckpt runs/<experiment>/latest.pt --episodes 20
```

## Dependencies and Notes
- **External Submodules**: Ensure DINO and VJEPA2 submodules are initialized:
  ```bash
  git submodule update --init --recursive
  ```
- **Python Version**: Python 3.11+ required.
- **setuptools**: Pinned to `<82` for compatibility with `torch 2.11.0`.
