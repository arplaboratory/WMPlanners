# World Model training via JEPA

- V-JEPA 2 experiments scaffolding for two-phase continual learning:
  - **Task 1**: self-supervised representation learning on broad video data.
  - **Task 2**: post-training/action-conditioned adaptation on robot data.

- Utilities to build V-JEPA 2 CSV datasets (`data/0001.mp4 label`).
- A small CLI (`cjepa`) to generate a reproducible local training
  command for `external/vjepa2/app/main.py`.

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

## Data preparation and training

### DINOv2 Video Models
Refer to [DINOv2 and JEPA Demo](jepa_dino_training.md) for more details.
#### Testing Models
DINOv2 Test
```bash
python tests/test_dino_load.py
```
Expected output: Model loads successfully, forward pass completes with output shape [1, 768]

<br>
Pairwise Dataset Loading Test

```bash
python tests/test_pairdataset.py
```
Expected output: Loads data successfully.

<br>
Video Dataset Loading Test:

```bash
python tests/test_videodataset.py
```
Expected: Loads data successfully.

<br>
World Model Modular Architecture Test:

```bash
python -m pytest tests/test_world_model.py
```
Expected: All 15 tests (backbones, components, architecture) pass.

### For CNN+MLP encoders

#### Download toy dataset
```bash
python scripts/world_model/make_toy_dataset.py --out_dir data/toy --episodes 200
```

#### Training

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

#### Evaluate MPC
```bash
python scripts/world_model/eval_mpc.py --ckpt runs/latest.pt --episodes 20
```

### For C-JEPA
Create at least two dataset roots/manifests:

- `datasets/task1/*.csv` (broad dynamics pre-training)
- `datasets/task2/*.csv` (robot/environment-specific adaptation)

V-JEPA 2 dataset manifests are plain CSV-like text files where each row is:

```text
/path/to/video.mp4 <label>
```

Labels are ignored in self-supervised pre-training, but the column is still
required by the loader contract.

You can generate these manifests with `cjepa.write_dataset_csv`.

#### CL mapping to V-JEPA 2

When adapting V-JEPA 2 as a world model, keep the following module boundaries:

- **Encoder (ViT context/target)**: latent state extraction; often frozen or
  regularized in CL to reduce catastrophic forgetting.
- **Predictor**: latent dynamics model; primary target for adaptation/expansion.
- **V-JEPA 2-AC predictor**: action-conditioned dynamics for control/robotics.

A practical CL setup is phase-based:

1. Train on general videos (Task 1).
2. Adapt to robot interaction videos (Task 2) while applying replay,
   regularization (EWC/MAS), or predictor-only expansion.

Generate a launch command:

```bash
cjepa \
  --config configs/train/vitl16/pretrain-256px-16f.yaml \
  --devices cuda:0 \
  --root-path datasets/task1 \
  --train-csv datasets/task1/train.csv \
  --val-csv datasets/task1/val.csv \
  --task-name task1_pretrain
```

Then run it from the V-JEPA 2 directory:

```bash
cd src/vjepa2
python -m app.main \
  --fname configs/train/vitl16/pretrain-256px-16f.yaml \
  --devices cuda:0 \
  --root_path ../../datasets/task1 \
  --train_csv ../../datasets/task1/train.csv \
  --val_csv ../../datasets/task1/val.csv \
  --tag task1_pretrain
```

For a task-2 adaptation pass, reuse the same command pattern with robot-specific
CSVs and a new `--tag`, then incorporate your CL method in the trainer/loss path
inside `src/vjepa2/app/vjepa/train.py` (or AC equivalent).

```bash
cd src/vjepa2
python -m app.main \
  --fname configs/train/vitg16/droid-256px-8f.yaml \
  --devices cuda:0 \
  --root_path ../../datasets/task2 \
  --train_csv ../../datasets/task2/train.csv \
  --val_csv ../../datasets/task2/val.csv \
  --tag task2_robot_adapt
```

#### Inject continual learning methods

Typical hooks:

- Freeze/partially freeze encoder weights.
- Add replay batches sampled from task 1 manifests.
- Add EWC/MAS-style penalties in the training loss.
- Swap or expand predictor heads (including action-conditioned predictor).

Good implementation entry points in the local V-JEPA 2 tree:

- `src/vjepa2/app/vjepa/train.py`
- `src/vjepa2/src/models/predictor.py`
- `src/vjepa2/src/models/ac_predictor.py`
