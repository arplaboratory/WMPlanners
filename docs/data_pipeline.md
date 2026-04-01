# World Model Data Pipeline

End-to-end pipeline: **MCAP bags → NPZ episodes → HuggingFace dataset**.

## Architecture

```
MCAP bags
  → Stage 0: Extract (extract_dataset) → sharded images + CSVs
  → Stage 1: Build Episodes (build_episodes) → fixed-length NPZ episodes
  → Stage 2: Select Episodes (select_episodes) → diverse subset via FPS
  → Stage 3: HF Upload (hf_upload) → push to Hugging Face Hub
```

## Quick Start

```bash
# Full pipeline: MCAP → HuggingFace (10s episodes at 5fps)
python -m scripts.world_model.data_pipeline.run_pipeline \
    --mcap path/to/race16_bag.mcap \
    --robot race16 \
    --frame-skip 6 --episode-len 50 --stride 50 \
    --target-count 5000 \
    --push-to-hub dmody1/wm-data

# From pre-extracted bags (skip Stage 0)
python -m scripts.world_model.data_pipeline.run_pipeline \
    --bags dataset_out \
    --frame-skip 6 --episode-len 50 --stride 50 \
    --target-count 5000
```

## Prerequisites

```bash
pip install -e .
# MCAP extraction requires: pip install mcap mcap-ros2-interfaces
# HF upload requires: pip install datasets
# HF auth: huggingface-cli login
```

---

## Stage 0: Extract (`extract_dataset.py`)

Reads raw ROS2 MCAP bag files and produces sharded images + CSV sensor logs.

**What it does:**
1. Opens each `.mcap` file, decodes colour/depth images and sensor messages.
2. Writes images into sharded directories (`shard_0000/`, `shard_0001/`, ...).
3. Writes sensor CSVs: `imu.csv`, `odometry.csv`, `position_cmd.csv`, `so3_cmd.csv`.
4. Produces `image_index.csv` and `manifest.json`.

**Robot config** (`configs/robots/race16.yaml`) defines which ROS topics to extract.

**Skipped** when using `--bags` to point at an already-extracted directory.

---

## Stage 1: Build Episodes (`build_episodes.py`)

Converts extracted data into fixed-length `.npz` episodes for training.

**What it does:**
1. Reads `manifest.json` to find all bags and their sensor config.
2. Time-syncs all sensors to image timestamps via nearest-neighbor merge.
3. Quality filtering: marks idle frames (speed < threshold) and timestamp gaps.
4. With `--frame-skip N`, takes every Nth frame (e.g. `--frame-skip 6` turns 30fps → 5fps).
5. Slices valid runs into episodes of `--episode-len` frames with `--stride` overlap.

**NPZ episode contents:**

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `rgb` | `(T, H, W, 3)` | uint8 | Colour frames |
| `depth` | `(T, H, W)` | float32 | Depth in metres |
| `imu` | `(T, 10)` | float32 | IMU readings |
| `odom` | `(T, 13)` | float32 | Odometry state |
| `action` | `(T, 14)` | float32 | Position commands |

---

## Stage 2: Select Episodes (`select_episodes.py`)

Selects a maximally diverse subset using Farthest Point Sampling (FPS).

**What it does:**
1. Computes per-episode descriptors: action mean/std + odom start→end delta.
2. Z-score normalises action and odom blocks independently.
3. FPS greedily picks episodes that maximise minimum distance to already-selected set.
4. Copies/symlinks selected episodes to output directory.

This ensures the training set covers the full diversity of behaviours rather than over-sampling common motions.

---

## Stage 3: HF Upload (`hf_upload.py`)

Creates a Hugging Face `Dataset` from selected episodes and pushes to Hub.

**What it does:**
1. Scans `ep_*.npz` files in the selected directory.
2. Explodes each episode into per-frame rows with columns for `episode_id`, `frame_idx`, `image`, and all sensor channels.
3. Saves locally and optionally pushes to Hub.

**Standalone usage:**
```bash
python -m scripts.world_model.data_pipeline.hf_upload \
    --episodes-dir pipeline_out/selected \
    --push-to-hub dmody1/wm-data
```

---

## Full CLI Reference

```bash
python -m scripts.world_model.data_pipeline.run_pipeline \
    --mcap PATH [PATH ...]        # MCAP files (Stage 0)
    --bags PATH                   # OR pre-extracted dir (skips Stage 0)
    --robot NAME                  # Robot config (default: race16)
    --output DIR                  # Output directory (default: pipeline_out)
    --episode-len N               # Frames per episode (default: 50)
    --stride N                    # Episode stride (default: 50)
    --frame-skip N                # Temporal subsampling (default: 6)
    --img-size N                  # Resize to NxN (default: 128)
    --min-speed FLOAT             # Idle threshold m/s (default: 0.1)
    --target-count N              # Episodes to select (default: 5000)
    --push-to-hub REPO_ID         # HF Hub repo ID
    --hf-private                  # Make HF repo private
    --skip-build                  # Skip Stage 1
    --skip-select                 # Skip Stage 2
    --skip-hf-upload              # Skip Stage 3
```

## Output Structure

```
pipeline_out/
├── extracted/                    # Stage 0: per-bag images + CSVs
│   ├── manifest.json
│   └── race16_bag_.../
│       ├── image_index.csv
│       ├── odometry.csv
│       └── shard_0000/
├── episodes/                     # Stage 1: all NPZ episodes
│   ├── ep_00000.npz
│   ├── ep_00001.npz
│   └── metadata.json
└── selected/                     # Stage 2: FPS-selected subset
    ├── ep_00000.npz
    ├── ep_00001.npz
    ├── metadata.json
    ├── selection_stats.json
    └── hf_dataset/               # Stage 3: HF dataset
```
