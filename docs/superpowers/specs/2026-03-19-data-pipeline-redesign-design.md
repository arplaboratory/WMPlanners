# Data Pipeline Redesign: Multi-Bag, Config-Driven, Dynamics-Weighted Selection

**Date:** 2026-03-19
**Branch:** collect-1-store-ahx-89
**Status:** Approved

## Problem

The current data pipeline has two issues:

1. **DINOv2-centric selection** — Stage 2 computes DINOv2 CLS tokens for data curation, but the world model learns its own latent space (CNN/DINOv2/V-JEPA2 backbone → fusion → dynamics). The pipeline's embeddings aren't the latent space used for MPC planning.
2. **Inflexible extraction** — Hardcoded race16 topics, single-bag input, no support for different robots or sensor configurations.

## Goals

1. Multi-bag input with per-bag output organization.
2. Nearest-neighbor time-sync of odom/IMU to image pairs, robust to gaps and missing data.
3. Configurable topics and namespaces via per-robot YAML configs with CLI overrides.
4. Dynamics-weighted trajectory window selection for world model training, with optional visual encoder support.

## Design Decisions

- **Separate models per robot** — no cross-robot action space alignment. Each robot's config maps its native sensors/actions to the standard NPZ keys.
- **Nearest-neighbor sync** — simple, sufficient given image framerate is the bottleneck. Max sync gap tolerance configurable via `--max-sync-gap-ms` (default: 100ms); frames exceeding this are zero-filled.
- **YAML config with CLI overrides** — reusable configs, one-off flexibility.
- **Staged pipeline** — each step is independently debuggable and rerunnable. Each stage always writes from scratch to `--out_dir` (must be empty or new).
- **Existing `data_pipeline/` preserved** — serves a different purpose (HF dataset curation). No deletions.
- **`preprocess_race16.py` preserved** — serves as a standalone shortcut for the HuggingFace-hosted race16 dataset (download + extract + NPZ in one step). The new pipeline is for local MCAP bags. Both produce the same NPZ format consumed by `train_world_model.py`.

## Architecture

```
configs/robots/race16.yaml
         |
         v
extract_dataset.py  -->  dataset_out/{bag_name}/ (images + CSVs + manifest.json)
         |
         v
build_episodes.py   -->  data/race16/ (NPZ episodes + metadata.json)
         |
         v
select_episodes.py  -->  data/race16_selected/ (subset + selection_stats.json)
         |
         v
train_world_model.py --> runs/ (checkpoints)
         |
         v
eval_mpc.py          --> MPC evaluation in learned latent space
```

## Component Designs

### 1. Robot Config Schema

Location: `configs/robots/{robot_name}.yaml`

```yaml
version: 1
name: race16

topics:
  color: /race16/cam1/color/image_raw/compressed
  depth: /race16/cam1/aligned_depth_to_color/image_raw
  imu: /race16/cam1/imu
  odom: /race16/odom
  position_cmd: /race16/position_cmd
  so3_cmd: /race16/so3_cmd        # optional

sensors:
  color_encoding: compressed       # "compressed" or "raw"
  depth_encoding: 16UC1
  imu_dim: 10
  odom_dim: 13

action:
  source: position_cmd             # which topic key to use as the action signal
  dim: 14                          # columns after timestamp_ns in the source CSV
  # CSV filename is derived from the topic key: position_cmd -> position_cmd.csv
```

**Action source mapping:** `action.source` refers to a key in `topics`. The extractor writes a CSV named `{topic_key}.csv` for each topic. `build_episodes.py` loads `{action.source}.csv` as the action signal.

Loaded by `scripts/world_model/robot_config.py`:
- `load_robot_config(name_or_path, cli_overrides=None) -> dict`
- Checks `configs/robots/{name}.yaml` first, falls back to file path
- Merges CLI overrides via dot-notation (e.g., `--override topics.color=/other/topic`)

Optional topics (like `so3_cmd`) that don't appear in the bag produce a warning and are skipped.

### 2. `extract_dataset.py`

Extracts one or more MCAP bags into a structured directory using the robot config for topic routing.

**CLI:**
```bash
python -m scripts.world_model.extract_dataset \
    --robot race16 \
    --mcap bag1.mcap bag2.mcap \
    --output dataset_out \
    --shard-size 500
```

**Changes from current:**
- `--mcap` accepts multiple paths. Each bag gets a subdirectory named by its stem.
- Topic-to-handler routing is driven by config, not hardcoded if/elif.
- Handles both `compressed` (raw JPEG/PNG bytes) and `raw` (sensor_msgs/Image) color based on `sensors.color_encoding`.
- Action topic is dynamic: extracts whatever `action.source` points to.
- CLI overrides: `--color-topic`, `--depth-topic`, etc. override the config for one-off runs.

**Output:**
```
dataset_out/
  {bag_stem}/
    shard_0000/
      color_000000.jpg
      depth_000000.png
    image_index.csv
    imu.csv
    odometry.csv
    position_cmd.csv
    so3_cmd.csv           # if present
  manifest.json           # bag list + robot config used
```

`manifest.json` schema:
```json
{
  "robot": "race16",
  "config": { ... },
  "bags": [
    {"name": "race16_bag_20260118_120207", "path": "race16_bag_20260118_120207/", "frames": 10173}
  ]
}
```

### 3. `build_episodes.py`

Converts extracted bag directories into fixed-length NPZ episodes with inline quality filtering.

**CLI:**
```bash
python -m scripts.world_model.build_episodes \
    --data_dir dataset_out \
    --out_dir data/race16 \
    --episode_len 64 \
    --stride 32 \
    --img_size 128 \
    --min_speed 0.1
```

**Processing per bag:**
1. Load `image_index.csv` + sensor CSVs from the bag subdirectory.
2. Sort all by timestamp.
3. Detect and remove idle segments: odom speed < `min_speed` for 10+ consecutive frames.
4. Detect timestamp gaps: dt > 2x median frame dt. Episodes don't span gaps.
5. Within each valid contiguous run, slice fixed-length windows at the given stride.
6. Nearest-neighbor sync all sensor CSVs to image timestamps.
7. Save NPZ episodes with global episode counter across all bags.

**Reads `manifest.json`** to discover bags and robot config. No need to re-specify `--robot`.

**Handles missing sensors:** If a CSV (e.g., depth, IMU) is absent, fills with zeros and logs a warning. For present CSVs, if the nearest-neighbor sync gap exceeds `--max-sync-gap-ms` (default: 100ms), that frame's sensor reading is zero-filled and a warning is logged.

**NPZ episode format (unchanged from current):**
```
rgb:       (T, H, W, 3) uint8
depth:     (T, H, W) float32
imu:       (T, D_imu) float32
odom:      (T, D_odom) float32
action:    (T, D_act) float32
```

**Output metadata:**
```
data/race16/
  ep_00000.npz
  ep_00001.npz
  ...
  metadata.json    # total episodes, per-bag counts, sensor dims, robot config, quality params
```

### 4. `select_episodes.py`

Selects a diverse subset of NPZ episodes using dynamics-weighted Farthest Point Sampling.

**CLI:**
```bash
python -m scripts.world_model.select_episodes \
    --data_dir data/race16 \
    --out_dir data/race16_selected \
    --target-count 5000 \
    --encoder none \
    --dynamics-weight 0.6
```

**Descriptor construction (per episode):**

| Component | Dims | Source | Default Weight |
|-----------|------|--------|----------------|
| Action statistics | 2 * D_act | mean + std of action over the episode | 0.6 |
| Odom delta | 6 | body-frame displacement start-to-end: dx, dy, dz, ax, ay, az (axis-angle rotation, consistent with `odom_delta.py`) | 0.2 |
| Visual summary | D_vis | depends on `--encoder` | 0.2 |

**Visual encoder options (`--encoder`):**
- `none` (default): No visual features. Descriptor = action stats + odom delta. Fast, no GPU needed.
- `cnn`: Lightweight CNN (or MobileNet) features from first + last frame. ~256-dim.
- `dinov2`: DINOv2 CLS token averaged over 4 evenly-spaced frames. 384-dim. Requires `external/dino`.
- `vjepa2`: V-JEPA2 spatiotemporal embedding of the full clip. Richest but slowest. Requires `external/vjepa2`.

When `--encoder none`, visual weight is redistributed proportionally to preserve the 3:1 action-to-odom ratio (action stats get 0.75, odom delta gets 0.25).

**Selection algorithm:**
1. Load all episodes, compute per-episode descriptors.
2. Z-score normalize each component independently.
3. Apply component weights.
4. Farthest Point Sampling to `--target-count`.
5. Copy selected episodes to output directory (or `--symlink` for disk savings).

**Output:**
```
data/race16_selected/
  ep_00000.npz
  ep_00001.npz
  ...
  selection_stats.json    # descriptor statistics, FPS distances, per-cluster breakdown
  metadata.json           # inherited from source + selection params
```

### 5. Shared Utilities

#### `robot_config.py`

Location: `scripts/world_model/robot_config.py`

```python
def load_robot_config(name_or_path: str, overrides: dict | None = None) -> dict:
    """Load a robot config from configs/robots/ or a file path, merge CLI overrides."""
    ...

def add_robot_args(parser: argparse.ArgumentParser) -> None:
    """Add --robot and common override args (--color-topic, etc.) to a parser."""
    ...

def apply_overrides(config: dict, overrides: dict) -> dict:
    """Merge dot-notation overrides into a config dict."""
    ...
```

Used by `extract_dataset.py` and `build_episodes.py`. The `select_episodes.py` reads metadata.json which already contains the resolved config.

#### `sync_utils.py`

Location: `scripts/world_model/sync_utils.py`

Shared time-synchronization logic, factored out of `preprocess_race16.py` and `build_episodes.py`:

```python
def nearest_merge(
    target_ts: np.ndarray,
    src_df: pd.DataFrame,
    max_gap_ns: int | None = None,
) -> np.ndarray:
    """Nearest-neighbor merge with optional max-gap zero-fill."""
    ...
```

Both `preprocess_race16.py` and `build_episodes.py` import from here instead of duplicating the function.

## Files Changed

| File | Change |
|------|--------|
| `scripts/world_model/extract_dataset.py` | Rewrite: multi-bag, config-driven topics, manifest output |
| `scripts/world_model/build_episodes.py` | Rewrite: manifest-aware, multi-bag, quality filtering, metadata output |

## New Files

| File | Purpose |
|------|---------|
| `configs/robots/race16.yaml` | Race16 drone topic/sensor config |
| `scripts/world_model/select_episodes.py` | Dynamics-weighted FPS episode selection |
| `scripts/world_model/robot_config.py` | Shared YAML config loader with CLI override merging |
| `scripts/world_model/sync_utils.py` | Shared `nearest_merge` with max-gap tolerance |

## Unchanged Files

| File | Reason |
|------|--------|
| `train_world_model.py` | Already consumes NPZ directories |
| `eval_mpc.py` | Already works with trained checkpoints |
| `src/world_model/` | Architecture, trainer, backbones, dataloaders all unchanged |
| `scripts/world_model/data_pipeline/` | Preserved — serves HF dataset curation purpose |
| `scripts/world_model/preprocess_race16.py` | Preserved — standalone shortcut for HF-hosted race16 data. Will import `nearest_merge` from `sync_utils.py` instead of its local copy. |
