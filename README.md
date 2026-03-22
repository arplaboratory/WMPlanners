# DS-GA 3001
This repo supports the DS-GA 3001 - Embodied Learning and Vision Course Project

| Authors              | Email                |
|-------------------|----------------------|
| **Long Quang**    | lq2146@nyu.edu       |
| **Akash Guna** | ar9799@nyu.edu     |
| **Deepanshu Mody** | dm6262@nyu.edu      |

<sup>
Authors are with New York University.
</sup>

[![Tests](https://github.com/lquan9/NYU_DS_GA_3001/actions/workflows/tests.yml/badge.svg)](https://github.com/lquan9/NYU_DS_GA_3001/actions/workflows/tests.yml)
![Pylint](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/lquan9/d09cfb9f249c9995b9fa1943d90a1d71/raw/ds3001-pylint.json)
[![Docs](https://github.com/lquan9/NYU_DS_GA_3001/actions/workflows/docs.yml/badge.svg)](https://github.com/lquan9/NYU_DS_GA_3001/actions/workflows/docs.yml)
[![Python Version](https://img.shields.io/badge/python-%203.11-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Code](#) | [Paper](#)

`src/cjepa` is a lightweight integration layer for adapting continual learning to JEPA.
<br>
`external/vjepa2` official [V-JEPA 2](https://github.com/facebookresearch/vjepa2) repository.
<br>
`src/rloclab` package provides an Isaac Lab reinforcement learning workflow for Unitree Go2 locomotion and wheeled robots.

## Inspiration and preliminary works

The `rloclab` package builds on and cites the following repositories as upstream references and preliminary implementations:

- **isaac-sim/IsaacLab**: core simulator/task framework and APIs that power `rloclab` [Link](https://github.com/isaac-sim/IsaacLab).  
- **machines-in-motion/rob6323_go2_project**: original course starter workflow and task framing for Go2 in Isaac Lab [Link](https://github.com/machines-in-motion/rob6323_go2_project).
  
- **wheeledlab/wheeledlab**: reference for wheeled drifting task structure and asset conventions integrated in this package [Link](https://github.com/wheeledlab/wheeledlab).

# Quick Start
## Pre-requisites
1. [Git LFS](https://git-lfs.github.com/)
2. [Python Virtual Env](https://docs.python.org/3/tutorial/venv.html)
3. [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/index.html)
4. [UV](https://docs.astral.sh/uv/getting-started/installation/) (optional)
5. [Dialog](https://invisible-island.net/dialog/dialog.html#download)

<details>
<summary><strong>Click for instructions on installing prerequisites</strong></summary>

---
**1. Git-LFS**

We rely on GIT Large File Storage for handling binary files that are required for tracking assets.

```bash
sudo apt install git-lfs
git lfs install
```

For non-sudo users:
```bash
wget https://github.com/git-lfs/git-lfs/releases/download/v3.2.0/git-lfs-linux-amd64-v3.2.0.tar.gz
tar -xvf git-lfs-linux-amd64-v3.2.0.tar.gz
cd git-lfs-3.2.0
chmod +x install.sh
sed -i 's|^prefix="/usr/local"$|prefix="$HOME/.local"|' install.sh
mkdir -p ~/.local/bin/
export PATH="$HOME/.local/bin:$PATH"
./install.sh
```

**2. Python3 Virtual Environment**
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

**3. IsaacLab** (Installed within Python environment)

**4. UV**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**5. Dialog**
On Debian/Ubuntu:
```bash
sudo apt install dialog
```

Source:
```bash
wget https://invisible-island.net/datafiles/release/dialog.tar.gz
tar -xvf dialog.tar.gz
cd dialog-1.3.20220216
./configure --prefix=$HOME/.local
mkdir -p ~/.local/bin
make
make install
export PATH="$HOME/.local/bin:$PATH"
```
---
</details>

### Setup Python workspace (Recommended)
The easiest way to set up the environment and all external dependencies (DINOv2, V-JEPA2) is using the provided setup script. This requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# This will create a .venv, install project dependencies, 
# and set up DINOv2/V-JEPA2 in editable mode.
bash scripts/python_setup.sh

# Activate the environment
source .venv/bin/activate
```

### Manual Setup (Alternative)
If you prefer to set up the environment manually:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
```

```bash
pip install pip --upgrade
# V-JEPA2 runtime dependencies
pip install -r external/vjepa2/requirements.txt
# DINOv2 runtime dependencies
pip install -r external/dino/requirements.txt

# Project dependencies
pip install -e .[dev,docs]
```
<sup>
If using <b>zsh</b>, you may need to run: <code>setopt nonomatch</code> to set glob behavior properly.
</sup>

### Run
```bash
dino --config configs/config.yaml
```
Run the example notebook by launching Jupyter Lab and opening
``docs/notebooks/getting_started.md``.  Build the documentation locally with:

```bash
sphinx-build -b html docs docs/_build/html
```

Execute the tests:

```bash
pytest
```

#### Headless
```bash
TBD
```


## World Model Data Pipeline

End-to-end pipeline that takes raw ROS2 MCAP bag recordings and produces a curated, diverse training dataset on Hugging Face Hub.

```
MCAP bags → Extract → Build Episodes → Select (FPS) → Upload to HF
```

### Quick Start

```bash
# Full pipeline: MCAP → HuggingFace (10-second episodes at 5 fps)
python -m scripts.world_model.data_pipeline.run_pipeline \
    --mcap path/to/race16_bag.mcap \
    --robot race16 \
    --frame-skip 6 --episode-len 50 --stride 50 \
    --target-count 5000 \
    --push-to-hub dmody1/wm-data

# From pre-extracted bags (skip MCAP extraction)
python -m scripts.world_model.data_pipeline.run_pipeline \
    --bags dataset_out \
    --frame-skip 6 --episode-len 50 --stride 50 \
    --target-count 5000
```

### Pipeline Stages

#### Stage 0: Extract (`extract_dataset.py`)

Reads raw ROS2 MCAP bag files and produces a structured directory of images and sensor CSVs. Each bag becomes a subdirectory containing:
- **Sharded colour images** (JPEG) and **depth images** (16-bit PNG) in `shard_0000/`, `shard_0001/`, etc.
- **Sensor CSVs**: `imu.csv` (10-dim: orientation quaternion + angular velocity + linear acceleration), `odometry.csv` (13-dim: position + orientation + velocities), `position_cmd.csv` (14-dim: position/velocity/acceleration commands + yaw).
- **`image_index.csv`**: Maps each `frame_id` to its `timestamp_ns`, `color_path`, and `depth_path`.
- **`manifest.json`**: Lists all extracted bags with frame counts and the robot config used.

Which ROS topics to extract is driven by a robot config file (e.g. `configs/robots/race16.yaml`) that maps topic names to sensor types and specifies encoding and dimensionality.

Skipped when you pass `--bags` to point at an already-extracted directory.

#### Stage 1: Build Episodes (`build_episodes.py`)

Converts the extracted images and CSVs into fixed-length `.npz` episode files ready for training.

1. **Reads `manifest.json`** to discover all bags and their sensor configuration (dimensions, action source).
2. **Time-synchronises** all sensor modalities (IMU, odometry, commands) to image timestamps using nearest-neighbour interpolation. Frames where the nearest sensor reading is more than `--max-sync-gap-ms` away get zero-filled.
3. **Quality filtering**:
   - *Idle detection*: frames where speed < `--min-speed` (default 0.1 m/s) for `--idle-persistence` consecutive frames are marked invalid.
   - *Gap detection*: timestamp jumps > 2× the median frame interval create segment boundaries.
4. **Temporal subsampling** (`--frame-skip N`): takes every Nth raw frame. At 30 fps source, `--frame-skip 6` produces 5 fps effective rate.
5. **Episode slicing**: valid (non-idle, non-gap) runs are sliced into episodes of `--episode-len` frames with `--stride` raw-frame overlap between consecutive episodes.

Each `.npz` episode contains:

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `rgb` | `(T, H, W, 3)` | uint8 | Resized colour frames |
| `depth` | `(T, H, W)` | float32 | Depth in metres (from 16-bit PNG / 1000) |
| `imu` | `(T, 10)` | float32 | IMU readings synced to frame timestamps |
| `odom` | `(T, 13)` | float32 | Odometry: pos(3) + quat(4) + lin_vel(3) + ang_vel(3) |
| `action` | `(T, 14)` | float32 | Position commands synced to frame timestamps |

A `metadata.json` file records episode count, per-bag counts, sensor dimensions, and all pipeline parameters.

#### Stage 2: Select Episodes (`select_episodes.py`)

Selects a maximally diverse subset of episodes using **Farthest Point Sampling (FPS)** so the training set covers the full range of behaviours rather than over-sampling common motions.

1. **Descriptor computation**: for each episode, concatenates `mean(action)`, `std(action)`, and a 6-dim odometry delta (body-frame displacement from first to last frame). This captures both *what the robot was commanded to do* and *where it actually went*.
2. **Z-score normalisation**: the action block and odom block are normalised independently so neither dominates by scale.
3. **FPS algorithm**:
   - Seeds with the episode farthest from the centroid of all descriptors.
   - Each step greedily picks the episode whose minimum distance to any already-selected episode is largest.
   - Repeats until `--target-count` episodes are selected.
4. **Output**: selected episodes are copied (or symlinked) to the output directory with sequential naming (`ep_00000.npz`, ...). Writes `selection_stats.json` with FPS spread metrics.

#### Stage 3: HF Upload (`hf_upload.py`)

Creates a Hugging Face `Dataset` from the selected NPZ episodes and optionally pushes to the Hub.

1. Scans all `ep_*.npz` files in the selected-episodes directory.
2. Explodes each episode into per-frame rows with columns: `episode_id`, `frame_idx`, `image` (PIL), plus numbered columns for each sensor dimension (`odom_0`..`odom_12`, `action_0`..`action_13`, `imu_0`..`imu_9`, `depth_mean`).
3. Saves locally to `hf_dataset/` and, if `--push-to-hub` is set, pushes to the Hub.

Standalone usage (if stages 0-2 are already done):
```bash
python -m scripts.world_model.data_pipeline.hf_upload \
    --episodes-dir pipeline_out/selected \
    --push-to-hub dmody1/wm-data
```

### CLI Reference

| Argument | Description | Default |
|----------|-------------|---------|
| `--mcap PATH [...]` | MCAP bag file(s) — runs Stage 0 | — |
| `--bags PATH` | Pre-extracted bags dir — skips Stage 0 | — |
| `--robot NAME` | Robot config name from `configs/robots/` | `race16` |
| `--episode-len N` | Frames per episode | `50` |
| `--stride N` | Episode stride in raw frames | `50` |
| `--frame-skip N` | Take every Nth frame (temporal subsampling) | `6` |
| `--img-size N` | Resize images to N×N | `128` |
| `--min-speed F` | Idle speed threshold (m/s) | `0.1` |
| `--target-count N` | Episodes to select via FPS | `5000` |
| `--push-to-hub ID` | HF Hub repo ID (e.g. `dmody1/wm-data`) | — |
| `--hf-private` | Make HF dataset private | off |
| `--output DIR` | Base output directory | `pipeline_out` |
| `--skip-build` | Skip Stage 1 | off |
| `--skip-select` | Skip Stage 2 | off |
| `--skip-hf-upload` | Skip Stage 3 | off |

### Output Structure

```
pipeline_out/
├── extracted/                  # Stage 0
│   ├── manifest.json
│   └── race16_bag_.../
│       ├── image_index.csv
│       ├── odometry.csv
│       └── shard_0000/
├── episodes/                   # Stage 1
│   ├── ep_00000.npz ... ep_NNNNN.npz
│   └── metadata.json
└── selected/                   # Stage 2
    ├── ep_00000.npz ... ep_NNNNN.npz
    ├── metadata.json
    ├── selection_stats.json
    └── hf_dataset/             # Stage 3
```

See `docs/data_pipeline.md` for additional details.

## Extra documentation

See `docs/getting_started.md` for a concise step-by-step.

## Resources
- [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/ecosystem.html) — Everything you need to know about IsaacLab, and more!
- [Isaac Lab ANYmal C environment](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c) — This targets ANYmal C (not Unitree Go2), so use it as a reference and adapt robot config, assets, and reward to Go2.
- [DMO (IsaacGym) Go2 walking project page](https://machines-in-motion.github.io/DMO/) • [Go2 walking environment used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/tasks/go2_terrain.py) • [Config file used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/cfg/task/Go2Terrain.yaml) — Look at the function `compute_reward_CaT` (beware that some reward terms have a weight of 0 and thus are deactivated, check weights in the config file); this implementation includes strong reward shaping, domain randomization, and training disturbances for robust sim‑to‑real, but it is written for legacy IsaacGym and the challenge is to re-implement it in Isaac Lab.
- **API References**:
    - [ArticulationData (`robot.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationData) — Contains `root_pos_w`, `joint_pos`, `projected_gravity_b`, etc.
    - [ContactSensorData (`_contact_sensor.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.ContactSensorData) — Contains `net_forces_w` (contact forces).
---

## License

This project is licensed under the MIT license. See [LICENSE](LICENSE).
