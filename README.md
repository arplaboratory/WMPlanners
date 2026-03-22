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

Converts the extracted images and CSVs into fixed-length `.npz` episode files ready for training. This is where raw sensor streams become aligned, filtered, multi-modal training samples.

**Step 1 — Load manifest and sensor config**

Reads `manifest.json` (written by Stage 0) to discover every bag and its sensor layout. The manifest stores which CSV column is the action source (e.g. `position_cmd`), and the dimensionality of each sensor (IMU=10, odom=13, action=14 for race16). This means the same code works for different robots — just change the config.

**Step 2 — Time-synchronise sensors to image timestamps**

Each sensor (IMU, odometry, position commands) is recorded asynchronously at its own rate. The script uses `nearest_merge()` from `sync_utils.py` to align them:
- For each image timestamp, find the sensor reading with the closest timestamp.
- If the closest reading is more than `--max-sync-gap-ms` (default 100 ms) away, zero-fill that sensor for that frame.
- This produces a single table where every row is one image frame with all sensor modalities aligned.

**Step 3 — Quality filtering**

Two filters remove unusable data:

*Idle detection:* Computes speed as `‖linear_velocity‖` from odometry. If speed stays below `--min-speed` (default 0.1 m/s) for `--idle-persistence` (default 10) consecutive frames, that entire run is marked invalid. The persistence requirement prevents single slow frames (e.g. during a turn) from being incorrectly flagged — only sustained stops are removed.

```
Frame speeds: [2.1, 1.8, 0.05, 0.03, 0.02, ...(10+ frames)..., 0.04, 1.9, 2.0]
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                This run is marked idle (10+ frames below 0.1 m/s)
```

*Gap detection:* Computes the time delta between consecutive frames and compares to the median delta. If any gap exceeds 2× the median, it marks a segment boundary. This catches recording pauses, dropped frames, or bag file boundaries. Episodes are never allowed to span a gap.

```
Frame timestamps: [100, 133, 166, 200, 233, 800, 833, 866]
                                              ^^^
                                              Gap > 2× median(33ms) → segment split
Valid runs:       [--------run 1---------]    [---run 2---]
```

**Step 4 — Temporal subsampling (frame_skip)**

With `--frame-skip N`, only every Nth frame is kept. This reduces the effective FPS and increases the temporal span of each episode. The subsampling happens *after* time-sync (so each kept frame has correct sensor data) but *before* episode slicing.

Example: source at 30 fps with `--frame-skip 6`:
- Effective rate: 30 / 6 = **5 fps**
- Each episode of 50 frames spans 50 × 0.2s = **10 seconds** of real time
- The stride is applied in raw-frame space, so `--stride 50` with `--frame-skip 6` means episodes are spaced 50 × 6 = 300 raw frames apart

**Step 5 — Episode slicing**

Each valid run (contiguous non-idle, non-gap frames) is sliced into fixed-length episodes:
- A sliding window of `episode_len × frame_skip` raw frames advances by `stride × frame_skip` raw frames.
- Only every `frame_skip`-th frame within each window is kept.
- If a valid run is too short for even one episode, it is skipped.

For each episode, images are loaded from disk, resized to `img_size × img_size`, and packed into arrays:

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `rgb` | `(T, H, W, 3)` | uint8 | Resized colour frames |
| `depth` | `(T, H, W)` | float32 | Depth in metres (16-bit PNG value / 1000) |
| `imu` | `(T, 10)` | float32 | Orientation quat (4) + angular vel (3) + linear accel (3) |
| `odom` | `(T, 13)` | float32 | Position (3) + orientation quat (4) + linear vel (3) + angular vel (3) |
| `action` | `(T, 14)` | float32 | Position/velocity/acceleration commands (12) + yaw + yaw_dot |

Episodes are saved as `ep_00000.npz`, `ep_00001.npz`, etc. A `metadata.json` records episode count, per-bag counts, sensor dimensions, and all pipeline parameters used.

#### Stage 2: Select Episodes (`select_episodes.py`)

Selects a maximally diverse subset of episodes using **Farthest Point Sampling (FPS)** so the training set covers the full range of behaviours rather than over-sampling common motions.

**Step 1 — Descriptor computation**

Each episode is summarised as a single vector that captures its dynamics:
- **Action mean** (D_act dims): average commanded action across all frames — captures the typical behaviour in this episode (e.g. "mostly turning left" vs "accelerating forward").
- **Action std** (D_act dims): variance of commands — captures whether the episode has steady or rapidly changing actions.
- **Odometry delta** (6 dims): body-frame displacement from the first frame to the last, computed as:
  - `(dx, dy, dz)`: position change rotated into the body frame of the first frame.
  - `(droll, dpitch, dyaw)`: total rotation decomposed into Euler angles.

This produces a `(2 × D_act + 6)`-dim descriptor per episode (e.g. 34-dim for race16 with D_act=14).

The descriptor captures both *what the robot was told to do* (action stats) and *what actually happened* (odom delta). Two episodes with similar commands but different outcomes (e.g. one hit an obstacle) will have different descriptors.

**Step 2 — Z-score normalisation with block weighting**

The action block and odom block are normalised independently using `StandardScaler` (zero mean, unit variance per column). This prevents high-magnitude columns (e.g. position commands in metres) from dominating low-magnitude ones (e.g. yaw rate in rad/s).

After normalisation, blocks are weighted:
- Action block: `√0.75` (75% of squared-distance contribution)
- Odom block: `√0.25` (25% of squared-distance contribution)

This means FPS prioritises action diversity (what the robot did) over trajectory diversity (where it ended up), while still ensuring spatial coverage.

**Step 3 — Farthest Point Sampling (FPS)**

FPS is a greedy algorithm that builds a maximally spread-out subset:

1. **Seed**: pick the episode farthest from the centroid of all descriptors. This starts the selection at an "extreme" episode rather than an average one.

2. **Initialise distances**: compute the distance from every episode to the seed. Store these as `min_dists[i]` — the distance from episode `i` to its nearest selected neighbour (initially just the seed).

3. **Greedy loop**: repeat until `target_count` episodes are selected:
   - Pick `next = argmax(min_dists)` — the episode most isolated from all currently selected episodes.
   - Compute distances from every episode to `next`.
   - Update: `min_dists[i] = min(min_dists[i], dist(i, next))` — each episode's distance to the selected set can only decrease or stay the same.
   - Mark `next` as selected.

The key invariant is that `min_dists[i]` always equals the distance from episode `i` to its *nearest* selected episode. The `np.minimum` update maintains this in O(N) per step — you only need to check the new point, since old nearest-neighbour distances can't increase.

**Concrete example** — selecting 3 from 6 episodes in a 1D descriptor space:

```
Episodes:    A(0)   B(1)   C(3)   D(7)   E(9)   F(10)
Centroid = 5.0

Step 0: Farthest from centroid = F(10). Select F.
        min_dists = [10, 9, 7, 3, 1, -]

Step 1: argmax(min_dists) = A(10). Select A.
        Update: min_dists = [-, 1, 3, 3, 1, -]
                             A  B  C  D  E  F
        (B is 1 from A, C is 3 from A which < 7 from F, etc.)

Step 2: argmax(min_dists) = C or D (tie at 3). Select C.
        Result: {A(0), C(3), F(10)} — maximally spread out.
        Skipped: B(1) too close to A, E(9) too close to F.
```

This runs in O(N × K) time where N = total episodes and K = selected count.

**Step 4 — Output**

Selected episodes are copied to the output directory with sequential naming (`ep_00000.npz`, `ep_00001.npz`, ...). Metadata is carried forward from Stage 1 and augmented with selection parameters. A `selection_stats.json` records FPS spread metrics (mean/std/min of nearest-neighbour distances among selected episodes) for monitoring selection quality.

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
