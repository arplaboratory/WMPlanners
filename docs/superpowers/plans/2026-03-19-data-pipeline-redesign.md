# Data Pipeline Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the data pipeline to support multi-bag extraction, per-robot YAML configs, and dynamics-weighted episode selection for world model training.

**Architecture:** Staged pipeline — `extract_dataset.py` (MCAP → images+CSVs) → `build_episodes.py` (images+CSVs → NPZ) → `select_episodes.py` (NPZ → diverse NPZ subset). Shared utilities in `robot_config.py` and `sync_utils.py`. Per-robot YAML configs in `configs/robots/`.

**Tech Stack:** Python, PyYAML, NumPy, Pandas, OpenCV, SciPy (Rotation), scikit-learn (StandardScaler), mcap/mcap_ros2, pytest

**Spec:** `docs/superpowers/specs/2026-03-19-data-pipeline-redesign-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `configs/robots/race16.yaml` | Create | Race16 drone topic/sensor config |
| `scripts/world_model/robot_config.py` | Create | YAML config loader + CLI override merging |
| `scripts/world_model/sync_utils.py` | Create | Shared `nearest_merge` with max-gap tolerance |
| `scripts/world_model/extract_dataset.py` | Rewrite | Multi-bag, config-driven MCAP extraction |
| `scripts/world_model/build_episodes.py` | Rewrite | Manifest-aware NPZ episode builder with quality filtering |
| `scripts/world_model/select_episodes.py` | Create | Dynamics-weighted FPS episode selection |
| `scripts/world_model/preprocess_race16.py` | Modify (minor) | Import `nearest_merge` from `sync_utils` |
| `tests/test_robot_config.py` | Create | Tests for config loading and overrides |
| `tests/test_sync_utils.py` | Create | Tests for nearest_merge with gap tolerance |
| `tests/test_build_episodes.py` | Create | Tests for quality filtering and episode slicing |
| `tests/test_select_episodes.py` | Create | Tests for descriptor construction and FPS selection |
| `tests/test_extract_dataset.py` | Create | Tests for manifest generation and topic dispatch |

---

### Task 1: Robot Config — YAML Schema + Loader

**Design note:** The spec mentions `--color-topic`, `--depth-topic` convenience flags. We use a single generic `--override topics.color=/new/topic` mechanism instead — simpler, covers all fields, no per-field CLI args to maintain.

**Files:**
- Create: `configs/robots/race16.yaml`
- Create: `scripts/world_model/robot_config.py`
- Create: `tests/test_robot_config.py`

- [ ] **Step 1: Create configs/robots/ directory and write the race16 YAML config**

Run: `mkdir -p configs/robots`

```yaml
# configs/robots/race16.yaml
version: 1
name: race16

topics:
  color: /race16/cam1/color/image_raw/compressed
  depth: /race16/cam1/aligned_depth_to_color/image_raw
  imu: /race16/cam1/imu
  odom: /race16/odom
  position_cmd: /race16/position_cmd
  so3_cmd: /race16/so3_cmd

sensors:
  color_encoding: compressed
  depth_encoding: 16UC1
  imu_dim: 10
  odom_dim: 13

action:
  source: position_cmd
  dim: 14
```

- [ ] **Step 2: Write tests for robot_config.py**

```python
# tests/test_robot_config.py
import os
import pytest
import yaml

from scripts.world_model.robot_config import (
    load_robot_config,
    apply_overrides,
)

def test_load_by_name():
    cfg = load_robot_config("race16")
    assert cfg["name"] == "race16"
    assert cfg["topics"]["color"] == "/race16/cam1/color/image_raw/compressed"
    assert cfg["action"]["source"] == "position_cmd"
    assert cfg["action"]["dim"] == 14

def test_load_by_path(tmp_path):
    custom = {
        "version": 1, "name": "test_bot",
        "topics": {"color": "/test/color", "odom": "/test/odom"},
        "sensors": {"color_encoding": "raw", "odom_dim": 6},
        "action": {"source": "odom", "dim": 6},
    }
    p = tmp_path / "test_bot.yaml"
    p.write_text(yaml.dump(custom))
    cfg = load_robot_config(str(p))
    assert cfg["name"] == "test_bot"

def test_apply_overrides():
    cfg = {"topics": {"color": "/old"}, "action": {"dim": 14}}
    result = apply_overrides(cfg, {"topics.color": "/new", "action.dim": "6"})
    assert result["topics"]["color"] == "/new"
    assert result["action"]["dim"] == "6"

def test_load_with_overrides():
    cfg = load_robot_config("race16", overrides={"topics.color": "/override/topic"})
    assert cfg["topics"]["color"] == "/override/topic"
    # Other fields unchanged
    assert cfg["action"]["source"] == "position_cmd"

def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        load_robot_config("nonexistent_robot_xyz")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_robot_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.world_model.robot_config'`

- [ ] **Step 4: Implement robot_config.py**

```python
# scripts/world_model/robot_config.py
"""Robot configuration loader — YAML files with CLI override support."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

# Resolve configs/robots/ relative to repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _REPO_ROOT / "configs" / "robots"


def load_robot_config(
    name_or_path: str,
    overrides: dict[str, str] | None = None,
) -> dict:
    """Load a robot config by name or file path, optionally applying overrides.

    Lookup order:
      1. configs/robots/{name_or_path}.yaml
      2. Treat name_or_path as a file path
    """
    yaml_path = _CONFIGS_DIR / f"{name_or_path}.yaml"
    if yaml_path.is_file():
        path = yaml_path
    elif Path(name_or_path).is_file():
        path = Path(name_or_path)
    else:
        raise FileNotFoundError(
            f"Robot config not found: tried {yaml_path} and {name_or_path}"
        )

    with open(path) as f:
        cfg = yaml.safe_load(f)

    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def apply_overrides(cfg: dict, overrides: dict[str, str]) -> dict:
    """Merge dot-notation overrides into a config dict.

    Example: {"topics.color": "/new/topic"} sets cfg["topics"]["color"].
    """
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        parts = key.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return cfg


def add_robot_args(parser) -> None:
    """Add --robot and common topic override args to an argparse parser."""
    parser.add_argument("--robot", required=True,
                        help="Robot config name (looks in configs/robots/) or path to YAML")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides in dot.notation=value format")


def parse_overrides(override_list: list[str]) -> dict[str, str]:
    """Parse ['key=value', ...] into {'key': 'value', ...}."""
    result = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        k, v = item.split("=", 1)
        result[k] = v
    return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_robot_config.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add configs/robots/race16.yaml scripts/world_model/robot_config.py tests/test_robot_config.py
git commit -m "feat: add robot config loader with YAML schema and CLI overrides"
```

---

### Task 2: Shared Sync Utilities

**Files:**
- Create: `scripts/world_model/sync_utils.py`
- Create: `tests/test_sync_utils.py`
- Modify: `scripts/world_model/preprocess_race16.py` (replace `_nearest_merge`)

- [ ] **Step 1: Write tests for sync_utils**

```python
# tests/test_sync_utils.py
import numpy as np
import pandas as pd
import pytest

from scripts.world_model.sync_utils import nearest_merge


def _make_sensor_df(timestamps, values):
    """Helper to build a sensor DataFrame with timestamp_ns + one value column."""
    return pd.DataFrame({"timestamp_ns": timestamps, "val": values})


def test_exact_match():
    src = _make_sensor_df([100, 200, 300], [1.0, 2.0, 3.0])
    target = np.array([100, 200, 300])
    result = nearest_merge(target, src)
    np.testing.assert_array_equal(result[:, 0], [1.0, 2.0, 3.0])


def test_nearest_neighbor():
    src = _make_sensor_df([100, 200, 300], [1.0, 2.0, 3.0])
    target = np.array([110, 260, 290])  # 110→100, 260→300, 290→300
    result = nearest_merge(target, src)
    np.testing.assert_array_equal(result[:, 0], [1.0, 3.0, 3.0])


def test_single_source_row():
    src = _make_sensor_df([100], [5.0])
    target = np.array([90, 100, 200])
    result = nearest_merge(target, src)
    np.testing.assert_array_equal(result[:, 0], [5.0, 5.0, 5.0])


def test_max_gap_zero_fill():
    src = _make_sensor_df([100, 200, 300], [1.0, 2.0, 3.0])
    target = np.array([100, 500])  # 500 is 200ns away from nearest (300)
    result = nearest_merge(target, src, max_gap_ns=150)
    assert result[0, 0] == 1.0  # within gap
    assert result[1, 0] == 0.0  # exceeds gap — zero-filled


def test_max_gap_none_no_zeroing():
    src = _make_sensor_df([100, 200], [1.0, 2.0])
    target = np.array([100, 9999])
    result = nearest_merge(target, src, max_gap_ns=None)
    assert result[1, 0] == 2.0  # no gap limit, takes nearest


def test_multi_column():
    src = pd.DataFrame({
        "timestamp_ns": [100, 200, 300],
        "a": [1.0, 2.0, 3.0],
        "b": [10.0, 20.0, 30.0],
    })
    target = np.array([100, 300])
    result = nearest_merge(target, src)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result[0], [1.0, 10.0])
    np.testing.assert_array_equal(result[1], [3.0, 30.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sync_utils.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement sync_utils.py**

```python
# scripts/world_model/sync_utils.py
"""Shared time-synchronization utilities for the data pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd


def nearest_merge(
    target_ts: np.ndarray,
    src_df: pd.DataFrame,
    max_gap_ns: int | None = None,
) -> np.ndarray:
    """For each target timestamp, pick the nearest row in src_df.

    Assumes *src_df* has a ``timestamp_ns`` column. All other columns are
    treated as value columns and returned as an ``(N, D)`` float32 array.

    Parameters
    ----------
    target_ts : (N,) int64 array of target timestamps in nanoseconds.
    src_df : DataFrame with ``timestamp_ns`` and D value columns.
    max_gap_ns : If set, zero-fill rows where the nearest source timestamp
        is farther than this many nanoseconds from the target.
    """
    src_ts = src_df["timestamp_ns"].values
    vals = src_df.drop(columns=["timestamp_ns"]).values.astype(np.float32)
    n_target = len(target_ts)
    n_src = len(src_ts)

    if n_src == 0:
        return np.zeros((n_target, vals.shape[1]), dtype=np.float32)
    if n_src == 1:
        result = np.tile(vals[0], (n_target, 1))
        if max_gap_ns is not None:
            gaps = np.abs(src_ts[0] - target_ts)
            result = result.copy()
            result[gaps > max_gap_ns] = 0.0
        return result

    idx = np.searchsorted(src_ts, target_ts, side="left")
    idx = np.clip(idx, 1, n_src - 1)

    left = np.abs(src_ts[idx - 1] - target_ts)
    right = np.abs(src_ts[idx] - target_ts)
    closer = np.where(left <= right, idx - 1, idx)

    result = vals[closer]

    if max_gap_ns is not None:
        gaps = np.minimum(left, right)
        exceed = gaps > max_gap_ns
        if exceed.any():
            result = result.copy()
            result[exceed] = 0.0

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sync_utils.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Update preprocess_race16.py to use shared sync_utils**

In `scripts/world_model/preprocess_race16.py`:
- Remove the local `_nearest_merge` function (lines 29–42).
- Add import: `from scripts.world_model.sync_utils import nearest_merge`
- Replace all calls from `_nearest_merge(...)` to `nearest_merge(...)`.

- [ ] **Step 6: Verify preprocess_race16.py still parses**

Run: `python -c "import scripts.world_model.preprocess_race16"`
Expected: No errors (won't run main, just imports)

- [ ] **Step 7: Commit**

```bash
git add scripts/world_model/sync_utils.py tests/test_sync_utils.py scripts/world_model/preprocess_race16.py
git commit -m "feat: extract shared nearest_merge into sync_utils with max-gap tolerance"
```

---

### Task 3: Rewrite `extract_dataset.py` — Config-Driven Multi-Bag Extraction

**Files:**
- Rewrite: `scripts/world_model/extract_dataset.py`

- [ ] **Step 1: Rewrite extract_dataset.py**

The rewrite must:
- Import `load_robot_config`, `add_robot_args`, `parse_overrides` from `robot_config`.
- Accept `--mcap` as `nargs="+"` for multi-bag.
- Accept `--robot` + `--override` for config.
- Loop over each MCAP path, create `output/{bag_stem}/` subdirectory.
- Build a topic-to-handler dispatch dict from the config's `topics` map instead of hardcoded if/elif.
- Handlers for each sensor type:
  - `color`: `save_compressed_image()` if `sensors.color_encoding == "compressed"`, else decode raw Image msg.
  - `depth`: `save_depth_image()` using `sensors.depth_encoding`.
  - `imu`: CSV writer with the existing IMU_FIELDS.
  - `odom`: CSV writer with ODOM_FIELDS.
  - Any other topic key (e.g., `position_cmd`, `so3_cmd`): generic CSV writer that auto-detects numeric fields from the first ROS message. This handles dynamic action topics.
- After all bags, write `manifest.json` at the output root with schema:
  ```json
  {
    "robot": "race16",
    "config": { <full config dict> },
    "bags": [
      {"name": "bag_stem", "path": "bag_stem/", "frames": 10173}
    ]
  }
  ```
- Preserve the `CSVManager` class and image save functions from current code.
- Keep `SHARD_SIZE = 500` default, configurable via `--shard-size`.

Key design for generic CSV handler — for topics like `position_cmd` and `so3_cmd` where field names vary by robot, use the existing hardcoded field lists for known topic types (`imu`, `odom`, `position_cmd`, `so3_cmd`). The CSV filename matches the topic key (e.g., topic key `position_cmd` → `position_cmd.csv`).

- [ ] **Step 2: Write tests for extract_dataset**

```python
# tests/test_extract_dataset.py
"""Tests for extract_dataset manifest generation and topic dispatch.

Note: MCAP reading is not tested here (requires binary bag files).
We test the config-driven dispatch logic and manifest output.
"""
import json
import os

import pytest


class TestManifest:
    def test_write_manifest(self, tmp_path):
        from scripts.world_model.extract_dataset import write_manifest

        config = {"version": 1, "name": "race16", "topics": {}, "action": {"source": "position_cmd", "dim": 14}}
        bags_info = [
            {"name": "bag1", "path": "bag1/", "frames": 100},
            {"name": "bag2", "path": "bag2/", "frames": 200},
        ]
        write_manifest(str(tmp_path), config, bags_info)

        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            m = json.load(f)
        assert m["robot"] == "race16"
        assert len(m["bags"]) == 2
        assert m["bags"][0]["frames"] == 100
        assert m["config"]["action"]["source"] == "position_cmd"

    def test_empty_bags_list(self, tmp_path):
        from scripts.world_model.extract_dataset import write_manifest

        config = {"version": 1, "name": "test"}
        write_manifest(str(tmp_path), config, [])
        with open(tmp_path / "manifest.json") as f:
            m = json.load(f)
        assert m["bags"] == []


class TestTopicDispatch:
    def test_build_topic_handlers(self):
        from scripts.world_model.extract_dataset import build_topic_handlers

        config = {
            "topics": {
                "color": "/robot/color",
                "depth": "/robot/depth",
                "imu": "/robot/imu",
                "odom": "/robot/odom",
                "position_cmd": "/robot/cmd",
            },
            "sensors": {"color_encoding": "compressed", "depth_encoding": "16UC1"},
        }
        handlers = build_topic_handlers(config)
        # Each topic string maps to a handler key
        assert "/robot/color" in handlers
        assert "/robot/imu" in handlers
        assert handlers["/robot/color"]["type"] == "color"
        assert handlers["/robot/imu"]["type"] == "imu"

    def test_optional_topic_excluded(self):
        from scripts.world_model.extract_dataset import build_topic_handlers

        config = {
            "topics": {"color": "/robot/color", "odom": "/robot/odom"},
            "sensors": {"color_encoding": "raw"},
        }
        handlers = build_topic_handlers(config)
        # Only the topics listed are included
        assert len(handlers) == 2
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_extract_dataset.py -v`
Expected: FAIL — `ImportError` (functions don't exist yet)

- [ ] **Step 4: Manually test with a dry import**

Run: `python -c "from scripts.world_model.extract_dataset import extract, main"`
Expected: No errors. (Actual extraction requires MCAP files which are large binary data.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_extract_dataset.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/world_model/extract_dataset.py tests/test_extract_dataset.py
git commit -m "feat: rewrite extract_dataset with multi-bag support and config-driven topics"
```

---

### Task 4: Rewrite `build_episodes.py` — Manifest-Aware with Quality Filtering

**Files:**
- Rewrite: `scripts/world_model/build_episodes.py`
- Create: `tests/test_build_episodes.py`

- [ ] **Step 1: Write tests for build_episodes core logic**

```python
# tests/test_build_episodes.py
import json
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import pytest


def _make_extracted_bag(bag_dir, n_frames=100, img_size=32):
    """Create a minimal extracted bag directory for testing."""
    os.makedirs(bag_dir, exist_ok=True)
    shard_dir = os.path.join(bag_dir, "shard_0000")
    os.makedirs(shard_dir, exist_ok=True)

    ts = np.linspace(1_000_000_000, 2_000_000_000, n_frames, dtype=np.int64)
    t_sec = (ts - ts[0]) / 1e9

    # Write color images (RGB) and depth images
    color_paths, depth_paths = [], []
    for i in range(n_frames):
        cp = os.path.join(shard_dir, f"color_{i:06d}.jpg")
        dp = os.path.join(shard_dir, f"depth_{i:06d}.png")
        cv2.imwrite(cp, np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8))
        cv2.imwrite(dp, np.random.randint(0, 1000, (img_size, img_size), dtype=np.uint16))
        color_paths.append(os.path.join("shard_0000", f"color_{i:06d}.jpg"))
        depth_paths.append(os.path.join("shard_0000", f"depth_{i:06d}.png"))

    # image_index.csv
    pd.DataFrame({
        "frame_id": range(n_frames), "timestamp_ns": ts,
        "color_path": color_paths, "depth_path": depth_paths,
    }).to_csv(os.path.join(bag_dir, "image_index.csv"), index=False)

    # odometry.csv — moving at 1 m/s in x
    pd.DataFrame({
        "timestamp_ns": ts, "pos_x": t_sec, "pos_y": np.zeros(n_frames),
        "pos_z": np.zeros(n_frames),
        "orient_x": np.zeros(n_frames), "orient_y": np.zeros(n_frames),
        "orient_z": np.zeros(n_frames), "orient_w": np.ones(n_frames),
        "linear_vel_x": np.ones(n_frames), "linear_vel_y": np.zeros(n_frames),
        "linear_vel_z": np.zeros(n_frames),
        "angular_vel_x": np.zeros(n_frames), "angular_vel_y": np.zeros(n_frames),
        "angular_vel_z": np.zeros(n_frames),
    }).to_csv(os.path.join(bag_dir, "odometry.csv"), index=False)

    # imu.csv
    pd.DataFrame({
        "timestamp_ns": ts,
        **{f"orientation_{c}": (np.ones(n_frames) if c == "w" else np.zeros(n_frames))
           for c in "xyzw"},
        **{f"angular_velocity_{c}": np.zeros(n_frames) for c in "xyz"},
        **{f"linear_acceleration_{c}": (np.full(n_frames, 9.81) if c == "z" else np.zeros(n_frames))
           for c in "xyz"},
    }).to_csv(os.path.join(bag_dir, "imu.csv"), index=False)

    # position_cmd.csv (action source)
    pd.DataFrame({
        "timestamp_ns": ts,
        **{f"position_{c}": t_sec for c in "xyz"},
        **{f"velocity_{c}": np.ones(n_frames) for c in "xyz"},
        **{f"acceleration_{c}": np.zeros(n_frames) for c in "xyz"},
        **{f"jerk_{c}": np.zeros(n_frames) for c in "xyz"},
        "yaw": np.zeros(n_frames), "yaw_dot": np.zeros(n_frames),
    }).to_csv(os.path.join(bag_dir, "position_cmd.csv"), index=False)

    return ts


def _make_manifest(data_dir, bag_names, config=None):
    """Write a manifest.json."""
    if config is None:
        config = {
            "version": 1, "name": "race16",
            "topics": {}, "sensors": {"odom_dim": 13, "imu_dim": 10},
            "action": {"source": "position_cmd", "dim": 14},
        }
    manifest = {
        "robot": config["name"],
        "config": config,
        "bags": [{"name": n, "path": f"{n}/"} for n in bag_names],
    }
    with open(os.path.join(data_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)


class TestBuildEpisodes:
    def test_single_bag_produces_episodes(self, tmp_path):
        from scripts.world_model.build_episodes import build_episodes

        bag_dir = str(tmp_path / "dataset" / "bag1")
        data_dir = str(tmp_path / "dataset")
        out_dir = str(tmp_path / "episodes")

        _make_extracted_bag(bag_dir, n_frames=100, img_size=16)
        _make_manifest(data_dir, ["bag1"])

        stats = build_episodes(data_dir, out_dir, episode_len=16, stride=16, img_size=16)
        assert stats["total_episodes"] > 0
        # Check NPZ content
        ep = np.load(os.path.join(out_dir, "ep_00000.npz"))
        assert ep["rgb"].shape == (16, 16, 16, 3)
        assert ep["odom"].shape[0] == 16
        assert ep["action"].shape[0] == 16

    def test_idle_frames_excluded(self, tmp_path):
        from scripts.world_model.build_episodes import build_episodes

        bag_dir = str(tmp_path / "dataset" / "bag1")
        data_dir = str(tmp_path / "dataset")
        out_dir = str(tmp_path / "episodes")

        _make_extracted_bag(bag_dir, n_frames=100, img_size=16)
        # Make all frames idle (zero velocity)
        odom_path = os.path.join(bag_dir, "odometry.csv")
        df = pd.read_csv(odom_path)
        df["linear_vel_x"] = 0.0
        df.to_csv(odom_path, index=False)

        _make_manifest(data_dir, ["bag1"])
        stats = build_episodes(data_dir, out_dir, episode_len=16, stride=16, img_size=16, min_speed=0.1)
        assert stats["total_episodes"] == 0

    def test_timestamp_gap_splits_episodes(self, tmp_path):
        from scripts.world_model.build_episodes import build_episodes

        bag_dir = str(tmp_path / "dataset" / "bag1")
        data_dir = str(tmp_path / "dataset")
        out_dir = str(tmp_path / "episodes")

        _make_extracted_bag(bag_dir, n_frames=100, img_size=16)
        # Insert a large timestamp gap at frame 50
        idx_path = os.path.join(bag_dir, "image_index.csv")
        df = pd.read_csv(idx_path)
        # Normal dt is ~10ms; make gap at frame 50 = 1 second
        df.loc[50:, "timestamp_ns"] += 1_000_000_000
        df.to_csv(idx_path, index=False)
        # Also update odom/imu/action timestamps to match
        for csv_name in ["odometry.csv", "imu.csv", "position_cmd.csv"]:
            csv_path = os.path.join(bag_dir, csv_name)
            sdf = pd.read_csv(csv_path)
            sdf.loc[50:, "timestamp_ns"] += 1_000_000_000
            sdf.to_csv(csv_path, index=False)

        _make_manifest(data_dir, ["bag1"])
        stats = build_episodes(data_dir, out_dir, episode_len=16, stride=16, img_size=16)
        # Should produce episodes from two separate runs, not spanning the gap
        # Each run has ~50 frames → ~3 episodes per run at stride 16
        assert stats["total_episodes"] > 0
        # Verify no episode spans the gap by checking timestamps
        for f in sorted(os.listdir(out_dir)):
            if not f.endswith(".npz"):
                continue
            ep = np.load(os.path.join(out_dir, f))
            # All odom timestamps within an episode should be from the same run
            # (no 1-second jumps)

    def test_missing_imu_zero_filled(self, tmp_path):
        from scripts.world_model.build_episodes import build_episodes

        bag_dir = str(tmp_path / "dataset" / "bag1")
        data_dir = str(tmp_path / "dataset")
        out_dir = str(tmp_path / "episodes")

        _make_extracted_bag(bag_dir, n_frames=100, img_size=16)
        # Delete imu.csv
        os.remove(os.path.join(bag_dir, "imu.csv"))

        _make_manifest(data_dir, ["bag1"])
        stats = build_episodes(data_dir, out_dir, episode_len=16, stride=16, img_size=16)
        assert stats["total_episodes"] > 0
        ep = np.load(os.path.join(out_dir, "ep_00000.npz"))
        # IMU should be zeros
        assert np.all(ep["imu"] == 0.0)

    def test_metadata_written(self, tmp_path):
        from scripts.world_model.build_episodes import build_episodes

        bag_dir = str(tmp_path / "dataset" / "bag1")
        data_dir = str(tmp_path / "dataset")
        out_dir = str(tmp_path / "episodes")

        _make_extracted_bag(bag_dir, n_frames=100, img_size=16)
        _make_manifest(data_dir, ["bag1"])
        build_episodes(data_dir, out_dir, episode_len=16, stride=16, img_size=16)

        meta_path = os.path.join(out_dir, "metadata.json")
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert "total_episodes" in meta
        assert "sensor_dims" in meta
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_build_episodes.py -v`
Expected: FAIL — `ImportError` (build_episodes doesn't have `build_episodes` function yet)

- [ ] **Step 3: Implement build_episodes.py**

Rewrite `scripts/world_model/build_episodes.py` with these key functions:

```python
def build_episodes(
    data_dir: str,
    out_dir: str,
    episode_len: int = 64,
    stride: int = 32,
    img_size: int = 128,
    min_speed: float = 0.1,
    max_sync_gap_ms: float = 100.0,
    idle_persistence: int = 10,
) -> dict:
    """Build NPZ episodes from all bags in data_dir. Returns stats dict."""
```

Logic:
1. Read `manifest.json` from `data_dir` → get bag list + config.
2. Determine action CSV name from `config["action"]["source"]` (e.g., `"position_cmd"` → `"position_cmd.csv"`).
3. For each bag in manifest:
   a. Load `image_index.csv`, `odometry.csv`, `imu.csv`, `{action_source}.csv`. Missing CSVs → warn + zero arrays.
   b. Sort by timestamp. Compute frame-to-frame speed from odom `linear_vel_*` columns.
   c. Mark idle frames: speed < `min_speed` for `idle_persistence`+ consecutive frames.
   d. Detect timestamp gaps: dt > 2× median dt.
   e. Split into contiguous valid runs (non-idle, no gaps).
   f. `nearest_merge` each sensor to image timestamps with `max_gap_ns = max_sync_gap_ms * 1_000_000`.
   g. Within each run, slice windows of `episode_len` at `stride`. Save NPZ.
4. Write `metadata.json` with total episodes, per-bag counts, sensor dims, config, quality params.
5. Return stats dict.

Use `sync_utils.nearest_merge` for time sync. Use the existing `_load_image` and `_resolve_path` helpers (adapted from current code).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_build_episodes.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/world_model/build_episodes.py tests/test_build_episodes.py
git commit -m "feat: rewrite build_episodes with manifest support, quality filtering, and gap detection"
```

---

### Task 5: `select_episodes.py` — Dynamics-Weighted FPS Selection

**Files:**
- Create: `scripts/world_model/select_episodes.py`
- Create: `tests/test_select_episodes.py`

- [ ] **Step 1: Write tests for descriptor construction and FPS**

```python
# tests/test_select_episodes.py
import json
import os
import tempfile

import numpy as np
import pytest


def _make_episodes(out_dir, n_episodes=50, ep_len=16, img_size=8, odom_dim=13, act_dim=14):
    """Create synthetic NPZ episodes with varying dynamics."""
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_episodes):
        speed = 0.5 + i * 0.1  # varying speeds
        angle = i * 0.2  # varying turns
        np.savez_compressed(
            os.path.join(out_dir, f"ep_{i:05d}.npz"),
            rgb=np.random.randint(0, 255, (ep_len, img_size, img_size, 3), dtype=np.uint8),
            depth=np.random.rand(ep_len, img_size, img_size).astype(np.float32),
            imu=np.random.randn(ep_len, 10).astype(np.float32),
            odom=np.column_stack([
                np.linspace(0, speed, ep_len),  # pos_x
                np.zeros((ep_len, 2)),  # pos_y, pos_z
                np.zeros((ep_len, 3)),  # orient_xyz
                np.ones((ep_len, 1)),   # orient_w
                np.full((ep_len, 1), speed),  # lin_vel_x
                np.zeros((ep_len, 2)),  # lin_vel_yz
                np.full((ep_len, 1), angle),  # ang_vel_x
                np.zeros((ep_len, 2)),  # ang_vel_yz
            ]).astype(np.float32),
            action=np.random.randn(ep_len, act_dim).astype(np.float32) * (i + 1),
        )
    # metadata
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({"total_episodes": n_episodes, "sensor_dims": {
            "odom": odom_dim, "imu": 10, "action": act_dim,
        }}, f)


class TestDescriptorConstruction:
    def test_action_stats_shape(self, tmp_path):
        from scripts.world_model.select_episodes import compute_descriptors
        _make_episodes(str(tmp_path), n_episodes=10, act_dim=14)
        descs, _ = compute_descriptors(str(tmp_path), encoder="none")
        # action stats (2*14) + odom delta (6) = 34
        assert descs.shape == (10, 34)

    def test_descriptors_vary(self, tmp_path):
        from scripts.world_model.select_episodes import compute_descriptors
        _make_episodes(str(tmp_path), n_episodes=10)
        descs, _ = compute_descriptors(str(tmp_path), encoder="none")
        # Not all identical
        assert not np.allclose(descs[0], descs[-1])


class TestFPSSelection:
    def test_selects_target_count(self, tmp_path):
        from scripts.world_model.select_episodes import select_episodes
        _make_episodes(str(tmp_path), n_episodes=50)
        out = str(tmp_path / "selected")
        stats = select_episodes(
            str(tmp_path), out, target_count=10, encoder="none", dynamics_weight=0.6,
        )
        assert stats["selected"] == 10
        # Count actual episode files
        eps = [f for f in os.listdir(out) if f.endswith(".npz")]
        assert len(eps) == 10

    def test_target_exceeds_available(self, tmp_path):
        from scripts.world_model.select_episodes import select_episodes
        _make_episodes(str(tmp_path), n_episodes=5)
        out = str(tmp_path / "selected")
        stats = select_episodes(
            str(tmp_path), out, target_count=100, encoder="none", dynamics_weight=0.6,
        )
        assert stats["selected"] == 5

    def test_selection_stats_written(self, tmp_path):
        from scripts.world_model.select_episodes import select_episodes
        _make_episodes(str(tmp_path), n_episodes=20)
        out = str(tmp_path / "selected")
        select_episodes(str(tmp_path), out, target_count=5, encoder="none", dynamics_weight=0.6)
        assert os.path.exists(os.path.join(out, "selection_stats.json"))
        assert os.path.exists(os.path.join(out, "metadata.json"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_select_episodes.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement select_episodes.py**

```python
# scripts/world_model/select_episodes.py
"""Dynamics-weighted Farthest Point Sampling for episode selection."""
```

Key functions:

```python
def compute_episode_odom_delta(odom: np.ndarray) -> np.ndarray:
    """Compute start-to-end body-frame displacement from odom array.

    odom: (T, D_odom) where first 7 cols are pos_x,y,z, orient_x,y,z,w.
    Returns: (6,) array [dx, dy, dz, ax, ay, az] in body frame.
    """
    # Use first and last row
    pos_start, pos_end = odom[0, :3], odom[-1, :3]
    quat_start = odom[0, 3:7]  # (x, y, z, w)
    quat_end = odom[-1, 3:7]
    r_start = Rotation.from_quat(quat_start)
    r_end = Rotation.from_quat(quat_end)
    dp_world = pos_end - pos_start
    dp_body = r_start.inv().apply(dp_world)
    r_delta = r_start.inv() * r_end
    axis_angle = r_delta.as_rotvec()
    return np.concatenate([dp_body, axis_angle])


def compute_descriptors(
    data_dir: str,
    encoder: str = "none",
) -> tuple[np.ndarray, list[str]]:
    """Compute per-episode descriptors. Returns (N, D) array + episode filenames."""
    # For each episode:
    #   action_stats = concat(mean(action), std(action))  # (2*D_act,)
    #   odom_delta = compute_episode_odom_delta(odom)      # (6,)
    #   descriptor = concat(action_stats, odom_delta)      # (2*D_act + 6,)
    # If encoder != "none": also compute visual features and append


def select_episodes(
    data_dir: str,
    out_dir: str,
    target_count: int,
    encoder: str = "none",
    dynamics_weight: float = 0.6,
    symlink: bool = False,
) -> dict:
    """Select diverse episodes via dynamics-weighted FPS."""
    # 1. compute_descriptors(data_dir, encoder)
    # 2. Z-score normalize each component block independently
    # 3. Apply weights:
    #    - If encoder == "none": action_weight=0.75, odom_weight=0.25
    #    - Else: action_weight=dynamics_weight, odom_weight=(1-dynamics_weight)/2,
    #            visual_weight=(1-dynamics_weight)/2
    #    Weight application: multiply each block by sqrt(weight) before FPS
    #    (so squared distances reflect weighted contributions)
    # 4. FPS (reuse algorithm from stage3_select.py — inline, not import)
    # 5. Copy or symlink selected episodes to out_dir, renumber sequentially
    # 6. Write selection_stats.json + metadata.json
```

The FPS algorithm is inlined (same logic as `scripts/world_model/data_pipeline/stage3_select.py:53-81`) to avoid coupling to the data_pipeline package.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_select_episodes.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/world_model/select_episodes.py tests/test_select_episodes.py
git commit -m "feat: add dynamics-weighted FPS episode selection with optional visual encoder support"
```

---

### Task 6: Integration Verification

**Files:** No new files — verify the full pipeline works end-to-end.

- [ ] **Step 1: Run all new tests together**

Run: `python -m pytest tests/test_robot_config.py tests/test_sync_utils.py tests/test_build_episodes.py tests/test_select_episodes.py -v`
Expected: All tests PASS

- [ ] **Step 2: Run existing tests to verify no regressions**

Run: `python -m pytest tests/ -v --ignore=tests/rloclab --ignore=tests/cjepa -x`
Expected: All existing tests still PASS (especially `tests/data_pipeline/` which depends on unmodified code)

- [ ] **Step 3: Verify preprocess_race16.py import works**

Run: `python -c "from scripts.world_model.preprocess_race16 import main; print('OK')"`
Expected: Prints "OK"

- [ ] **Step 4: Verify CLI help for all three pipeline scripts**

Run:
```bash
python -m scripts.world_model.extract_dataset --help
python -m scripts.world_model.build_episodes --help
python -m scripts.world_model.select_episodes --help
```
Expected: Each prints usage with the expected arguments.

- [ ] **Step 5: Commit any final fixes**

If any fixes were needed, commit them:
```bash
git add -u
git commit -m "fix: integration fixes for data pipeline redesign"
```

- [ ] **Step 6: Final commit — all tests green**

If everything passes clean, no additional commit needed. Verify with:
```bash
git log --oneline -10
```
Expected: See the 5 commits from Tasks 1-5.
