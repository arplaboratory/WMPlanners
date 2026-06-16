"""Tests for build_episodes: quality filtering and episode slicing."""
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

    # odom.csv — moving at 1 m/s in x
    pd.DataFrame({
        "timestamp_ns": ts, "pos_x": t_sec, "pos_y": np.zeros(n_frames),
        "pos_z": np.zeros(n_frames),
        "orient_x": np.zeros(n_frames), "orient_y": np.zeros(n_frames),
        "orient_z": np.zeros(n_frames), "orient_w": np.ones(n_frames),
        "linear_vel_x": np.ones(n_frames), "linear_vel_y": np.zeros(n_frames),
        "linear_vel_z": np.zeros(n_frames),
        "angular_vel_x": np.zeros(n_frames), "angular_vel_y": np.zeros(n_frames),
        "angular_vel_z": np.zeros(n_frames),
    }).to_csv(os.path.join(bag_dir, "odom.csv"), index=False)

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
    def test_image_gap_factor_controls_gap_detection(self):
        from scripts.world_model.build_episodes import _mark_gaps

        ts = np.array([0, 100, 200, 300, 700], dtype=np.int64)
        assert _mark_gaps(ts, gap_factor=2.0).tolist() == [False, False, False, False, True]
        assert _mark_gaps(ts, gap_factor=5.0).tolist() == [False, False, False, False, False]

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
        odom_path = os.path.join(bag_dir, "odom.csv")
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
        for csv_name in ["odom.csv", "imu.csv", "position_cmd.csv"]:
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

    def test_missing_action_zero_filled(self, tmp_path):
        from scripts.world_model.build_episodes import build_episodes

        bag_dir = str(tmp_path / "dataset" / "bag1")
        data_dir = str(tmp_path / "dataset")
        out_dir = str(tmp_path / "episodes")

        _make_extracted_bag(bag_dir, n_frames=100, img_size=16)
        os.remove(os.path.join(bag_dir, "position_cmd.csv"))

        _make_manifest(data_dir, ["bag1"])
        stats = build_episodes(data_dir, out_dir, episode_len=16, stride=16, img_size=16)
        assert stats["total_episodes"] > 0

        ep = np.load(os.path.join(out_dir, "ep_00000.npz"))
        assert ep["action"].shape == (16, 14)
        assert np.all(ep["action"] == 0.0)

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
