"""Tests for data_pipeline.hf_upload — HF dataset creation from selected windows."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_window(
    bag_name: str = "bag_a",
    window_id: int = 0,
    n_frames: int = 4,
    start_frame: int = 0,
) -> dict:
    """Return a window metadata dict matching Stage 3 output."""
    frame_ids = list(range(start_frame, start_frame + n_frames))
    timestamps = [1_000_000_000 + i * 33_000_000 for i in frame_ids]
    # odom_deltas has n_frames - 1 entries
    odom_deltas = [[0.1 * (i + 1), 0.0, 0.0, 0.0, 0.0, 0.01 * (i + 1)]
                   for i in range(n_frames - 1)]
    return {
        "bag_name": bag_name,
        "window_id": window_id,
        "start_frame_id": frame_ids[0],
        "end_frame_id": frame_ids[-1],
        "start_timestamp": timestamps[0],
        "end_timestamp": timestamps[-1],
        "n_frames": n_frames,
        "frame_ids": frame_ids,
        "timestamps": timestamps,
        "odom_deltas": odom_deltas,
        "descriptor": [0.0] * 390,
    }


def _setup_bag(bags_root: str, bag_name: str, frame_ids: list[int]) -> None:
    """Create a fake bag directory with image_index.csv and dummy images."""
    bag_dir = os.path.join(bags_root, bag_name)
    os.makedirs(bag_dir, exist_ok=True)
    shard_dir = os.path.join(bag_dir, "shard_0000")
    os.makedirs(shard_dir, exist_ok=True)

    rows = []
    for fid in frame_ids:
        color_rel = f"shard_0000/color_{fid:06d}.jpg"
        color_abs = os.path.join(bag_dir, color_rel)
        # Create a tiny 4x4 image
        img = PILImage.new("RGB", (4, 4), color=(fid * 10 % 256, 0, 0))
        img.save(color_abs)
        rows.append({
            "frame_id": fid,
            "timestamp_ns": 1_000_000_000 + fid * 33_000_000,
            "color_path": color_rel,
            "depth_path": "",
        })

    pd.DataFrame(rows).to_csv(os.path.join(bag_dir, "image_index.csv"), index=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReadJsonCorrectly:
    """1. The function reads the JSON correctly."""

    def test_loads_windows_from_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = os.path.join(tmp, "corpus")
            bags_root = os.path.join(tmp, "bags")
            os.makedirs(corpus_dir)

            windows = [_make_window(window_id=0, n_frames=3),
                        _make_window(window_id=1, n_frames=3, start_frame=3)]
            with open(os.path.join(corpus_dir, "selected_windows_full.json"), "w") as f:
                json.dump(windows, f)

            _setup_bag(bags_root, "bag_a", list(range(6)))

            from scripts.world_model.data_pipeline.hf_upload import run_hf_upload
            save_dir = run_hf_upload(corpus_dir, bags_root)

            assert os.path.isdir(save_dir)
            from datasets import load_from_disk
            ds = load_from_disk(save_dir)
            # 2 windows x 3 frames = 6 rows
            assert len(ds) == 6


class TestDatasetColumns:
    """2. The dataset has the right columns and types."""

    def test_columns_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = os.path.join(tmp, "corpus")
            bags_root = os.path.join(tmp, "bags")
            os.makedirs(corpus_dir)

            windows = [_make_window(n_frames=3)]
            with open(os.path.join(corpus_dir, "selected_windows_full.json"), "w") as f:
                json.dump(windows, f)

            _setup_bag(bags_root, "bag_a", list(range(3)))

            from scripts.world_model.data_pipeline.hf_upload import run_hf_upload
            save_dir = run_hf_upload(corpus_dir, bags_root)

            from datasets import load_from_disk
            ds = load_from_disk(save_dir)

            expected_cols = {
                "bag_name", "window_id", "frame_idx_in_window", "frame_id",
                "timestamp_ns", "odom_dx", "odom_dy", "odom_dz",
                "odom_ax", "odom_ay", "odom_az", "image",
            }
            assert set(ds.column_names) == expected_cols

    def test_column_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = os.path.join(tmp, "corpus")
            bags_root = os.path.join(tmp, "bags")
            os.makedirs(corpus_dir)

            windows = [_make_window(n_frames=3)]
            with open(os.path.join(corpus_dir, "selected_windows_full.json"), "w") as f:
                json.dump(windows, f)

            _setup_bag(bags_root, "bag_a", list(range(3)))

            from scripts.world_model.data_pipeline.hf_upload import run_hf_upload
            save_dir = run_hf_upload(corpus_dir, bags_root)

            from datasets import load_from_disk, Image as HFImage, Value
            ds = load_from_disk(save_dir)

            assert isinstance(ds.features["image"], HFImage)
            assert ds.features["bag_name"].dtype == "string"
            assert ds.features["window_id"].dtype == "int64"
            assert ds.features["odom_dx"].dtype == "float64"
            assert ds.features["timestamp_ns"].dtype == "int64"


class TestWindowRowCount:
    """3. Each window produces window_size rows."""

    def test_row_count_per_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = os.path.join(tmp, "corpus")
            bags_root = os.path.join(tmp, "bags")
            os.makedirs(corpus_dir)

            n_frames = 5
            windows = [_make_window(window_id=0, n_frames=n_frames)]
            with open(os.path.join(corpus_dir, "selected_windows_full.json"), "w") as f:
                json.dump(windows, f)

            _setup_bag(bags_root, "bag_a", list(range(n_frames)))

            from scripts.world_model.data_pipeline.hf_upload import run_hf_upload
            save_dir = run_hf_upload(corpus_dir, bags_root)

            from datasets import load_from_disk
            ds = load_from_disk(save_dir)
            assert len(ds) == n_frames

            # Verify frame_idx_in_window goes 0..n_frames-1
            frame_idxs = ds["frame_idx_in_window"]
            assert frame_idxs == list(range(n_frames))

    def test_multiple_windows_multiple_bags(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = os.path.join(tmp, "corpus")
            bags_root = os.path.join(tmp, "bags")
            os.makedirs(corpus_dir)

            windows = [
                _make_window(bag_name="bag_a", window_id=0, n_frames=3, start_frame=0),
                _make_window(bag_name="bag_b", window_id=1, n_frames=4, start_frame=0),
            ]
            with open(os.path.join(corpus_dir, "selected_windows_full.json"), "w") as f:
                json.dump(windows, f)

            _setup_bag(bags_root, "bag_a", list(range(3)))
            _setup_bag(bags_root, "bag_b", list(range(4)))

            from scripts.world_model.data_pipeline.hf_upload import run_hf_upload
            save_dir = run_hf_upload(corpus_dir, bags_root)

            from datasets import load_from_disk
            ds = load_from_disk(save_dir)
            assert len(ds) == 7  # 3 + 4


class TestOdomDeltas:
    """4. Odom deltas are correctly assigned (last frame gets zeros)."""

    def test_last_frame_odom_zeros(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = os.path.join(tmp, "corpus")
            bags_root = os.path.join(tmp, "bags")
            os.makedirs(corpus_dir)

            n_frames = 4
            windows = [_make_window(n_frames=n_frames)]
            with open(os.path.join(corpus_dir, "selected_windows_full.json"), "w") as f:
                json.dump(windows, f)

            _setup_bag(bags_root, "bag_a", list(range(n_frames)))

            from scripts.world_model.data_pipeline.hf_upload import run_hf_upload
            save_dir = run_hf_upload(corpus_dir, bags_root)

            from datasets import load_from_disk
            ds = load_from_disk(save_dir)

            # Last frame (index 3) should have zero odom
            last = ds[n_frames - 1]
            assert last["odom_dx"] == 0.0
            assert last["odom_dy"] == 0.0
            assert last["odom_dz"] == 0.0
            assert last["odom_ax"] == 0.0
            assert last["odom_ay"] == 0.0
            assert last["odom_az"] == 0.0

            # First frame should have non-zero odom
            first = ds[0]
            assert first["odom_dx"] == pytest.approx(0.1)
            assert first["odom_az"] == pytest.approx(0.01)

    def test_interior_frame_odom_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = os.path.join(tmp, "corpus")
            bags_root = os.path.join(tmp, "bags")
            os.makedirs(corpus_dir)

            n_frames = 4
            windows = [_make_window(n_frames=n_frames)]
            with open(os.path.join(corpus_dir, "selected_windows_full.json"), "w") as f:
                json.dump(windows, f)

            _setup_bag(bags_root, "bag_a", list(range(n_frames)))

            from scripts.world_model.data_pipeline.hf_upload import run_hf_upload
            save_dir = run_hf_upload(corpus_dir, bags_root)

            from datasets import load_from_disk
            ds = load_from_disk(save_dir)

            # Frame index 1 (second frame) should have odom_deltas[1]
            row1 = ds[1]
            assert row1["odom_dx"] == pytest.approx(0.2)
            assert row1["odom_az"] == pytest.approx(0.02)


class TestEmptySelection:
    """5. The function handles empty selection gracefully."""

    def test_empty_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = os.path.join(tmp, "corpus")
            bags_root = os.path.join(tmp, "bags")
            os.makedirs(corpus_dir)
            os.makedirs(bags_root)

            with open(os.path.join(corpus_dir, "selected_windows_full.json"), "w") as f:
                json.dump([], f)

            from scripts.world_model.data_pipeline.hf_upload import run_hf_upload
            save_dir = run_hf_upload(corpus_dir, bags_root)

            assert os.path.isdir(save_dir)
            from datasets import load_from_disk
            ds = load_from_disk(save_dir)
            assert len(ds) == 0
            # All expected columns should be present
            for col in ["image", "bag_name", "window_id", "odom_dx"]:
                assert col in ds.column_names


class TestHubPush:
    """Push to hub is called when repo_id is provided."""

    def test_push_called(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = os.path.join(tmp, "corpus")
            bags_root = os.path.join(tmp, "bags")
            os.makedirs(corpus_dir)

            windows = [_make_window(n_frames=2)]
            with open(os.path.join(corpus_dir, "selected_windows_full.json"), "w") as f:
                json.dump(windows, f)

            _setup_bag(bags_root, "bag_a", list(range(2)))

            with patch("datasets.Dataset.push_to_hub") as mock_push:
                from scripts.world_model.data_pipeline.hf_upload import run_hf_upload
                run_hf_upload(corpus_dir, bags_root, repo_id="user/test-dataset", private=False)
                mock_push.assert_called_once_with("user/test-dataset", private=False)

    def test_no_push_when_no_repo_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = os.path.join(tmp, "corpus")
            bags_root = os.path.join(tmp, "bags")
            os.makedirs(corpus_dir)

            windows = [_make_window(n_frames=2)]
            with open(os.path.join(corpus_dir, "selected_windows_full.json"), "w") as f:
                json.dump(windows, f)

            _setup_bag(bags_root, "bag_a", list(range(2)))

            with patch("datasets.Dataset.push_to_hub") as mock_push:
                from scripts.world_model.data_pipeline.hf_upload import run_hf_upload
                run_hf_upload(corpus_dir, bags_root)
                mock_push.assert_not_called()
