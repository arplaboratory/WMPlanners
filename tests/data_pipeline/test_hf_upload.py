"""Tests for data_pipeline.hf_upload — HF dataset creation from NPZ episodes."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest


def _make_episode(out_dir: str, ep_id: int, n_frames: int = 4, img_size: int = 4) -> str:
    """Create a fake NPZ episode file."""
    path = os.path.join(out_dir, f"ep_{ep_id:05d}.npz")
    np.savez_compressed(
        path,
        rgb=np.random.randint(0, 255, (n_frames, img_size, img_size, 3), dtype=np.uint8),
        depth=np.random.rand(n_frames, img_size, img_size).astype(np.float32),
        imu=np.random.rand(n_frames, 10).astype(np.float32),
        odom=np.random.rand(n_frames, 13).astype(np.float32),
        action=np.random.rand(n_frames, 14).astype(np.float32),
    )
    return path


class TestBasicUpload:
    """HF dataset is created from NPZ episodes."""

    def test_creates_dataset_from_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_episode(tmp, 0, n_frames=3)
            _make_episode(tmp, 1, n_frames=4)

            from scripts.world_model.data_pipeline.hf_upload import run_hf_upload
            save_dir = run_hf_upload(tmp)

            assert os.path.isdir(save_dir)
            from datasets import load_from_disk
            ds = load_from_disk(save_dir)
            assert len(ds) == 7  # 3 + 4 frames


class TestDatasetColumns:
    """Dataset has the right columns."""

    def test_columns_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_episode(tmp, 0, n_frames=2)

            from scripts.world_model.data_pipeline.hf_upload import run_hf_upload
            save_dir = run_hf_upload(tmp)

            from datasets import load_from_disk
            ds = load_from_disk(save_dir)

            assert "episode_id" in ds.column_names
            assert "frame_idx" in ds.column_names
            assert "image" in ds.column_names
            # Check sensor columns exist (odom has 13 dims)
            assert "odom_0" in ds.column_names
            assert "odom_12" in ds.column_names
            # action has 14 dims
            assert "action_0" in ds.column_names
            assert "action_13" in ds.column_names
            # imu has 10 dims
            assert "imu_0" in ds.column_names
            assert "imu_9" in ds.column_names
            # depth
            assert "depth_mean" in ds.column_names


class TestEmptyDirectory:
    """Handles empty episodes directory."""

    def test_no_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            from scripts.world_model.data_pipeline.hf_upload import run_hf_upload
            save_dir = run_hf_upload(tmp)
            assert os.path.isdir(save_dir)


class TestHubPush:
    """Push to hub is called when repo_id is provided."""

    def test_push_called(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_episode(tmp, 0, n_frames=2)

            with patch("datasets.Dataset.push_to_hub") as mock_push:
                from scripts.world_model.data_pipeline.hf_upload import run_hf_upload
                run_hf_upload(tmp, repo_id="user/test-dataset", private=False)
                mock_push.assert_called_once_with("user/test-dataset", private=False)

    def test_no_push_when_no_repo_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_episode(tmp, 0, n_frames=2)

            with patch("datasets.Dataset.push_to_hub") as mock_push:
                from scripts.world_model.data_pipeline.hf_upload import run_hf_upload
                run_hf_upload(tmp)
                mock_push.assert_not_called()
