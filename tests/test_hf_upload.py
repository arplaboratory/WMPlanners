"""HF dataset creation from small synthetic NPZ episodes."""

from unittest.mock import patch

import numpy as np
from datasets import load_from_disk

from scripts.world_model.data_pipeline.hf_upload import run_hf_upload


def _make_episode(directory, episode_id, n_frames=4, img_size=4):
    rng = np.random.default_rng(episode_id)
    path = directory / f"ep_{episode_id:05d}.npz"
    np.savez_compressed(
        path,
        rgb=rng.integers(0, 255, (n_frames, img_size, img_size, 3), dtype=np.uint8),
        depth=rng.random((n_frames, img_size, img_size)).astype(np.float32),
        imu=rng.random((n_frames, 10)).astype(np.float32),
        odom=rng.random((n_frames, 13)).astype(np.float32),
        action=rng.random((n_frames, 14)).astype(np.float32),
    )
    return path


def test_upload_creates_dataset_with_expected_columns(tmp_path):
    _make_episode(tmp_path, 0, n_frames=3)
    _make_episode(tmp_path, 1, n_frames=4)

    saved = run_hf_upload(str(tmp_path))
    dataset = load_from_disk(saved)

    assert len(dataset) == 7
    assert {"episode_id", "frame_idx", "image", "depth", "depth_valid"} <= set(dataset.column_names)
    assert {f"odom_{i}" for i in range(13)} <= set(dataset.column_names)
    assert {f"imu_{i}" for i in range(10)} <= set(dataset.column_names)
    assert {f"action_{i}" for i in range(14)} <= set(dataset.column_names)
    assert np.asarray(dataset[0]["depth"]).shape == (4, 4)


def test_empty_directory_creates_output_without_upload(tmp_path):
    with patch("datasets.Dataset.push_to_hub") as push:
        saved = run_hf_upload(str(tmp_path))

    assert (tmp_path / "hf_dataset").is_dir()
    assert saved == str(tmp_path / "hf_dataset")
    push.assert_not_called()


def test_repo_id_pushes_dataset_to_hub(tmp_path):
    _make_episode(tmp_path, 0, n_frames=2)

    with patch("datasets.Dataset.push_to_hub") as push:
        run_hf_upload(str(tmp_path), repo_id="user/test-dataset", private=False)

    push.assert_called_once_with("user/test-dataset", private=False)


def test_no_repo_id_does_not_push(tmp_path):
    _make_episode(tmp_path, 0, n_frames=2)

    with patch("datasets.Dataset.push_to_hub") as push:
        run_hf_upload(str(tmp_path))

    push.assert_not_called()
