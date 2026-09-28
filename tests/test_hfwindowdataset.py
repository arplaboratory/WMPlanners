"""Offline tests for the Hugging Face window loader."""

import numpy as np
from datasets import Dataset

from world_model.data import HFWindowDataset
from world_model.data.normalization import NormStats


def _frames(include_actions=True):
    # Deliberately shuffle frame order to exercise window indexing.
    frame_order = [2, 0, 1]
    rows = {
        "episode_id": [7] * 3,
        "frame_idx": frame_order,
        "image": [None] * 3,
        "depth": [[[0.0, 0.0], [0.0, 0.0]]] * 3,
    }
    for dim in range(13):
        rows[f"odom_{dim}"] = [
            float(frame if dim == 0 else 10 + frame if dim == 7 else 0)
            for frame in frame_order
        ]
    rows["imu_0"] = [float(frame) for frame in frame_order]
    if include_actions:
        rows["action_0"] = [float(frame + 20) for frame in frame_order]
    return Dataset.from_dict(rows)


def test_windows_sort_frames_and_read_actions():
    dataset = HFWindowDataset(_frames(), img_size=2)

    assert len(dataset) == 1
    window = dataset[0]
    np.testing.assert_array_equal(window["odom"][:, 0], [0, 1, 2])
    np.testing.assert_array_equal(window["action"][:, 0], [20, 21, 22])
    assert window["rgb"].shape == (3, 3, 2, 2)


def test_local_upload_directory_resolves_nested_dataset(tmp_path):
    _frames().save_to_disk(str(tmp_path / "hf_dataset"))

    dataset = HFWindowDataset(str(tmp_path), img_size=2)

    assert len(dataset) == 1
    assert dataset.D_act == 1


def test_norm_stats_argument_is_applied():
    odom = np.zeros((2, 13), dtype=np.float32)
    imu = np.zeros((2, 1), dtype=np.float32)
    action = np.zeros((2, 1), dtype=np.float32)
    stats = NormStats.from_arrays(odom, imu, action)

    dataset = HFWindowDataset(_frames(), img_size=2, norm_stats=stats)

    assert dataset._norm_stats is stats
    assert dataset.action_stats is not None


def test_missing_action_columns_use_odom_velocity():
    dataset = HFWindowDataset(_frames(include_actions=False), img_size=2)

    assert dataset.D_act == 6
    assert dataset.action_dim == 6
    window = dataset[0]
    assert window["action"].shape == (3, 6)
    np.testing.assert_array_equal(window["action"][:, 0], [10, 11, 12])
