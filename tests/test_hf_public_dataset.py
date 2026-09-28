"""Small end-to-end loader check using real rows from public collect0."""

import json
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from datasets import Dataset, Image

from world_model.data import HFWindowDataset


DATASET_ID = "lquan9/collect0"


def _public_frames():
    query = urlencode({
        "dataset": DATASET_ID, "config": "default", "split": "train",
        "offset": 0, "length": 4,
    })
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    with urlopen(url, timeout=30) as response:
        page = json.load(response)

    rows = [entry["row"] for entry in page["rows"]]
    assert len(rows) == 4, "collect0 should expose four sample frames"
    for row in rows:
        with urlopen(row["image"]["src"], timeout=30) as response:
            image_bytes = response.read()
        row["image"] = {"bytes": image_bytes, "path": None}
    return Dataset.from_list(rows).cast_column("image", Image())


def test_public_collect0_loads_sampled_window():
    frames = _public_frames()
    expected_columns = {
        "episode_id", "frame_idx", "image", "odom_12", "imu_9", "action_12"
    }
    assert expected_columns <= set(frames.column_names)

    dataset = HFWindowDataset(
        frames, img_size=32, sample_mode=True, sample_k=4
    )
    assert len(dataset) == 1

    sample = dataset[0]
    assert sample["rgb"].shape == (4, 3, 32, 32)
    assert sample["odom"].shape == (4, dataset.D_odom)
    assert sample["imu"].shape == (4, dataset.D_imu)
    assert sample["action"].shape == (4, dataset.D_act)
    assert np.isfinite(sample["rgb"]).all()
    assert sample["rgb"].max() > 0
    assert np.isfinite(sample["odom"]).all()
