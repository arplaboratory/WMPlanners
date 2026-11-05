"""Local Parquet schema mappings should not require a second dataset reader."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image

from world_model.data import (
    CachedWindowDataset, ParquetDinoWMDataset, collate_for_dataset, open_window_dataset,
)


def test_configured_parquet_tables_and_sensor_policy(tmp_path: Path):
    Image.fromarray(np.full((4, 4, 3), 50, dtype=np.uint8)).save(tmp_path / "frame.png")
    frames = pa.Table.from_pylist([
        {"session": "r1", "frame_index": index, "image_file": "frame.png",
         "odom": [float(index), 0.0], "imu": [10.0], "action": [99.0]}
        for index in range(3)
    ])
    windows = pa.Table.from_pylist([
        {"window_id": f"w{index}", "session": "r1", "first_frame_index": index,
         "frame_count": 2, "frame_step": 1}
        for index in range(2)
    ])
    pq.write_table(frames, tmp_path / "images.parquet")
    pq.write_table(windows, tmp_path / "clips.parquet")
    config = {
        "files": {"frames": "images.parquet", "windows": "clips.parquet"},
        "columns": {"frames": {"recording_id": "session", "rgb_path": "image_file"},
                    "windows": {"recording_id": "session", "length": "frame_count"}},
        "sensors": {"depth": False, "imu": False, "action_source": "odom_delta"},
        "sensor_dimensions": {"action": 1},
        "split": {"holdout_fraction": 0.5},
    }
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(yaml.safe_dump(config))

    train = open_window_dataset(dataset_type="hf", path=str(tmp_path), split="train",
                                image_size=4, schema_config=str(schema_path))
    val = open_window_dataset(dataset_type="hf", path=str(tmp_path), split="val",
                              image_size=4, schema_config=str(schema_path))
    assert len(train) == len(val) == 1
    sample = train[0]
    assert sample["rgb"].shape == (2, 4, 4, 3)
    np.testing.assert_array_equal(sample["odom"][:, 0], [0, 1])
    np.testing.assert_array_equal(sample["imu"], 0)
    np.testing.assert_array_equal(sample["depth"], 0)
    np.testing.assert_array_equal(sample["action"][:, 0], [0, 1])
    dino = ParquetDinoWMDataset(str(tmp_path), img_size=4, sample_k=2,
                                schema_config=str(schema_path))
    assert dino[0]["rgb"].shape == (2, 3, 4, 4)
    np.testing.assert_array_equal(dino[0]["odom"][:, 0], [0, 1])
    cached = CachedWindowDataset(train, max_windows=1)
    assert collate_for_dataset(cached)([cached[0]]).rgb.shape == (1, 2, 3, 4, 4)
