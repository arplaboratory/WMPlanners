"""Direct loader for manifest-first wm_planner Gold datasets."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .dataloaders import collate_npz


class GoldWindowDataset(Dataset):
    """Load windows whose payload frames are stored exactly once in Gold."""

    def __init__(
        self,
        dataset: str,
        *,
        split: str = "train",
        img_size: Optional[int] = None,
    ) -> None:
        value = Path(dataset).expanduser()
        self.manifest_path = value if value.is_file() else value / "dataset_manifest.json"
        self.root = self.manifest_path.parent.resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ValueError(f"unsupported wm_planner Gold schema: {self.manifest.get('schema_version')}")
        parquet = __import__("pyarrow.parquet", fromlist=["read_table"])
        frames_path = self.root / self.manifest["files"]["frames"]
        windows_path = self.root / self.manifest["files"]["windows"]
        frames = parquet.read_table(str(frames_path)).to_pylist()
        self.windows = [
            row for row in parquet.read_table(str(windows_path)).to_pylist()
            if row["split"] == split
        ]
        self.frames = {
            (str(row["recording_id"]), int(row["frame_index"])): row for row in frames
        }
        self.img_size = img_size
        dimensions = self.manifest["sensor_dimensions"]
        self.D_imu = int(dimensions["imu"])
        self.D_odom = int(dimensions["odom"])
        self.D_act = int(dimensions["action"])
        self.proprio_dim = self.D_imu
        self.action_dim = self.D_act
        if not self.windows:
            raise ValueError(f"Gold dataset has no windows in split {split!r}")

    def __len__(self) -> int:
        return len(self.windows)

    def _rgb(self, relative: str) -> np.ndarray:
        with Image.open(self.root / relative) as image:
            image = image.convert("RGB")
            if self.img_size is not None:
                image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
            return np.asarray(image, dtype=np.uint8)

    def _depth(self, relative: str | None, shape: tuple[int, int]) -> np.ndarray:
        if not relative:
            return np.zeros(shape, dtype=np.float32)
        path = self.root / relative
        if path.name.endswith(".npy.zst"):
            import pyarrow as pa

            payload = path.read_bytes()
            size = int.from_bytes(payload[:8], "little")
            raw = pa.decompress(payload[8:], decompressed_size=size, codec="zstd").to_pybytes()
            value = np.load(io.BytesIO(raw), allow_pickle=False).astype(np.float32)
        elif path.suffix == ".npy":
            value = np.load(path).astype(np.float32)
        else:
            with Image.open(path) as image:
                raw = np.asarray(image)
            value = raw.astype(np.float32) / 1000.0 if raw.dtype == np.uint16 else raw.astype(np.float32)
        if self.img_size is not None and value.shape != (self.img_size, self.img_size):
            tensor = torch.from_numpy(value).unsqueeze(0).unsqueeze(0)
            value = F.interpolate(tensor, size=(self.img_size, self.img_size), mode="nearest")[0, 0].numpy()
        return value

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        window = self.windows[index]
        recording_id = str(window["recording_id"])
        first = int(window["first_frame_index"])
        length = int(window["length"])
        step = int(window["frame_step"])
        rows = [self.frames[(recording_id, first + offset * step)] for offset in range(length)]
        rgb = np.stack([self._rgb(str(row["rgb_path"])) for row in rows])
        shape = rgb.shape[1:3]
        return {
            "rgb": rgb,
            "depth": np.stack([self._depth(row.get("depth_path"), shape) for row in rows]),
            "depth_valid": np.asarray([bool(row.get("depth_valid")) for row in rows]),
            "imu_valid": np.asarray([bool(row.get("imu_valid", True)) for row in rows]),
            "odom_valid": np.asarray([bool(row.get("odom_valid", True)) for row in rows]),
            "action_valid": np.asarray([bool(row.get("action_valid", True)) for row in rows]),
            "collision_valid": np.asarray([bool(row.get("collision_valid")) for row in rows]),
            "imu": np.asarray([row["imu"] for row in rows], dtype=np.float32),
            "odom": np.asarray([row["odom"] for row in rows], dtype=np.float32),
            "action": np.asarray([row["action"] for row in rows], dtype=np.float32),
            "collision": np.asarray([row.get("collision", 0.0) for row in rows], dtype=np.float32),
            "timestamp_ns": np.asarray([row["timestamp_ns"] for row in rows], dtype=np.int64),
        }


collate_gold = collate_npz
