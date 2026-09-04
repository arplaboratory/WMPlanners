"""Direct loader for manifest-first wm_planner Gold datasets."""

from __future__ import annotations

import io
import json
from collections import OrderedDict
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
        recording_cache_size: int = 2,
    ) -> None:
        value = Path(dataset).expanduser()
        self.manifest_path = value if value.is_file() else value / "dataset_manifest.json"
        self.root = self.manifest_path.parent.resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") not in {1, 2}:
            raise ValueError(f"unsupported wm_planner Gold schema: {self.manifest.get('schema_version')}")
        if self.manifest.get("consumer_format") != "wm_planner":
            raise ValueError("Gold manifest is not a wm_planner product")
        parquet = __import__("pyarrow.parquet", fromlist=["read_table"])
        self.frames_path = self.root / self.manifest["files"]["frames"]
        windows_path = self.root / self.manifest["files"]["windows"]
        self.windows = [
            row for row in parquet.read_table(
                str(windows_path), filters=[("split", "=", split)]
            ).to_pylist()
            if row["split"] == split
        ]
        self._parquet = parquet
        self._frame_cache: OrderedDict[str, dict[int, dict]] = OrderedDict()
        self.recording_cache_size = max(1, int(recording_cache_size))
        self.img_size = img_size
        dimensions = self.manifest["sensor_dimensions"]
        self.D_imu = int(dimensions["imu"])
        self.D_odom = int(dimensions["odom"])
        self.D_act = int(dimensions["action"])
        self.proprio_dim = self.D_imu
        self.action_dim = self.D_act
        if not self.windows:
            raise ValueError(f"Gold dataset has no windows in split {split!r}")

    def _recording_frames(self, recording_id: str) -> dict[int, dict]:
        cached = self._frame_cache.pop(recording_id, None)
        if cached is None:
            rows = self._parquet.read_table(
                str(self.frames_path), filters=[("recording_id", "=", recording_id)]
            ).to_pylist()
            cached = {int(row["frame_index"]): row for row in rows}
            if len(cached) != len(rows):
                raise ValueError(f"duplicate frame indexes in recording {recording_id!r}")
        self._frame_cache[recording_id] = cached
        while len(self._frame_cache) > self.recording_cache_size:
            self._frame_cache.popitem(last=False)
        return cached

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
        frames = self._recording_frames(recording_id)
        frame_indices = [first + offset * step for offset in range(length)]
        try:
            rows = [frames[value] for value in frame_indices]
        except KeyError as exc:
            raise ValueError(f"window {window['window_id']!r} references a missing frame") from exc
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
            "window_id": str(window["window_id"]),
            "recording_id": recording_id,
            "frame_indices": np.asarray(frame_indices, dtype=np.int64),
            "split": str(window["split"]),
        }


collate_gold = collate_npz
