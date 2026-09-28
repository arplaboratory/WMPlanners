"""Shared reader for frame/window Parquet datasets."""

from __future__ import annotations

import io
import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class ParquetWindowDataset(Dataset):
    """Read one canonical frames/windows layout using optional schema mappings."""

    def __init__(
        self,
        dataset: str,
        *,
        split: str | None = "train",
        img_size: Optional[int] = None,
        recording_cache_size: int = 2,
        schema_config: str | None = None,
    ) -> None:
        value = Path(dataset).expanduser()
        self.root = (value.parent if value.is_file() else value).resolve()
        self.manifest_path = self.root / "dataset_manifest.json"
        self.manifest = (
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest_path.is_file() else {}
        )
        if self.manifest:
            if self.manifest.get("schema_version") not in {1, 2}:
                raise ValueError(f"unsupported wm_planner Parquet schema: {self.manifest.get('schema_version')}")
            if self.manifest.get("consumer_format") != "wm_planner":
                raise ValueError("Parquet manifest is not a wm_planner product")
        config = {}
        if schema_config is not None:
            config = yaml.safe_load(Path(schema_config).expanduser().read_text(encoding="utf-8")) or {}
            if not isinstance(config, dict):
                raise ValueError("schema config must be a YAML mapping")
        unknown = set(config) - {"files", "columns", "sensors", "split", "sensor_dimensions"}
        if unknown:
            raise ValueError(f"unknown Parquet schema config keys: {sorted(unknown)}")
        self.frame_columns = config.get("columns", {}).get("frames", {})
        self.window_columns = config.get("columns", {}).get("windows", {})
        self.sensors = config.get("sensors", {})
        files = {**self.manifest.get("files", {}), **config.get("files", {})}
        self.frames_path = self.root / files.get("frames", "frames.parquet")
        windows_path = self.root / files.get("windows", "windows.parquet")
        if not self.frames_path.is_file() or not windows_path.is_file():
            raise FileNotFoundError(f"Parquet frames/windows tables are missing under {self.root}")
        parquet = __import__("pyarrow.parquet", fromlist=["read_table"])
        self._parquet = parquet
        all_windows = parquet.read_table(str(windows_path)).to_pylist()
        split_column = self._column("split", "windows")
        split_config = config.get("split", {})
        holdout = split_config.get("holdout_fraction", 0.2 if not self.manifest else None)
        if split is None:
            self.windows = all_windows
        elif holdout is not None and not {"train", "val"}.issubset(
            {str(row.get(split_column)) for row in all_windows}
        ):
            fraction = float(holdout)
            if not 0 < fraction < 1:
                raise ValueError("split.holdout_fraction must be between 0 and 1")
            ordered = sorted(all_windows, key=lambda row: (
                str(row[self._column("recording_id", "windows")]),
                int(row[self._column("first_frame_index", "windows")]),
            ))
            holdout_count = max(1, round(len(ordered) * fraction)) if len(ordered) > 1 else 0
            boundary = len(ordered) - holdout_count
            self.windows = ordered[:boundary] if split == "train" else ordered[boundary:] if split in {"val", "test"} else []
        else:
            self.windows = [row for row in all_windows if row.get(split_column) == split]
        self._frame_cache: OrderedDict[str, dict[int, dict]] = OrderedDict()
        self.recording_cache_size = max(1, int(recording_cache_size))
        self.img_size = img_size
        first = parquet.ParquetFile(str(self.frames_path)).read_row_group(0).slice(0, 1).to_pylist()
        if not first:
            raise ValueError("Parquet frames table is empty")
        dimensions = {**self.manifest.get("sensor_dimensions", {}), **config.get("sensor_dimensions", {})}
        def dimension(name: str) -> int:
            if name in dimensions:
                return int(dimensions[name])
            column = self._column(name, "frames")
            value = first[0].get(column)
            if value is None:
                if name == "action" and self.sensors.get("action_source") == "odom_delta":
                    return dimension("odom")
                raise ValueError(f"missing {name!r} dimension and frame column {column!r}")
            return int(np.asarray(value).shape[-1])
        self.D_imu = dimension("imu")
        self.D_odom = dimension("odom")
        self.D_act = dimension("action")
        self.proprio_dim = self.D_imu
        self.action_dim = self.D_act
        if not self.windows:
            raise ValueError(f"Parquet dataset has no windows in split {split!r}")

    def _column(self, name: str, table: str) -> str:
        mapping = self.frame_columns if table == "frames" else self.window_columns
        return mapping.get(name, name)

    def _field(self, row: dict, name: str, table: str, default=None):
        return row.get(self._column(name, table), default)

    def _recording_frames(self, recording_id: str) -> dict[int, dict]:
        cached = self._frame_cache.pop(recording_id, None)
        if cached is None:
            rows = self._parquet.read_table(
                str(self.frames_path), filters=[(self._column("recording_id", "frames"), "=", recording_id)]
            ).to_pylist()
            cached = {int(self._field(row, "frame_index", "frames")): row for row in rows}
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

    def get_window(
        self, index: int, frame_positions: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        window = self.windows[index]
        w = lambda name, default=None: self._field(window, name, "windows", default)
        recording_id = str(w("recording_id"))
        first = int(w("first_frame_index"))
        length = int(w("length"))
        step = int(w("frame_step", 1))
        frames = self._recording_frames(recording_id)
        positions = range(length) if frame_positions is None else frame_positions
        frame_indices = [first + int(offset) * step for offset in positions]
        try:
            rows = [frames[value] for value in frame_indices]
        except KeyError as exc:
            raise ValueError(f"window {w('window_id', index)!r} references a missing frame") from exc
        f = lambda row, name, default=None: self._field(row, name, "frames", default)
        rgb = np.stack([self._rgb(str(f(row, "rgb_path"))) for row in rows])
        shape = rgb.shape[1:3]
        use_depth = self.sensors.get("depth", True)
        use_imu = self.sensors.get("imu", True)
        depth = np.stack([
            self._depth(f(row, "depth_path") if use_depth else None, shape)
            for row in rows
        ])
        imu = np.asarray([f(row, "imu") for row in rows], dtype=np.float32) if use_imu else np.zeros((len(rows), self.D_imu), dtype=np.float32)
        odom = np.asarray([f(row, "odom") for row in rows], dtype=np.float32)
        if self.sensors.get("action_source", "column") == "odom_delta":
            action = np.zeros((len(rows), self.D_act), dtype=np.float32)
            n = min(self.D_act, self.D_odom)
            action[1:, :n] = odom[1:, :n] - odom[:-1, :n]
        else:
            action = np.asarray([f(row, "action") for row in rows], dtype=np.float32)
        collision_valid = np.asarray([bool(f(row, "collision_valid", False)) for row in rows])
        return {
            "rgb": rgb,
            "depth": depth,
            "depth_valid": np.asarray([bool(f(row, "depth_valid", bool(f(row, "depth_path")))) and use_depth for row in rows]),
            "imu_valid": np.asarray([bool(f(row, "imu_valid", True)) and use_imu for row in rows]),
            "odom_valid": np.asarray([bool(f(row, "odom_valid", True)) for row in rows]),
            "action_valid": np.asarray([bool(f(row, "action_valid", True)) for row in rows]),
            "collision_valid": collision_valid,
            "imu": imu,
            "odom": odom,
            "action": action,
            "collision": np.asarray([float(f(row, "collision", 0.0) or 0.0) for row in rows], dtype=np.float32),
            "timestamp_ns": np.asarray([int(f(row, "timestamp_ns", 0) or 0) for row in rows], dtype=np.int64),
            "window_id": str(w("window_id", f"{recording_id}:{first:06d}")),
            "recording_id": recording_id,
            "frame_indices": np.asarray(frame_indices, dtype=np.int64),
            "split": str(w("split", "")),
        }

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return self.get_window(index)


class ParquetDinoWMDataset(Dataset):
    """Present Parquet windows in the sequence format used by DinoWM."""

    def __init__(
        self, path: str, *, img_size: int, num_hist: int = 3,
        num_pred: int = 1, frameskip: int = 1, sample_mode: bool = True,
        odom_stats=None, imu_stats=None, action_stats=None, collision_stats=None,
        split: str = "train", sample_k: int | None = None,
        schema_config: str | None = None,
    ) -> None:
        self.base = ParquetWindowDataset(path, split=split, img_size=img_size, schema_config=schema_config)
        self.D_odom = self.base.D_odom
        self.D_imu = self.base.D_imu
        self.D_act = self.base.D_act
        self.proprio_dim = self.D_imu
        self.action_dim = self.D_act
        self.sample_mode = sample_mode
        self.sample_k = sample_k if sample_k is not None else num_hist + num_pred
        self.frameskip = frameskip
        self.stats = {
            "odom": odom_stats, "imu": imu_stats, "action": action_stats,
            "collision": collision_stats,
        }
        if num_hist < 1 or num_pred < 1 or frameskip < 1 or self.sample_k < 1:
            raise ValueError("num_hist, num_pred, and frameskip must be positive")

    def __len__(self) -> int:
        return len(self.base)

    @staticmethod
    def _normalize(value: np.ndarray, stats) -> np.ndarray:
        if stats is None:
            return value
        if "mean" in stats and "std" in stats:
            return (value - np.asarray(stats["mean"])) / (np.asarray(stats["std"]) + 1e-8)
        if "min" in stats and "max" in stats:
            return (value - np.asarray(stats["min"])) / (
                np.asarray(stats["max"]) - np.asarray(stats["min"]) + 1e-8
            )
        return value

    def __getitem__(self, index: int) -> dict:
        n_frames = int(self.base._field(self.base.windows[index], "length", "windows"))
        if self.sample_mode:
            span = (self.sample_k - 1) * self.frameskip + 1
            if n_frames >= span:
                start = np.random.randint(0, n_frames - span + 1)
                indices = start + np.arange(self.sample_k) * self.frameskip
            else:
                indices = np.minimum(
                    np.arange(self.sample_k) * self.frameskip, n_frames - 1
                )
        else:
            indices = np.arange(n_frames)
        source = self.base.get_window(index, indices)
        rgb = np.ascontiguousarray(
            source["rgb"].transpose(0, 3, 1, 2), dtype=np.float32
        ) / 255.0
        odom = source["odom"].astype(np.float32, copy=True)
        imu = source["imu"].astype(np.float32, copy=True)
        action = source["action"].astype(np.float32, copy=True)
        collision = source["collision"].astype(np.float32, copy=True).reshape(-1, 1)
        if odom.shape[-1] >= 3:
            origin = odom[0, :3].copy()
            odom[:, :3] -= origin
            if action.shape[-1] >= 3:
                action[:, :3] -= origin
        state = np.concatenate([odom, imu, collision], axis=-1)
        return {
            "rgb": rgb,
            "odom": self._normalize(odom, self.stats["odom"]).astype(np.float32),
            "imu": self._normalize(imu, self.stats["imu"]).astype(np.float32),
            "action": self._normalize(action, self.stats["action"]).astype(np.float32),
            "collision": self._normalize(collision, self.stats["collision"]).astype(np.float32),
            "state": torch.from_numpy(state),
        }

