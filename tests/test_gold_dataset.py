from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
from PIL import Image

from world_model.data.gold_cli import (
    _depth_bytes,
    _nearest,
    export_hf,
    export_npz,
    finalize,
    materialize,
)
from world_model.data.gold_dataset import GoldWindowDataset, collate_gold
from world_model.data.hf_datasets import HFWindowDataset, collate_hf


def _rgb_message() -> NS:
    buffer = io.BytesIO()
    Image.fromarray(np.full((8, 10, 3), 127, dtype=np.uint8)).save(buffer, format="JPEG")
    return NS(format="jpeg", data=buffer.getvalue())


def _messages(count: int = 10):
    rgb = _rgb_message()
    depth = NS(height=8, width=10, step=20, encoding="16UC1",
               data=np.full((8, 10), 1500, dtype=np.uint16).tobytes())
    imu = NS(orientation=NS(x=0, y=0, z=0, w=1), angular_velocity=NS(x=1, y=2, z=3),
             linear_acceleration=NS(x=4, y=5, z=6))
    odom = NS(pose=NS(pose=NS(position=NS(x=0, y=0, z=0), orientation=NS(x=0, y=0, z=0, w=1))),
              twist=NS(twist=NS(linear=NS(x=1, y=0, z=0), angular=NS(x=0, y=0, z=0))))
    action = NS(position=NS(x=0, y=0, z=0), velocity=NS(x=1, y=0, z=0),
                acceleration=NS(x=0, y=0, z=0), jerk=NS(x=0, y=0, z=0), yaw=0, yaw_dot=0)
    sequence = 0
    for index in range(count):
        timestamp = 1_000_000_000 + index * 100_000_000
        for topic, message in (("/camera", rgb), ("/depth", depth), ("/imu", imu), ("/odom", odom), ("/cmd", action)):
            yield topic, timestamp, sequence, message
            sequence += 1


def test_materialize_direct_loader_and_npz_export_are_equivalent(tmp_path: Path, monkeypatch) -> None:
    profile = tmp_path / "robot.yaml"
    profile.write_text("""name: test
topics:
  color: /camera
  depth: /depth
  imu: /imu
  odom: /odom
  position_cmd: /cmd
action:
  source: position_cmd
""", encoding="utf-8")
    source = tmp_path / "bronze.mcap"
    source.write_bytes(b"fixture")
    dataset_root = tmp_path / "gold/product"
    recording_output = dataset_root / "recordings/rec"
    job = {
        "schema_version": 1, "consumer": "wm_planner", "recording_id": "rec",
        "split": "train", "approved_ranges": [{"start_ns": 1_000_000_000, "end_ns": 2_000_000_000}],
        "sources": [{"path": str(source)}], "output_dir": str(recording_output),
        "dataset_output": str(dataset_root), "object_store": str(tmp_path / "gold/.objects"),
        "topic_profile": str(profile), "fingerprint": "fingerprint",
        "recipe": {"options": {"window_length": 4, "window_stride": 2, "min_speed": 0.0}},
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    monkeypatch.setattr("world_model.data.gold_cli._iter_messages", lambda _: _messages())
    materialize(job_path)
    build = {
        "output_root": str(dataset_root), "release_id": "release", "release_hash": "hash",
        "recipe_sha256": "recipe", "adapter": {"adapter": "test"},
        "adapter_executable": "wm-dataset", "recipe_path": "recipe.yaml",
        "object_store": str(tmp_path / "gold/.objects"),
        "jobs": [{"output_dir": str(recording_output), "fingerprint": "fingerprint"}],
    }
    build_path = dataset_root / "gold_build.json"
    build_path.write_text(json.dumps(build), encoding="utf-8")
    manifest = finalize(build_path)
    direct = GoldWindowDataset(str(manifest), split="train")
    assert len(direct) == 4
    sample = direct[0]
    assert sample["rgb"].shape == (4, 8, 10, 3)
    assert np.allclose(sample["depth"], 1.5)
    assert sample["depth_valid"].all()
    assert sample["imu_valid"].all()
    assert not sample["collision_valid"].any()
    export_root = tmp_path / "npz"
    export_npz(manifest, export_root)
    with np.load(export_root / "train/ep_000000.npz") as exported:
        for key in (
            "rgb", "depth", "imu", "odom", "action", "depth_valid", "imu_valid",
            "odom_valid", "action_valid", "collision_valid", "timestamp_ns",
        ):
            assert np.array_equal(exported[key], sample[key])
    rgb_assets = sorted((recording_output / "rgb").iterdir())
    assert len({path.stat().st_ino for path in rgb_assets}) == 1

    hf_root = tmp_path / "hf"
    export_hf(manifest, hf_root)
    from datasets import load_from_disk
    hf = load_from_disk(str(hf_root))["train"]
    assert np.array_equal(np.asarray(hf[0]["image"]), sample["rgb"][0])
    assert np.allclose(np.asarray(hf[0]["depth"]), sample["depth"][0])
    assert hf[0]["timestamp_ns"] == sample["timestamp_ns"][0]
    assert hf[0]["recording_id"] == "rec"
    direct_resized = GoldWindowDataset(str(manifest), split="train", img_size=8)
    hf_loader = HFWindowDataset(str(hf_root), split="train", img_size=8)
    hf_sample = hf_loader[0]
    assert np.array_equal(hf_sample["timestamp_ns"], direct_resized[0]["timestamp_ns"])
    direct_batch = collate_gold([direct_resized[0]])
    hf_batch = collate_hf([hf_sample])
    for name in ("rgb", "depth", "imu", "odom", "action", "collision"):
        torch.testing.assert_close(getattr(hf_batch, name), getattr(direct_batch, name))
    for name in (
        "depth_valid", "imu_valid", "odom_valid", "action_valid", "collision_valid",
    ):
        assert torch.equal(getattr(hf_batch, name), getattr(direct_batch, name))


def test_float_depth_is_zstd_compressed_npy() -> None:
    import pyarrow as pa

    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    message = NS(height=3, width=4, step=16, encoding="32FC1", data=expected.tobytes())
    payload, suffix, media_type = _depth_bytes(message)
    assert suffix == ".npy.zst"
    assert "zstd" in media_type
    size = int.from_bytes(payload[:8], "little")
    raw = pa.decompress(payload[8:], decompressed_size=size, codec="zstd").to_pybytes()
    assert np.array_equal(np.load(io.BytesIO(raw), allow_pickle=False), expected)


def test_nearest_association_breaks_equidistant_ties_earlier() -> None:
    selected = _nearest([(90, 1, "earlier"), (110, 0, "later")], 100, 20)
    assert selected == (90, 1, "earlier")


def test_windows_do_not_cross_approved_range_boundaries(tmp_path: Path, monkeypatch) -> None:
    import pyarrow.parquet as pq

    profile = tmp_path / "robot.yaml"
    profile.write_text("""topics:
  color: /camera
  depth: /depth
  imu: /imu
  odom: /odom
  position_cmd: /cmd
action:
  source: position_cmd
  dim: 14
""", encoding="utf-8")
    source = tmp_path / "bronze.mcap"
    source.write_bytes(b"fixture")
    output = tmp_path / "gold/recordings/rec"
    job = {
        "schema_version": 1, "consumer": "wm_planner", "recording_id": "rec",
        "split": "train", "approved_ranges": [
            {"start_ns": 1_000_000_000, "end_ns": 1_500_000_000},
            {"start_ns": 1_500_000_000, "end_ns": 2_000_000_000},
        ],
        "sources": [{"path": str(source)}], "output_dir": str(output),
        "dataset_output": str(tmp_path / "gold"),
        "object_store": str(tmp_path / ".objects"), "topic_profile": str(profile),
        "fingerprint": "ranges", "recipe": {"options": {
            "window_length": 4, "window_stride": 1, "min_speed": 0.0,
        }},
    }
    path = tmp_path / "ranges-job.json"
    path.write_text(json.dumps(job), encoding="utf-8")
    monkeypatch.setattr("world_model.data.gold_cli._iter_messages", lambda _: _messages())
    materialize(path)
    frames = pq.read_table(output / "frames.parquet").to_pylist()
    windows = pq.read_table(output / "windows.parquet").to_pylist()
    assert len(windows) == 4
    segments = {int(row["frame_index"]): int(row["segment_id"]) for row in frames}
    for window in windows:
        indexes = range(
            int(window["first_frame_index"]),
            int(window["first_frame_index"])
            + (int(window["length"]) - 1) * int(window["frame_step"]) + 1,
            int(window["frame_step"]),
        )
        assert len({segments[index] for index in indexes}) == 1

    missing_output = tmp_path / "gold/recordings/rec-no-depth"
    missing_job = {
        **job, "recording_id": "rec-no-depth", "output_dir": str(missing_output),
        "fingerprint": "optional-depth",
    }
    missing_path = tmp_path / "missing-depth-job.json"
    missing_path.write_text(json.dumps(missing_job), encoding="utf-8")
    monkeypatch.setattr(
        "world_model.data.gold_cli._iter_messages",
        lambda _: (row for row in _messages() if row[0] != "/depth"),
    )
    materialize(missing_path)
    missing_frames = pq.read_table(missing_output / "frames.parquet").to_pylist()
    assert missing_frames
    assert all(not row["depth_valid"] and row["depth_path"] is None for row in missing_frames)
