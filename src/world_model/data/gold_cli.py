"""Consumer-owned materializer and exporters for wm_planner Gold datasets."""

from __future__ import annotations

import argparse
import bisect
import errno
import hashlib
import importlib.metadata
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

PROTOCOL_VERSION = 1
DATASET_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _code_identity() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in (Path(__file__).resolve(), root / "gold_dataset.py", root / "dataloaders.py"):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def protocol() -> dict[str, Any]:
    packages = {}
    for distribution in ("ds-ga-3001", "numpy", "pyarrow", "mcap", "mcap-ros2-support"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    environment = {
        "python": sys.version.split()[0],
        "implementation": sys.implementation.name,
        "platform": sys.platform,
        "packages": packages,
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "consumers": ["wm_planner"],
        "adapter": "wm-dataset",
        "adapter_version": "1",
        "code_revision": _revision(),
        "code_identity_sha256": _code_identity(),
        "environment": environment,
        "environment_sha256": hashlib.sha256(
            json.dumps(environment, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _timestamp(message: Any, fallback: int) -> int:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return fallback
    value = int(getattr(stamp, "sec", 0)) * 1_000_000_000 + int(
        getattr(stamp, "nanosec", getattr(stamp, "nsec", 0))
    )
    return value or fallback


def _iter_messages(paths: Iterable[Path]):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    sequence = 0
    for path in paths:
        with path.open("rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            for _, channel, message, decoded in reader.iter_decoded_messages():
                yield channel.topic, _timestamp(decoded, int(message.log_time)), sequence, decoded
                sequence += 1


def _image_bytes(
    message: Any, *, image_size: int | None, encoding: str, jpeg_quality: int
) -> tuple[bytes, str, str]:
    compressed = hasattr(message, "format")
    if hasattr(message, "format"):
        payload = bytes(message.data)
        fmt = str(message.format).casefold()
        suffix = ".png" if "png" in fmt or payload.startswith(b"\x89PNG") else ".jpg"
        if image_size is None and encoding == "preserve":
            return payload, suffix, "image/png" if suffix == ".png" else "image/jpeg"
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    else:
        height, width = int(message.height), int(message.width)
        source_encoding = str(message.encoding).casefold()
        channels = {
            "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1,
        }.get(source_encoding)
        if channels is None:
            raise ValueError(f"unsupported RGB encoding: {source_encoding}")
        raw = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(height, int(message.step))
        array = raw[:, : width * channels].reshape(height, width, channels)
        if source_encoding in {"bgr8", "bgra8"}:
            array = array[..., [2, 1, 0, 3][:channels]]
        if channels == 4:
            array = array[..., :3]
        elif channels == 1:
            array = array[..., 0]
        image = Image.fromarray(array).convert("RGB")
        suffix = ".png"
    if image_size is not None:
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    selected = encoding if encoding != "preserve" else ("jpeg" if compressed and suffix == ".jpg" else "png")
    buffer = io.BytesIO()
    if selected == "jpeg":
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        return buffer.getvalue(), ".jpg", "image/jpeg"
    if selected != "png":
        raise ValueError(f"rgb_encoding must be preserve, png, or jpeg; got {selected!r}")
    image.save(buffer, format="PNG")
    return buffer.getvalue(), ".png", "image/png"


def _depth_bytes(message: Any, *, image_size: int | None = None) -> tuple[bytes, str, str]:
    if hasattr(message, "format"):
        import cv2
        import struct

        payload = bytes(message.data)
        depth_format = str(message.format).casefold()
        if "compresseddepth" in depth_format:
            png_offset = payload.find(b"\x89PNG")
            if png_offset < 0:
                raise ValueError("compressedDepth RVL payloads are not supported")
            encoded = payload[png_offset:]
        else:
            encoded = payload
        decoded = cv2.imdecode(
            np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        if decoded is None:
            raise ValueError("could not decode compressed depth")
        if "32fc1" in depth_format and "compresseddepth" in depth_format:
            if len(payload) < 12:
                raise ValueError("compressedDepth float header is truncated")
            _, depth_a, depth_b = struct.unpack("<iff", payload[:12])
            inverse = decoded.astype(np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                array = np.where(inverse > 0, depth_a / (inverse - depth_b), np.nan)
        else:
            array = decoded
    else:
        height, width = int(message.height), int(message.width)
        encoding = str(message.encoding).upper()
        dtype = np.uint16 if encoding == "16UC1" else np.float32 if encoding == "32FC1" else None
        if dtype is None:
            raise ValueError(f"unsupported depth encoding: {encoding}")
        array = np.frombuffer(bytes(message.data), dtype=dtype).reshape(height, int(message.step) // np.dtype(dtype).itemsize)[:, :width]
    if image_size is not None and array.shape != (image_size, image_size):
        array = np.asarray(
            Image.fromarray(array).resize(
                (image_size, image_size), Image.Resampling.NEAREST
            )
        )
    if array.dtype == np.uint16:
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        return buffer.getvalue(), ".png", "image/png; semantics=depth-millimeters"
    import pyarrow as pa

    buffer = io.BytesIO()
    np.save(buffer, array.astype(np.float32), allow_pickle=False)
    raw = buffer.getvalue()
    compressed = pa.compress(raw, codec="zstd").to_pybytes()
    payload = len(raw).to_bytes(8, "little") + compressed
    return payload, ".npy.zst", "application/x-npy+zstd; semantics=depth-meters"


def _imu(message: Any) -> list[float]:
    q, av, la = message.orientation, message.angular_velocity, message.linear_acceleration
    return [q.x, q.y, q.z, q.w, av.x, av.y, av.z, la.x, la.y, la.z]


def _odom(message: Any) -> list[float]:
    pose, twist = message.pose.pose, message.twist.twist
    p, q, lv, av = pose.position, pose.orientation, twist.linear, twist.angular
    return [p.x, p.y, p.z, q.x, q.y, q.z, q.w, lv.x, lv.y, lv.z, av.x, av.y, av.z]


def _position_action(message: Any) -> list[float]:
    p, v, a, j = message.position, message.velocity, message.acceleration, message.jerk
    return [p.x, p.y, p.z, v.x, v.y, v.z, a.x, a.y, a.z, j.x, j.y, j.z, message.yaw, message.yaw_dot]


def _twist_action(message: Any) -> list[float]:
    twist = getattr(message, "twist", message)
    linear, angular = twist.linear, twist.angular
    return [linear.x, linear.y, linear.z, angular.x, angular.y, angular.z]


def _so3_action(message: Any) -> list[float]:
    force, q, av = message.force, message.orientation, message.angular_velocity
    return [force.x, force.y, force.z, q.x, q.y, q.z, q.w, av.x, av.y, av.z,
            *message.kx, *message.kv, *message.kr, *message.kom]


def _inside(timestamp: int, ranges: list[dict[str, Any]]) -> bool:
    return not ranges or any(int(item["start_ns"]) <= timestamp < int(item["end_ns"]) for item in ranges)


def _range_index(timestamp: int, ranges: list[dict[str, Any]]) -> int:
    if not ranges:
        return 0
    return next(
        index for index, item in enumerate(ranges)
        if int(item["start_ns"]) <= timestamp < int(item["end_ns"])
    )


def _false_spans(rows: list[dict[str, Any]], key: str) -> int:
    spans, active = 0, False
    for row in rows:
        rejected = not bool(row[key])
        if rejected and not active:
            spans += 1
        active = rejected
    return spans


def _nearest(rows: list[tuple[int, int, Any]], timestamp: int, tolerance: int):
    if not rows:
        return None
    times = [row[0] for row in rows]
    index = bisect.bisect_left(times, timestamp)
    candidates = rows[max(index - 1, 0):min(index + 1, len(rows))]
    selected = min(candidates, key=lambda row: (abs(row[0] - timestamp), row[0], row[1]))
    return selected if abs(selected[0] - timestamp) <= tolerance else None


def _matched_values(match: tuple[int, int, Any] | None, dimension: int) -> list[float]:
    if match is None:
        return [0.0] * dimension
    values = list(map(float, match[2]))
    if len(values) != dimension:
        raise ValueError(f"sensor dimension changed within recording: {len(values)} != {dimension}")
    return values


def _offset(match: tuple[int, int, Any] | None, timestamp: int) -> int | None:
    return int(match[0] - timestamp) if match is not None else None


def _cas_ingest(source: Path, object_root: Path, destination: Path) -> tuple[str, str]:
    digest = _sha256(source)
    obj = object_root / "sha256" / digest[:2] / digest
    obj.parent.mkdir(parents=True, exist_ok=True)
    if not obj.exists():
        temporary = obj.with_name(f".{digest}.tmp-{os.getpid()}")
        shutil.copyfile(source, temporary)
        if _sha256(temporary) != digest:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"CAS ingest checksum changed while reading {source}")
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            os.link(temporary, obj)
        except FileExistsError:
            if _sha256(obj) != digest:
                raise ValueError(f"corrupt concurrently published CAS object: {obj}") from None
        finally:
            temporary.unlink(missing_ok=True)
    elif _sha256(obj) != digest:
        raise ValueError(f"corrupt CAS object: {obj}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(obj, destination)
        return digest, "hardlink"
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP}:
            raise
        shutil.copy2(obj, destination)
        return digest, "copy"


def _write_parquet(rows: list[dict[str, Any]], path: Path, schema: Any) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def _schemas(imu_dim: int, odom_dim: int, action_dim: int):
    import pyarrow as pa
    frames = pa.schema([
        ("recording_id", pa.string()), ("split", pa.string()), ("frame_index", pa.int64()),
        ("segment_id", pa.int64()), ("timestamp_ns", pa.int64()),
        ("rgb_path", pa.string()), ("depth_path", pa.string()), ("depth_valid", pa.bool_()),
        ("imu", pa.list_(pa.float32(), imu_dim)), ("odom", pa.list_(pa.float32(), odom_dim)),
        ("action", pa.list_(pa.float32(), action_dim)), ("collision", pa.float32()),
        ("imu_valid", pa.bool_()), ("odom_valid", pa.bool_()), ("action_valid", pa.bool_()),
        ("collision_valid", pa.bool_()), ("imu_offset_ns", pa.int64()),
        ("odom_offset_ns", pa.int64()), ("action_offset_ns", pa.int64()),
        ("depth_offset_ns", pa.int64()), ("valid", pa.bool_()),
    ])
    windows = pa.schema([
        ("window_id", pa.string()), ("recording_id", pa.string()), ("split", pa.string()),
        ("first_frame_index", pa.int64()), ("length", pa.int64()), ("frame_step", pa.int64()),
    ])
    assets = pa.schema([
        ("logical_path", pa.string()), ("sha256", pa.string()), ("size_bytes", pa.int64()),
        ("media_type", pa.string()), ("storage_mode", pa.string()), ("object_path", pa.string()),
    ])
    return frames, windows, assets


def materialize(job_path: Path) -> None:
    import yaml
    job = json.loads(job_path.read_text(encoding="utf-8"))
    if job.get("schema_version") != 1 or job.get("consumer") != "wm_planner":
        raise ValueError("unsupported wm-dataset job specification")
    output = Path(job["output_dir"])
    existing = output / "recording_manifest.json"
    if existing.is_file():
        existing_payload = json.loads(existing.read_text(encoding="utf-8"))
        if existing_payload.get("fingerprint") == job["fingerprint"] and all(
            (output / relative).is_file() and _sha256(output / relative) == digest
            for relative, digest in existing_payload.get("outputs", {}).items()
        ) and existing_payload.get("outputs"):
            return
    profile_value = job.get("topic_profile") or job["recipe"].get("consumer_inputs", {}).get("robot_config")
    if not profile_value or not Path(profile_value).is_file():
        raise ValueError("wm_planner Gold requires a Silver topic profile/robot config path")
    config = yaml.safe_load(Path(profile_value).read_text(encoding="utf-8")) or {}
    topics = dict(config.get("topics", {}))
    action_name = str(config.get("action", {}).get("source", "position_cmd"))
    options = {
        "max_sync_gap_ms": 100.0, "window_length": 64, "window_stride": 32,
        "frame_step": 1, "image_gap_factor": 2.0, "min_speed": 0.1,
        "idle_persistence": 10, "depth_required": False,
        "image_size": None, "rgb_encoding": "preserve", "jpeg_quality": 95,
        "required_modalities": ["rgb", "imu", "odom", "action"],
        **job["recipe"].get("options", {}),
    }
    required = {str(value).casefold() for value in options["required_modalities"]}
    if options["depth_required"]:
        required.add("depth")
    supported = {"rgb", "depth", "imu", "odom", "action"}
    if unknown := sorted(required - supported):
        raise ValueError(f"unsupported required modalities: {unknown}")
    required.add("rgb")
    topic_keys = {"rgb": "color", "depth": "depth", "imu": "imu", "odom": "odom",
                  "action": action_name}
    missing_topics = sorted(name for name in required if not topics.get(topic_keys[name]))
    if missing_topics:
        raise ValueError(f"topic profile is missing required mappings: {missing_topics}")
    tolerance = int(float(options["max_sync_gap_ms"]) * 1_000_000)
    ranges = list(job.get("approved_ranges", []))
    streams: dict[str, list[tuple[int, int, Any]]] = {
        "color": [], "depth": [], "imu": [], "odom": [], "action": [],
    }
    topic_roles = {str(value): key for key, value in topics.items()}
    for topic, timestamp, sequence, message in _iter_messages(Path(item["path"]) for item in job["sources"]):
        if not _inside(timestamp, ranges):
            continue
        role = topic_roles.get(topic)
        if role == "color":
            streams["color"].append((timestamp, sequence, _image_bytes(
                message,
                image_size=int(options["image_size"]) if options["image_size"] else None,
                encoding=str(options["rgb_encoding"]).casefold(),
                jpeg_quality=int(options["jpeg_quality"]),
            )))
        elif role == "depth":
            streams["depth"].append((timestamp, sequence, _depth_bytes(
                message,
                image_size=int(options["image_size"]) if options["image_size"] else None,
            )))
        elif role == "imu":
            streams["imu"].append((timestamp, sequence, _imu(message)))
        elif role == "odom":
            decoded_odom = _odom(message)
            streams["odom"].append((timestamp, sequence, decoded_odom))
            if action_name == "odom":
                streams["action"].append((timestamp, sequence, decoded_odom))
        elif role == action_name:
            decoders = {
                "position_cmd": _position_action,
                "so3_cmd": _so3_action,
                "cmd_vel": _twist_action,
            }
            if action_name not in decoders:
                raise ValueError(f"unsupported action source: {action_name}")
            decoder = decoders[action_name]
            streams["action"].append((timestamp, sequence, decoder(message)))
    for rows in streams.values():
        rows.sort(key=lambda row: (row[0], row[1]))
    if not streams["color"]:
        raise ValueError("recording has no RGB frames in approved ranges")

    candidates = []
    for candidate_index, (timestamp, sequence, image) in enumerate(streams["color"]):
        matches = {name: _nearest(streams[name], timestamp, tolerance) for name in ("imu", "odom", "action", "depth")}
        required_valid = all(matches[name] is not None for name in required - {"rgb"})
        speed = np.linalg.norm(matches["odom"][2][7:10]) if matches["odom"] else float("inf")
        candidates.append({"timestamp": timestamp, "sequence": sequence,
                           "candidate_index": candidate_index, "image": image,
                           "range_index": _range_index(timestamp, ranges),
                           "matches": matches, "required_valid": required_valid,
                           "slow": speed < float(options["min_speed"])})
    persistence = int(options["idle_persistence"])
    index = 0
    while index < len(candidates):
        if not candidates[index]["slow"]:
            index += 1
            continue
        end = index + 1
        while end < len(candidates) and candidates[end]["slow"]:
            end += 1
        if end - index >= persistence:
            for row in candidates[index:end]:
                row["required_valid"] = False
        index = end
    timestamps = np.asarray([row["timestamp"] for row in candidates], dtype=np.int64)
    median_dt = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else float("inf")
    gap_limit = median_dt * float(options["image_gap_factor"])

    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    object_store = Path(job["object_store"])
    recording_rel = Path("recordings") / job["recording_id"]
    frame_rows, asset_rows = [], []
    sensors = config.get("sensors", {})
    action_config = config.get("action", {})
    imu_dim = len(streams["imu"][0][2]) if streams["imu"] else int(sensors.get("imu_dim", 10))
    odom_dim = len(streams["odom"][0][2]) if streams["odom"] else int(sensors.get("odom_dim", 13))
    action_dim = len(streams["action"][0][2]) if streams["action"] else (
        int(action_config.get("dim", 22 if action_name == "so3_cmd" else 14))
    )
    declared_action_dim = action_config.get("dim")
    if declared_action_dim is not None and action_dim != int(declared_action_dim):
        raise ValueError(
            f"decoded action dimension {action_dim} differs from profile dimension {declared_action_dim}"
        )
    segment, previous_candidate, previous_timestamp, previous_range = -1, None, None, None
    with tempfile.TemporaryDirectory(prefix="wm-gold-assets-") as scratch_value:
        scratch = Path(scratch_value)
        for candidate in candidates:
            if not candidate["required_valid"]:
                previous_candidate = None
                continue
            timestamp = int(candidate["timestamp"])
            if (previous_candidate is None
                    or candidate["candidate_index"] != previous_candidate + 1
                    or candidate["range_index"] != previous_range
                    or timestamp - int(previous_timestamp) > gap_limit):
                segment += 1
            frame_index = len(frame_rows)
            matches = candidate["matches"]
            rgb_bytes, rgb_suffix, rgb_media = candidate["image"]
            asset_stem = f"{frame_index:08d}-{timestamp}"
            rgb_source = scratch / f"rgb-{asset_stem}{rgb_suffix}"
            rgb_source.write_bytes(rgb_bytes)
            rgb_rel = recording_rel / "rgb" / f"{asset_stem}{rgb_suffix}"
            rgb_dest = temporary / "rgb" / f"{asset_stem}{rgb_suffix}"
            rgb_digest, rgb_mode = _cas_ingest(rgb_source, object_store, rgb_dest)
            asset_rows.append({"logical_path": rgb_rel.as_posix(), "sha256": rgb_digest,
                               "size_bytes": rgb_source.stat().st_size, "media_type": rgb_media,
                               "storage_mode": rgb_mode, "object_path": str(object_store / "sha256" / rgb_digest[:2] / rgb_digest)})
            depth_rel = None
            depth_offset = None
            if matches["depth"] is not None:
                depth_bytes, depth_suffix, depth_media = matches["depth"][2]
                depth_source = scratch / f"depth-{asset_stem}{depth_suffix}"
                depth_source.write_bytes(depth_bytes)
                depth_rel_path = recording_rel / "depth" / f"{asset_stem}{depth_suffix}"
                depth_dest = temporary / "depth" / f"{asset_stem}{depth_suffix}"
                depth_digest, depth_mode = _cas_ingest(depth_source, object_store, depth_dest)
                asset_rows.append({"logical_path": depth_rel_path.as_posix(), "sha256": depth_digest,
                                   "size_bytes": depth_source.stat().st_size, "media_type": depth_media,
                                   "storage_mode": depth_mode, "object_path": str(object_store / "sha256" / depth_digest[:2] / depth_digest)})
                depth_rel = depth_rel_path.as_posix()
                depth_offset = int(matches["depth"][0] - timestamp)
            frame_rows.append({
                "recording_id": job["recording_id"], "split": job["split"],
                "frame_index": frame_index, "segment_id": segment, "timestamp_ns": timestamp,
                "rgb_path": rgb_rel.as_posix(), "depth_path": depth_rel,
                "depth_valid": matches["depth"] is not None,
                "imu": _matched_values(matches["imu"], imu_dim),
                "odom": _matched_values(matches["odom"], odom_dim),
                "action": _matched_values(matches["action"], action_dim),
                "collision": 0.0,
                "imu_valid": matches["imu"] is not None,
                "odom_valid": matches["odom"] is not None,
                "action_valid": matches["action"] is not None,
                "collision_valid": False,
                "imu_offset_ns": _offset(matches["imu"], timestamp),
                "odom_offset_ns": _offset(matches["odom"], timestamp),
                "action_offset_ns": _offset(matches["action"], timestamp),
                "depth_offset_ns": depth_offset, "valid": True,
            })
            previous_candidate, previous_timestamp = candidate["candidate_index"], timestamp
            previous_range = candidate["range_index"]
    if not frame_rows:
        raise ValueError("recording has no frames satisfying required modality/quality policies")
    frame_schema, window_schema, asset_schema = _schemas(imu_dim, odom_dim, action_dim)
    window_rows = []
    length, stride, step = int(options["window_length"]), int(options["window_stride"]), int(options["frame_step"])
    for segment_id in sorted({int(row["segment_id"]) for row in frame_rows}):
        segment_rows = [row for row in frame_rows if row["segment_id"] == segment_id]
        span = (length - 1) * step + 1
        for offset in range(0, max(0, len(segment_rows) - span + 1), stride):
            first = int(segment_rows[offset]["frame_index"])
            window_rows.append({"window_id": f"{job['recording_id']}:{len(window_rows):06d}",
                                "recording_id": job["recording_id"], "split": job["split"],
                                "first_frame_index": first, "length": length, "frame_step": step})
    _write_parquet(frame_rows, temporary / "frames.parquet", frame_schema)
    _write_parquet(window_rows, temporary / "windows.parquet", window_schema)
    _write_parquet(asset_rows, temporary / "assets.parquet", asset_schema)
    output_checksums = {
        name: _sha256(temporary / name)
        for name in ("frames.parquet", "windows.parquet", "assets.parquet")
    }
    _atomic_json(temporary / "recording_manifest.json", {
        "schema_version": 1, "recording_id": job["recording_id"], "split": job["split"],
        "fingerprint": job["fingerprint"], "frame_count": len(frame_rows),
        "window_count": len(window_rows), "sensor_dimensions": {
            "imu": imu_dim, "odom": odom_dim, "action": action_dim,
        }, "dropped_frames": len(candidates) - len(frame_rows),
        "rejected_spans": _false_spans(candidates, "required_valid"),
        "outputs": output_checksums,
    })
    if output.exists():
        raise FileExistsError(f"stale Gold recording output already exists: {output}")
    temporary.replace(output)


def _sync_quantiles(frames: Any) -> dict[str, dict[str, float]]:
    report = {}
    for modality in ("imu", "odom", "action", "depth"):
        column = f"{modality}_offset_ns"
        if column not in frames.column_names:
            continue
        values = np.asarray(
            [abs(int(value)) for value in frames[column].to_pylist() if value is not None],
            dtype=np.float64,
        )
        if values.size:
            report[modality] = {
                "p50": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
                "p99": float(np.quantile(values, 0.99)),
                "max": float(values.max()),
            }
    return report


def _file_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "export_manifest.json"
    ]


def finalize(build_manifest: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq
    build = json.loads(build_manifest.read_text(encoding="utf-8"))
    root = Path(build["output_root"])
    frame_tables, window_tables, asset_tables, records = [], [], [], []
    for job in build["jobs"]:
        output = Path(job["output_dir"])
        record = json.loads((output / "recording_manifest.json").read_text(encoding="utf-8"))
        if record.get("fingerprint") != job["fingerprint"]:
            raise ValueError(f"stale Gold recording output: {output}")
        for relative, digest in record.get("outputs", {}).items():
            if not (output / relative).is_file() or _sha256(output / relative) != digest:
                raise ValueError(f"recording output checksum mismatch: {output / relative}")
        records.append(record)
        frame_tables.append(pq.read_table(output / "frames.parquet"))
        window_tables.append(pq.read_table(output / "windows.parquet"))
        asset_tables.append(pq.read_table(output / "assets.parquet"))
    frames = pa.concat_tables(frame_tables) if frame_tables else pa.table({})
    windows = pa.concat_tables(window_tables) if window_tables else pa.table({})
    assets = pa.concat_tables(asset_tables) if asset_tables else pa.table({})
    index_tables = {"frames.parquet": frames, "windows.parquet": windows, "assets.parquet": assets}
    temporary_indexes = {}
    for name, table in index_tables.items():
        temporary = root / f".{name}.tmp-{os.getpid()}"
        pq.write_table(table, temporary, compression="zstd")
        temporary_indexes[name] = temporary
    for name, temporary in temporary_indexes.items():
        temporary.replace(root / name)
    dimensions = records[0]["sensor_dimensions"] if records else {"imu": 0, "odom": 0, "action": 0}
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION, "consumer_format": "wm_planner",
        "release_id": build["release_id"], "release_hash": build["release_hash"],
        "created_at_ns": time.time_ns(), "recipe_sha256": build["recipe_sha256"],
        "adapter": build["adapter"], "recording_count": len(records),
        "provenance": {
            "adapter_executable": build.get("adapter_executable"),
            "recipe_path": build.get("recipe_path"),
            "recipe_sha256": build["recipe_sha256"],
            "build_manifest": str(build_manifest.resolve()),
            "orchestration": build.get("orchestration", {}),
            "invocations": {
                "materialize": [build.get("adapter_executable"), "materialize", "--job", "JOB_SPEC"],
                "finalize": [build.get("adapter_executable"), "finalize", "--build-manifest", str(build_manifest.resolve())],
            },
        },
        "object_store": build.get("object_store", str(root.parent / ".objects")),
        "frame_count": frames.num_rows, "window_count": windows.num_rows,
        "sensor_dimensions": dimensions,
        "semantic_schema": {
            "version": 1, "timestamp": {"dtype": "int64", "unit": "ns"},
            "rgb": {"dtype": "uint8", "layout": "T,H,W,C", "required": True},
            "depth": {
                "dtype": "float32", "layout": "T,H,W", "unit": "m",
                "validity": "depth_valid",
            },
            "imu": {"dtype": "float32", "dimension": dimensions["imu"],
                    "validity": "imu_valid"},
            "odom": {"dtype": "float32", "dimension": dimensions["odom"],
                     "validity": "odom_valid"},
            "action": {"dtype": "float32", "dimension": dimensions["action"],
                       "validity": "action_valid"},
            "collision": {"dtype": "float32", "validity": "collision_valid"},
        },
        "files": {"frames": "frames.parquet", "windows": "windows.parquet", "assets": "assets.parquet"},
        "checksums": {name: _sha256(root / name) for name in ("frames.parquet", "windows.parquet", "assets.parquet")},
        "quality": {
            "rejected_frames": sum(int(record.get("dropped_frames", 0)) for record in records),
            "rejected_spans": sum(int(record.get("rejected_spans", 0)) for record in records),
            "synchronization_quantiles_ns": _sync_quantiles(frames),
        },
    }
    path = root / "dataset_manifest.json"
    _atomic_json(path, manifest)
    return path


def _load_gold(manifest: Path, split: str):
    from world_model.data.gold_dataset import GoldWindowDataset
    return GoldWindowDataset(str(manifest), split=split)


def export_npz(manifest: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"export destination already exists: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.mkdir(parents=True)
    try:
        for split in ("train", "val", "test"):
            try:
                dataset = _load_gold(manifest, split)
            except ValueError:
                continue
            split_root = temporary / split
            split_root.mkdir()
            for index in range(len(dataset)):
                row = dataset[index]
                np.savez_compressed(split_root / f"ep_{index:06d}.npz", **row)
        _atomic_json(temporary / "export_manifest.json", {
            "schema_version": 1, "format": "npz",
            "source_manifest": str(manifest.resolve()),
            "source_sha256": _sha256(manifest), "files": _file_inventory(temporary),
        })
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def export_hf(manifest: Path, output: Path) -> None:
    from datasets import Array2D, Dataset, DatasetDict, Features, Image as HFImage, Value
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    dims = payload["sensor_dimensions"]
    features = {"window_id": Value("string"), "recording_id": Value("string"),
                "split": Value("string"), "frame_idx_in_window": Value("int64"),
                "timestamp_ns": Value("int64"), "image": HFImage(),
                "depth_valid": Value("bool"), "imu_valid": Value("bool"),
                "odom_valid": Value("bool"), "action_valid": Value("bool"),
                "collision": Value("float32"), "collision_valid": Value("bool")}
    for prefix, dimension in (("imu", dims["imu"]), ("odom", dims["odom"]), ("action", dims["action"])):
        for index in range(int(dimension)):
            features[f"{prefix}_{index}"] = Value("float32")
    datasets = {}
    for split in ("train", "val", "test"):
        try:
            gold = _load_gold(manifest, split)
        except ValueError:
            continue
        rows = []
        for window_index in range(len(gold)):
            item = gold[window_index]
            for frame_index in range(len(item["rgb"])):
                window = gold.windows[window_index]
                row = {"window_id": str(window["window_id"]),
                       "recording_id": str(window["recording_id"]), "split": split,
                       "frame_idx_in_window": frame_index,
                       "timestamp_ns": int(item["timestamp_ns"][frame_index]),
                       "image": Image.fromarray(item["rgb"][frame_index]),
                       "depth_valid": bool(item["depth_valid"][frame_index]),
                       "imu_valid": bool(item["imu_valid"][frame_index]),
                       "odom_valid": bool(item["odom_valid"][frame_index]),
                       "action_valid": bool(item["action_valid"][frame_index]),
                       "collision": float(item["collision"][frame_index]),
                       "collision_valid": bool(item["collision_valid"][frame_index])}
                for prefix in ("imu", "odom", "action"):
                    for column, value in enumerate(item[prefix][frame_index]):
                        row[f"{prefix}_{column}"] = float(value)
                # Dense depth is retained as an ordinary nested column; infer its fixed shape.
                row["depth"] = item["depth"][frame_index].tolist()
                rows.append(row)
        if rows:
            local_features = dict(features)
            shape = np.asarray(rows[0]["depth"]).shape
            local_features["depth"] = Array2D(shape=shape, dtype="float32")
            datasets["validation" if split == "val" else split] = Dataset.from_list(rows, features=Features(local_features))
    if not datasets:
        raise ValueError("Gold dataset has no windows to export")
    if output.exists():
        raise FileExistsError(f"export destination already exists: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        DatasetDict(datasets).save_to_disk(str(temporary))
        _atomic_json(temporary / "export_manifest.json", {
            "schema_version": 1, "format": "hf",
            "source_manifest": str(manifest.resolve()),
            "source_sha256": _sha256(manifest), "splits": sorted(datasets),
            "files": _file_inventory(temporary),
        })
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol")
    materialize_parser = sub.add_parser("materialize")
    materialize_parser.add_argument("--job", type=Path, required=True)
    finalize_parser = sub.add_parser("finalize")
    finalize_parser.add_argument("--build-manifest", type=Path, required=True)
    export_parser = sub.add_parser("export")
    export_parser.add_argument("--dataset-manifest", type=Path, required=True)
    export_parser.add_argument("--format", choices=("npz", "hf"), required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "protocol":
        print(json.dumps(protocol()))
    elif args.command == "materialize":
        materialize(args.job)
    elif args.command == "finalize":
        print(finalize(args.build_manifest))
    elif args.format == "npz":
        export_npz(args.dataset_manifest, args.output)
    else:
        export_hf(args.dataset_manifest, args.output)


if __name__ == "__main__":
    main()
