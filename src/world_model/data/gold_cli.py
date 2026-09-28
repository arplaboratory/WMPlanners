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
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image

PROTOCOL_VERSION = 2
DATASET_SCHEMA_VERSION = 2
SUPPORTED_JOB_SCHEMA_VERSIONS = (1, 2)
SUPPORTED_DATASET_SCHEMA_VERSIONS = (1, 2)


class McapStructureError(RuntimeError):
    """An ordered Bronze storage part cannot be traversed as valid MCAP."""

    def __init__(
        self, message: str, *, storage_part: str, storage_part_index: int,
        error_type: str,
    ) -> None:
        super().__init__(message)
        self.storage_part = storage_part
        self.storage_part_index = storage_part_index
        self.error_type = error_type


class McapMessageDecodeError(RuntimeError):
    """One MCAP message cannot be decoded using its embedded ROS schema."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _revision() -> str | None:
    for name in ("WM_PLANNER_REVISION", "SOURCE_REVISION", "GIT_COMMIT"):
        if os.environ.get(name):
            return os.environ[name]
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
    for path in (
        Path(__file__).resolve(), root / "parquet_dataset.py", root / "dataloaders.py",
        root / "hf_datasets.py", root / "factory.py", root / "types.py",
    ):
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
        "container": {
            key: os.environ[key]
            for key in ("APPTAINER_CONTAINER", "WM_DATASET_CONTAINER_DIGEST")
            if os.environ.get(key)
        },
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "consumers": ["wm_planner"],
        "adapter": "wm-dataset",
        "adapter_version": "3",
        "job_schema_versions": list(SUPPORTED_JOB_SCHEMA_VERSIONS),
        "dataset_schema_versions": list(SUPPORTED_DATASET_SCHEMA_VERSIONS),
        "export_formats": ["npz", "hf"],
        "validation_commands": ["validate-recipe"],
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


def _iter_messages(
    paths: Iterable[Path], *,
    on_decode_error: Callable[[dict[str, Any]], bool] | None = None,
):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    sequence = 0
    for part_index, path in enumerate(paths):
        # Decoder factories retain schema state. Ordered MCAP parts may embed
        # distinct definitions for the same ROS type, so isolate every part.
        decoder_factory = DecoderFactory()
        with path.open("rb") as stream:
            try:
                reader = make_reader(stream, decoder_factories=[decoder_factory])
                records = iter(reader.iter_messages())
            except Exception as exc:
                raise McapStructureError(
                    f"failed to open MCAP structure in {path}: {exc}",
                    storage_part=str(path), storage_part_index=part_index,
                    error_type=type(exc).__name__,
                ) from exc
            decoders: dict[int, Any] = {}
            while True:
                try:
                    schema, channel, message = next(records)
                except StopIteration:
                    break
                except Exception as exc:
                    raise McapStructureError(
                        f"failed to read MCAP structure in {path}: {exc}",
                        storage_part=str(path), storage_part_index=part_index,
                        error_type=type(exc).__name__,
                    ) from exc
                try:
                    decoder = decoders.get(message.channel_id)
                    if decoder is None:
                        decoder = decoder_factory.decoder_for(
                            channel.message_encoding, schema,
                        )
                        if decoder is None:
                            raise RuntimeError(
                                "no decoder for message encoding "
                                f"{channel.message_encoding!r} and schema "
                                f"{getattr(schema, 'name', '')!r}"
                            )
                        decoders[message.channel_id] = decoder
                    decoded = decoder(message.data)
                except Exception as exc:
                    detail = {
                        "topic": str(channel.topic),
                        "timestamp_ns": int(message.log_time),
                        "sequence": sequence,
                        "storage_part": str(path),
                        "storage_part_index": part_index,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    sequence += 1
                    if on_decode_error is not None and on_decode_error(detail):
                        continue
                    raise McapMessageDecodeError(
                        f"failed to decode {channel.topic} at {message.log_time} "
                        f"in {path}: {exc}"
                    ) from exc
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


def _collision(message: Any) -> float:
    for name in ("data", "collision", "value"):
        if hasattr(message, name):
            return float(getattr(message, name))
    raise ValueError("collision message has no supported scalar field")


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


def _cas_ingest(source: Path, object_root: Path, destination: Path) -> tuple[str, str, str]:
    digest = _sha256(source)
    obj = object_root / "sha256" / digest[:2] / digest
    obj.parent.mkdir(parents=True, exist_ok=True)
    if not obj.exists():
        temporary = obj.with_name(f".{digest}.tmp-{os.getpid()}-{time.time_ns()}")
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
        return digest, "hardlink", f"sha256/{digest[:2]}/{digest}"
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP}:
            raise
        shutil.copy2(obj, destination)
        if _sha256(destination) != digest:
            destination.unlink(missing_ok=True)
            raise ValueError(f"CAS copy verification failed: {destination}")
        return digest, "copy", f"sha256/{digest[:2]}/{digest}"


def _write_parquet(rows: list[dict[str, Any]], path: Path, schema: Any) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def _write_poses(frames: list[dict[str, Any]], output: Path) -> None:
    """Export the odometry match used by each RGB-anchored frame, in XYZW order."""
    import pyarrow as pa

    components = ("x", "y", "z", "qx", "qy", "qz", "qw")
    schema = pa.schema([
        ("recording_id", pa.string()), ("frame_index", pa.int64()),
        ("timestamp_ns", pa.int64()), ("odom_timestamp_ns", pa.int64()),
        ("odom_offset_ns", pa.int64()), ("rgb_path", pa.string()),
        ("depth_path", pa.string()), ("depth_offset_ns", pa.int64()),
        ("valid", pa.bool_()),
        *[(name, pa.float64()) for name in components],
    ])
    rows = []
    for frame in frames:
        valid = frame["odom_valid"]
        offset = frame["odom_offset_ns"]
        rows.append({
            **{key: frame[key] for key in (
                "recording_id", "frame_index", "timestamp_ns", "odom_offset_ns",
                "rgb_path", "depth_path", "depth_offset_ns",
            )},
            "odom_timestamp_ns": frame["timestamp_ns"] + offset if valid else None,
            "valid": valid,
            **dict(zip(components, frame["odom"][:7] if valid else [None] * 7)),
        })
    _write_parquet(rows, output / "poses/poses.parquet", schema)


def _schemas(imu_dim: int, odom_dim: int, action_dim: int):
    import pyarrow as pa
    frames = pa.schema([
        ("recording_id", pa.string()), ("split", pa.string()), ("frame_index", pa.int64()),
        ("segment_id", pa.int64()), ("approved_range_index", pa.int64()),
        ("timestamp_ns", pa.int64()),
        ("rgb_path", pa.string()), ("depth_path", pa.string()), ("depth_valid", pa.bool_()),
        ("imu", pa.list_(pa.float32(), imu_dim)), ("odom", pa.list_(pa.float32(), odom_dim)),
        ("action", pa.list_(pa.float32(), action_dim)), ("collision", pa.float32()),
        ("imu_valid", pa.bool_()), ("odom_valid", pa.bool_()), ("action_valid", pa.bool_()),
        ("collision_valid", pa.bool_()), ("imu_offset_ns", pa.int64()),
        ("odom_offset_ns", pa.int64()), ("action_offset_ns", pa.int64()),
        ("depth_offset_ns", pa.int64()), ("collision_offset_ns", pa.int64()),
        ("valid", pa.bool_()),
    ])
    windows = pa.schema([
        ("window_id", pa.string()), ("recording_id", pa.string()), ("split", pa.string()),
        ("first_frame_index", pa.int64()), ("length", pa.int64()), ("frame_step", pa.int64()),
    ])
    assets = pa.schema([
        ("logical_path", pa.string()), ("sha256", pa.string()), ("size_bytes", pa.int64()),
        ("media_type", pa.string()), ("encoding", pa.string()),
        ("storage_mode", pa.string()), ("object_key", pa.string()),
    ])
    return frames, windows, assets


def _declared_sensor_contract(
    config: dict[str, Any], action_name: str,
) -> tuple[int, int, int, dict[str, Any]]:
    sensors = config.get("sensors", {})
    action = config.get("action", {})
    imu_dim = int(sensors.get("imu_dim", 10))
    odom_dim = int(sensors.get("odom_dim", 13))
    default_action_dim = (
        22 if action_name == "so3_cmd"
        else 7 if action_name == "future_odom_pose"
        else 14
    )
    action_dim = int(action.get("dim", default_action_dim))
    semantics = {
        "imu": sensors.get("imu", {"dimension": imu_dim}),
        "odom": sensors.get("odom", {"dimension": odom_dim}),
        "action": {
            "source": action_name,
            "dimension": action_dim,
            "semantic": action.get("semantic", action_name),
            "components": action.get("components", []),
            "units": action.get("units", []),
            "frame": action.get("frame"),
            "derived_from": action.get("derived_from"),
        },
    }
    return imu_dim, odom_dim, action_dim, semantics


def _materialize_early_exclusion(
    job: dict[str, Any], output: Path, *, reason_code: str,
    details: dict[str, Any], config: dict[str, Any], action_name: str,
) -> None:
    """Atomically emit the normal WM recording contract for an exclusion."""

    temporary = output.with_name(
        f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    imu_dim, odom_dim, action_dim, semantics = _declared_sensor_contract(
        config, action_name,
    )
    frame_schema, window_schema, asset_schema = _schemas(
        imu_dim, odom_dim, action_dim,
    )
    try:
        temporary.mkdir(parents=True)
        _write_parquet([], temporary / "frames.parquet", frame_schema)
        _write_parquet([], temporary / "windows.parquet", window_schema)
        _write_parquet([], temporary / "assets.parquet", asset_schema)
        _write_poses([], temporary)
        outputs = {
            name: _sha256(temporary / name)
            for name in ("frames.parquet", "windows.parquet", "assets.parquet", "poses/poses.parquet")
        }
        _atomic_json(temporary / "recording_manifest.json", {
            "schema_version": 1,
            "recording_id": job["recording_id"],
            "split": job["split"],
            "status": "excluded",
            "reason_codes": [reason_code],
            "exclusion_details": details,
            "fingerprint": job["fingerprint"],
            "frame_count": 0,
            "window_count": 0,
            "sensor_dimensions": {
                "imu": imu_dim, "odom": odom_dim, "action": action_dim,
            },
            "sensor_semantics": semantics,
            "dropped_frames": 0,
            "sampling_dropped_frames": 0,
            "rejected_spans": 0,
            "decode_errors": {},
            "outputs": outputs,
        })
        if output.exists():
            archive = output.parent / ".stale" / f"{output.name}-{time.time_ns()}"
            archive.parent.mkdir(parents=True, exist_ok=True)
            output.replace(archive)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validated_options(recipe: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "max_sync_gap_ms": 100.0, "window_length": 64, "window_stride": 32,
        "frame_step": 1, "image_gap_factor": 2.0, "min_speed": 0.1,
        "idle_persistence": 10, "depth_required": False,
        "image_size": None, "rgb_encoding": "preserve", "jpeg_quality": 95,
        "required_modalities": ["rgb", "imu", "odom", "action"],
        "on_ineligible_recording": "exclude",
        "action_contract": None,
    }
    supplied = recipe.get("options", {})
    if not isinstance(supplied, dict):
        raise ValueError("wm_planner recipe options must be a mapping")
    unknown = sorted(set(supplied) - set(defaults))
    if unknown:
        raise ValueError(f"unknown wm_planner recipe options: {unknown}")
    options = {**defaults, **supplied}
    for name in ("window_length", "window_stride", "frame_step", "idle_persistence"):
        if int(options[name]) < 1:
            raise ValueError(f"{name} must be >= 1")
    if float(options["max_sync_gap_ms"]) < 0:
        raise ValueError("max_sync_gap_ms must be >= 0")
    if float(options["image_gap_factor"]) <= 1:
        raise ValueError("image_gap_factor must be > 1")
    if options["image_size"] is not None and int(options["image_size"]) < 1:
        raise ValueError("image_size must be null or >= 1")
    if str(options["rgb_encoding"]).casefold() not in {"preserve", "png", "jpeg"}:
        raise ValueError("rgb_encoding must be preserve, png, or jpeg")
    if not 1 <= int(options["jpeg_quality"]) <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")
    if options["on_ineligible_recording"] not in {"exclude", "fail"}:
        raise ValueError("on_ineligible_recording must be exclude or fail")
    if not isinstance(options["required_modalities"], list):
        raise ValueError("required_modalities must be a list")
    contract = options["action_contract"]
    if contract is not None:
        if not isinstance(contract, dict):
            raise ValueError("action_contract must be null or a mapping")
        if unknown := sorted(set(contract) - {"source", "dimension", "semantic"}):
            raise ValueError(f"unknown action_contract fields: {unknown}")
        if not isinstance(contract.get("source"), str) or not contract["source"]:
            raise ValueError("action_contract.source must be a nonempty string")
        if int(contract.get("dimension", 0)) < 1:
            raise ValueError("action_contract.dimension must be positive")
        options["action_contract"] = {
            "source": str(contract["source"]),
            "dimension": int(contract["dimension"]),
            "semantic": str(contract.get("semantic", contract["source"])),
        }
    return options


def _resolved_profile_topics(
    config: dict[str, Any], bindings: dict[str, Any] | None,
) -> dict[str, str]:
    """Resolve legacy scalar/list profiles, preferring Silver-frozen choices."""

    raw = config.get("topics", {})
    if not isinstance(raw, dict):
        raise ValueError("Silver topic profile topics must be a mapping")
    topics: dict[str, str] = {}
    for key, value in raw.items():
        candidates = value if isinstance(value, list) else [value]
        if not candidates or any(not isinstance(item, str) or not item for item in candidates):
            raise ValueError(
                f"Silver topic profile topics.{key} must be a string or nonempty list of strings"
            )
        topics[str(key)] = str(candidates[0])

    if not bindings:
        return topics
    if not isinstance(bindings, dict):
        raise TypeError("job modality_bindings must be a mapping")
    canonical = {
        "color": "rgb", "image": "rgb", "rgb": "rgb",
        "camera_info": "camera_info", "depth": "depth", "imu": "imu",
        "odom": "odom", "odometry": "odom", "collision": "collision",
    }
    action_name = str(config.get("action", {}).get("source", "position_cmd"))
    for key in topics:
        binding_name = "action" if key == action_name else canonical.get(key, key)
        binding = bindings.get(binding_name)
        if not isinstance(binding, dict):
            continue
        selected = binding.get("topic")
        topics[key] = (
            str(selected)
            if binding.get("present") is True and isinstance(selected, str) and selected
            else ""
        )
    return topics


def _sample_candidates(
    candidates: list[dict[str, Any]],
    ranges: list[dict[str, Any]],
    sampling_policy: dict[str, Any],
) -> list[dict[str, Any]]:
    sample_hz = float(sampling_policy.get("sample_hz", 0) or 0)
    maximum = int(sampling_policy.get("max_frames", 0) or 0)
    if sample_hz <= 0:
        selected = list(candidates)
    else:
        period = max(1, int(round(1_000_000_000 / sample_hz)))
        selected = []
        for range_index in sorted({int(row["range_index"]) for row in candidates}):
            rows = [row for row in candidates if int(row["range_index"]) == range_index]
            if not rows:
                continue
            times = [int(row["timestamp"]) for row in rows]
            anchor = int(ranges[range_index]["start_ns"]) if ranges else times[0]
            used: set[int] = set()
            grid = anchor
            while grid <= times[-1]:
                pos = bisect.bisect_left(times, grid)
                indexes = range(max(0, pos - 1), min(len(rows), pos + 1))
                if indexes:
                    chosen = min(
                        indexes,
                        key=lambda idx: (abs(times[idx] - grid), times[idx], rows[idx]["sequence"]),
                    )
                    if chosen not in used:
                        selected.append(rows[chosen])
                        used.add(chosen)
                grid += period
    selected.sort(key=lambda row: (row["timestamp"], row["sequence"]))
    return selected[:maximum] if maximum > 0 else selected


def materialize(job_path: Path) -> None:
    import yaml
    job = json.loads(job_path.read_text(encoding="utf-8"))
    if job.get("schema_version") not in SUPPORTED_JOB_SCHEMA_VERSIONS or job.get("consumer") != "wm_planner":
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
    profile_artifact = job.get("topic_profile_artifact")
    if isinstance(profile_artifact, dict) and profile_artifact.get("text"):
        text = str(profile_artifact["text"])
        if _sha256_bytes(text.encode()) != profile_artifact.get("sha256"):
            raise ValueError("job topic profile snapshot checksum mismatch")
        config = yaml.safe_load(text) or {}
    else:
        profile_value = job.get("topic_profile") or job["recipe"].get("consumer_inputs", {}).get("robot_config")
        if not profile_value or not Path(profile_value).is_file():
            raise ValueError("wm_planner Gold requires a Silver topic profile/robot config path")
        config = yaml.safe_load(Path(profile_value).read_text(encoding="utf-8")) or {}
    topics = _resolved_profile_topics(config, job.get("modality_bindings"))
    action_name = str(config.get("action", {}).get("source", "position_cmd"))
    options = _validated_options(job["recipe"])
    action_config = config.get("action", {})
    if (
        action_name == "future_odom_pose"
        and str(action_config.get("derived_from", "odom")) != "odom"
    ):
        raise ValueError("future_odom_pose actions must be derived_from: odom")
    default_action_dim = (
        22 if action_name == "so3_cmd"
        else 7 if action_name == "future_odom_pose"
        else 14
    )
    actual_action_contract = {
        "source": action_name,
        "dimension": int(action_config.get("dim", default_action_dim)),
        "semantic": str(action_config.get("semantic", action_name)),
    }
    required_action_contract = options["action_contract"]
    if (
        required_action_contract is not None
        and actual_action_contract != required_action_contract
    ):
        if options["on_ineligible_recording"] == "fail":
            raise ValueError(
                "recording action contract does not match recipe: "
                f"expected {required_action_contract}, got {actual_action_contract}"
            )
        _materialize_early_exclusion(
            job, output,
            reason_code="incompatible_action_contract",
            details={
                "expected": required_action_contract,
                "actual": actual_action_contract,
            },
            config=config,
            action_name=action_name,
        )
        return
    required = {str(value).casefold() for value in options["required_modalities"]}
    if options["depth_required"]:
        required.add("depth")
    supported = {"rgb", "depth", "imu", "odom", "action", "collision"}
    if unknown := sorted(required - supported):
        raise ValueError(f"unsupported required modalities: {unknown}")
    required.add("rgb")
    depth_is_required = "depth" in required
    topic_keys = {"rgb": "color", "depth": "depth", "imu": "imu", "odom": "odom",
                  "action": str(action_config.get("derived_from", action_name)),
                  "collision": "collision"}
    missing_topics = sorted(name for name in required if not topics.get(topic_keys[name]))
    if missing_topics:
        raise ValueError(f"topic profile is missing required mappings: {missing_topics}")
    tolerance = int(float(options["max_sync_gap_ms"]) * 1_000_000)
    ranges = list(job.get("approved_ranges", []))
    streams: dict[str, list[tuple[int, int, Any]]] = {
        "color": [], "depth": [], "imu": [], "odom": [], "action": [], "collision": [],
    }
    decode_scratch = output.parent / f".{output.name}.decode-scratch"
    if decode_scratch.exists():
        shutil.rmtree(decode_scratch)
    decode_scratch.mkdir(parents=True)
    payload_index = 0

    def spool(payload: tuple[bytes, str, str], role: str) -> tuple[Path, str, str]:
        nonlocal payload_index
        encoded, suffix, media_type = payload
        path = decode_scratch / f"{role}-{payload_index:012d}{suffix}"
        payload_index += 1
        path.write_bytes(encoded)
        return path, suffix, media_type
    topic_roles = {str(value): key for key, value in topics.items() if value}
    depth_decode_errors: dict[str, Any] = {
        "total": 0, "by_error_type": {}, "examples": [],
    }

    def record_depth_error(detail: dict[str, Any]) -> None:
        error_type = str(detail["error_type"])
        depth_decode_errors["total"] += 1
        counts = depth_decode_errors["by_error_type"]
        counts[error_type] = counts.get(error_type, 0) + 1
        if len(depth_decode_errors["examples"]) < 20:
            depth_decode_errors["examples"].append(detail)

    def handle_message_decode_error(detail: dict[str, Any]) -> bool:
        if topic_roles.get(str(detail["topic"])) != "depth":
            return False
        record_depth_error({**detail, "stage": "ros_message_decode", "format": ""})
        return True

    try:
        for topic, timestamp, sequence, message in _iter_messages(
            (Path(item["path"]) for item in job["sources"]),
            on_decode_error=handle_message_decode_error,
        ):
            if not _inside(timestamp, ranges):
                continue
            role = topic_roles.get(topic)
            if role == "color":
                streams["color"].append((timestamp, sequence, spool(_image_bytes(
                    message,
                    image_size=int(options["image_size"]) if options["image_size"] else None,
                    encoding=str(options["rgb_encoding"]).casefold(),
                    jpeg_quality=int(options["jpeg_quality"]),
                ), "rgb")))
            elif role == "depth":
                try:
                    payload = _depth_bytes(
                        message,
                        image_size=int(options["image_size"])
                        if options["image_size"] else None,
                    )
                except Exception as exc:
                    error_type = type(exc).__name__
                    record_depth_error({
                        "topic": topic,
                        "timestamp_ns": int(timestamp),
                        "sequence": int(sequence),
                        "error_type": error_type,
                        "error": str(exc),
                        "stage": "depth_payload_decode",
                        "format": str(getattr(message, "format", "")),
                    })
                    continue
                streams["depth"].append((
                    timestamp, sequence, spool(payload, "depth"),
                ))
            elif role == "imu":
                streams["imu"].append((timestamp, sequence, _imu(message)))
            elif role == "odom":
                decoded_odom = _odom(message)
                streams["odom"].append((timestamp, sequence, decoded_odom))
                if action_name == "odom":
                    streams["action"].append((timestamp, sequence, decoded_odom))
                elif action_name == "future_odom_pose":
                    # The future portion of a window is a sequence of realized
                    # pose waypoints. Twist remains in the odometry state.
                    streams["action"].append((
                        timestamp, sequence, decoded_odom[:7],
                    ))
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
            elif role == "collision":
                streams["collision"].append((timestamp, sequence, _collision(message)))
    except McapStructureError as exc:
        shutil.rmtree(decode_scratch, ignore_errors=True)
        if options["on_ineligible_recording"] == "fail":
            raise
        _materialize_early_exclusion(
            job, output,
            reason_code="unreadable_mcap_structure",
            details={
                "message": str(exc),
                "storage_part": exc.storage_part,
                "storage_part_index": exc.storage_part_index,
                "error_type": exc.error_type,
            },
            config=config,
            action_name=action_name,
        )
        return
    for rows in streams.values():
        rows.sort(key=lambda row: (row[0], row[1]))
    candidates = []
    for candidate_index, (timestamp, sequence, image) in enumerate(streams["color"]):
        matches = {
            name: _nearest(streams[name], timestamp, tolerance)
            for name in ("imu", "odom", "action", "depth", "collision")
        }
        required_valid = all(matches[name] is not None for name in required - {"rgb"})
        speed = np.linalg.norm(matches["odom"][2][7:10]) if matches["odom"] else float("inf")
        candidates.append({"timestamp": timestamp, "sequence": sequence,
                           "candidate_index": candidate_index, "image": image,
                           "range_index": _range_index(timestamp, ranges),
                           "matches": matches, "required_valid": required_valid,
                           "slow": speed < float(options["min_speed"])})
    source_candidate_count = len(candidates)
    candidates = _sample_candidates(candidates, ranges, job.get("sampling_policy", {}))
    sampling_dropped = source_candidate_count - len(candidates)
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

    if output.exists():
        archive = output.parent / ".stale" / f"{output.name}-{time.time_ns()}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        output.replace(archive)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}")
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
    default_action_dim = (
        22 if action_name == "so3_cmd"
        else 7 if action_name == "future_odom_pose"
        else 14
    )
    action_dim = len(streams["action"][0][2]) if streams["action"] else (
        int(action_config.get("dim", default_action_dim))
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
                    or candidate["range_index"] != previous_range
                    or timestamp - int(previous_timestamp) > gap_limit):
                segment += 1
            frame_index = len(frame_rows)
            matches = candidate["matches"]
            rgb_spooled, rgb_suffix, rgb_media = candidate["image"]
            asset_stem = f"{frame_index:08d}-{timestamp}"
            rgb_source = scratch / f"rgb-{asset_stem}{rgb_suffix}"
            shutil.copyfile(rgb_spooled, rgb_source)
            rgb_rel = recording_rel / "rgb" / f"{asset_stem}{rgb_suffix}"
            rgb_dest = temporary / "rgb" / f"{asset_stem}{rgb_suffix}"
            rgb_digest, rgb_mode, rgb_key = _cas_ingest(rgb_source, object_store, rgb_dest)
            asset_rows.append({"logical_path": rgb_rel.as_posix(), "sha256": rgb_digest,
                               "size_bytes": rgb_source.stat().st_size, "media_type": rgb_media,
                               "encoding": rgb_suffix.removeprefix("."),
                               "storage_mode": rgb_mode, "object_key": rgb_key})
            depth_rel = None
            depth_offset = None
            if matches["depth"] is not None:
                depth_spooled, depth_suffix, depth_media = matches["depth"][2]
                depth_source = scratch / f"depth-{asset_stem}{depth_suffix}"
                shutil.copyfile(depth_spooled, depth_source)
                depth_rel_path = recording_rel / "depth" / f"{asset_stem}{depth_suffix}"
                depth_dest = temporary / "depth" / f"{asset_stem}{depth_suffix}"
                depth_digest, depth_mode, depth_key = _cas_ingest(depth_source, object_store, depth_dest)
                asset_rows.append({"logical_path": depth_rel_path.as_posix(), "sha256": depth_digest,
                                   "size_bytes": depth_source.stat().st_size, "media_type": depth_media,
                                   "encoding": depth_suffix.removeprefix("."),
                                   "storage_mode": depth_mode, "object_key": depth_key})
                depth_rel = depth_rel_path.as_posix()
                depth_offset = int(matches["depth"][0] - timestamp)
            frame_rows.append({
                "recording_id": job["recording_id"], "split": job["split"],
                "frame_index": frame_index, "segment_id": segment,
                "approved_range_index": int(candidate["range_index"]),
                "timestamp_ns": timestamp,
                "rgb_path": rgb_rel.as_posix(), "depth_path": depth_rel,
                "depth_valid": matches["depth"] is not None,
                "imu": _matched_values(matches["imu"], imu_dim),
                "odom": _matched_values(matches["odom"], odom_dim),
                "action": _matched_values(matches["action"], action_dim),
                "collision": float(matches["collision"][2]) if matches["collision"] else 0.0,
                "imu_valid": matches["imu"] is not None,
                "odom_valid": matches["odom"] is not None,
                "action_valid": matches["action"] is not None,
                "collision_valid": matches["collision"] is not None,
                "imu_offset_ns": _offset(matches["imu"], timestamp),
                "odom_offset_ns": _offset(matches["odom"], timestamp),
                "action_offset_ns": _offset(matches["action"], timestamp),
                "depth_offset_ns": depth_offset, "valid": True,
                "collision_offset_ns": _offset(matches["collision"], timestamp),
            })
            previous_candidate, previous_timestamp = candidate["candidate_index"], timestamp
            previous_range = candidate["range_index"]
    shutil.rmtree(decode_scratch, ignore_errors=True)
    excluded_reason = None
    if not frame_rows:
        if not streams["color"]:
            excluded_reason = "no_rgb_in_approved_ranges"
        elif depth_is_required and depth_decode_errors["total"] and not streams["depth"]:
            excluded_reason = "unusable_required_depth"
        else:
            excluded_reason = "no_valid_frames"
        if options["on_ineligible_recording"] == "fail":
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
    _write_poses(frame_rows, temporary)
    _write_parquet(frame_rows, temporary / "frames.parquet", frame_schema)
    _write_parquet(window_rows, temporary / "windows.parquet", window_schema)
    _write_parquet(asset_rows, temporary / "assets.parquet", asset_schema)
    output_checksums = {
        name: _sha256(temporary / name)
        for name in ("frames.parquet", "windows.parquet", "assets.parquet", "poses/poses.parquet")
    }
    _atomic_json(temporary / "recording_manifest.json", {
        "schema_version": 1, "recording_id": job["recording_id"], "split": job["split"],
        "status": "excluded" if excluded_reason else "included",
        "reason_codes": [excluded_reason] if excluded_reason else [],
        "fingerprint": job["fingerprint"], "frame_count": len(frame_rows),
        "window_count": len(window_rows), "sensor_dimensions": {
            "imu": imu_dim, "odom": odom_dim, "action": action_dim,
        }, "dropped_frames": len(candidates) - len(frame_rows),
        "sampling_dropped_frames": sampling_dropped,
        "rejected_spans": _false_spans(candidates, "required_valid"),
        "decode_errors": {
            "depth": {
                **depth_decode_errors,
                "examples_truncated": (
                    depth_decode_errors["total"]
                    > len(depth_decode_errors["examples"])
                ),
            },
        },
        "sensor_semantics": {
            "imu": sensors.get("imu", {"dimension": imu_dim}),
            "odom": sensors.get("odom", {"dimension": odom_dim}),
            "action": {
                "source": action_name, "dimension": action_dim,
                "semantic": action_config.get("semantic", action_name),
                "components": action_config.get("components", []),
                "units": action_config.get("units", []),
                "frame": action_config.get("frame"),
                "derived_from": action_config.get("derived_from"),
            },
        },
        "outputs": output_checksums,
    })
    temporary.replace(output)


def _sync_quantiles(frames: Any) -> dict[str, dict[str, float]]:
    report = {}
    for modality in ("imu", "odom", "action", "depth", "collision"):
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


def _verify_inventory(root: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        path = root / str(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"standalone export is incomplete: {path}")
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"standalone export checksum mismatch: {path}")


def finalize(build_manifest: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq
    build = json.loads(build_manifest.read_text(encoding="utf-8"))
    final_root = Path(build["output_root"])
    root = Path(build.get("staging_root", build["output_root"]))
    root.mkdir(parents=True, exist_ok=True)
    frame_tables, window_tables, asset_tables = [], [], []
    records, included_records, recording_rows = [], [], []
    for job in build["jobs"]:
        output = Path(job["output_dir"])
        record = json.loads((output / "recording_manifest.json").read_text(encoding="utf-8"))
        if record.get("fingerprint") != job["fingerprint"]:
            raise ValueError(f"stale Gold recording output: {output}")
        for relative, digest in record.get("outputs", {}).items():
            if not (output / relative).is_file() or _sha256(output / relative) != digest:
                raise ValueError(f"recording output checksum mismatch: {output / relative}")
        records.append(record)
        if record.get("status", "included") == "included":
            included_records.append(record)
            frame_tables.append(pq.read_table(output / "frames.parquet"))
            window_tables.append(pq.read_table(output / "windows.parquet"))
            asset_tables.append(pq.read_table(output / "assets.parquet"))
        spec = (
            json.loads(Path(job["spec_path"]).read_text(encoding="utf-8"))
            if job.get("spec_path") else {}
        )
        recording_rows.append({
            "recording_id": str(record["recording_id"]),
            "split": str(record["split"]),
            "status": str(record.get("status", "included")),
            "reason_codes": list(record.get("reason_codes", [])),
            "frame_count": int(record.get("frame_count", 0)),
            "window_count": int(record.get("window_count", 0)),
            "dropped_frames": int(record.get("dropped_frames", 0)),
            "sampling_dropped_frames": int(record.get("sampling_dropped_frames", 0)),
            "rejected_spans": int(record.get("rejected_spans", 0)),
            "depth_decode_errors": int(
                record.get("decode_errors", {}).get("depth", {}).get("total", 0)
            ),
            "exclusion_details_json": json.dumps(
                record.get("exclusion_details", {}),
                sort_keys=True, separators=(",", ":"),
            ),
            "source_parts_json": json.dumps([
                {key: part[key] for key in ("logical_path", "sha256", "size_bytes")}
                for part in spec.get("sources", [])
            ], sort_keys=True, separators=(",", ":")),
            "approved_ranges_json": json.dumps(
                spec.get("approved_ranges", []), sort_keys=True, separators=(",", ":")
            ),
            "topic_profile_sha256": spec.get("topic_profile_sha256"),
            "fingerprint": str(record["fingerprint"]),
        })
    dimension_values = {
        json.dumps(record["sensor_dimensions"], sort_keys=True)
        for record in included_records
    }
    semantic_values = {
        json.dumps(record.get("sensor_semantics", {}), sort_keys=True)
        for record in included_records
    }
    if len(dimension_values) > 1 or len(semantic_values) > 1:
        contracts: dict[str, list[str]] = {}
        for record in included_records:
            contract = json.dumps({
                "dimensions": record["sensor_dimensions"],
                "semantics": record.get("sensor_semantics", {}),
            }, sort_keys=True, separators=(",", ":"))
            contracts.setdefault(contract, []).append(str(record["recording_id"]))
        summary = "; ".join(
            f"{contract}: {', '.join(recording_ids[:5])}"
            for contract, recording_ids in sorted(contracts.items())
        )
        raise ValueError(
            "included WM recordings have incompatible sensor contracts; configure "
            "recipe.options.action_contract so one Gold product has one action shape "
            f"and meaning ({summary})"
        )
    frames = pa.concat_tables(frame_tables) if frame_tables else pa.table({})
    windows = pa.concat_tables(window_tables) if window_tables else pa.table({})
    assets = pa.concat_tables(asset_tables) if asset_tables else pa.table({})
    if windows.num_rows == 0:
        raise ValueError("Gold release has no valid training windows")
    if "train" not in set(windows["split"].to_pylist()):
        raise ValueError("Gold release has no windows in the Silver train split")
    sensor_semantics = (
        included_records[0].get("sensor_semantics", {}) if included_records else {}
    )
    recordings = pa.Table.from_pylist(recording_rows)
    index_tables = {
        "recordings.parquet": recordings,
        "frames.parquet": frames,
        "windows.parquet": windows,
        "assets.parquet": assets,
    }
    temporary_indexes = {}
    for name, table in index_tables.items():
        temporary = root / f".{name}.tmp-{os.getpid()}"
        pq.write_table(table, temporary, compression="zstd")
        temporary_indexes[name] = temporary
    for name, temporary in temporary_indexes.items():
        temporary.replace(root / name)
    dimensions = (
        included_records[0]["sensor_dimensions"]
        if included_records else {"imu": 0, "odom": 0, "action": 0}
    )
    checksums = {name: _sha256(root / name) for name in index_tables}
    identity = {
        "consumer_format": "wm_planner",
        "release_hash": build["release_hash"],
        "recipe_sha256": build["recipe_sha256"],
        "adapter_code_identity": build.get("adapter", {}).get("code_identity_sha256"),
        "adapter_environment_identity": build.get("adapter", {}).get("environment_sha256"),
        "checksums": checksums,
    }
    dataset_hash = _sha256_bytes(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    )
    split_counts = {
        split: {
            "recordings": sum(
                row["split"] == split and row["status"] == "included"
                for row in recording_rows
            ),
            "frames": sum(1 for row in frames.to_pylist() if row["split"] == split),
            "windows": sum(1 for row in windows.to_pylist() if row["split"] == split),
        }
        for split in ("train", "val", "test")
    }
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION, "consumer_format": "wm_planner",
        "dataset_id": f"{build['release_id']}-{dataset_hash[:12]}",
        "dataset_hash": dataset_hash,
        "identity": identity,
        "release_id": build["release_id"], "release_hash": build["release_hash"],
        "created_at_ns": time.time_ns(), "recipe_sha256": build["recipe_sha256"],
        "policy": build.get("recipe", {}).get("options", {}),
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
        "object_store": os.path.relpath(
            Path(build.get("object_store", final_root.parent / ".objects")), final_root
        ),
        "frame_count": frames.num_rows, "window_count": windows.num_rows,
        "split_counts": split_counts,
        "sensor_dimensions": dimensions,
        "sensor_semantics": sensor_semantics,
        "poses": {
            "path_template": "recordings/{recording_id}/poses/poses.parquet",
            "source": "odom", "timestamp_anchor": "rgb",
            "components": ["x", "y", "z", "qx", "qy", "qz", "qw"],
            "frame": "source odometry pose frame (no camera extrinsic applied)",
            "validity": "valid",
        },
        "semantic_schema": {
            "version": 2, "timestamp": {"dtype": "int64", "unit": "ns"},
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
        "files": {
            "recordings": "recordings.parquet", "frames": "frames.parquet",
            "windows": "windows.parquet", "assets": "assets.parquet",
        },
        "checksums": checksums,
        "quality": {
            "rejected_frames": sum(int(record.get("dropped_frames", 0)) for record in records),
            "sampling_dropped_frames": sum(
                int(record.get("sampling_dropped_frames", 0)) for record in records
            ),
            "rejected_spans": sum(int(record.get("rejected_spans", 0)) for record in records),
            "excluded_recordings": sum(record.get("status") == "excluded" for record in records),
            "incompatible_action_recordings": sum(
                "incompatible_action_contract" in record.get("reason_codes", [])
                for record in records
            ),
            "unreadable_mcap_recordings": sum(
                "unreadable_mcap_structure" in record.get("reason_codes", [])
                for record in records
            ),
            "depth_decode_errors": sum(
                int(record.get("decode_errors", {}).get("depth", {}).get("total", 0))
                for record in records
            ),
            "synchronization_quantiles_ns": _sync_quantiles(frames),
        },
    }
    path = root / "dataset_manifest.json"
    _atomic_json(path, manifest)
    if root.resolve() != final_root.resolve():
        if final_root.exists():
            existing = final_root / "dataset_manifest.json"
            if existing.is_file() and json.loads(existing.read_text()).get("dataset_hash") == dataset_hash:
                shutil.rmtree(root)
                return existing
            raise FileExistsError(f"immutable Gold product already exists: {final_root}")
        root.replace(final_root)
        path = final_root / "dataset_manifest.json"
    return path


def _load_gold(manifest: Path, split: str):
    from world_model.data.parquet_dataset import ParquetWindowDataset
    return ParquetWindowDataset(str(manifest), split=split)


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
        files = _file_inventory(temporary)
        _verify_inventory(temporary, files)
        source = json.loads(manifest.read_text(encoding="utf-8"))
        _atomic_json(temporary / "export_manifest.json", {
            "schema_version": 2, "format": "npz", "window_semantics": "canonical",
            "source_manifest": str(manifest.resolve()),
            "source_sha256": _sha256(manifest), "dataset_hash": source.get("dataset_hash"),
            "files": files,
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
                "source_frame_index": Value("int64"),
                "release_id": Value("string"), "dataset_id": Value("string"),
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
                       "source_frame_index": int(item["frame_indices"][frame_index]),
                       "release_id": str(payload["release_id"]),
                       "dataset_id": str(payload.get("dataset_id", payload["release_id"])),
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
        files = _file_inventory(temporary)
        _verify_inventory(temporary, files)
        _atomic_json(temporary / "export_manifest.json", {
            "schema_version": 2, "format": "hf", "window_semantics": "canonical",
            "source_manifest": str(manifest.resolve()),
            "source_sha256": _sha256(manifest), "dataset_hash": payload.get("dataset_hash"),
            "splits": sorted(datasets), "files": files,
        })
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol")
    validate_parser = sub.add_parser("validate-recipe")
    validate_parser.add_argument("--recipe", type=Path, required=True)
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
    elif args.command == "validate-recipe":
        import yaml
        recipe = yaml.safe_load(args.recipe.read_text(encoding="utf-8")) or {}
        _validated_options(recipe)
        print(json.dumps({"valid": True, "consumer": "wm_planner"}))
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
