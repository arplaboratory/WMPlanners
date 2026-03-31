#!/usr/bin/env python3
"""Stage 0: Extract ROS2 MCAP bags into a sharded dataset.

Produces the directory structure expected by the data pipeline (Stage 1+):
  - Color images saved as JPEGs in shards (folders of N images each)
  - Depth images saved as 16-bit PNGs alongside color
  - Proprioceptive data (IMU, odometry, commands) saved as CSV
  - image_index.csv mapping frame IDs to file paths and timestamps
  - manifest.json at the output root summarising all bags extracted

Usage:
    python -m scripts.world_model.extract_dataset \\
        --robot race16 \\
        --mcap path/to/bag1.mcap path/to/bag2.mcap \\
        --output dataset_out

    # With config overrides:
    python -m scripts.world_model.extract_dataset \\
        --robot race16 --override topics.color=/other/topic \\
        --mcap path/to/bag.mcap --output dataset_out

    # Then feed into the pipeline:
    python -m scripts.world_model.data_pipeline.run_pipeline \\
        --bags dataset_out --output pipeline_out
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from pathlib import Path

import cv2
import numpy as np

from rosbags.highlevel import AnyReader

from scripts.world_model.robot_config import add_robot_args, load_robot_config, parse_overrides

SHARD_SIZE = 500  # images per shard

# ---------------------------------------------------------------------------
# CSV field definitions
# ---------------------------------------------------------------------------
IMU_FIELDS = [
    "timestamp_ns",
    "orientation_x", "orientation_y", "orientation_z", "orientation_w",
    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
    "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
]

ODOM_FIELDS = [
    "timestamp_ns",
    "pos_x", "pos_y", "pos_z",
    "orient_x", "orient_y", "orient_z", "orient_w",
    "linear_vel_x", "linear_vel_y", "linear_vel_z",
    "angular_vel_x", "angular_vel_y", "angular_vel_z",
]

CMD_FIELDS = [
    "timestamp_ns",
    "position_x", "position_y", "position_z",
    "velocity_x", "velocity_y", "velocity_z",
    "acceleration_x", "acceleration_y", "acceleration_z",
    "jerk_x", "jerk_y", "jerk_z",
    "yaw", "yaw_dot",
]

SO3_CMD_FIELDS = [
    "timestamp_ns",
    "force_x", "force_y", "force_z",
    "orientation_x", "orientation_y", "orientation_z", "orientation_w",
    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
    "kx_0", "kx_1", "kx_2",
    "kv_0", "kv_1", "kv_2",
    "kR_0", "kR_1", "kR_2",
    "kOm_0", "kOm_1", "kOm_2",
]

INDEX_FIELDS = ["frame_id", "timestamp_ns", "color_path", "depth_path"]


# ---------------------------------------------------------------------------
# CSV writer helper
# ---------------------------------------------------------------------------
class CSVManager:
    """Lazily creates and manages CSV writers for each sensor modality."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self._files: dict[str, object] = {}
        self._writers: dict[str, csv.DictWriter] = {}

    def get(self, name: str, fieldnames: list[str]) -> csv.DictWriter:
        if name not in self._writers:
            path = os.path.join(self.output_dir, f"{name}.csv")
            f = open(path, "w", newline="")
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            self._files[name] = f
            self._writers[name] = writer
        return self._writers[name]

    def close(self) -> None:
        for fh in self._files.values():
            fh.close()

    @property
    def names(self) -> list[str]:
        return list(self._writers.keys())


# ---------------------------------------------------------------------------
# Image savers
# ---------------------------------------------------------------------------
def save_compressed_image(
    msg, output_dir: str, color_count: int, shard_size: int,
) -> str:
    shard_id = color_count // shard_size
    shard_dir = os.path.join(output_dir, f"shard_{shard_id:04d}")
    os.makedirs(shard_dir, exist_ok=True)
    img_path = os.path.join(shard_dir, f"color_{color_count:06d}.jpg")
    with open(img_path, "wb") as f:
        f.write(bytes(msg.data))
    return img_path


def save_depth_image(
    msg, output_dir: str, depth_count: int, shard_size: int,
) -> str:
    shard_id = depth_count // shard_size
    shard_dir = os.path.join(output_dir, f"shard_{shard_id:04d}")
    os.makedirs(shard_dir, exist_ok=True)
    img_path = os.path.join(shard_dir, f"depth_{depth_count:06d}.png")
    h, w = msg.height, msg.width
    encoding = msg.encoding
    raw = bytes(msg.data)
    if encoding == "16UC1":
        arr = np.frombuffer(raw, dtype=np.uint16).reshape(h, w)
    elif encoding == "32FC1":
        arr = np.frombuffer(raw, dtype=np.float32).reshape(h, w)
        arr = (arr * 1000).astype(np.uint16)  # metres → mm
    else:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, -1)
    cv2.imwrite(img_path, arr)
    return img_path


# ---------------------------------------------------------------------------
# Sensor processors
# ---------------------------------------------------------------------------
def process_imu(msg, timestamp_ns: int, csv_mgr: CSVManager) -> None:
    w = csv_mgr.get("imu", IMU_FIELDS)
    o, av, la = msg.orientation, msg.angular_velocity, msg.linear_acceleration
    w.writerow({
        "timestamp_ns": timestamp_ns,
        "orientation_x": o.x, "orientation_y": o.y, "orientation_z": o.z, "orientation_w": o.w,
        "angular_velocity_x": av.x, "angular_velocity_y": av.y, "angular_velocity_z": av.z,
        "linear_acceleration_x": la.x, "linear_acceleration_y": la.y, "linear_acceleration_z": la.z,
    })


def process_odom(msg, timestamp_ns: int, csv_mgr: CSVManager) -> None:
    w = csv_mgr.get("odom", ODOM_FIELDS)
    p = msg.pose.pose.position
    o = msg.pose.pose.orientation
    lv = msg.twist.twist.linear
    av = msg.twist.twist.angular
    w.writerow({
        "timestamp_ns": timestamp_ns,
        "pos_x": p.x, "pos_y": p.y, "pos_z": p.z,
        "orient_x": o.x, "orient_y": o.y, "orient_z": o.z, "orient_w": o.w,
        "linear_vel_x": lv.x, "linear_vel_y": lv.y, "linear_vel_z": lv.z,
        "angular_vel_x": av.x, "angular_vel_y": av.y, "angular_vel_z": av.z,
    })


def process_position_cmd(msg, timestamp_ns: int, csv_mgr: CSVManager) -> None:
    w = csv_mgr.get("position_cmd", CMD_FIELDS)
    p, v, a, j = msg.position, msg.velocity, msg.acceleration, msg.jerk
    w.writerow({
        "timestamp_ns": timestamp_ns,
        "position_x": p.x, "position_y": p.y, "position_z": p.z,
        "velocity_x": v.x, "velocity_y": v.y, "velocity_z": v.z,
        "acceleration_x": a.x, "acceleration_y": a.y, "acceleration_z": a.z,
        "jerk_x": j.x, "jerk_y": j.y, "jerk_z": j.z,
        "yaw": msg.yaw, "yaw_dot": msg.yaw_dot,
    })


def process_so3_cmd(msg, timestamp_ns: int, csv_mgr: CSVManager) -> None:
    w = csv_mgr.get("so3_cmd", SO3_CMD_FIELDS)
    f, o, av = msg.force, msg.orientation, msg.angular_velocity
    kx, kv, kr, kom = msg.kx, msg.kv, msg.kr, msg.kom
    w.writerow({
        "timestamp_ns": timestamp_ns,
        "force_x": f.x, "force_y": f.y, "force_z": f.z,
        "orientation_x": o.x, "orientation_y": o.y, "orientation_z": o.z, "orientation_w": o.w,
        "angular_velocity_x": av.x, "angular_velocity_y": av.y, "angular_velocity_z": av.z,
        "kx_0": kx[0], "kx_1": kx[1], "kx_2": kx[2],
        "kv_0": kv[0], "kv_1": kv[1], "kv_2": kv[2],
        "kR_0": kr[0], "kR_1": kr[1], "kR_2": kr[2],
        "kOm_0": kom[0], "kOm_1": kom[1], "kOm_2": kom[2],
    })


# ---------------------------------------------------------------------------
# Config-driven topic handler builder
# ---------------------------------------------------------------------------
def discover_topics(path: str) -> list[str]:
    """Pre-scan bag to discover available topics."""
    topics = []
    if path.endswith(".bag"):
        from rosbags.highlevel import AnyReader
        from pathlib import Path
        try:
            with AnyReader([Path(path)]) as reader:
                topics = [c.topic for c in reader.connections]
        except Exception: pass
    else:
        from mcap.stream_reader import StreamReader
        from mcap.records import Channel
        try:
            with open(path, "rb") as f:
                reader = StreamReader(f, emit_chunks=True)
                it = reader.records
                while True:
                    try:
                        record = next(it)
                    except StopIteration: break
                    except Exception: continue
                    if isinstance(record, Channel):
                        if record.topic not in topics:
                            topics.append(record.topic)
        except Exception: pass
    return topics


def build_topic_handlers(config: dict, bag_path: str = None) -> dict:
    """Build a mapping from ROS topic string to handler info dict.
    
    If bag_path is provided, attempts fuzzy matching for missing topics.
    """
    handlers: dict[str, dict] = {}
    bag_topics = discover_topics(bag_path) if bag_path else []
    
    # Fallback suffixes for common sensors
    FALLBACKS = {
        "imu": ["/imu", "/imu/data", "/mavros/imu/data"],
        "odom": ["/odom", "/local_position/odom", "/mavros/local_position/odom"],
        "depth": ["/depth/image_raw", "/aligned_depth_to_color/image_raw", "/depth/image_rect_raw"],
        "color": ["/color/image_raw/compressed", "/color/image_raw"],
    }
    
    robot_name = config.get("name", "")

    for key, topic in config["topics"].items():
        chosen_topic = None
        if not bag_path or topic in bag_topics:
            chosen_topic = topic
        elif bag_topics:
            # Try fuzzy matching
            # 1. Look for robot-prefixed fallbacks
            if key in FALLBACKS:
                for suffix in FALLBACKS[key]:
                    full = f"/{robot_name}{suffix}" if not suffix.startswith(f"/{robot_name}") else suffix
                    if full in bag_topics:
                        chosen_topic = full
                        print(f"  [Fuzzy] Matched {key} -> {chosen_topic} (fallback)")
                        break
            # 2. Look for any topic ending in a common suffix if still not found
            if not chosen_topic and key in FALLBACKS:
                for suffix in FALLBACKS[key]:
                    for bt in bag_topics:
                        if bt.endswith(suffix):
                            chosen_topic = bt
                            print(f"  [Fuzzy] Matched {key} -> {chosen_topic} (approximate)")
                            break
                    if chosen_topic: break

        if chosen_topic:
            handlers[chosen_topic] = {"type": key, "key": key}
    
    return handlers


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------
def write_manifest(output_dir: str, config: dict, bags_info: list[dict]) -> None:
    """Write manifest.json with robot name, config, and bag list."""
    manifest = {
        "robot": config.get("name", "unknown"),
        "config": config,
        "bags": bags_info,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Format-specific message iterators
# ---------------------------------------------------------------------------
def iter_mcap_messages(path: str):
    """Generator for MCAP (ROS2) messages using streaming reader to avoid summary/index errors
    and handle corrupted chunks gracefully."""
    from mcap.stream_reader import StreamReader, breakup_chunk
    from mcap_ros2.decoder import DecoderFactory
    from mcap.records import Schema, Channel, Message, Chunk
    
    decoder_factory = DecoderFactory()
    channels: dict[int, Channel] = {}
    schemas: dict[int, Schema] = {}
    
    def process_record(record):
        nonlocal channels, schemas
        if isinstance(record, Schema):
            schemas[record.id] = record
        elif isinstance(record, Channel):
            channels[record.id] = record
        elif isinstance(record, Message):
            if record.channel_id not in channels:
                return None
            channel = channels[record.channel_id]
            if channel.schema_id not in schemas:
                return None
            schema = schemas[channel.schema_id]
            decoder = decoder_factory.decoder_for(channel.message_encoding, schema)
            if not decoder:
                return None
            ros_msg = decoder(record.data)
            return channel.topic, record.log_time, ros_msg
        return None

    with open(path, "rb") as f:
        reader = StreamReader(f, emit_chunks=True)
        it = reader.records
        while True:
            try:
                record = next(it)
            except StopIteration:
                break
            except Exception as e:
                print(f"Warning: Top-level MCAP corruption detected: {e}. Stopping extraction for this file.")
                break

            if isinstance(record, Chunk):
                try:
                    for subrecord in breakup_chunk(record):
                        res = process_record(subrecord)
                        if res:
                            yield res
                except Exception as e:
                    print(f"Warning: Skipping corrupted MCAP chunk: {e}")
            else:
                res = process_record(record)
                if res:
                    yield res


def iter_rosbag_messages(path: str):
    """Generator for ROS1 bag messages using rosbags library."""
    from rosbags.highlevel import AnyReader
    from pathlib import Path
    with AnyReader([Path(path)]) as reader:
        for connection, timestamp, rawdata in reader.messages():
            try:
                ros_msg = reader.deserialize(rawdata, connection.msgtype)
                yield connection.topic, timestamp, ros_msg
            except (AssertionError, Exception) as e:
                print(f"Warning: Failed to deserialize ROS1 message on topic {connection.topic}: {e}. Skipping message.")


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------
def extract(mcap_path: str, output_dir: str, config: dict, shard_size: int = SHARD_SIZE) -> int:
    """Extract all sensor data from an MCAP bag into *output_dir*.

    Args:
        mcap_path: Path to the .mcap bag file.
        output_dir: Directory to write extracted data into.
        config: Robot config dict (topics, sensors sections expected).
        shard_size: Number of images per shard directory.

    Returns:
        Number of color frames extracted.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_mgr = CSVManager(output_dir)
    index_rows: list[dict] = []
    color_count = 0
    depth_count = 0

    # Build topic handlers (with fuzzy matching if possible)
    handlers = build_topic_handlers(config, bag_path=mcap_path)
    color_encoding = config.get("sensors", {}).get("color_encoding", "compressed")

    # Track unknown topics to warn only once
    warned_topics: set[str] = set()

    print(f"Extracting from {mcap_path} ...")

    if str(mcap_path).endswith(".bag"):
        msg_iterator = iter_rosbag_messages(mcap_path)
    else:
        msg_iterator = iter_mcap_messages(mcap_path)

    for topic, ts, ros_msg in msg_iterator:
        if topic not in handlers:
            continue

        handler = handlers[topic]
        htype = handler["type"]

        if htype == "color":
            if color_encoding == "compressed":
                path = save_compressed_image(ros_msg, output_dir, color_count, shard_size)
            else:
                # Raw Image message — encode to JPEG via OpenCV
                h, w = ros_msg.height, ros_msg.width
                raw = bytes(ros_msg.data)
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, -1)
                shard_id = color_count // shard_size
                shard_dir = os.path.join(output_dir, f"shard_{shard_id:04d}")
                os.makedirs(shard_dir, exist_ok=True)
                path = os.path.join(shard_dir, f"color_{color_count:06d}.jpg")
                cv2.imwrite(path, arr)
            color_count += 1
            index_rows.append({
                "frame_id": color_count - 1,
                "timestamp_ns": ts,
                "color_path": path,
                "depth_path": "",
            })
            if color_count % 1000 == 0:
                print(f"  Color frames: {color_count}")

        elif htype == "depth":
            path = save_depth_image(ros_msg, output_dir, depth_count, shard_size)
            depth_count += 1
            if index_rows and index_rows[-1]["depth_path"] == "":
                index_rows[-1]["depth_path"] = path
            if depth_count % 1000 == 0:
                print(f"  Depth frames: {depth_count}")

        elif htype == "imu":
            process_imu(ros_msg, ts, csv_mgr)

        elif htype == "odom":
            process_odom(ros_msg, ts, csv_mgr)

        elif htype == "position_cmd":
            process_position_cmd(ros_msg, ts, csv_mgr)

        elif htype == "so3_cmd":
            process_so3_cmd(ros_msg, ts, csv_mgr)

        else:
            if htype not in warned_topics:
                warnings.warn(
                    f"No handler for topic key '{htype}' (topic: {topic}); skipping.",
                    stacklevel=2,
                )
                warned_topics.add(htype)

    # Write image index
    index_path = os.path.join(output_dir, "image_index.csv")
    with open(index_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(index_rows)

    csv_mgr.close()

    n_shards = color_count // shard_size + (1 if color_count % shard_size else 0)
    print(f"\nDone!")
    print(f"  Color frames: {color_count}")
    print(f"  Depth frames: {depth_count}")
    print(f"  Image shards: {n_shards}")
    print(f"  CSVs: {csv_mgr.names}")
    print(f"  Output: {output_dir}/")

    return color_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ROS2 MCAP bags into sharded dataset",
    )
    add_robot_args(parser)  # adds --robot and --override
    parser.add_argument("--mcap", nargs="+", required=True, help="Path(s) to .mcap bag file(s)")
    parser.add_argument("--output", default="dataset_out", help="Output directory (default: dataset_out)")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE, help=f"Images per shard (default: {SHARD_SIZE})")
    args = parser.parse_args()

    overrides = parse_overrides(args.override)
    config = load_robot_config(args.robot, overrides=overrides)

    os.makedirs(args.output, exist_ok=True)

    bags_info = []
    for mcap_path in args.mcap:
        bag_stem = Path(mcap_path).stem
        bag_output = os.path.join(args.output, bag_stem)
        frames = extract(mcap_path, bag_output, config, args.shard_size)
        bags_info.append({"name": bag_stem, "path": f"{bag_stem}/", "frames": frames})

    write_manifest(args.output, config, bags_info)
    print(f"\nManifest written to {os.path.join(args.output, 'manifest.json')}")


if __name__ == "__main__":
    main()
