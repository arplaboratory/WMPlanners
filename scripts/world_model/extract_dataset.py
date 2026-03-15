"""
Extract a ROS2 MCAP bag into a sharded dataset:
  - Color images saved as JPEGs in shards (folders of N images each)
  - Depth images saved as 16-bit PNGs alongside color
  - Proprioceptive data (IMU, odometry, commands) saved as CSV
"""

import os
import csv
import struct
import numpy as np
import cv2
from collections import defaultdict
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

MCAP_PATH = "race16_bag_20260118_120207.mcap"
OUTPUT_DIR = "dataset_out"
SHARD_SIZE = 500  # images per shard

# Topics to extract
COLOR_TOPIC = "/race16/cam1/color/image_raw/compressed"
DEPTH_TOPIC = "/race16/cam1/aligned_depth_to_color/image_raw"
IMU_TOPIC = "/race16/cam1/imu"
ODOM_TOPIC = "/race16/odom"
CMD_TOPIC = "/race16/position_cmd"
SO3_CMD_TOPIC = "/race16/so3_cmd"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CSV writers setup ---
csv_files = {}
csv_writers = {}

def get_csv_writer(name, fieldnames):
    if name not in csv_writers:
        path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        f = open(path, "w", newline="")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        csv_files[name] = f
        csv_writers[name] = writer
    return csv_writers[name]

imu_fields = [
    "timestamp_ns",
    "orientation_x", "orientation_y", "orientation_z", "orientation_w",
    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
    "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
]

odom_fields = [
    "timestamp_ns",
    "pos_x", "pos_y", "pos_z",
    "orient_x", "orient_y", "orient_z", "orient_w",
    "linear_vel_x", "linear_vel_y", "linear_vel_z",
    "angular_vel_x", "angular_vel_y", "angular_vel_z",
]

cmd_fields = [
    "timestamp_ns",
    "position_x", "position_y", "position_z",
    "velocity_x", "velocity_y", "velocity_z",
    "acceleration_x", "acceleration_y", "acceleration_z",
    "jerk_x", "jerk_y", "jerk_z",
    "yaw", "yaw_dot",
]

so3_cmd_fields = [
    "timestamp_ns",
    "force_x", "force_y", "force_z",
    "orientation_x", "orientation_y", "orientation_z", "orientation_w",
    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
    "kx_0", "kx_1", "kx_2",
    "kv_0", "kv_1", "kv_2",
    "kR_0", "kR_1", "kR_2",
    "kOm_0", "kOm_1", "kOm_2",
]

# --- Counters ---
color_count = 0
depth_count = 0

def current_shard_dir():
    shard_id = color_count // SHARD_SIZE
    d = os.path.join(OUTPUT_DIR, f"shard_{shard_id:04d}")
    os.makedirs(d, exist_ok=True)
    return d

def save_compressed_image(msg, timestamp_ns):
    global color_count
    shard_dir = current_shard_dir()
    img_path = os.path.join(shard_dir, f"color_{color_count:06d}.jpg")
    # CompressedImage data is raw JPEG/PNG bytes
    img_bytes = bytes(msg.data)
    with open(img_path, "wb") as f:
        f.write(img_bytes)
    color_count += 1
    return img_path

def save_depth_image(msg, timestamp_ns):
    global depth_count
    shard_dir = current_shard_dir()
    img_path = os.path.join(shard_dir, f"depth_{depth_count:06d}.png")
    # sensor_msgs/Image: decode raw buffer
    h, w = msg.height, msg.width
    encoding = msg.encoding
    raw = bytes(msg.data)
    if encoding == "16UC1":
        arr = np.frombuffer(raw, dtype=np.uint16).reshape(h, w)
    elif encoding == "32FC1":
        arr = np.frombuffer(raw, dtype=np.float32).reshape(h, w)
        arr = (arr * 1000).astype(np.uint16)  # meters to mm
    else:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, -1)
    cv2.imwrite(img_path, arr)
    depth_count += 1
    return img_path

def process_imu(msg, timestamp_ns):
    w = get_csv_writer("imu", imu_fields)
    o = msg.orientation
    av = msg.angular_velocity
    la = msg.linear_acceleration
    w.writerow({
        "timestamp_ns": timestamp_ns,
        "orientation_x": o.x, "orientation_y": o.y, "orientation_z": o.z, "orientation_w": o.w,
        "angular_velocity_x": av.x, "angular_velocity_y": av.y, "angular_velocity_z": av.z,
        "linear_acceleration_x": la.x, "linear_acceleration_y": la.y, "linear_acceleration_z": la.z,
    })

def process_odom(msg, timestamp_ns):
    w = get_csv_writer("odometry", odom_fields)
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

def process_position_cmd(msg, timestamp_ns):
    w = get_csv_writer("position_cmd", cmd_fields)
    p = msg.position
    v = msg.velocity
    a = msg.acceleration
    j = msg.jerk
    w.writerow({
        "timestamp_ns": timestamp_ns,
        "position_x": p.x, "position_y": p.y, "position_z": p.z,
        "velocity_x": v.x, "velocity_y": v.y, "velocity_z": v.z,
        "acceleration_x": a.x, "acceleration_y": a.y, "acceleration_z": a.z,
        "jerk_x": j.x, "jerk_y": j.y, "jerk_z": j.z,
        "yaw": msg.yaw, "yaw_dot": msg.yaw_dot,
    })

def process_so3_cmd(msg, timestamp_ns):
    w = get_csv_writer("so3_cmd", so3_cmd_fields)
    f = msg.force
    o = msg.orientation
    av = msg.angular_velocity
    kr = msg.kr
    kom = msg.kom
    kx = msg.kx
    kv = msg.kv
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

# --- Image index CSV (maps frame number to file path + timestamp) ---
index_fields = ["frame_id", "timestamp_ns", "color_path", "depth_path"]
index_rows = []

# --- Main extraction loop ---
print(f"Extracting from {MCAP_PATH} ...")

# We need to pair color and depth by timestamp. Depth is raw Image, color is compressed.
# They may not be perfectly synced, so we save them independently and record in index.

with open(MCAP_PATH, "rb") as f:
    reader = make_reader(f, decoder_factories=[DecoderFactory()])
    for schema, channel, message, ros_msg in reader.iter_decoded_messages():
        ts = message.log_time
        topic = channel.topic

        if topic == COLOR_TOPIC:
            path = save_compressed_image(ros_msg, ts)
            index_rows.append({
                "frame_id": color_count - 1,
                "timestamp_ns": ts,
                "color_path": path,
                "depth_path": "",
            })
            if color_count % 1000 == 0:
                print(f"  Color frames: {color_count}")

        elif topic == DEPTH_TOPIC:
            path = save_depth_image(ros_msg, ts)
            # Update the closest index row if timestamps are close
            if index_rows and index_rows[-1]["depth_path"] == "":
                index_rows[-1]["depth_path"] = path
            if depth_count % 1000 == 0:
                print(f"  Depth frames: {depth_count}")

        elif topic == IMU_TOPIC:
            process_imu(ros_msg, ts)

        elif topic == ODOM_TOPIC:
            process_odom(ros_msg, ts)

        elif topic == CMD_TOPIC:
            process_position_cmd(ros_msg, ts)

        elif topic == SO3_CMD_TOPIC:
            process_so3_cmd(ros_msg, ts)

# Write image index
index_path = os.path.join(OUTPUT_DIR, "image_index.csv")
with open(index_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=index_fields)
    w.writeheader()
    w.writerows(index_rows)

# Close CSV files
for fh in csv_files.values():
    fh.close()

print(f"\nDone!")
print(f"  Color frames: {color_count}")
print(f"  Depth frames: {depth_count}")
print(f"  Image shards: {color_count // SHARD_SIZE + (1 if color_count % SHARD_SIZE else 0)}")
print(f"  CSVs: {list(csv_writers.keys())}")
print(f"  Output: {OUTPUT_DIR}/")
