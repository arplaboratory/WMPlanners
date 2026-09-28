"""
Build a "collect1"-style dataset (image + ego position/odom + action + imu + depth_mean)
from nuScenes, matching this schema:

    episode_id   int    - scene index
    frame_idx    int    - frame index within the scene (per selected camera)
    image        Image  - camera frame
    odom_0..12   float  - [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
    action_0..12 float  - odom of the NEXT frame (same 13-dim layout), used as the
                          supervised "action" target. Last frame in an episode repeats
                          its own odom (no next step to predict).
    imu_0..9     float  - [ax, ay, az, gx, gy, gz, qw, qx, qy, qz] from the CAN bus
                          `ms_imu` message nearest in time to the frame (zeros + a
                          `has_imu` flag if the scene has no CAN bus data).
    depth_mean   float  - mean depth (meters) of LIDAR points projected into the image

Requirements
------------
    pip install nuscenes-devkit datasets pillow pyarrow pyquaternion --break-system-packages

Data
----
Download nuScenes (e.g. the `v1.0-mini` split to try this out, or a full trainval split)
from https://www.nuscenes.org/download and unzip so you have a `dataroot` directory
containing `maps/`, `samples/`, `sweeps/`, `v1.0-mini/` (or `v1.0-trainval/`), etc.

For IMU, also grab the "CAN bus expansion" from the same page and unzip it as
`<dataroot>/can_bus/`.

Usage
-----
    python build_nuscenes.py \
        --dataroot /path/to/nuscenes \
        --version v1.0-mini \
        --camera CAM_FRONT \
        --out ./nuscenes_collect_parquet
"""

import argparse
import io
import os

import numpy as np
from PIL import Image as PILImage
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

try:
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    HAVE_CAN_BUS = True
except Exception:
    HAVE_CAN_BUS = False


# ---------------------------------------------------------------------------
# Ego pose -> odom vector
# ---------------------------------------------------------------------------

def ego_pose_vec(ego_pose):
    """[x, y, z, qx, qy, qz, qw] from a nuScenes ego_pose record."""
    x, y, z = ego_pose["translation"]
    qw, qx, qy, qz = ego_pose["rotation"]  # nuScenes stores quaternions as (w, x, y, z)
    return np.array([x, y, z, qx, qy, qz, qw], dtype=np.float32)


def finite_diff_velocity(pose_a, pose_b, dt):
    """Linear + angular velocity between two ego poses, expressed in the global frame.

    Angular velocity is derived from the relative rotation between the two
    orientations divided by dt (small-angle / axis-angle approximation).
    """
    if dt <= 0:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    pa = np.array(pose_a["translation"])
    pb = np.array(pose_b["translation"])
    lin_vel = (pb - pa) / dt

    qa = Quaternion(pose_a["rotation"])
    qb = Quaternion(pose_b["rotation"])
    q_rel = qb * qa.inverse
    angle = q_rel.angle  # radians, in [0, pi]
    axis = q_rel.axis if angle > 1e-9 else np.array([0.0, 0.0, 1.0])
    ang_vel = axis * (angle / dt)

    return lin_vel.astype(np.float32), ang_vel.astype(np.float32)


def full_odom(ego_pose, lin_vel, ang_vel):
    pose7 = ego_pose_vec(ego_pose)
    return np.concatenate([pose7, lin_vel, ang_vel]).astype(np.float32)  # 13-dim


# ---------------------------------------------------------------------------
# IMU (CAN bus) lookup
# ---------------------------------------------------------------------------

class ImuLookup:
    """Nearest-timestamp IMU sample per scene, from the CAN bus `ms_imu` message."""

    def __init__(self, dataroot):
        self.ok = HAVE_CAN_BUS
        self.cache = {}
        if self.ok:
            try:
                self.can_bus = NuScenesCanBus(dataroot=dataroot)
            except Exception:
                self.ok = False

    def get(self, scene_name, timestamp_us):
        if not self.ok:
            return np.zeros(10, dtype=np.float32), False

        if scene_name not in self.cache:
            try:
                msgs = self.can_bus.get_messages(scene_name, "ms_imu")
            except Exception:
                msgs = []
            self.cache[scene_name] = msgs

        msgs = self.cache[scene_name]
        if not msgs:
            return np.zeros(10, dtype=np.float32), False

        # msgs are dicts with 'utime', 'linear_accel', 'rotation_rate', 'q'
        idx = int(np.argmin([abs(m["utime"] - timestamp_us) for m in msgs]))
        m = msgs[idx]
        ax, ay, az = m["linear_accel"]
        gx, gy, gz = m["rotation_rate"]
        qw, qx, qy, qz = m["q"]  # (w, x, y, z)
        vec = np.array([ax, ay, az, gx, gy, gz, qw, qx, qy, qz], dtype=np.float32)
        return vec, True


# ---------------------------------------------------------------------------
# Depth from projected LIDAR
# ---------------------------------------------------------------------------

def mean_depth_for_camera(nusc, cam_sd_token):
    cam_sd = nusc.get("sample_data", cam_sd_token)
    sample = nusc.get("sample", cam_sd["sample_token"])
    lidar_token = sample["data"].get("LIDAR_TOP")
    if lidar_token is None:
        return 0.0

    lidar_sd = nusc.get("sample_data", lidar_token)
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_sd["filename"]))

    # Transform lidar points: lidar -> ego (at lidar time) -> global -> ego (at cam time) -> cam
    cs_lidar = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs_lidar["rotation"]).rotation_matrix)
    pc.translate(np.array(cs_lidar["translation"]))

    ego_lidar = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    pc.rotate(Quaternion(ego_lidar["rotation"]).rotation_matrix)
    pc.translate(np.array(ego_lidar["translation"]))

    ego_cam = nusc.get("ego_pose", cam_sd["ego_pose_token"])
    pc.translate(-np.array(ego_cam["translation"]))
    pc.rotate(Quaternion(ego_cam["rotation"]).rotation_matrix.T)

    cs_cam = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    pc.translate(-np.array(cs_cam["translation"]))
    pc.rotate(Quaternion(cs_cam["rotation"]).rotation_matrix.T)

    depths = pc.points[2, :]
    in_front = depths > 0.1
    if not np.any(in_front):
        return 0.0

    cam_intrinsic = np.array(cs_cam["camera_intrinsic"])
    points_2d = view_points(pc.points[:3, in_front], cam_intrinsic, normalize=True)
    img_w, img_h = nusc.get("sample_data", cam_sd_token)["width"], nusc.get("sample_data", cam_sd_token)["height"]

    valid = (
        (points_2d[0, :] > 0) & (points_2d[0, :] < img_w) &
        (points_2d[1, :] > 0) & (points_2d[1, :] < img_h)
    )
    valid_depths = depths[in_front][valid]
    if valid_depths.size == 0:
        return 0.0
    return float(valid_depths.mean())


# ---------------------------------------------------------------------------
# Main build loop
# ---------------------------------------------------------------------------
def scene_has_camera_data(nusc, scene, camera, dataroot):
    """True if every keyframe's `camera` image for this scene is on disk.

    Trainval ships as multiple blob archives; a partially-downloaded dataset
    has some scenes fully present and the rest entirely absent (blobs are
    packed per source log), so a whole-scene check is sufficient.
    """
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        sd = nusc.get("sample_data", sample["data"][camera])
        if not os.path.exists(os.path.join(dataroot, sd["filename"])):
            return False
        sample_token = sample["next"]
    return True


def build(dataroot, version, camera, out_dir, with_depth=True, with_imu=True):
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    imu_lookup = ImuLookup(dataroot) if with_imu else None

    rows = []
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    scenes = [s for s in nusc.scene if scene_has_camera_data(nusc, s, camera, dataroot)]
    n_skipped = len(nusc.scene) - len(scenes)
    if n_skipped:
        print(f"Skipping {n_skipped}/{len(nusc.scene)} scenes missing {camera} data "
              f"(not yet downloaded from this blob)")

    for episode_id, scene in enumerate(scenes):
        scene_name = scene["name"]

        # Walk the camera's sample_data chain for this scene, keyframes only.
        sample_token = scene["first_sample_token"]
        cam_sd_tokens = []
        while sample_token:
            sample = nusc.get("sample", sample_token)
            cam_sd_tokens.append(sample["data"][camera])
            sample_token = sample["next"]

        ego_poses = [
            nusc.get("ego_pose", nusc.get("sample_data", t)["ego_pose_token"])
            for t in cam_sd_tokens
        ]
        timestamps = [nusc.get("sample_data", t)["timestamp"] for t in cam_sd_tokens]

        odoms = []
        for i, sd_token in enumerate(cam_sd_tokens):
            if i < len(cam_sd_tokens) - 1:
                dt = (timestamps[i + 1] - timestamps[i]) / 1e6
                lin_vel, ang_vel = finite_diff_velocity(ego_poses[i], ego_poses[i + 1], dt)
            else:
                lin_vel, ang_vel = (odoms[-1][7:10], odoms[-1][10:13]) if odoms else (
                    np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
                )
            odoms.append(full_odom(ego_poses[i], lin_vel, ang_vel))

        for frame_idx, sd_token in enumerate(cam_sd_tokens):
            sd = nusc.get("sample_data", sd_token)
            img_path = os.path.join(dataroot, sd["filename"])

            odom = odoms[frame_idx]
            action = odoms[frame_idx + 1] if frame_idx + 1 < len(odoms) else odoms[frame_idx]

            if with_imu:
                imu_vec, has_imu = imu_lookup.get(scene_name, timestamps[frame_idx])
            else:
                imu_vec, has_imu = np.zeros(10, dtype=np.float32), False

            depth_mean = mean_depth_for_camera(nusc, sd_token) if with_depth else 0.0

            out_img_name = f"{episode_id:04d}_{frame_idx:04d}.jpg"
            with PILImage.open(img_path) as im:
                im.convert("RGB").save(os.path.join(images_dir, out_img_name), quality=90)

            row = {
                "episode_id": episode_id,
                "frame_idx": frame_idx,
                "image": out_img_name,
                "has_imu": has_imu,
            }
            for i in range(13):
                row[f"odom_{i}"] = float(odom[i])
                row[f"action_{i}"] = float(action[i])
            for i in range(10):
                row[f"imu_{i}"] = float(imu_vec[i])
            row["depth_mean"] = depth_mean

            rows.append(row)

        print(f"scene {episode_id} ({scene_name}): {len(cam_sd_tokens)} frames")

    # Write out as a HF datasets-compatible parquet with an Image feature.
    import pandas as pd
    from datasets import Dataset, Image as HFImage

    df = pd.DataFrame(rows)
    df["image"] = df["image"].apply(lambda name: os.path.join(images_dir, name))
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds = ds.cast_column("image", HFImage())

    ds.to_parquet(os.path.join(out_dir, "train.parquet"))
    print(f"Wrote {len(ds)} rows to {os.path.join(out_dir, 'train.parquet')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", required=True, help="Path to unzipped nuScenes data")
    p.add_argument("--version", default="v1.0-mini", help="e.g. v1.0-mini, v1.0-trainval")
    p.add_argument("--camera", default="CAM_FRONT", help="nuScenes camera channel")
    p.add_argument("--out", default="./nuscenes_collect_parquet")
    p.add_argument("--with-depth", action="store_true",
                    help="Compute LIDAR depth_mean (slower). Default: zero-filled.")
    p.add_argument("--no-imu", action="store_true", help="Skip CAN bus IMU lookup")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    build(
        dataroot=args.dataroot,
        version=args.version,
        camera=args.camera,
        out_dir=args.out,
        with_depth=args.with_depth,
        with_imu=not args.no_imu,
    )