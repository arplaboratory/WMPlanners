"""Convert the race16-ros2-bag HuggingFace dataset into .npz episodes.

Downloads the dataset, time-synchronises images + CSVs, and slices the
continuous recording into fixed-length episodes the world-model pipeline
can consume.

Usage:
    python scripts/world_model/preprocess_race16.py \
        --out_dir data/race16 \
        --episode_len 64 \
        --stride 32 \
        --img_size 128
"""
from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from scripts.world_model.sync_utils import nearest_merge


# ── helpers ──────────────────────────────────────────────────────────────

def _load_image(path: str, size: tuple[int, int]) -> np.ndarray:
    """Load an image, resize, return as uint8 (RGB) or float32 (depth)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        # depth: 16-bit PNG → float metres (assume mm encoding)
        return img.astype(np.float32) / 1000.0
    # colour: BGR → RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess race16-ros2-bag → .npz episodes")
    ap.add_argument("--repo_id", default="dmody1/race16-ros2-bag")
    ap.add_argument("--cache_dir", default=None, help="HF cache directory (default: HF default)")
    ap.add_argument("--out_dir", required=True, help="Where to write .npz episodes")
    ap.add_argument("--episode_len", type=int, default=64,
                    help="Timesteps per episode (at image rate ~30 fps)")
    ap.add_argument("--stride", type=int, default=32,
                    help="Stride between episode start indices (overlapping if < episode_len)")
    ap.add_argument("--img_size", type=int, default=128,
                    help="Resize images to (img_size, img_size)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img_sz = (args.img_size, args.img_size)

    # ── 1. download dataset ─────────────────────────────────────────────
    print("Downloading dataset from HuggingFace …")
    snap = snapshot_download(repo_id=args.repo_id, repo_type="dataset",
                             cache_dir=args.cache_dir)
    data_root = Path(snap) / "data"
    print(f"Dataset cached at {snap}")

    # ── 2. extract image shards ─────────────────────────────────────────
    images_dir = data_root / "images"
    extract_root = Path(args.out_dir) / "_extracted_images"
    if not extract_root.exists():
        extract_root.mkdir(parents=True, exist_ok=True)
        shards = sorted(images_dir.glob("shard_*.tar.gz"))
        print(f"Extracting {len(shards)} image shards …")
        for shard in shards:
            with tarfile.open(shard, "r:gz") as tar:
                tar.extractall(extract_root)
        print("Extraction done.")
    else:
        print("Images already extracted, skipping.")

    # ── 3. load CSVs ────────────────────────────────────────────────────
    img_index = pd.read_csv(data_root / "image_index.csv", skipinitialspace=True)
    imu_df = pd.read_csv(data_root / "imu.csv")
    odom_df = pd.read_csv(data_root / "odometry.csv")
    pos_cmd_df = pd.read_csv(data_root / "position_cmd.csv")

    # Sort all by timestamp
    img_index = img_index.sort_values("timestamp_ns").reset_index(drop=True)
    imu_df = imu_df.sort_values("timestamp_ns").reset_index(drop=True)
    odom_df = odom_df.sort_values("timestamp_ns").reset_index(drop=True)
    pos_cmd_df = pos_cmd_df.sort_values("timestamp_ns").reset_index(drop=True)

    image_ts = img_index["timestamp_ns"].values
    N_frames = len(image_ts)
    print(f"Total image frames: {N_frames}")

    # ── 4. time-synchronise CSVs to image timestamps ────────────────────
    print("Time-synchronising sensors to image timestamps …")
    imu_sync = nearest_merge(image_ts, imu_df)       # (N, 10)
    odom_sync = nearest_merge(image_ts, odom_df)      # (N, 13)
    action_sync = nearest_merge(image_ts, pos_cmd_df) # (N, 14)

    print(f"  IMU dims:    {imu_sync.shape[1]}")
    print(f"  Odom dims:   {odom_sync.shape[1]}")
    print(f"  Action dims: {action_sync.shape[1]}")

    # ── 5. resolve image paths ──────────────────────────────────────────
    # image_index.csv paths look like "dataset_out/shard_XXXX/color_XXXXXX.jpg"
    # After extraction they land under extract_root with the same relative structure.
    def _resolve(rel_path: str) -> str:
        # Try the path as-is under extract_root
        p = extract_root / rel_path
        if p.exists():
            return str(p)
        # Try just the filename under a flat search
        name = Path(rel_path).name
        shard = Path(rel_path).parent.name
        p2 = extract_root / shard / name
        if p2.exists():
            return str(p2)
        raise FileNotFoundError(f"Cannot find image: {rel_path} (tried {p} and {p2})")

    color_paths = img_index["color_path"].values
    depth_paths = img_index["depth_path"].values

    # ── 6. slice into episodes and save .npz ────────────────────────────
    ep_len = args.episode_len
    ep_count = 0

    starts = range(0, N_frames - ep_len, args.stride)
    print(f"Generating {len(list(starts))} episodes (len={ep_len}, stride={args.stride}) …")

    for start in starts:
        sl = slice(start, start + ep_len)

        # Load and resize images for this episode
        rgb = np.zeros((ep_len, args.img_size, args.img_size, 3), dtype=np.uint8)
        depth = np.zeros((ep_len, args.img_size, args.img_size), dtype=np.float32)

        ok = True
        for t in range(ep_len):
            frame_idx = start + t
            try:
                rgb[t] = _load_image(_resolve(color_paths[frame_idx]), img_sz)
                depth[t] = _load_image(_resolve(depth_paths[frame_idx]), img_sz)
            except FileNotFoundError as e:
                print(f"  Skipping episode at frame {start}: {e}")
                ok = False
                break
        if not ok:
            continue

        out_path = os.path.join(args.out_dir, f"ep_{ep_count:05d}.npz")
        np.savez_compressed(
            out_path,
            rgb=rgb,                    # (T, H, W, 3) uint8
            depth=depth,                # (T, H, W) float32
            imu=imu_sync[sl],           # (T, D_imu) float32
            odom=odom_sync[sl],         # (T, D_odom) float32
            action=action_sync[sl],     # (T, D_act) float32
        )
        ep_count += 1

    print(f"Wrote {ep_count} episodes to {args.out_dir}")
    print(f"Shapes: rgb=({ep_len},{args.img_size},{args.img_size},3) "
          f"imu=({ep_len},{imu_sync.shape[1]}) "
          f"odom=({ep_len},{odom_sync.shape[1]}) "
          f"action=({ep_len},{action_sync.shape[1]})")


if __name__ == "__main__":
    main()
