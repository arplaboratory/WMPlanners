"""Stage 3 -- Upload selected NPZ episodes to Hugging Face Hub."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Generator

import numpy as np
from PIL import Image as PILImage


def _iter_episode_frames(
    episodes_dir: str,
    odom_dim: int | None = None,
    action_dim: int | None = None,
    imu_dim: int | None = None,
) -> Generator[dict, None, None]:
    """Yield one dict per frame across all selected episodes.

    Each NPZ episode contains: rgb (T,H,W,3), depth (T,H,W),
    imu (T,D), odom (T,D), action (T,D).
    """
    ep_files = sorted(Path(episodes_dir).glob("ep_*.npz"))

    for ep_path in ep_files:
        ep_id = int(ep_path.stem.split("_")[1])
        data = np.load(ep_path)
        rgb = data["rgb"]         # (T, H, W, 3) uint8
        odom = data["odom"]       # (T, D_odom) float32
        action = data["action"]   # (T, D_act) float32
        imu = data["imu"]         # (T, D_imu) float32
        depth = data.get("depth")  # (T, H, W) float32 or None
        n_frames = len(rgb)

        for t in range(n_frames):
            image = PILImage.fromarray(rgb[t])

            row = {
                "episode_id": ep_id,
                "frame_idx": t,
                "image": image,
            }

            # Flatten odom, action, imu into named columns.  Pad shorter legacy
            # episodes so one HF schema can cover the whole selected set.
            for j in range(odom_dim or odom.shape[1]):
                row[f"odom_{j}"] = float(odom[t, j]) if j < odom.shape[1] else 0.0
            for j in range(action_dim or action.shape[1]):
                row[f"action_{j}"] = float(action[t, j]) if j < action.shape[1] else 0.0
            for j in range(imu_dim or imu.shape[1]):
                row[f"imu_{j}"] = float(imu[t, j]) if j < imu.shape[1] else 0.0

            if depth is not None:
                row["depth_mean"] = float(depth[t].mean())

            yield row


def run_hf_upload(
    episodes_dir: str,
    repo_id: str | None = None,
    private: bool = True,
) -> str:
    """Create HF dataset from selected NPZ episodes and optionally push to Hub.

    Parameters
    ----------
    episodes_dir : str
        Directory containing ep_*.npz files (output of select_episodes).
    repo_id : str | None
        If provided, push the dataset to this Hugging Face Hub repository.
    private : bool
        Whether the Hub repository should be private (default True).

    Returns
    -------
    str
        Path to the saved dataset directory.
    """
    from datasets import Dataset, Features, Image, Value

    save_dir = os.path.join(episodes_dir, "hf_dataset")

    ep_files = sorted(Path(episodes_dir).glob("ep_*.npz"))
    if not ep_files:
        print("[hf_upload] No episodes found — nothing to upload.")
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    # Determine a single schema that can cover all selected episodes.
    odom_dim = 0
    action_dim = 0
    imu_dim = 0
    has_depth = False
    for ep_file in ep_files:
        probe = np.load(ep_file)
        odom_dim = max(odom_dim, probe["odom"].shape[1])
        action_dim = max(action_dim, probe["action"].shape[1])
        imu_dim = max(imu_dim, probe["imu"].shape[1])
        has_depth = has_depth or "depth" in probe

    # Build features schema
    feature_dict = {
        "episode_id": Value("int64"),
        "frame_idx": Value("int64"),
        "image": Image(),
    }
    for j in range(odom_dim):
        feature_dict[f"odom_{j}"] = Value("float32")
    for j in range(action_dim):
        feature_dict[f"action_{j}"] = Value("float32")
    for j in range(imu_dim):
        feature_dict[f"imu_{j}"] = Value("float32")
    if has_depth:
        feature_dict["depth_mean"] = Value("float32")

    features = Features(feature_dict)

    ds = Dataset.from_generator(
        _iter_episode_frames,
        gen_kwargs={
            "episodes_dir": episodes_dir,
            "odom_dim": odom_dim,
            "action_dim": action_dim,
            "imu_dim": imu_dim,
        },
        features=features,
        cache_dir=os.path.join(save_dir, ".cache"),
    )

    ds.save_to_disk(save_dir)
    n_episodes = len(ep_files)
    print(f"[hf_upload] Saved {len(ds)} frames ({n_episodes} episodes) to {save_dir}")

    if repo_id:
        repo_id = repo_id.strip("'\"")
        ds.push_to_hub(repo_id, private=private)
        print(f"[hf_upload] Pushed to https://huggingface.co/datasets/{repo_id}")

    return save_dir


def main() -> None:
    """Standalone CLI for creating / uploading the HF dataset."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a Hugging Face dataset from selected NPZ episodes",
    )
    parser.add_argument(
        "--episodes-dir", required=True,
        help="Directory containing ep_*.npz files",
    )
    parser.add_argument(
        "--push-to-hub", default=None,
        help="HF Hub repo ID (e.g. dmody1/wm-data). Omit to save locally only.",
    )
    parser.add_argument(
        "--private", action="store_true", default=False,
        help="Make the Hub repo private",
    )
    args = parser.parse_args()

    save_dir = run_hf_upload(
        episodes_dir=args.episodes_dir,
        repo_id=args.push_to_hub,
        private=args.private,
    )
    print(f"[hf_upload] Dataset saved to {save_dir}")


if __name__ == "__main__":
    main()
