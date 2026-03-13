"""Stage 3b -- Export DINOv2 patch tokens for selected transitions."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from .utils import load_csv, resolve_path

DINO_MODEL = "dinov2_vits14"
SHARD_SIZE = 10_000
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 16  # smaller than Stage 2 because patch tokens use more memory


def _select_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _extract_patch_tokens(
    image_paths: list[str],
    model: torch.nn.Module,
    transform: T.Compose,
    device: torch.device,
) -> np.ndarray:
    """Extract DINOv2 patch tokens for a batch of images.

    Returns (B, n_patches, 384) float16 array.
    """
    n = len(image_paths)
    all_patches = []

    for batch_start in range(0, n, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n)
        batch_images = []
        for idx in range(batch_start, batch_end):
            try:
                img = Image.open(image_paths[idx]).convert("RGB")
                batch_images.append(transform(img))
            except Exception:
                batch_images.append(torch.zeros(3, 224, 224))

        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            # Get intermediate features (patch tokens without CLS)
            features = model.forward_features(batch_tensor)
            patch_tokens = features["x_norm_patchtokens"]  # (B, n_patches, 384)

        all_patches.append(patch_tokens.cpu().numpy().astype(np.float16))

    return np.concatenate(all_patches, axis=0) if all_patches else np.zeros((0, 256, 384), dtype=np.float16)


def run_stage3b(
    selected_csv_path: str,
    bags_root: str,
    output_dir: str,
    device: str | None = None,
) -> str:
    """Export DINOv2 patch tokens for selected transitions.

    Parameters
    ----------
    selected_csv_path : str
        Path to selected_transitions.csv from Stage 3.
    bags_root : str
        Root directory for bag datasets.
    output_dir : str
        Where to write output.
    device : str | None
        Torch device.

    Returns
    -------
    Path to the selected_patches directory.
    """
    dev = _select_device(device)
    selected_df = pd.read_csv(selected_csv_path)
    n = len(selected_df)
    print(f"[stage3b] {n} selected transitions")

    if n == 0:
        print("[stage3b] Nothing to export.")
        return output_dir

    # Resolve image paths
    print("[stage3b] Resolving image paths ...")
    image_index_cache = {}
    paths_t = []
    paths_t1 = []

    for _, row in selected_df.iterrows():
        bag_name = row["bag_name"]
        if bag_name not in image_index_cache:
            idx_path = os.path.join(bags_root, bag_name, "image_index.csv")
            idx_df = load_csv(idx_path)
            image_index_cache[bag_name] = idx_df.set_index("frame_id")

        idx_df = image_index_cache[bag_name]
        bag_dir = os.path.join(bags_root, bag_name)

        color_t = resolve_path(str(idx_df.loc[row["frame_t_id"], "color_path"]), bag_dir)
        color_t1 = resolve_path(str(idx_df.loc[row["frame_t1_id"], "color_path"]), bag_dir)
        paths_t.append(color_t)
        paths_t1.append(color_t1)

    # Load DINOv2
    print(f"[stage3b] Loading DINOv2 ({DINO_MODEL}) on {dev} ...")
    model = torch.hub.load("facebookresearch/dinov2", DINO_MODEL)
    model = model.to(dev)
    model.eval()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Extract patches and save as shards
    patches_dir = os.path.join(output_dir, "selected_patches")
    os.makedirs(patches_dir, exist_ok=True)

    shard_idx = 0
    for shard_start in range(0, n, SHARD_SIZE):
        shard_end = min(shard_start + SHARD_SIZE, n)
        shard_paths_t = paths_t[shard_start:shard_end]
        shard_paths_t1 = paths_t1[shard_start:shard_end]

        print(f"[stage3b] Extracting patches for shard {shard_idx} "
              f"({shard_start}-{shard_end}/{n}) ...")

        patches_t = _extract_patch_tokens(shard_paths_t, model, transform, dev)
        patches_t1 = _extract_patch_tokens(shard_paths_t1, model, transform, dev)

        shard_df = selected_df.iloc[shard_start:shard_end]
        odom_deltas = shard_df[["odom_dx", "odom_dy", "odom_dz",
                                 "odom_ax", "odom_ay", "odom_az"]].values.astype(np.float32)

        shard_path = os.path.join(patches_dir, f"shard_{shard_idx:04d}.npz")
        np.savez_compressed(
            shard_path,
            patches_t=patches_t,
            patches_t1=patches_t1,
            odom_deltas=odom_deltas,
            frame_t_ids=shard_df["frame_t_id"].values,
            frame_t1_ids=shard_df["frame_t1_id"].values,
            timestamps_t=shard_df["timestamp_t"].values,
            timestamps_t1=shard_df["timestamp_t1"].values,
            bag_names=shard_df["bag_name"].values,
        )
        shard_idx += 1

    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    print(f"[stage3b] Saved {shard_idx} shard(s) to {patches_dir}")
    return patches_dir
