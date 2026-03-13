"""Stage 2 -- DINOv2 CLS tokens + body-frame odom deltas -> transition descriptors."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from .odom_delta import compute_odom_deltas_for_frames
from .utils import load_csv, resolve_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DINO_MODEL = "dinov2_vits14"
VISUAL_DIM = 384
ODOM_DELTA_DIM = 6
DESCRIPTOR_DIM = VISUAL_DIM + ODOM_DELTA_DIM  # 390
BATCH_SIZE = 32
CHECKPOINT_INTERVAL = 1000

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _select_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# DINOv2 CLS token extraction
# ---------------------------------------------------------------------------

def _build_dino_transform() -> T.Compose:
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _load_dino_model(device: torch.device) -> torch.nn.Module:
    print(f"[stage2] Loading DINOv2 ({DINO_MODEL}) on {device} ...")
    model = torch.hub.load("facebookresearch/dinov2", DINO_MODEL)
    model = model.to(device)
    model.eval()
    return model


def _compute_cls_tokens(
    color_paths: list[str],
    model: torch.nn.Module,
    transform: T.Compose,
    device: torch.device,
    checkpoint_path: str | None = None,
) -> np.ndarray:
    """Compute DINOv2 CLS-token embeddings (384-dim) for a list of images.

    Supports checkpointing for resumption.
    """
    n = len(color_paths)
    cls_tokens = np.zeros((n, VISUAL_DIM), dtype=np.float32)
    start_idx = 0

    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = np.load(checkpoint_path)
        done = int(ckpt["done"])
        cls_tokens[:done] = ckpt["embeddings"][:done]
        start_idx = done
        print(f"[stage2] Resuming CLS tokens from frame {start_idx}/{n}")

    for batch_start in range(start_idx, n, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n)
        batch_images = []
        for idx in range(batch_start, batch_end):
            try:
                img = Image.open(color_paths[idx]).convert("RGB")
                batch_images.append(transform(img))
            except Exception:
                batch_images.append(torch.zeros(3, 224, 224))

        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            feats = model(batch_tensor)
        cls_tokens[batch_start:batch_end] = feats.cpu().numpy()

        if checkpoint_path and (batch_end % CHECKPOINT_INTERVAL < BATCH_SIZE or batch_end == n):
            np.savez_compressed(checkpoint_path, embeddings=cls_tokens, done=np.array(batch_end))

        if (batch_end - start_idx) % (BATCH_SIZE * 10) == 0 or batch_end == n:
            print(f"[stage2]   CLS tokens: {batch_end}/{n}")

    return cls_tokens


# ---------------------------------------------------------------------------
# Transition descriptor construction
# ---------------------------------------------------------------------------

def build_transition_descriptors(
    cls_tokens: np.ndarray,
    odom_deltas: np.ndarray,
    valid_transition: np.ndarray,
) -> np.ndarray:
    """Build 390-dim transition descriptors from CLS tokens and odom deltas.

    Parameters
    ----------
    cls_tokens : (N, 384) float32
    odom_deltas : (N-1, 6) float64
    valid_transition : (N-1,) bool -- True where transition is valid

    Returns
    -------
    (N-1, 390) float32 transition descriptors (zeroed for invalid transitions)
    """
    n = len(cls_tokens)
    assert len(odom_deltas) == n - 1
    assert len(valid_transition) == n - 1

    descriptors = np.zeros((n - 1, DESCRIPTOR_DIM), dtype=np.float32)

    valid_indices = np.where(valid_transition)[0]
    descriptors[valid_indices, :VISUAL_DIM] = cls_tokens[valid_indices]
    descriptors[valid_indices, VISUAL_DIM:] = odom_deltas[valid_indices].astype(np.float32)

    return descriptors


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_stage2(
    dataset_dir: str,
    metrics_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    output_dir: str,
    base_dir: str | None = None,
    device: str | None = None,
) -> dict | None:
    """Run Stage 2: compute CLS tokens + odom deltas for transition descriptors.

    Parameters
    ----------
    dataset_dir : str
        Directory containing odometry.csv.
    metrics_df : pd.DataFrame
        Per-frame metrics with frame_id, timestamp_ns, color_path, depth_path, valid_transition.
    segments_df : pd.DataFrame
        Segment info with start_idx, end_idx, keep columns.
    output_dir : str
        Where to write the output .npz file.
    base_dir : str | None
        Root for resolving relative image paths.
    device : str | None
        Torch device string.

    Returns
    -------
    dict with keys: cls_tokens, odom_deltas, descriptors, etc. Or None if no kept frames.
    """
    if base_dir is None:
        base_dir = os.getcwd()

    dev = _select_device(device)
    os.makedirs(output_dir, exist_ok=True)
    bag_name = os.path.basename(os.path.normpath(dataset_dir))

    # Identify kept frames
    kept_segments = segments_df[segments_df["keep"] == True]  # noqa: E712
    if kept_segments.empty:
        print("[stage2] No kept segments -- skipping.")
        return None

    kept_mask = np.zeros(len(metrics_df), dtype=bool)
    for _, seg in kept_segments.iterrows():
        s, e = int(seg["start_idx"]), int(seg["end_idx"])
        kept_mask[s:e] = True

    kept_df = metrics_df[kept_mask].reset_index(drop=True)
    n_frames = len(kept_df)
    print(f"[stage2] {n_frames} kept frames from {len(kept_segments)} segments.")

    if n_frames == 0:
        return None

    frame_ids = kept_df["frame_id"].values
    timestamps = kept_df["timestamp_ns"].values
    color_paths = [resolve_path(p, base_dir) for p in kept_df["color_path"].values]
    depth_paths = [resolve_path(p, base_dir) for p in kept_df["depth_path"].values]

    # Valid transition mask for kept frames
    if "valid_transition" in kept_df.columns:
        valid_transition = kept_df["valid_transition"].values[:-1].astype(bool)
    else:
        valid_transition = np.ones(n_frames - 1, dtype=bool)

    # CLS tokens
    checkpoint_path = os.path.join(output_dir, f"{bag_name}_checkpoint.npz")
    model = _load_dino_model(dev)
    transform = _build_dino_transform()
    cls_tokens = _compute_cls_tokens(color_paths, model, transform, dev, checkpoint_path)
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # Body-frame odom deltas
    print("[stage2] Computing body-frame odom deltas ...")
    odom_df = load_csv(os.path.join(dataset_dir, "odometry.csv"))
    odom_deltas = compute_odom_deltas_for_frames(odom_df, timestamps)

    # Zero out cross-segment deltas (invalid transitions produce garbage values)
    odom_deltas[~valid_transition] = 0.0

    # Build transition descriptors
    print("[stage2] Building transition descriptors ...")
    descriptors = build_transition_descriptors(cls_tokens, odom_deltas, valid_transition)

    # Save output
    output_path = os.path.join(output_dir, f"{bag_name}_transitions_compact.npz")
    np.savez_compressed(
        output_path,
        descriptors=descriptors,                # (N-1, 390) float32
        odom_deltas=odom_deltas,                # (N-1, 6) float64
        cls_tokens=cls_tokens,                  # (N, 384) float32
        frame_t_ids=frame_ids[:-1],             # (N-1,) int
        frame_t1_ids=frame_ids[1:],             # (N-1,) int
        timestamps_t=timestamps[:-1],           # (N-1,) int64
        timestamps_t1=timestamps[1:],           # (N-1,) int64
        valid_mask=valid_transition,            # (N-1,) bool
        bag_name=np.array(bag_name),            # scalar string
        color_paths=np.array(color_paths),      # (N,)
        depth_paths=np.array(depth_paths),      # (N,)
    )

    print(f"[stage2] Saved {output_path}")
    print(f"[stage2]   CLS tokens: {cls_tokens.shape}")
    print(f"[stage2]   Odom deltas: {odom_deltas.shape}")
    print(f"[stage2]   Descriptors: {descriptors.shape}")

    # Clean up checkpoint
    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)

    return {
        "cls_tokens": cls_tokens,
        "odom_deltas": odom_deltas,
        "descriptors": descriptors,
        "frame_ids": frame_ids,
        "timestamps": timestamps,
        "valid_mask": valid_transition,
    }
