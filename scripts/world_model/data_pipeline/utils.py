"""Shared utilities for the data-selection pipeline."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DINOv2 constants
# ---------------------------------------------------------------------------
DINO_MODEL = "dinov2_vits14"
VISUAL_DIM = 384
ODOM_DELTA_DIM = 6
DESCRIPTOR_DIM = VISUAL_DIM + ODOM_DELTA_DIM  # 390

# TD-JEPA rich embedding dimensions
DEPTH_DIM = 12
PROPRIO_DIM = 20     # 13 odom + 7 SO3 cmd
ACTION_DIM = 10      # 8 position cmd + 2 derived force
FRAME_EMBEDDING_DIM = VISUAL_DIM + DEPTH_DIM + PROPRIO_DIM  # 416

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Depth feature constants
# ---------------------------------------------------------------------------
DEPTH_HIST_BINS = 8
DEPTH_HIST_RANGE = (0.0, 10.0)  # metres
DEPTH_NEAR_THRESH = 1.0  # metres
DEPTH_FAR_THRESH = 5.0   # metres

# ---------------------------------------------------------------------------
# Telemetry column names
# ---------------------------------------------------------------------------
ODOM_COLS = [
    "pos_x", "pos_y", "pos_z",
    "orient_x", "orient_y", "orient_z", "orient_w",
    "linear_vel_x", "linear_vel_y", "linear_vel_z",
    "angular_vel_x", "angular_vel_y", "angular_vel_z",
]
SO3_PROPRIO_COLS = [
    "force_x", "force_y", "force_z",
    "orientation_x", "orientation_y", "orientation_z", "orientation_w",
]
PCMD_ACTION_COLS = [
    "velocity_x", "velocity_y", "velocity_z",
    "acceleration_x", "acceleration_y", "acceleration_z",
    "yaw", "yaw_dot",
]
SO3_ACTION_COLS = ["force_x", "force_y", "force_z"]


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device(device: str | None) -> torch.device:
    """Pick the best available torch device."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# DINOv2 helpers
# ---------------------------------------------------------------------------

def build_dino_transform() -> T.Compose:
    """Standard DINOv2 preprocessing: resize 224, normalize with ImageNet stats."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_dino_model(device: torch.device) -> torch.nn.Module:
    """Load DINOv2 ViT-S/14 from local external/dino submodule."""
    print(f"[dino] Loading DINOv2 ({DINO_MODEL}) on {device} (local) ...")
    # Determine the path to the hubconf.py in the submodule
    repo_dir = os.path.join(os.getcwd(), "external/dino")
    model = torch.hub.load(repo_dir, DINO_MODEL, source="local")
    model = model.to(device)
    model.eval()
    return model


def cleanup_model(model: torch.nn.Module, device: torch.device) -> None:
    """Delete a model and free GPU memory if applicable."""
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_image_batch(
    paths: list[str],
    transform: T.Compose,
) -> torch.Tensor:
    """Load images from *paths*, apply *transform*, return a stacked tensor.

    Corrupt/missing images are replaced with zero tensors and a warning is logged.
    """
    tensors = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(transform(img))
        except Exception:
            logger.warning("Failed to load image: %s — substituting zeros", p)
            tensors.append(torch.zeros(3, 224, 224))
    return torch.stack(tensors)


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV, cast timestamp_ns to int64, and fill NaN depth_path with ''."""
    df = pd.read_csv(path)
    if "timestamp_ns" in df.columns:
        df["timestamp_ns"] = df["timestamp_ns"].astype(np.int64)
    if "depth_path" in df.columns:
        df["depth_path"] = df["depth_path"].fillna("")
    return df


def interpolate_to_timestamps(
    source_df: pd.DataFrame,
    target_timestamps_ns: np.ndarray,
    value_columns: list[str],
) -> pd.DataFrame:
    """Linearly interpolate *value_columns* from source_df onto target timestamps.

    Returns a DataFrame indexed by the target timestamps with one column per
    requested value column.
    """
    src_ts = source_df["timestamp_ns"].values.astype(np.float64)
    tgt_ts = np.asarray(target_timestamps_ns, dtype=np.float64)

    result = pd.DataFrame({"timestamp_ns": target_timestamps_ns})
    for col in value_columns:
        result[col] = np.interp(tgt_ts, src_ts, source_df[col].values.astype(np.float64))
    return result


def extract_track_id(bag_name: str) -> str:
    """Extract the track ID from a bag filename like ``{track_id}_bag_YYYYMMDD_HHMMSS``."""
    m = re.match(r"^(.+?)_bag_\d{8}_\d{6}", bag_name)
    if m:
        return m.group(1)
    return bag_name


def resolve_path(relative_path: str, base_dir: str) -> str:
    """Resolve a relative image path against *base_dir*."""
    if not relative_path:
        return ""
    return os.path.join(base_dir, relative_path)
