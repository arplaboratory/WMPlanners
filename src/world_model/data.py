from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

def _as_float32(x):
    return np.asarray(x, dtype=np.float32)

@dataclass
class SampleBatch:
    # Shapes (B, T, ...)
    rgb: torch.Tensor       # (B,T,3,H,W) float in [0,1]
    depth: torch.Tensor     # (B,T,1,H,W)
    imu: torch.Tensor       # (B,T,D_imu)
    odom: torch.Tensor      # (B,T,D_odom)
    action: torch.Tensor    # (B,T,D_act)
    collision: torch.Tensor # (B,T,1)

class NPZEpisodeDataset(Dataset):
    """Loads episodes from .npz files and returns random windows.

    Expected keys per episode:
      rgb: (T,H,W,3) uint8
      depth: (T,H,W) float32   [optional]
      imu: (T,D_imu) float32
      odom: (T,D_odom) float32
      action: (T,D_act) float32
      collision: (T,) float32 [optional]
    """
    def __init__(self, data_dir: str, window: int = 16, stride: int = 1):
        self.data_dir = data_dir
        self.window = window
        self.stride = stride
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])
        if not self.files:
            raise FileNotFoundError(f"No .npz episode files found in: {data_dir}")

        # Light scan for shapes
        ex = np.load(self.files[0])
        self.has_depth = 'depth' in ex
        self.has_collision = 'collision' in ex
        self.H, self.W = ex['rgb'].shape[1], ex['rgb'].shape[2]
        self.D_imu = ex['imu'].shape[-1]
        self.D_odom = ex['odom'].shape[-1]
        self.D_act = ex['action'].shape[-1]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        ep = np.load(self.files[idx])
        rgb = ep['rgb']  # (T,H,W,3) uint8
        T = rgb.shape[0]
        if T < self.window + 1:
            raise ValueError(f"Episode too short: {self.files[idx]} with T={T}")

        # pick a random start
        start = np.random.randint(0, T - self.window)
        sl = slice(start, start + self.window)

        out = {}
        out['rgb'] = rgb[sl]  # (W,H,...)
        if 'depth' in ep:
            out['depth'] = _as_float32(ep['depth'][sl])
        else:
            out['depth'] = np.zeros((self.window, self.H, self.W), dtype=np.float32)

        out['imu'] = _as_float32(ep['imu'][sl])
        out['odom'] = _as_float32(ep['odom'][sl])
        out['action'] = _as_float32(ep['action'][sl])
        if 'collision' in ep:
            out['collision'] = _as_float32(ep['collision'][sl])
        else:
            out['collision'] = np.zeros((self.window,), dtype=np.float32)
        return out

def collate_npz(batch: List[Dict[str, np.ndarray]]) -> SampleBatch:
    # batch: list of dicts with numpy arrays
    rgb = np.stack([b['rgb'] for b in batch], axis=0)  # (B,T,H,W,3)
    depth = np.stack([b['depth'] for b in batch], axis=0)  # (B,T,H,W)
    imu = np.stack([b['imu'] for b in batch], axis=0)  # (B,T,D)
    odom = np.stack([b['odom'] for b in batch], axis=0)
    action = np.stack([b['action'] for b in batch], axis=0)
    collision = np.stack([b['collision'] for b in batch], axis=0)  # (B,T)

    # convert to torch
    rgb_t = torch.from_numpy(rgb).permute(0,1,4,2,3).float() / 255.0  # (B,T,3,H,W)
    depth_t = torch.from_numpy(depth).unsqueeze(2).float()            # (B,T,1,H,W)
    imu_t = torch.from_numpy(imu).float()
    odom_t = torch.from_numpy(odom).float()
    action_t = torch.from_numpy(action).float()
    collision_t = torch.from_numpy(collision).unsqueeze(-1).float()
    return SampleBatch(rgb=rgb_t, depth=depth_t, imu=imu_t, odom=odom_t, action=action_t, collision=collision_t)
