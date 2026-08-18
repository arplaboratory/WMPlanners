from __future__ import annotations
import os
from typing import Dict, Optional, List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T

from .types import SampleBatch

def _as_float32(x):
    return np.asarray(x, dtype=np.float32)

class NPZEpisodeDataset(Dataset):
    """Loads episodes from .npz files and returns sliding windows.

    Expected keys per episode:
      rgb: (T,H,W,3) uint8
      depth: (T,H,W) float32   [optional]
      imu: (T,D_imu) float32
      odom: (T,D_odom) float32
      action: (T,D_act) float32
      collision: (T,) float32 [optional]
    """
    def __init__(self, data_dir: str, window: int = 16, stride: int = 1, img_size: Optional[int] = None):
        self.data_dir = data_dir
        self.window = window
        self.stride = stride
        self.img_size = img_size
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])
        if not self.files:
            raise FileNotFoundError(f"No .npz episode files found in: {data_dir}")

        self._transform = None
        if img_size is not None:
            self._transform = T.Compose([
                T.ToPILImage(),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
            ])

        # Light scan: read first file for dtype/dim metadata
        ex = np.load(self.files[0])
        self.has_depth = 'depth' in ex
        self.has_collision = 'collision' in ex
        self.H = img_size if img_size is not None else ex['rgb'].shape[1]
        self.W = img_size if img_size is not None else ex['rgb'].shape[2]
        self.D_imu = ex['imu'].shape[-1]
        self.D_odom = ex['odom'].shape[-1]
        self.D_act = ex['action'].shape[-1]

        # Pre-index all valid (file_idx, start_frame) windows across all episodes.
        # This makes __len__ accurate and __getitem__ deterministic — DataLoader
        # shuffling is then the only source of randomness, which is the correct design.
        self._windows: List[tuple] = []
        for file_idx, path in enumerate(self.files):
            ep = np.load(path, mmap_mode='r')
            T_len = ep['rgb'].shape[0]
            for start in range(0, T_len - self.window, self.stride):
                self._windows.append((file_idx, start))
        if not self._windows:
            raise ValueError(
                f"No valid windows found in {data_dir} with window={window}, stride={stride}. "
                "All episodes may be shorter than the requested window size."
            )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        file_idx, start = self._windows[idx]
        ep = np.load(self.files[file_idx], mmap_mode='r')
        rgb = ep['rgb']  # (T,H,W,3) uint8
        sl = slice(start, start + self.window)

        out = {}
        if self._transform is not None:
            # Resize frame by frame
            rgb_resized = []
            for frame in rgb[sl]:
                # ToPILImage expects (H,W,C)
                rgb_resized.append(self._transform(frame).permute(1, 2, 0).numpy()) # back to (H,W,3) for collation
            out['rgb'] = np.stack(rgb_resized)
        else:
            out['rgb'] = rgb[sl]  # (T,H,W,3)

        if 'depth' in ep:
            depth = _as_float32(ep['depth'][sl])
            if self.img_size is not None:
                # Simple resize for depth (nearest)
                depth_t = torch.from_numpy(depth).unsqueeze(1) # (T,1,H,W)
                depth_t = F.interpolate(depth_t, size=(self.img_size, self.img_size), mode='nearest')
                out['depth'] = depth_t.squeeze(1).numpy()
            else:
                out['depth'] = depth
            out['depth_valid'] = np.asarray(
                ep['depth_valid'][sl] if 'depth_valid' in ep
                else np.ones(self.window, dtype=bool), dtype=bool,
            )
        else:
            out['depth'] = np.zeros((self.window, self.H, self.W), dtype=np.float32)
            out['depth_valid'] = np.zeros(self.window, dtype=bool)

        out['imu'] = _as_float32(ep['imu'][sl])
        out['odom'] = _as_float32(ep['odom'][sl])
        out['action'] = _as_float32(ep['action'][sl])
        for name in ('imu', 'odom', 'action'):
            out[f'{name}_valid'] = np.asarray(
                ep[f'{name}_valid'][sl] if f'{name}_valid' in ep
                else np.ones(self.window, dtype=bool), dtype=bool,
            )
        if 'collision' in ep:
            out['collision'] = _as_float32(ep['collision'][sl])
            out['collision_valid'] = np.asarray(
                ep['collision_valid'][sl] if 'collision_valid' in ep
                else np.ones(self.window, dtype=bool), dtype=bool,
            )
        else:
            out['collision'] = np.zeros((self.window,), dtype=np.float32)
            out['collision_valid'] = np.zeros(self.window, dtype=bool)
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
    def validity(name: str) -> torch.Tensor:
        values = np.stack([
            np.asarray(item.get(name, np.ones(item["rgb"].shape[0], dtype=bool)), dtype=bool)
            for item in batch
        ])
        return torch.from_numpy(values).unsqueeze(-1)

    return SampleBatch(
        rgb=rgb_t, depth=depth_t, imu=imu_t, odom=odom_t, action=action_t,
        collision=collision_t, depth_valid=validity("depth_valid"),
        imu_valid=validity("imu_valid"), odom_valid=validity("odom_valid"),
        action_valid=validity("action_valid"), collision_valid=validity("collision_valid"),
    )
