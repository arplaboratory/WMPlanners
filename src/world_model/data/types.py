from dataclasses import dataclass, field
from typing import Optional
import torch

@dataclass
class SampleBatch:
    """Collated batch of trajectory windows.

    All tensors have shape ``(B, T, ...)`` where *B* is the batch size and
    *T* is the number of frames (window size).
    """

    rgb: torch.Tensor        # (B, T, 3, H, W) float in [0, 1]
    depth: torch.Tensor      # (B, T, 1, H, W)
    imu: torch.Tensor        # (B, T, D_imu)
    odom: torch.Tensor       # (B, T, D_odom)
    action: torch.Tensor     # (B, T, D_act)
    collision: torch.Tensor  # (B, T, 1)
    # DinoWM extras (optional)
    proprio: Optional[torch.Tensor] = None  # (B, T, D_odom+D_imu+1) concatenated state
    state: Optional[torch.Tensor] = None    # (B, T, D_odom+D_imu+1) raw (pre-norm) state
