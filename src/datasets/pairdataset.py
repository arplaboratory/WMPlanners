import random
from dataclasses import dataclass
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset

try:
    from decord import VideoReader, cpu
except ImportError as e:
    raise ImportError("Please install decord.") from e

from .datasample import VideoSample



class PairDataset(Dataset):
    """
    Returns two consecutive frames from a video.

    Output:
    {
        "frame_t": Tensor[C, H, W],
        "frame_t1": Tensor[C, H, W],
        "label": Optional[int],
        "path": str,
        "indices": Tensor[2],   # [t, t+1]
    }
    """

    def __init__(
        self,
        samples: list[VideoSample],
        transform: Optional[Callable] = None,
        random_pair: bool = True,
        image_size: int = 224,
    ):
        self.samples = samples
        self.random_pair = random_pair
        self.transform = transform or self._default_transform(image_size)

    def __len__(self):
        return len(self.samples)

    def _default_transform(self, image_size: int):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def _sample_indices(self, num_frames: int) -> torch.Tensor:
        if num_frames < 2:
            raise ValueError("Video must contain at least 2 frames.")

        if self.random_pair:
            t = random.randint(0, num_frames - 2)
        else:
            t = (num_frames - 2) // 2

        return torch.tensor([t, t + 1], dtype=torch.long)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        vr = VideoReader(sample.video_path, ctx=cpu(0))
        num_frames = len(vr)

        indices = self._sample_indices(num_frames)
        frames = vr.get_batch(indices.tolist()).asnumpy()  # [2, H, W, C]

        frame_t = frames[0]
        frame_t1 = frames[1]

        from PIL import Image
        frame_t = Image.fromarray(frame_t)
        frame_t1 = Image.fromarray(frame_t1)

        frame_t = self.transform(frame_t)     # [C, H, W]
        frame_t1 = self.transform(frame_t1)   # [C, H, W]

        return {
            "frame_t": frame_t,
            "frame_t1": frame_t1,
            "label": sample.label,
            "path": sample.video_path,
            "indices": indices,
        }