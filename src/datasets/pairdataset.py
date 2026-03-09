import random
from dataclasses import dataclass
from typing import Optional, Callable
import torch
from torch.utils.data import Dataset
import numpy as np
from .datasample import VideoSample
from logging import getLogger
logger = getLogger(__name__)


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
        loader: str = "decord",  # "decord" or "opencv"
    ):
        self.samples = samples
        self.random_pair = random_pair
        self.transform = transform or self._default_transform(image_size)
        self.loader = loader

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
            logger.error(f"Video has {num_frames} frames, but at least 2 are required for pairing.")
            raise ValueError(f"Video has {num_frames} frames, but at least 2 are required for pairing.")
        

        if self.random_pair:
            t = random.randint(0, num_frames - 2)
        else:
            t = (num_frames - 2) // 2

        return torch.tensor([t, t + 1], dtype=torch.long)

    def _load_video(self, path: str) -> tuple:
        # choose loader
        if self.loader == "decord":
            from decord import VideoReader, cpu
            vr = VideoReader(path, ctx=cpu(0))
            num_frames = len(vr)

            indices = self._sample_indices(num_frames)
            frames = vr.get_batch(indices.tolist()).asnumpy()  # [2, H, W, C]

            return frames, indices
        else:
            import cv2
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {path}")
                raise ValueError(f"Could not open video: {path}")

            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = self._sample_indices(num_frames)

            # read frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0].item())
            ret1, frame_t = cap.read()
            if not ret1:
                logger.error(f"Could not read frame {indices[0]} from {path}")
                raise ValueError(f"Could not read frame {indices[0]} from {path}")

            cap.set(cv2.CAP_PROP_POS_FRAMES, indices[1].item())
            ret2, frame_t1 = cap.read()
            if not ret2:
                logger.error(f"Could not read frame {indices[1]} from {path}")
                raise ValueError(f"Could not read frame {indices[1]} from {path}")

            cap.release()

            # Convert to RGB
            frame_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2RGB)
            frame_t1 = cv2.cvtColor(frame_t1, cv2.COLOR_BGR2RGB)

            frames = np.stack([frame_t, frame_t1], axis=0)  # [2, H, W, C]
            return frames, indices

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        frames, indices = self._load_video(sample.video_path)

        frame_t = frames[0]
        frame_t1 = frames[1]

        # convert to PIL and transform
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