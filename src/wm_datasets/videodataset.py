import os
import random
from dataclasses import dataclass
from typing import Callable, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from logging import getLogger
from .datasample import VideoSample

logger = getLogger(__name__)
class VideoDataset(Dataset):
    """
    Minimal skeleton for V-JEPA2-style video loading.

    Returns:
        {
            "video": Tensor[C, T, H, W],
            "label": Optional[int],
            "path": str,
            "frame_indices": Tensor[T],
        }
    """

    def __init__(
        self,
        samples: list[VideoSample],
        num_frames: int = 16,
        sampling_stride: int = 4,
        crop_size: int = 256,
        transform: Optional[Callable] = None,
        random_clip: bool = True,
        loader: str = "decord",  # "decord" or "opencv"
    ) -> None:
        self.samples = samples
        self.num_frames = num_frames
        self.sampling_stride = sampling_stride
        self.crop_size = crop_size
        self.transform = transform or self._default_transform()
        self.random_clip = random_clip
        self.loader = loader

    def __len__(self) -> int:
        return len(self.samples)

    def _default_transform(self) -> Callable:
        # Expects frames as Tensor[T, C, H, W] in [0,1]
        def _transform(frames: torch.Tensor) -> torch.Tensor:
            # Simple resize-center crop skeleton without torchvision video ops.
            # Replace with the repo's official preprocessor if you want exact behavior.
            frames = torch.nn.functional.interpolate(
                frames,
                size=(self.crop_size, self.crop_size),
                mode="bilinear",
                align_corners=False,
            )
            # Normalize with ImageNet stats as a simple placeholder.
            mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device).view(
                1, 3, 1, 1
            )
            std = torch.tensor([0.229, 0.224, 0.225], device=frames.device).view(
                1, 3, 1, 1
            )
            frames = (frames - mean) / std
            return frames

        return _transform

    def _sample_indices(self, num_total_frames: int) -> torch.Tensor:
        clip_span = (self.num_frames - 1) * self.sampling_stride + 1

        if num_total_frames <= 0:
            logger.error("Video has no frames.")
            raise ValueError("Video has no frames.")

        if num_total_frames < clip_span:
            # Loop/pad indices for short videos.
            base = torch.arange(self.num_frames) * self.sampling_stride
            indices = base % max(num_total_frames, 1)
            return indices.long()

        max_start = num_total_frames - clip_span
        if self.random_clip:
            start = random.randint(0, max_start)
        else:
            start = max_start // 2

        indices = start + torch.arange(self.num_frames) * self.sampling_stride
        return indices.long()

    def _load_video(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        # choose loader
        if self.loader == "decord":
            from decord import VideoReader, cpu
            vr = VideoReader(path, ctx=cpu(0))
            num_total_frames = len(vr)

            frame_indices = self._sample_indices(num_total_frames)
            frames = vr.get_batch(frame_indices.tolist()).asnumpy()  # [T, H, W, C]

            frames = torch.from_numpy(frames).float() / 255.0
            frames = frames.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]

            return frames, frame_indices
        else:
            # OpenCV loader fallback
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {path}") 
                raise ValueError(f"Could not open video: {path}")

            num_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = self._sample_indices(num_total_frames)
            frames_list = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx.item())
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame).float() / 255.0
                frame = frame.permute(2, 0, 1)  # [C, H, W]
                frames_list.append(frame)
            cap.release()
            if not frames_list:
                logger.error(f"No frames could be read from {path}")
                raise ValueError(f"No frames could be read from {path}")
            frames = torch.stack(frames_list, dim=0)
            return frames, frame_indices

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        frames, frame_indices = self._load_video(sample.video_path)
        frames = self.transform(frames)  # [T, C, H, W]
        frames = frames.permute(1, 0, 2, 3)  # [C, T, H, W]

        return {
            "video": frames,
            "label": sample.label,
            "path": sample.video_path,
            "frame_indices": frame_indices,
        }
