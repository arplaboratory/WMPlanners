"""Convenience builders for video clip and adjacent-frame datasets."""

from __future__ import annotations

import glob
import os
from logging import getLogger

from torch.utils.data import DataLoader

from .datasample import VideoSample
from .pairdataset import PairDataset
from .utils.collate import pair_collate, video_collate
from .videodataset import VideoDataset

logger = getLogger(__name__)


def discover_videos(data_dir: str) -> list[str]:
    """Recursively collect common video file extensions from the directory."""
    patterns = ["**/*.mp4", "**/*.avi", "**/*.mov", "**/*.mkv"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pattern), recursive=True))
    return sorted(files)


def build_dataloader(
    data_dir: str,
    labels: list[int] | None = None,
    batch_size: int = 4,
    num_workers: int = 4,
    datasetargs: dict | None = None,
    type: str = "pair",
) -> DataLoader:
    """Build a video loader; OpenCV works with the core installation.

    Pass ``datasetargs={"loader": "decord"}`` after installing the optional
    ``video`` extra to use Decord instead.
    """
    if type not in {"pair", "video"}:
        raise ValueError(f"Unsupported dataset type {type!r}; use 'pair' or 'video'")
    options = dict(datasetargs or {})
    loader = options.get("loader", "opencv")
    if loader not in {"opencv", "decord"}:
        raise ValueError(f"Unsupported video loader {loader!r}")
    if loader == "decord":
        try:
            import decord  # noqa: F401
        except ImportError as exc:
            raise ImportError("Decord requires the optional 'video' extra") from exc

    video_paths = discover_videos(data_dir)
    if not video_paths:
        raise RuntimeError(f"No video files found under {data_dir}")
    logger.info("Found %s videos in %s", len(video_paths), data_dir)

    if labels is not None and len(video_paths) != len(labels):
        raise ValueError("video_paths and labels must have the same length")
    samples = [
        VideoSample(video_path=path, label=None if labels is None else labels[index])
        for index, path in enumerate(video_paths)
    ]

    if type == "video":
        dataset = VideoDataset(
            samples=samples,
            num_frames=options.get("num_frames", 16),
            sampling_stride=options.get("sampling_stride", 4),
            crop_size=options.get("crop_size", 256),
            random_clip=options.get("random_clip", True),
            loader=loader,
        )
        collate_fn = video_collate
        shuffle = dataset.random_clip
    else:
        dataset = PairDataset(
            samples=samples,
            random_pair=options.get("random_pair", True),
            image_size=options.get("image_size", 256),
            loader=loader,
        )
        collate_fn = pair_collate
        shuffle = dataset.random_pair

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
