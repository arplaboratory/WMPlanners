"""The public video-loader builder works with core dependencies and defaults."""

import numpy as np
import pytest
import torch

from wm_datasets import build_dataloader


def test_default_pair_loader_reads_video(tmp_path):
    import cv2

    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (16, 16))
    assert writer.isOpened()
    for value in (0, 80, 160, 240):
        writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.release()

    loader = build_dataloader(str(tmp_path), num_workers=0)
    assert loader.dataset.loader == "opencv"
    batch = next(iter(loader))
    assert batch["frame_t"].shape == (1, 3, 256, 256)
    assert batch["frame_t1"].shape == (1, 3, 256, 256)
    assert torch.equal(batch["indices"][0, 1:], batch["indices"][0, :1] + 1)


def test_invalid_type_and_loader_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported dataset type"):
        build_dataloader(str(tmp_path), type="unknown")
    with pytest.raises(ValueError, match="Unsupported video loader"):
        build_dataloader(str(tmp_path), datasetargs={"loader": "unknown"})
