"""PairDataset behavior with small, deterministic frame arrays."""

import numpy as np
import torch

from wm_datasets.datasample import VideoSample
from wm_datasets.pairdataset import PairDataset


def test_pairdataset_selects_consecutive_frames():
    dataset = PairDataset(
        [VideoSample(video_path="unused.avi", label=3)],
        random_pair=False,
        image_size=16,
        loader="opencv",
    )

    assert len(dataset) == 1
    torch.testing.assert_close(dataset._sample_indices(6), torch.tensor([2, 3]))


def test_pairdataset_transforms_loaded_frames(monkeypatch):
    sample = VideoSample(video_path="frames.avi", label=3)
    dataset = PairDataset([sample], random_pair=False, image_size=16, loader="opencv")
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    frames[1, :, :, :] = 255
    indices = torch.tensor([2, 3])
    monkeypatch.setattr(dataset, "_load_video", lambda path: (frames, indices))

    item = dataset[0]

    assert item["frame_t"].shape == (3, 16, 16)
    assert item["frame_t1"].shape == (3, 16, 16)
    assert item["label"] == 3
    assert item["path"] == sample.video_path
    torch.testing.assert_close(item["indices"], indices)
    assert not torch.equal(item["frame_t"], item["frame_t1"])
