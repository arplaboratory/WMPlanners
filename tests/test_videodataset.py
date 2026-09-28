"""VideoDataset sampling and tensor layout with deterministic frames."""

import torch

from wm_datasets.datasample import VideoSample
from wm_datasets.videodataset import VideoDataset


def test_videodataset_samples_fixed_stride():
    dataset = VideoDataset(
        [VideoSample(video_path="unused.avi")],
        num_frames=4,
        sampling_stride=2,
        crop_size=16,
        random_clip=False,
        loader="opencv",
    )

    assert len(dataset) == 1
    torch.testing.assert_close(
        dataset._sample_indices(8), torch.tensor([0, 2, 4, 6])
    )


def test_videodataset_transforms_loaded_frames(monkeypatch):
    sample = VideoSample(video_path="frames.avi", label=4)
    dataset = VideoDataset(
        [sample], num_frames=4, sampling_stride=2, crop_size=16,
        random_clip=False, loader="opencv",
    )
    frames = torch.full((4, 3, 8, 8), 0.5)
    indices = torch.tensor([0, 2, 4, 6])
    monkeypatch.setattr(dataset, "_load_video", lambda path: (frames, indices))

    item = dataset[0]

    assert item["video"].shape == (3, 4, 16, 16)
    assert item["label"] == 4
    assert item["path"] == sample.video_path
    torch.testing.assert_close(item["frame_indices"], indices)
    assert torch.isfinite(item["video"]).all()
