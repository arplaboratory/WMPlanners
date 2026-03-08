import pytest
import yaml
import os
from pathlib import Path
from datasets.videodataset import VideoDataset
from datasets.datasample import VideoSample


def load_test_config():
    config_path = Path(__file__).parent / "configs" / "test_videodataset.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def test_videodataset_init():
    config = load_test_config()
    dataset_config = config["dataset"]
    
    # Create dummy samples for testing init
    samples = [VideoSample(video_path="/dummy/path.mp4", label=0)]
    
    dataset = VideoDataset(
        samples=samples,
        num_frames=dataset_config["num_frames"],
        sampling_stride=dataset_config["sampling_stride"],
        crop_size=dataset_config["crop_size"],
        random_clip=dataset_config["random_clip"],
        loader=dataset_config["loader"]
    )
    
    assert len(dataset) == 1
    assert dataset.loader == "opencv"


@pytest.mark.skipif(not os.path.exists("./data"), reason="Test data directory not found")
def test_videodataset_loading():
    config = load_test_config()
    dataset_config = config["dataset"]
    data_dir = dataset_config["data_dir"]
    
    # Discover videos
    import glob
    patterns = ["**/*.mp4", "**/*.avi", "**/*.mov", "**/*.mkv"]
    video_paths = []
    for pat in patterns:
        video_paths.extend(glob.glob(os.path.join(data_dir, pat), recursive=True))
    
    if not video_paths:
        pytest.skip("No video files found for testing")
    
    samples = [VideoSample(video_path=path, label=0) for path in video_paths[:1]]  # Test with one video
    
    dataset = VideoDataset(
        samples=samples,
        num_frames=dataset_config["num_frames"],
        sampling_stride=dataset_config["sampling_stride"],
        crop_size=dataset_config["crop_size"],
        random_clip=dataset_config["random_clip"],
        loader=dataset_config["loader"]
    )
    
    # Test __getitem__
    item = dataset[0]
    
    assert "video" in item
    assert "label" in item
    assert "path" in item
    assert "frame_indices" in item
    
    video = item["video"]
    assert video.shape[0] == 3  # C
    assert video.shape[1] == dataset_config["num_frames"]  # T
    assert video.shape[2] == dataset_config["crop_size"]  # H
    assert video.shape[3] == dataset_config["crop_size"]  # W