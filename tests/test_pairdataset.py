import pytest
import yaml
import os
from pathlib import Path
from src.datasets.pairdataset import PairDataset
from src.datasets.datasample import VideoSample


def load_test_config():
    config_path = Path(__file__).parent / "configs" / "test_pairdataset.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def test_pairdataset_init():
    config = load_test_config()
    dataset_config = config["dataset"]
    
    # Create dummy samples for testing init
    samples = [VideoSample(video_path="/dummy/path.mp4", label=0)]
    
    dataset = PairDataset(
        samples=samples,
        random_pair=dataset_config["random_pair"],
        image_size=dataset_config["image_size"],
        loader=dataset_config["loader"]
    )
    
    assert len(dataset) == 1
    assert dataset.loader == "opencv"


@pytest.mark.skipif(not os.path.exists("./data"), reason="Test data directory not found")
def test_pairdataset_loading():
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
    
    dataset = PairDataset(
        samples=samples,
        random_pair=dataset_config["random_pair"],
        image_size=dataset_config["image_size"],
        loader=dataset_config["loader"]
    )
    
    # Test __getitem__
    item = dataset[0]
    
    assert "frame_t" in item
    assert "frame_t1" in item
    assert "label" in item
    assert "path" in item
    assert "indices" in item
    
    frame_t = item["frame_t"]
    frame_t1 = item["frame_t1"]
    
    assert frame_t.shape[0] == 3  # C
    assert frame_t.shape[1] == dataset_config["image_size"]  # H
    assert frame_t.shape[2] == dataset_config["image_size"]  # W
    
    assert frame_t1.shape == frame_t.shape