# JEPA and DINO Model Training Demo

This project demonstrates how to load JEPA and DINO models with pair/video dataloaders and perform forward passes.

## Files Created

- `config.yaml`: Configuration file for model and dataset settings
- `main.py`: Main script that loads the configuration, creates synthetic data, loads the model, and runs forward passes

## Configuration

The `config.yaml` file contains:

```yaml
# Model configuration
model:
  name: "dinov2"  # Options: "vjepa2", "dinov2"
  model_name: "dinov2_vitb14"  # Model variant
  pretrained: true
  device: "cpu"  # "cuda" or "cpu"

# Dataset configuration
dataset:
  type: "video"  # "video" or "pair"
  batch_size: 2
  num_workers: 2

  # Video dataset specific args
  video_args:
    num_frames: 16
    sampling_stride: 4
    crop_size: 256
    random_clip: true

  # Pair dataset specific args
  pair_args:
    random_pair: true
    image_size: 224

# Training configuration
training:
  epochs: 1
  log_every: 10
```

## Usage

1. Install dependencies:
```bash
pip install pyyaml requests tqdm torch torchvision opencv-python
```

2. Run the script:
```bash
python3 main.py
```

## Current Implementation

- Uses synthetic data instead of downloading real videos (for testing)
- Includes a mock model that simulates JEPA/DINO behavior
- Supports both video and pair dataset types
- Runs forward passes for 1 epoch

## Real Model Usage

To use real JEPA and DINO models:

1. Initialize git submodules:
```bash
git submodule update --init --recursive
```

2. Comment out the mock model code in `main.py` and uncomment the real model loading:
```python
from src.models.registry import create_model
model_bundle = create_model(
    name=model_config['name'],
    model_name=model_config['model_name'],
    pretrained=model_config['pretrained'],
    device=model_config['device']
)
```

## Dependencies

- PyTorch
- Torchvision
- OpenCV (for video processing)
- PyYAML (for config)
- tqdm (for progress bars)
- requests (for downloading)

## Notes

- The external submodules (DINO, VJEPA2) require Python 3.10+ due to type annotation syntax
- For testing purposes, synthetic data and mock models are used
- The script demonstrates the complete pipeline from config loading to forward passes