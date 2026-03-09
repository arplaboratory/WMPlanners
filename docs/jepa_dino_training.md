# JEPA and DINO Model Training Demo

This project demonstrates how to load JEPA and DINO models with pair/video dataloaders and perform forward passes.

## Files

- `configs/config.yaml`: Configuration file for model and dataset settings
- `src/world_model/dino_demo.py`: Main script that loads the configuration, builds the dataloader, loads the model, and runs forward passes.

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

Run the script using the installed CLI command:
```bash
dino --config configs/config.yaml
```

## Current Implementation

- Builds dataloaders from actual data located in the configured `data_dir`
- Loads real JEPA/DINO models via the backbone registry in `src/world_model/backbones/`
- Supports both video and pair dataset types
- Runs forward passes for configured epochs

## Real Model Usage

To ensure real JEPA and DINO models are loaded successfully:

1. Initialize git submodules for external model code:
```bash
git submodule update --init --recursive
```

The script now automatically attempts to load the real model based on `configs/config.yaml`. If model loading fails, it gracefully falls back to an Identity model to allow the pipeline to continue.

## Dependencies

- PyTorch
- Torchvision
- OpenCV (for video processing)
- PyYAML (for config)
- tqdm (for progress bars)
- requests (for downloading)

## Notes

- The external submodules (DINO, VJEPA2) require Python 3.10+ due to type annotation syntax
- The script demonstrates the complete pipeline from config loading to forward passes