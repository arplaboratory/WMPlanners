#!/usr/bin/env python3
"""
Entry point demonstrating model + dataloader usage.
This script:
  * reads configuration from config.yaml
  * lists video files under configured data directory
  * builds a DataLoader via datasets.datasets.build_dataloader
  * loads a model via world_model.backbones.create_model
  * runs a single forward pass epoch over the loader

This version relies on the modular world model architecture under `src/world_model`.
"""

import yaml
import torch
import logging

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="DINO Demo Script")
    # Default to configs/config.yaml in the project root if it exists, otherwise require argument
    default_config = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    parser.add_argument("--config", "-c", type=str, 
                        default=str(default_config) if default_config.exists() else "configs/config.yaml",
                        help="Path to config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    logger.info(f"Loaded config: {config}")

    data_dir = config["data"]["data_dir"]
    # imports that depend on project structure
    from world_model.backbones import create_model
    from datasets.datasets import build_dataloader

    # choose model
    model_cfg = config["model"]
    logger.info(f"Loading model {model_cfg['name']} ({model_cfg['model_name']})")
    try:
        model_bundle = create_model(
            name=model_cfg["name"],
            model_name=model_cfg["model_name"],
            pretrained=model_cfg.get("pretrained", True),
            device=model_cfg.get("device", "cpu"),
        )
        logger.info(f"Model loaded successfully: {model_cfg['model_name']}")
    except Exception as exc:
        logger.error(f"Model load failed: {exc}")
        logger.info("Using Identity fallback model to continue")
        from world_model.backbones import ModelBundle
        model_bundle = ModelBundle(
            model=torch.nn.Identity(),
            preprocess=lambda x: x,
            feature_dim=None,
            modality="image",
        )

    # build dataloader
    ds_cfg = config["dataset"]
    # use zero workers to avoid pickling issues with dynamic transforms
    num_workers = ds_cfg.get("num_workers", 0)
    if num_workers > 0:
        logger.warning("Setting num_workers to 0 because custom transform may not be picklable")
        num_workers = 0

    loader = build_dataloader(
        data_dir=data_dir,
        batch_size=ds_cfg.get("batch_size", 4),
        num_workers=num_workers,
        datasetargs={
            **ds_cfg.get("video_args", {}),
            **ds_cfg.get("pair_args", {}),
            "loader": ds_cfg.get("loader", "decord"),
        },
        type=ds_cfg.get("type", "pair"),
    )

    # training loop
    device = torch.device(model_cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")
    model_bundle.model.to(device).eval()
    epochs = config["training"].get("epochs", 1)
    log_every = config["training"].get("log_every", 10)

    with torch.no_grad():
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            for batch_idx, batch in enumerate(loader):
                if ds_cfg["type"] == "video":
                    video = batch["video"].to(device)
                    if hasattr(model_bundle.model, "forward_video"):
                        out = model_bundle.model.forward_video(video)
                    else:
                        B, C, T, H, W = video.shape
                        frames = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                        out = model_bundle.model(frames)
                        # if output is flat features we can reshape, otherwise keep original
                        if out.ndim == 2:
                            feat = out.shape[-1]
                            out = out.reshape(B, T, feat)
                        else:
                            # e.g. identity returned same spatial tensor
                            out = out.view(B, T, *out.shape[1:])
                else:  # pair
                    f0 = batch["frame_t"].to(device)
                    f1 = batch["frame_t1"].to(device)
                    o0 = model_bundle.model(f0)
                    o1 = model_bundle.model(f1)
                    out = {"frame_t": o0, "frame_t1": o1}

                if batch_idx % log_every == 0:
                    if isinstance(out, dict):
                        shapes = {k: v.shape for k, v in out.items()}
                        logger.info(f"Batch {batch_idx} output shapes: {shapes}")
                    else:
                        logger.info(f"Batch {batch_idx} output shape: {out.shape}")

    logger.info("Done forward pass")


if __name__ == "__main__":
    main()
