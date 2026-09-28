"""Compatibility import for image metrics used by earlier backbone callers."""

from world_model.metrics import (
    create_window,
    eval_images,
    gaussian,
    l1_loss,
    l2_loss,
    mse,
    psnr,
    ssim,
)

__all__ = [
    "create_window", "eval_images", "gaussian", "l1_loss", "l2_loss",
    "mse", "psnr", "ssim",
]
