#!/usr/bin/env python3
"""Tests for loading DINOv2 through the backbone registry.

Updated to use the canonical ``world_model.backbones`` import path.
"""
import os
import sys
import logging

import pytest
import yaml
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _get_test_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "configs", "test_dinov2.yaml")
    return _load_config(config_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDINOv2Registry:
    """Test that the backbone registry can import and load DINOv2."""

    def test_import_registry(self):
        """Verify the backbone registry is importable."""
        from world_model.backbones import create_model  # noqa: F401

    def test_dinov2_registered(self):
        """Verify ``dinov2`` is a registered backbone name."""
        from world_model.backbones.registry import _BUILDERS

        assert "dinov2" in _BUILDERS, (
            f"'dinov2' not in registry. Available: {sorted(_BUILDERS.keys())}"
        )

    def test_load_dinov2_from_config(self):
        """Load DINOv2 through the registry using the test config."""
        from world_model.backbones import create_model

        config = _get_test_config()
        model_cfg = config["model"]

        try:
            bundle = create_model(
                name=model_cfg["name"],
                model_name=model_cfg["model_name"],
                pretrained=model_cfg.get("pretrained", False),
                device=model_cfg.get("device", "cpu"),
            )
        except (RuntimeError, OSError) as exc:
            pytest.skip(f"Cannot load DINOv2 in this environment: {exc}")

        assert bundle.modality == "image"
        assert bundle.feature_dim is not None
        logger.info(
            f"Loaded DINOv2: feature_dim={bundle.feature_dim}, modality={bundle.modality}"
        )

    def test_forward_pass(self):
        """Run a dummy forward pass through DINOv2."""
        from world_model.backbones import create_model

        config = _get_test_config()
        model_cfg = config["model"]

        try:
            bundle = create_model(
                name=model_cfg["name"],
                model_name=model_cfg["model_name"],
                pretrained=model_cfg.get("pretrained", False),
                device=model_cfg.get("device", "cpu"),
            )
        except (RuntimeError, OSError) as exc:
            pytest.skip(f"Cannot load DINOv2 in this environment: {exc}")

        device = torch.device(model_cfg.get("device", "cpu"))
        dummy = torch.randn(1, 3, 224, 224, device=device)

        with torch.no_grad():
            output = bundle.model(dummy)

        assert output is not None
        logger.info(f"Forward pass OK — output shape: {output.shape}")
