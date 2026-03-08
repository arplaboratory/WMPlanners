#!/usr/bin/env python3
"""
Test script to verify if main.py can load DINO v2 model
"""
import sys
import os
import yaml
import torch
import logging

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    """Load YAML configuration file"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def test_dinov2_loading():
    """Test loading DINOv2 model through the registry"""
    logger.info("Starting DINOv2 model loading test...")
    
    # Test 1: Import the model registry
    logger.info("Test 1: Importing model registry...")
    try:
        from src.models.registry import create_model
        logger.info("✓ Model registry imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import model registry: {e}")
        return False
    
    # Test 2: Load configuration
    logger.info("Test 2: Loading test configuration...")
    try:
        config_path = os.path.join(os.path.dirname(__file__), "configs", "test_dinov2.yaml")
        config = load_config(config_path)
        logger.info(f"✓ Configuration loaded: {config}")
    except Exception as e:
        logger.error(f"✗ Failed to load configuration: {e}")
        return False
    
    # Test 3: Try to load dinov2 model
    logger.info("Test 3: Loading dinov2 model...")
    try:
        model_cfg = config["model"]
        model_bundle = create_model(
            name=model_cfg["name"],
            model_name=model_cfg["model_name"],
            pretrained=model_cfg.get("pretrained", False),
            device=model_cfg.get("device", "cpu"),
        )
        logger.info(f"✓ DINO v2 model loaded successfully!")
        logger.info(f"  - Model type: {type(model_bundle.model)}")
        logger.info(f"  - Feature dim: {model_bundle.feature_dim}")
        logger.info(f"  - Modality: {model_bundle.modality}")
        logger.info(f"  - Preprocess: {model_bundle.preprocess}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load dinov2 model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dinov2_forward_pass():
    """Test running a forward pass with DINOv2"""
    logger.info("\nTest 4: Running forward pass with DINOv2...")
    try:
        from src.models.registry import create_model
        import torch
        
        config_path = os.path.join(os.path.dirname(__file__), "configs", "test_dinov2.yaml")
        config = load_config(config_path)
        
        model_cfg = config["model"]
        model_bundle = create_model(
            name=model_cfg["name"],
            model_name=model_cfg["model_name"],
            pretrained=model_cfg.get("pretrained", False),
            device=model_cfg.get("device", "cpu"),
        )
        
        # Create a dummy input (batch of images)
        device = torch.device(model_cfg.get("device", "cpu"))
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        # Run forward pass
        with torch.no_grad():
            output = model_bundle.model(dummy_input)
        
        logger.info(f"✓ Forward pass successful!")
        logger.info(f"  - Input shape: {dummy_input.shape}")
        logger.info(f"  - Output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dinov2_loading()
    if success:
        forward_pass_success = test_dinov2_forward_pass()
        sys.exit(0 if forward_pass_success else 1)
    else:
        sys.exit(1)
