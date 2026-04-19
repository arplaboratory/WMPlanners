"""Standalone test for HFWindowDataset with lquan9/collect1."""
import sys
import os

# Ensure src/ is on the path (whether running from project root or scripts/)
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import torch
from world_model.data.hf_datasets import HFWindowDataset, collate_dinowm

DATASET_ID = "lquan9/collect1"

def make_dummy_stats(dim, mode="mean_std"):
    if mode == "mean_std":
        return {"mean": np.zeros(dim, dtype=np.float32), "std": np.ones(dim, dtype=np.float32)}
    return {"min": np.zeros(dim, dtype=np.float32), "max": np.ones(dim, dtype=np.float32)}

def test_default_mode():
    print("\n=== test_default_mode ===")
    ds = HFWindowDataset(DATASET_ID, img_size=224)
    print(f"  len(ds) = {len(ds)}")
    sample = ds[0]
    assert isinstance(sample, dict), f"Expected dict, got {type(sample)}"
    assert "rgb" in sample, "Missing 'rgb' key"
    rgb = sample["rgb"]
    print(f"  rgb shape : {rgb.shape}")   # (T, 3, 224, 224)
    assert rgb.ndim == 4 and rgb.shape[1] == 3 and rgb.shape[2] == 224
    print("  PASSED")

def test_sample_mode():
    print("\n=== test_sample_mode ===")
    ds_hf = HFWindowDataset(DATASET_ID, img_size=224)
    D_odom = ds_hf.D_odom
    D_imu  = ds_hf.D_imu
    D_act  = ds_hf.D_act

    odom_stats      = make_dummy_stats(D_odom)
    imu_stats       = make_dummy_stats(D_imu)
    collision_stats = make_dummy_stats(1)
    action_stats    = make_dummy_stats(D_act)

    ds = HFWindowDataset(
        DATASET_ID,
        img_size=224,
        sample_mode=True,
        sample_k=8,
        odom_stats=odom_stats,
        imu_stats=imu_stats,
        collision_stats=collision_stats,
        action_stats=action_stats,
    )
    print(f"  len(ds) = {len(ds)}")
    sample = ds[0]
    assert isinstance(sample, dict), f"Expected dict, got {type(sample)}"
    assert "rgb" in sample and "state" in sample, f"Missing keys: {list(sample.keys())}"
    rgb   = sample["rgb"]
    state = sample["state"]
    print(f"  rgb shape   : {rgb.shape}")    # (8, 3, 224, 224)
    print(f"  state shape : {state.shape}")  # (8, D_odom+D_imu+1)
    assert rgb.shape == (8, 3, 224, 224), f"Unexpected rgb shape: {rgb.shape}"
    assert state.shape == (8, D_odom + D_imu + 1), f"Unexpected state shape: {state.shape}"
    print("  PASSED")

def test_collate_dinowm():
    print("\n=== test_collate_dinowm ===")
    ds_hf = HFWindowDataset(DATASET_ID, img_size=224)
    D_odom = ds_hf.D_odom
    D_imu  = ds_hf.D_imu
    D_act  = ds_hf.D_act

    ds = HFWindowDataset(
        DATASET_ID,
        img_size=224,
        sample_mode=True,
        sample_k=8,
        odom_stats=make_dummy_stats(D_odom),
        imu_stats=make_dummy_stats(D_imu),
        collision_stats=make_dummy_stats(1),
        action_stats=make_dummy_stats(D_act),
    )
    batch = [ds[i] for i in range(2)]
    sb = collate_dinowm(batch)
    print(f"  rgb shape    : {sb.rgb.shape}")     # (2, 8, 3, 224, 224)
    print(f"  state shape  : {sb.state.shape}")   # (2, 8, D_odom+D_imu+1)
    print(f"  action shape : {sb.action.shape}")  # (2, 8, D_act)
    assert sb.rgb.shape == (2, 8, 3, 224, 224)
    print("  PASSED")

if __name__ == "__main__":
    test_default_mode()
    test_sample_mode()
    test_collate_dinowm()
    print("\nAll tests passed!")
