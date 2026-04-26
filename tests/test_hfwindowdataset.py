import sys
import numpy as np
from world_model.data import HFWindowDataset

def main():
    repo = "lquan9/collect1"
    print(f"Loading {repo} via HFWindowDataset (sample_mode=True, sample_k=8)...")
    # Dummy stats for normalization (replace with real stats if available)
    odom_stats = {"mean": 0.0, "std": 1.0}
    imu_stats = {"mean": 0.0, "std": 1.0}
    collision_stats = {"mean": 0.0, "std": 1.0}
    action_stats = {"mean": 0.0, "std": 1.0}
    ds = HFWindowDataset(
        repo, img_size=224, sample_mode=True, sample_k=8,
        odom_stats=odom_stats, imu_stats=imu_stats, collision_stats=collision_stats, action_stats=action_stats
    )
    print(f"Total segments: {len(ds)}")
    print(f"Total frames: {len(ds._ds)}")
    print(f"Columns: {ds._ds.column_names}")
    print(f"Dimensions: Odom={ds.D_odom}, IMU={ds.D_imu}, Act={ds.D_act}")
    print("\nExtracting first sampled window...")
    try:
        window = ds[0]
        print("window keys:", list(window.keys()))
        rgb = window["rgb"]
        print(f"RGB shape: {rgb.shape}")
        print(f"Sampled window stats: min={rgb.min()}, max={rgb.max()}, mean={rgb.mean()}")
        if "state" in window:
            print(f"State shape: {window['state'].shape}")
    except Exception as e:
        print(f"Error extracting sampled window: {e}")

if __name__ == "__main__":
    main()
