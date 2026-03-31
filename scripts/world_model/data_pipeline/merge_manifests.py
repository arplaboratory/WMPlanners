#!/usr/bin/env python3
import json
import os
import argparse
from pathlib import Path

def merge_manifests(staging_dir, output_dir):
    staging_path = Path(staging_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_bags = []
    final_config = None
    final_robot = None
    
    print(f"Searching for manifests in {staging_dir}...")
    
    # Iterate through each bag's staging folder
    # Expected structure: staging_dir/bag_stem/manifest.json
    for bag_entry in sorted(staging_path.iterdir()):
        if not bag_entry.is_dir():
            continue
        
        manifest_file = bag_entry / "manifest.json"
        if not manifest_file.exists():
            continue
            
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
            
        if final_config is None:
            final_config = manifest.get("config")
            final_robot = manifest.get("robot")
            
        # Add bags to the master list
        bags = manifest.get("bags", [])
        all_bags.extend(bags)
        
        # Symlink the actual bag data folder into output_dir
        for bag_info in bags:
            bag_name = bag_info["name"]
            # extract_dataset.py creates a folder with the bag name inside its output directory
            src_bag_dir = bag_entry / bag_name
            dst_bag_dir = output_path / bag_name
            
            if src_bag_dir.exists():
                if dst_bag_dir.exists():
                    if dst_bag_dir.is_symlink():
                        os.unlink(dst_bag_dir)
                    else:
                        print(f"Warning: {dst_bag_dir} already exists and is not a symlink. Skipping.")
                        continue
                
                # Use absolute paths for symlinks to ensure they work regardless of CWD
                os.symlink(os.path.abspath(src_bag_dir), os.path.abspath(dst_bag_dir))
                # print(f"  Linked {bag_name}")

    if not all_bags:
        print("Error: No bags found to merge.")
        return

    master_manifest = {
        "robot": final_robot,
        "config": final_config,
        "bags": all_bags
    }
    
    with open(output_path / "manifest.json", 'w') as f:
        json.dump(master_manifest, f, indent=2)
    
    print(f"Successfully merged {len(all_bags)} bags into {output_dir}/manifest.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple manifest.json files and symlink data.")
    parser.add_argument("--staging", required=True, help="Directory containing per-bag extractions")
    parser.add_argument("--output", required=True, help="Directory to write the merged manifest and symlinks")
    args = parser.parse_args()
    merge_manifests(args.staging, args.output)
