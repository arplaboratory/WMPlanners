"""Stage 0 -- Extract ROS2 MCAP bags into sharded dataset.

Thin wrapper around scripts.world_model.extract_dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

from scripts.world_model.extract_dataset import extract, write_manifest
from scripts.world_model.robot_config import load_robot_config, parse_overrides


def run_stage0(
    mcap_paths: list[str],
    robot: str,
    output_dir: str,
    shard_size: int = 500,
    overrides: dict | None = None,
) -> list[str]:
    """Extract MCAP bags into per-bag directories.

    Returns list of per-bag output directory paths.
    """
    config = load_robot_config(robot, overrides=overrides or {})
    os.makedirs(output_dir, exist_ok=True)

    bags_info: list[dict] = []
    bag_dirs: list[str] = []

    for mcap_path in mcap_paths:
        bag_stem = Path(mcap_path).stem
        bag_output = os.path.join(output_dir, bag_stem)
        frames = extract(mcap_path, bag_output, config, shard_size)
        bags_info.append({"name": bag_stem, "path": f"{bag_stem}/", "frames": frames})
        bag_dirs.append(bag_output)

    write_manifest(output_dir, config, bags_info)
    print(f"[stage0] Manifest written to {os.path.join(output_dir, 'manifest.json')}")

    return bag_dirs
