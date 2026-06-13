#!/usr/bin/env python3
"""Run the Collect3 Race16 archive through the world-model data pipeline.

The Collect3 bundle on the lab machine is stored as:

    /media/data/corl/DS3001/Collect3/2026_04_22.tar.gz

Despite the suffix, that file is a plain tar archive. This script handles both
plain tar and compressed tar formats, finds the contained MCAP files, runs the
repo's Stage 0 extractor, builds fixed-length NPZ episodes, selects a diverse
subset, and creates a Hugging Face dataset from the selected episodes. If
``--push-to-hub`` is provided, the dataset is pushed to that Hub repo.

Example:
    python -m scripts.world_model.data_pipeline.extract_collect3_npz
"""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
from pathlib import Path

from scripts.world_model.build_episodes import build_episodes
from scripts.world_model.data_pipeline.stage0_extract import run_stage0
from scripts.world_model.data_pipeline.hf_upload import run_hf_upload
from scripts.world_model.robot_config import load_robot_config
from scripts.world_model.select_episodes import select_episodes


DEFAULT_COLLECT_DIR = "/media/data/corl/DS3001/Collect3"
DEFAULT_ARCHIVE = "2026_04_22.tar.gz"
DEFAULT_OUTPUT = "/media/data/corl/DS3001/Collect3_npz"


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract tar contents while preventing path traversal."""
    dest_resolved = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest_resolved) + os.sep):
            raise RuntimeError(f"Unsafe tar member path: {member.name}")
    tar.extractall(dest)


def extract_archive(archive_path: Path, unpack_dir: Path, force: bool = False) -> None:
    if force and unpack_dir.exists():
        shutil.rmtree(unpack_dir)

    if unpack_dir.exists() and any(unpack_dir.iterdir()):
        print(f"[collect3] Reusing existing unpacked archive: {unpack_dir}")
        return

    unpack_dir.mkdir(parents=True, exist_ok=True)
    print(f"[collect3] Extracting archive: {archive_path}")
    print(f"[collect3] Destination:       {unpack_dir}")
    with tarfile.open(archive_path, mode="r:*") as tf:
        _safe_extract(tf, unpack_dir)


def find_mcaps(unpack_dir: Path) -> list[str]:
    mcaps = sorted(str(p) for p in unpack_dir.rglob("*.mcap"))
    if not mcaps:
        raise FileNotFoundError(f"No .mcap files found under {unpack_dir}")
    return mcaps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Collect3 Race16 archive into NPZ world-model episodes.",
    )
    parser.add_argument("--collect-dir", default=DEFAULT_COLLECT_DIR)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--robot", default="race16")
    parser.add_argument("--episode-len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--frame-skip", type=int, default=6)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--min-speed", type=float, default=0.1)
    parser.add_argument("--max-sync-gap-ms", type=float, default=100.0)
    parser.add_argument("--idle-persistence", type=int, default=10)
    parser.add_argument(
        "--image-gap-factor",
        type=float,
        default=2.0,
        help="Image stream gap threshold as a multiple of median image dt (default: 2.0)",
    )
    parser.add_argument(
        "--action-mode",
        default="auto",
        choices=["auto", "config", "odom_velocity"],
        help="Action source policy for episode building (default: auto)",
    )
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--target-count", type=int, default=5000)
    parser.add_argument("--encoder", default="none")
    parser.add_argument(
        "--profile",
        default="generic",
        choices=["generic", "spot", "rover"],
        help="Episode selection profile (default: generic)",
    )
    parser.add_argument("--push-to-hub", default=None)
    parser.add_argument("--hf-private", action="store_true", default=False)
    parser.add_argument(
        "--force-unpack",
        action="store_true",
        help="Delete and recreate the unpacked archive directory.",
    )
    parser.add_argument(
        "--skip-stage0",
        action="store_true",
        help="Reuse output/extracted and only rebuild episodes.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse output/episodes and only run later stages.",
    )
    parser.add_argument(
        "--skip-select",
        action="store_true",
        help="Reuse output/selected and only run HF dataset creation.",
    )
    parser.add_argument(
        "--skip-hf-upload",
        action="store_true",
        help="Skip local HF dataset creation and optional Hub upload.",
    )
    args = parser.parse_args()

    collect_dir = Path(args.collect_dir)
    archive_path = Path(args.archive)
    if not archive_path.is_absolute():
        archive_path = collect_dir / archive_path

    out_dir = Path(args.output)
    unpack_dir = out_dir / "archive_contents"
    extracted_dir = out_dir / "extracted"
    episodes_dir = out_dir / "episodes"
    selected_dir = out_dir / "selected"

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    extract_archive(archive_path, unpack_dir, force=args.force_unpack)
    mcaps = find_mcaps(unpack_dir)

    print(f"[collect3] Found {len(mcaps)} MCAP files")
    for mcap in mcaps:
        print(f"  {mcap}")

    cfg = load_robot_config(args.robot)
    configured_depth = cfg.get("topics", {}).get("depth")
    print(f"[collect3] Robot config: {args.robot}")
    print(f"[collect3] Configured depth topic: {configured_depth}")
    print(
        "[collect3] Note: Collect3 metadata shows compressedDepth, not the "
        "configured aligned-depth raw Image topic. If no matching depth topic is "
        "present in the MCAPs, episodes will contain zero-filled depth arrays.",
    )

    if not args.skip_stage0:
        run_stage0(
            mcap_paths=mcaps,
            robot=args.robot,
            output_dir=str(extracted_dir),
            shard_size=args.shard_size,
            overrides=None,
        )
    else:
        print(f"[collect3] Skipping Stage 0; using {extracted_dir}")

    if not args.skip_build:
        stats = build_episodes(
            data_dir=str(extracted_dir),
            out_dir=str(episodes_dir),
            episode_len=args.episode_len,
            stride=args.stride,
            img_size=args.img_size,
            min_speed=args.min_speed,
            max_sync_gap_ms=args.max_sync_gap_ms,
            idle_persistence=args.idle_persistence,
            frame_skip=args.frame_skip,
            action_mode=args.action_mode,
            image_gap_factor=args.image_gap_factor,
        )
    else:
        print(f"[collect3] Skipping Stage 1; using {episodes_dir}")
        stats = {"total_episodes": len(list(episodes_dir.glob("ep_*.npz")))}

    if not args.skip_select:
        sel_stats = select_episodes(
            data_dir=str(episodes_dir),
            out_dir=str(selected_dir),
            target_count=args.target_count,
            encoder=args.encoder,
            profile=args.profile,
        )
    else:
        print(f"[collect3] Skipping Stage 2; using {selected_dir}")
        sel_stats = {"selected": len(list(selected_dir.glob("ep_*.npz")))}

    hf_dir = None
    if not args.skip_hf_upload:
        upload_dir = episodes_dir if args.skip_select else selected_dir
        hf_dir = run_hf_upload(
            episodes_dir=str(upload_dir),
            repo_id=args.push_to_hub,
            private=args.hf_private,
        )
    else:
        print("[collect3] Skipping Stage 3 HF dataset creation/upload")

    print("[collect3] Complete")
    print(f"  Extracted data: {extracted_dir}")
    print(f"  NPZ episodes:   {episodes_dir}")
    print(f"  Episodes:       {stats.get('total_episodes', '?')}")
    print(f"  Selected:       {selected_dir}")
    print(f"  Selected count: {sel_stats.get('selected', '?')}")
    if hf_dir is not None:
        print(f"  HF dataset:     {hf_dir}")


if __name__ == "__main__":
    main()
