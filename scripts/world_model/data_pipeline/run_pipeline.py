#!/usr/bin/env python3
"""End-to-end data pipeline: MCAP or ROS1 bags → NPZ episodes → HuggingFace dataset.

Stages:
  0. Extract  — MCAP or ROS1 bags → sharded images + CSVs  (skip with --bags)
  1. Build    — images + CSVs → fixed-length NPZ episodes
  2. Select   — FPS diversity selection on episodes
  3. Upload   — push selected episodes to Hugging Face Hub

Usage:
    # Full pipeline from MCAP
    python -m scripts.world_model.data_pipeline.run_pipeline \\
        --mcap path/to/bag.mcap --robot race16 \\
        --frame-skip 6 --episode-len 50 --target-count 5000 \\
        --push-to-hub dmody1/wm-data

    # From pre-extracted bags (skip Stage 0)
    python -m scripts.world_model.data_pipeline.run_pipeline \\
        --bags dataset_out \\
        --frame-skip 6 --episode-len 50 --target-count 5000
"""

from __future__ import annotations

import argparse
import os

from scripts.world_model.build_episodes import build_episodes
from scripts.world_model.select_episodes import select_episodes

from .hf_upload import run_hf_upload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end data pipeline: MCAP → episodes → HuggingFace",
    )

    # --- Source (one of --mcap or --bags required) ---
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--mcap", nargs="+", help="MCAP or ROS1 bag file path(s) — runs extraction (Stage 0)")
    src.add_argument("--bags", help="Pre-extracted bags directory (has manifest.json) — skips Stage 0")

    # --- Stage 0: Extract options ---
    parser.add_argument("--robot", default="race16", help="Robot config name (default: race16)")
    parser.add_argument("--shard-size", type=int, default=500, help="Images per shard (default: 500)")
    parser.add_argument("--override", nargs="*", default=[], help="Robot config overrides (key=value)")

    # --- Stage 1: Build episodes options ---
    parser.add_argument("--episode-len", type=int, default=50, help="Frames per episode (default: 50)")
    parser.add_argument("--stride", type=int, default=50, help="Episode stride in raw frames (default: 50)")
    parser.add_argument("--frame-skip", type=int, default=6, help="Take every Nth frame for temporal subsampling (default: 6)")
    parser.add_argument("--img-size", type=int, default=128, help="Resize images to NxN (default: 128)")
    parser.add_argument("--min-speed", type=float, default=0.1, help="Idle speed threshold m/s (default: 0.1)")
    parser.add_argument("--max-sync-gap-ms", type=float, default=100.0, help="Max sensor sync gap in ms (default: 100)")
    parser.add_argument("--idle-persistence", type=int, default=10, help="Consecutive idle frames to mark (default: 10)")
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

    # --- Stage 2: Select episodes options ---
    parser.add_argument("--target-count", type=int, default=5000, help="Target number of episodes to select (default: 5000)")
    parser.add_argument("--encoder", default="none", help="Visual encoder for selection descriptors (default: none)")
    parser.add_argument(
        "--selection-profile",
        default="generic",
        choices=["generic", "spot", "rover"],
        help="Episode selection profile (default: generic)",
    )

    # --- Stage 3: HF upload options ---
    parser.add_argument("--push-to-hub", default=None, help="HF Hub repo ID (e.g. dmody1/wm-data)")
    parser.add_argument("--hf-private", action="store_true", default=False, help="Make HF dataset private")

    # --- General ---
    parser.add_argument("--output", default="pipeline_out", help="Base output directory (default: pipeline_out)")
    parser.add_argument("--skip-build", action="store_true", help="Skip Stage 1 (reuse existing episodes)")
    parser.add_argument("--skip-select", action="store_true", help="Skip Stage 2 (reuse existing selection)")
    parser.add_argument("--skip-hf-upload", action="store_true", help="Skip Stage 3 (no HF upload)")

    args = parser.parse_args()

    extracted_dir = os.path.join(args.output, "extracted")
    episodes_dir = os.path.join(args.output, "episodes")
    selected_dir = os.path.join(args.output, "selected")

    # ── Stage 0: Extract bags ─────────────────────────────────────────────
    if args.mcap:
        print(f"\n{'='*60}")
        print("[Stage 0] Extract bags (MCAP or ROS1)")
        print(f"{'='*60}")
        from .stage0_extract import run_stage0
        from scripts.world_model.robot_config import parse_overrides

        overrides = parse_overrides(args.override)
        run_stage0(
            mcap_paths=args.mcap,
            robot=args.robot,
            output_dir=extracted_dir,
            shard_size=args.shard_size,
            overrides=overrides,
        )
        data_dir = extracted_dir
    else:
        data_dir = args.bags

    # ── Stage 1: Build episodes ───────────────────────────────────────────
    if not args.skip_build:
        print(f"\n{'='*60}")
        print("[Stage 1] Build episodes")
        print(f"{'='*60}")
        stats = build_episodes(
            data_dir=data_dir,
            out_dir=episodes_dir,
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
        print(f"[Stage 1] Built {stats.get('total_episodes', '?')} episodes")

    # ── Stage 2: Select diverse episodes ──────────────────────────────────
    if not args.skip_select:
        print(f"\n{'='*60}")
        print("[Stage 2] Select diverse episodes (FPS)")
        print(f"{'='*60}")
        sel_stats = select_episodes(
            data_dir=episodes_dir,
            out_dir=selected_dir,
            target_count=args.target_count,
            encoder=args.encoder,
            profile=args.selection_profile,
        )
        print(f"[Stage 2] Selected {sel_stats.get('selected', '?')} episodes")

    # ── Stage 3: Upload to Hugging Face ───────────────────────────────────
    if not args.skip_hf_upload:
        print(f"\n{'='*60}")
        print("[Stage 3] Upload to Hugging Face")
        print(f"{'='*60}")
        upload_dir = episodes_dir if args.skip_select else selected_dir
        run_hf_upload(
            episodes_dir=upload_dir,
            repo_id=args.push_to_hub,
            private=args.hf_private,
        )

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[Pipeline complete]")
    print(f"  Extracted:  {data_dir}")
    print(f"  Episodes:   {episodes_dir}")
    print(f"  Selected:   {selected_dir if not args.skip_select else '<skipped>'}")
    if args.push_to_hub:
        print(f"  HF Hub:     https://huggingface.co/datasets/{args.push_to_hub}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
