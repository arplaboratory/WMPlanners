#!/usr/bin/env python3
"""Pipeline orchestrator CLI for DINO-WM / TD-JEPA data selection.

Supports two modes:
  - dino-wm (default): Window-level selection with 390-dim transition descriptors
  - td-jepa: Segment-level selection with 832-dim rich embeddings

Runs: Stage 1 (quality filter) -> Stage 2 (embed) ->
      Stage 3 (diversity selection) -> HF upload (dino-wm only).

Usage:
    # DINO-WM mode (default)
    python -m scripts.world_model.data_pipeline.run_pipeline \
        --bags dataset_out --output pipeline_out --target-count 5000

    # TD-JEPA mode
    python -m scripts.world_model.data_pipeline.run_pipeline \
        --mode td-jepa --bags dataset_out --output pipeline_out --target-frames 500000
"""

from __future__ import annotations

import argparse
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path

from .stage1_quality_filter import run_stage1
from .stage2_embed import run_stage2
from .stage3_select import run_stage3, run_stage3_tdjepa
from .hf_upload import run_hf_upload


def process_bag(
    dataset_dir: str,
    per_bag_dir: str,
    base_dir: str | None,
    device: str | None,
) -> bool:
    """Run Stages 1-2 on a single bag directory."""
    dataset_path = Path(dataset_dir)
    bag_name = dataset_path.name

    required = ["image_index.csv", "odometry.csv"]
    for fname in required:
        if not (dataset_path / fname).exists():
            print(f"[SKIP] {bag_name}: missing {fname}")
            return False

    bag_output = os.path.join(per_bag_dir, bag_name)
    os.makedirs(bag_output, exist_ok=True)

    # --- Stage 1 ---
    print(f"\n{'='*60}")
    print(f"[Stage 1] Quality filter: {bag_name}")
    print(f"{'='*60}")
    metrics_df, segments_df = run_stage1(
        dataset_dir=dataset_dir,
        output_dir=bag_output,
        base_dir=base_dir,
    )

    kept = segments_df[segments_df["keep"]] if "keep" in segments_df.columns else segments_df
    if len(kept) == 0:
        print(f"[SKIP] {bag_name}: no segments kept after Stage 1")
        return True

    # --- Stage 2 ---
    print(f"\n{'='*60}")
    print(f"[Stage 2] Embed transitions: {bag_name}")
    print(f"{'='*60}")
    result = run_stage2(
        dataset_dir=dataset_dir,
        metrics_df=metrics_df,
        segments_df=segments_df,
        output_dir=bag_output,
        base_dir=base_dir,
        device=device,
    )
    if result is None:
        print(f"[WARN] {bag_name}: Stage 2 returned None")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DINO-WM / TD-JEPA data selection pipeline",
    )
    parser.add_argument("--mode", choices=["dino-wm", "td-jepa"], default="dino-wm",
                        help="Pipeline mode: dino-wm (window selection) or td-jepa (segment selection)")
    parser.add_argument("--bags", nargs="+", required=True, help="Paths to extracted bag directories")
    parser.add_argument("--output", default="pipeline_out", help="Base output directory")
    parser.add_argument("--device", default=None, help="Torch device override")
    parser.add_argument("--skip-stage12", action="store_true", help="Skip Stages 1-2")
    parser.add_argument("--base-dir", default=None, help="Base directory for resolving image paths")

    # DINO-WM options
    parser.add_argument("--target-count", type=int, default=50_000, help="Target window count (dino-wm mode)")
    parser.add_argument("--bags-root", default=None, help="Root directory containing bag subdirs (for HF upload)")
    parser.add_argument("--window-size", type=int, default=16, help="Frames per trajectory window (default: 16)")
    parser.add_argument("--window-stride", type=int, default=16, help="Stride between windows (default: 16)")
    parser.add_argument("--push-to-hub", default=None, help="HF Hub repo ID to push dataset")
    parser.add_argument("--hf-private", action="store_true", default=True, help="Make HF dataset private")
    parser.add_argument("--skip-hf-upload", action="store_true", help="Skip HF dataset creation")

    # TD-JEPA options
    parser.add_argument("--target-frames", type=int, default=500_000, help="Target frame count (td-jepa mode)")

    args = parser.parse_args()

    per_bag_dir = os.path.join(args.output, "per_bag")
    corpus_dir = os.path.join(args.output, "corpus")
    os.makedirs(per_bag_dir, exist_ok=True)
    os.makedirs(corpus_dir, exist_ok=True)

    # --- Stages 1-2 (shared between modes) ---
    if not args.skip_stage12:
        successes = 0
        for bag_path in args.bags:
            ok = process_bag(bag_path, per_bag_dir, args.base_dir, args.device)
            if ok:
                successes += 1
        print(f"\n[Stages 1-2] Processed {successes}/{len(args.bags)} bag(s)")

    # --- Stage 3 (mode-specific) ---
    print(f"\n{'='*60}")
    print(f"[Stage 3] {args.mode.upper()} diversity selection")
    print(f"{'='*60}")

    if args.mode == "td-jepa":
        selected_df, stats = run_stage3_tdjepa(
            per_bag_dir=per_bag_dir,
            output_dir=corpus_dir,
            target_frames=args.target_frames,
        )
    else:
        selected_df, stats = run_stage3(
            per_bag_dir=per_bag_dir,
            output_dir=corpus_dir,
            target_count=args.target_count,
            window_size=args.window_size,
            window_stride=args.window_stride,
        )

    # --- HF Upload (dino-wm only) ---
    if args.mode == "dino-wm":
        bags_root = args.bags_root or (os.path.dirname(os.path.abspath(args.bags[0])) if args.bags else ".")

        if not args.skip_hf_upload and len(selected_df) > 0:
            print(f"\n{'='*60}")
            print(f"[HF Upload] Create Hugging Face dataset")
            print(f"{'='*60}")
            run_hf_upload(
                corpus_dir=corpus_dir,
                bags_root=bags_root,
                repo_id=args.push_to_hub,
                private=args.hf_private,
            )

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"[Pipeline complete] Mode: {args.mode}")
    print(f"  Selected: {len(selected_df)} {'segments' if args.mode == 'td-jepa' else 'windows'}")
    if args.mode == "td-jepa":
        print(f"  Target frames: {args.target_frames}")
    else:
        print(f"  Window size: {args.window_size} frames")
        print(f"  Target count: {args.target_count}")
    print(f"  Output: {corpus_dir}")
    if args.push_to_hub:
        print(f"  HF Hub: https://huggingface.co/datasets/{args.push_to_hub}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
