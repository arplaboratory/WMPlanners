"""Unified entry point for world-model training.

Usage
-----
# DinoWM (DINOv2-based predictor, default)
python train_wm.py --method dinowm --data_path lquan9/collect1 --epochs 100

# Standard WorldModel
python train_wm.py --method wm --data_path lquan9/collect1 --epochs 20 --backbone dinov2

Both underlying scripts remain independently runnable:
    python train_world_model_dinowm.py --data_path ...
    python train_world_model.py        --data_path ...
"""
from __future__ import annotations
import argparse
import sys


_METHODS = {
    "dinowm": "train_world_model_dinowm",
    "wm": "train_world_model",
}


def main():
    # ── 1. Pre-parse just --method so we know which full parser to load ──────
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--method",
        choices=list(_METHODS),
        default="wm",
        help="Training backend: 'dinowm' (DINOv2-based) or 'wm' (standard WorldModel)",
    )
    pre_args, _ = pre.parse_known_args()

    # ── 2. Import the chosen module and get its parser + Trainer ─────────────
    import importlib
    module = importlib.import_module(_METHODS[pre_args.method])
    build_parser = module.build_parser
    Trainer = module.Trainer

    # ── 3. Build the full parser, injecting --method as a documented arg ─────
    parser = build_parser()
    parser.add_argument(
        "--method",
        choices=list(_METHODS),
        default=pre_args.method,
        help="Training backend (already selected; kept here for --help clarity)",
    )

    # ── 4. Parse and dispatch ─────────────────────────────────────────────────
    args = parser.parse_args()
    Trainer(args).run()


if __name__ == "__main__":
    main()
