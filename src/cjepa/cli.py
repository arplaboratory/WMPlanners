"""CLI entry points"""

from __future__ import annotations

import argparse

from .continual import ContinualExperimentConfig, build_pretrain_command, format_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print a V-JEPA2 launch command"
    )
    parser.add_argument(
        "--config", default="configs/train/vitl16/pretrain-256px-16f.yaml"
    )
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--root-path", default="datasets/task1")
    parser.add_argument("--train-csv", default="datasets/task1/train.csv")
    parser.add_argument("--val-csv", default="datasets/task1/val.csv")
    parser.add_argument("--task-name", default="task1_pretrain")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ContinualExperimentConfig(
        config_path=args.config,
        devices=args.devices,
        root_path=args.root_path,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        task_name=args.task_name,
    )
    print(format_command(build_pretrain_command(config)))


if __name__ == "__main__":
    main()
