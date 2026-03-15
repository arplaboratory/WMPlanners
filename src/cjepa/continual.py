"""Core"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ContinualExperimentConfig:
    """Configuration for one training phase.

    The defaults target a lightweight local run that can be used as a smoke test.
    """

    config_path: str = "configs/train/vitl16/pretrain-256px-16f.yaml"
    devices: str = "cuda:0"
    root_path: str = "datasets/task1"
    train_csv: str = "datasets/task1/train.csv"
    val_csv: str = "datasets/task1/val.csv"
    task_name: str = "task1_pretrain"


def write_dataset_csv(
    entries: Sequence[tuple[str, int]], destination: str | Path
) -> Path:
    """JEPA compatible CSV file from `(video_path, label)` rows."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{video_path} {label}" for video_path, label in entries]
    csv_content = "\n".join(lines) + ("\n" if lines else "")
    destination_path.write_text(csv_content, encoding="utf-8")
    return destination_path


def build_pretrain_command(config: ContinualExperimentConfig) -> list[str]:
    """Launch command for vjepa main."""

    return [
        "python",
        "-m",
        "app.main",
        "--fname",
        config.config_path,
        "--devices",
        config.devices,
        "--root_path",
        config.root_path,
        "--train_csv",
        config.train_csv,
        "--val_csv",
        config.val_csv,
        "--tag",
        config.task_name,
    ]


def format_command(argv: Iterable[str]) -> str:
    """Output command"""

    return " ".join(argv)
