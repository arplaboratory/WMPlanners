"""Robot configuration loader — YAML files with CLI override support."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _REPO_ROOT / "configs" / "robots"


def load_robot_config(
    name_or_path: str,
    overrides: dict[str, str] | None = None,
) -> dict:
    """Load a robot config by name or file path, optionally applying overrides.

    Lookup order:
      1. configs/robots/{name_or_path}.yaml
      2. Treat name_or_path as a file path
    """
    yaml_path = _CONFIGS_DIR / f"{name_or_path}.yaml"
    if yaml_path.is_file():
        path = yaml_path
    elif Path(name_or_path).is_file():
        path = Path(name_or_path)
    else:
        raise FileNotFoundError(
            f"Robot config not found: tried {yaml_path} and {name_or_path}"
        )

    with open(path) as f:
        cfg = yaml.safe_load(f)

    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def apply_overrides(cfg: dict, overrides: dict[str, str]) -> dict:
    """Merge dot-notation overrides into a config dict."""
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        parts = key.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        # Parse value with YAML to preserve types (e.g., "6" → 6, "true" → True)
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed = value
        d[parts[-1]] = parsed
    return cfg


def add_robot_args(parser) -> None:
    """Add --robot and common topic override args to an argparse parser."""
    parser.add_argument("--robot", required=True,
                        help="Robot config name (looks in configs/robots/) or path to YAML")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides in dot.notation=value format")


def parse_overrides(override_list: list[str]) -> dict[str, str]:
    """Parse ['key=value', ...] into {'key': 'value', ...}."""
    result = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        k, v = item.split("=", 1)
        result[k] = v
    return result
