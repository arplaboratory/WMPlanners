#!/usr/bin/env python3
import os
from pathlib import Path
import urllib.request
import yaml

ASSETS_YML = "assets.yml"

def download_asset(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        print(f"[skip] {target_path} already exists")
        return

    print(f"[download] {url} -> {target_path}")
    urllib.request.urlretrieve(url, target_path.as_posix())
    print(f"[ok] {target_path}")

def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    assets_dir = repo_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    assets_yml_path = assets_dir / ASSETS_YML
    with open(assets_yml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assets = config.get("assets", [])
    if not assets:
        print("No assets found in the configuration.")
        return
    for asset in assets:
        name = asset.get("name")
        url = asset.get("url")

        if not name or not url:
            print(f"Invalid asset configuration: {asset}")
            continue

        target_path = assets_dir / name
        download_asset(url, target_path)

if __name__ == "__main__":
    main()
