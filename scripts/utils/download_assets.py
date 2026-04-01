#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import urllib.request
import yaml
from huggingface_hub import snapshot_download

ASSETS_YML = "assets.yml"

def download_asset(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        print(f"[skip] {target_path} already exists")
        return

    print(f"[download] {url} -> {target_path}")
    try:
        urllib.request.urlretrieve(url, target_path.as_posix())
        print(f"[ok] {target_path}")
    except Exception as e:
        print(f"[fail] Could not download {url}: {e}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Download project assets and Hugging Face datasets.")
    parser.add_argument("--repo", type=str, help="Hugging Face repository name (e.g., lquan9/collect1)")
    parser.add_argument("--out_dir", type=str, help="Local directory to store the downloaded assets or dataset.")
    parser.add_argument("--private", action="store_true", help="Whether the HF repository is private (requires HF_TOKEN).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    
    if args.repo:
        # Download HF dataset
        target_dir = Path(args.out_dir) if args.out_dir else repo_root / "data" / args.repo.split("/")[-1]
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[hf-download] {args.repo} -> {target_dir}")
        token = os.environ.get("HF_TOKEN")
        
        try:
            snapshot_download(
                repo_id=args.repo,
                local_dir=str(target_dir),
                repo_type="dataset",
                token=token,
                local_dir_use_symlinks=False
            )
            print(f"[ok] Dataset saved to {target_dir}")
        except Exception as e:
            print(f"[fail] Could not download dataset {args.repo}: {e}")
            
    else:
        # Download assets from assets.yml
        assets_dir = repo_root / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        assets_yml_path = assets_dir / ASSETS_YML
        if not assets_yml_path.exists():
            print(f"No assets.yml found at {assets_yml_path}. Skipping asset download.")
            return

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
