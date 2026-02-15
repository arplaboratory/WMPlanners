#!/usr/bin/env python3
"""Curate dataset manifests from video.

Output format:
    /absolute/or/relative/path/to/video.mp4 <label>
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Root directory to scan recursively.")
    parser.add_argument("--output-manifest", required=True, help="Where to write the output manifest.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".mp4", ".avi", ".mov", ".mkv"],
        help="File extensions to include.",
    )
    parser.add_argument("--label", type=int, default=0, help="Label value for every discovered sample.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle files before writing output.")
    parser.add_argument("--limit", type=int, help="Optional max number of files to include.")
    parser.add_argument(
        "--relative-to",
        default="",
        help="If set, write paths relative to this directory instead of absolute paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_manifest = Path(args.output_manifest).expanduser().resolve()
    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions}

    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist or is not a directory: {input_root}")

    files = [p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in extensions]
    files.sort()

    if args.shuffle:
        random.shuffle(files)

    if args.limit is not None:
        files = files[: args.limit]

    relative_to = Path(args.relative_to).expanduser().resolve() if args.relative_to else None

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as f:
        for path in files:
            out_path = path
            if relative_to is not None:
                out_path = path.relative_to(relative_to)
            f.write(f"{out_path} {args.label}\n")

    print(f"Wrote {len(files)} samples -> {output_manifest}")


if __name__ == "__main__":
    main()
