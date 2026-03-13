"""Stage 4 -- Create a Hugging Face dataset from selected windows for DINO-WM training."""

from __future__ import annotations

import json
import os
from typing import Generator

from PIL import Image as PILImage

from .utils import load_csv, resolve_path


def _iter_rows(
    windows: list[dict],
    bags_root: str,
    image_index_cache: dict[str, dict[int, str]],
) -> Generator[dict, None, None]:
    """Yield one dict per frame across all selected windows.

    Each window dict is expected to have (at minimum):
        bag_name, window_id, frame_ids, timestamps, odom_deltas

    ``odom_deltas`` has length ``len(frame_ids) - 1``; the last frame in
    each window is padded with zeros.
    """
    for win in windows:
        bag_name: str = win["bag_name"]
        window_id: int = win["window_id"]
        frame_ids: list[int] = win["frame_ids"]
        timestamps: list[int] = win["timestamps"]
        odom_deltas: list[list[float]] = win.get("odom_deltas", [])
        n_frames = len(frame_ids)

        # Resolve colour image paths via image_index.csv
        if bag_name not in image_index_cache:
            idx_path = os.path.join(bags_root, bag_name, "image_index.csv")
            if os.path.isfile(idx_path):
                df = load_csv(idx_path)
                image_index_cache[bag_name] = dict(
                    zip(df["frame_id"].tolist(), df["color_path"].tolist())
                )
            else:
                image_index_cache[bag_name] = {}

        fid_to_color = image_index_cache[bag_name]
        bag_dir = os.path.join(bags_root, bag_name)

        for i, fid in enumerate(frame_ids):
            # Odom delta: use window odom_deltas for frames 0..n-2, zeros for last
            if i < len(odom_deltas):
                dx, dy, dz, ax, ay, az = odom_deltas[i]
            else:
                dx = dy = dz = ax = ay = az = 0.0

            # Resolve image path
            color_rel = fid_to_color.get(fid, "")
            color_abs = resolve_path(color_rel, bag_dir) if color_rel else ""

            # Load image (or None if missing)
            if color_abs and os.path.isfile(color_abs):
                image = PILImage.open(color_abs).convert("RGB")
            else:
                image = None

            yield {
                "bag_name": bag_name,
                "window_id": window_id,
                "frame_idx_in_window": i,
                "frame_id": fid,
                "timestamp_ns": timestamps[i],
                "odom_dx": float(dx),
                "odom_dy": float(dy),
                "odom_dz": float(dz),
                "odom_ax": float(ax),
                "odom_ay": float(ay),
                "odom_az": float(az),
                "image": image,
            }


def run_hf_upload(
    corpus_dir: str,
    bags_root: str,
    repo_id: str | None = None,
    private: bool = True,
) -> str:
    """Create HF dataset from selected windows and optionally push to Hub.

    Parameters
    ----------
    corpus_dir : str
        Directory containing ``selected_windows_full.json`` (Stage 3 output).
    bags_root : str
        Root directory containing per-bag folders, each with ``image_index.csv``.
    repo_id : str | None
        If provided, push the dataset to this Hugging Face Hub repository.
    private : bool
        Whether the Hub repository should be private (default True).

    Returns
    -------
    str
        Path to the saved dataset directory (``{corpus_dir}/hf_dataset/``).
    """
    from datasets import Dataset, Features, Image, Value

    json_path = os.path.join(corpus_dir, "selected_windows_full.json")
    with open(json_path) as f:
        windows: list[dict] = json.load(f)

    save_dir = os.path.join(corpus_dir, "hf_dataset")

    if not windows:
        # Empty selection: create an empty dataset with the correct schema.
        # Use Value("string") for image column to avoid a datasets bug
        # with empty Image() columns on save_to_disk.
        empty_features = Features({
            "bag_name": Value("string"),
            "window_id": Value("int64"),
            "frame_idx_in_window": Value("int64"),
            "frame_id": Value("int64"),
            "timestamp_ns": Value("int64"),
            "odom_dx": Value("float64"),
            "odom_dy": Value("float64"),
            "odom_dz": Value("float64"),
            "odom_ax": Value("float64"),
            "odom_ay": Value("float64"),
            "odom_az": Value("float64"),
            "image": Value("string"),
        })
        ds = Dataset.from_dict(
            {col: [] for col in empty_features},
            features=empty_features,
        )
        ds.save_to_disk(save_dir)
        print(f"[hf_upload] Empty selection — saved empty dataset to {save_dir}")
        return save_dir

    # Collect all rows via generator (materialise into lists for from_dict)
    image_index_cache: dict[str, dict[int, str]] = {}
    rows = list(_iter_rows(windows, bags_root, image_index_cache))

    # Transpose list-of-dicts into dict-of-lists
    columns: dict[str, list] = {key: [] for key in rows[0]}
    for row in rows:
        for key in columns:
            columns[key].append(row[key])

    features = Features({
        "bag_name": Value("string"),
        "window_id": Value("int64"),
        "frame_idx_in_window": Value("int64"),
        "frame_id": Value("int64"),
        "timestamp_ns": Value("int64"),
        "odom_dx": Value("float64"),
        "odom_dy": Value("float64"),
        "odom_dz": Value("float64"),
        "odom_ax": Value("float64"),
        "odom_ay": Value("float64"),
        "odom_az": Value("float64"),
        "image": Image(),
    })

    ds = Dataset.from_dict(columns, features=features)
    ds.save_to_disk(save_dir)
    print(f"[hf_upload] Saved {len(ds)} rows ({len(windows)} windows) to {save_dir}")

    if repo_id:
        ds.push_to_hub(repo_id, private=private)
        print(f"[hf_upload] Pushed to https://huggingface.co/datasets/{repo_id}")

    return save_dir
