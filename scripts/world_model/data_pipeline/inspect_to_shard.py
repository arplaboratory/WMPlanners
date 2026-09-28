#!/usr/bin/env python3
"""Inspect wm_planner Rerun logs and export approved episodes to Parquet shards.

Typical manual review flow:

    python -m scripts.world_model.data_pipeline.inspect_to_shard \
        --rrd-dir outputs/rerun \
        --manifest outputs/curation/inspection_manifest.jsonl \
        --output outputs/parquet \
        --open-viewer \
        --review-only

After marking rows as "keep" or "discard" in the manifest:

    python -m scripts.world_model.data_pipeline.inspect_to_shard \
        --manifest outputs/curation/inspection_manifest.jsonl \
        --output outputs/parquet \
        --export-only
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import time_ns

from rerun_curation.exporter import (
    CurationDecision,
    ExportConfig,
    ValidationResult,
    export_manifest_to_parquet,
    mark_and_filter,
    read_curation_manifest,
    write_curation_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the inspect-to-shard sample pipeline parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rrd-dir", type=Path, help="Directory containing wm_planner .rrd recordings.")
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL keep/discard manifest.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for Parquet shards.")
    parser.add_argument("--open-viewer", action="store_true", help="Open pending recordings in Rerun.")
    parser.add_argument("--review-only", action="store_true", help="Stop after manifest generation/viewer launch.")
    parser.add_argument("--export-only", action="store_true", help="Skip scanning and export existing keep rows.")
    parser.add_argument("--keep-pending", action="store_true", help="Mark remaining pending rows as keep before export.")
    parser.add_argument("--reviewer", help="Reviewer or automation id stored on keep-pending updates.")
    parser.add_argument("--s3-prefix", help="Optional S3 prefix retained in new manifest rows.")
    parser.add_argument("--entity-prefix", action="append", default=[], help="Rerun entity prefix to export. Repeatable.")
    parser.add_argument("--sync-reference-entity", help="Entity path used as the time-alignment reference.")
    parser.add_argument("--max-sync-gap-ms", type=float, default=100.0)
    parser.add_argument("--batch-size-rows", type=int, default=2048)
    parser.add_argument("--max-shard-mb", type=int, default=256)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--timeline", default="ros_time")
    parser.add_argument(
        "--min-recording-mb",
        type=float,
        default=0.0,
        help="Discard pending recordings smaller than this size during automated validation.",
    )
    parser.add_argument(
        "--wm-context",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="World-model pipeline metadata stored in Parquet file metadata. Repeatable.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the inspect/filter/export sample pipeline."""

    args = build_arg_parser().parse_args(argv)
    if not args.export_only and args.rrd_dir is None:
        raise SystemExit("--rrd-dir is required unless --export-only is set")

    if not args.export_only:
        validators = []
        if args.min_recording_mb > 0.0:
            validators.append(_min_recording_size_validator(args.min_recording_mb))
        kept = mark_and_filter(
            args.rrd_dir,
            args.manifest,
            open_viewer=args.open_viewer,
            validators=validators,
            s3_prefix=args.s3_prefix,
            reviewer=args.reviewer,
        )
        print(f"[inspect] Manifest: {args.manifest}")
        print(f"[inspect] Currently kept recordings: {len(kept)}")

    if args.keep_pending:
        kept = _mark_pending_keep(args.manifest, reviewer=args.reviewer)
        print(f"[inspect] Marked pending recordings as keep: {len(kept)} total kept")

    if args.review_only:
        print("[inspect] Review-only mode complete; edit the manifest decisions before export.")
        return

    config = ExportConfig(
        output_dir=args.output,
        batch_size_rows=args.batch_size_rows,
        max_shard_mb=args.max_shard_mb,
        max_sync_gap_ms=args.max_sync_gap_ms,
        selected_entity_prefixes=tuple(args.entity_prefix or ["world/"]),
        compression=args.compression,
        timeline=args.timeline,
        sync_reference_entity=args.sync_reference_entity,
        wm_context=_parse_key_value_items(args.wm_context),
    )
    summary = export_manifest_to_parquet(args.manifest, args.output, config)
    print("[export] Parquet export complete")
    print(f"[export] Rows:        {summary.rows_exported}")
    print(f"[export] Recordings:  {summary.recordings_exported}/{summary.recordings_kept} kept")
    print(f"[export] Shards:      {len(summary.shards)}")
    print(f"[export] Output dir:  {summary.output_dir}")


def _min_recording_size_validator(min_mb: float):
    min_bytes = int(min_mb * 1024 * 1024)

    def validate(decision: CurationDecision) -> ValidationResult | None:
        if not decision.recording_path.exists():
            return ValidationResult(decision="discard", reason="local .rrd file is missing")
        size_bytes = decision.recording_path.stat().st_size
        if size_bytes < min_bytes:
            return ValidationResult(
                decision="discard",
                reason=f"recording is smaller than {min_mb:g} MB",
            )
        return None

    return validate


def _mark_pending_keep(manifest_path: Path, *, reviewer: str | None) -> list[CurationDecision]:
    decisions = read_curation_manifest(manifest_path)
    for decision in decisions:
        if decision.decision == "pending":
            decision.decision = "keep"
            decision.source = "inspect_to_shard"
            decision.reviewer = reviewer
            decision.updated_at_ns = time_ns()
    write_curation_manifest(manifest_path, decisions)
    return [decision for decision in decisions if decision.decision == "keep"]


def _parse_key_value_items(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--wm-context values must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--wm-context key cannot be empty: {item}")
        parsed[key] = value.strip()
    return parsed


if __name__ == "__main__":
    main()
