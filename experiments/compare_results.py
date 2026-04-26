"""Aggregate trajectory_eval.json from all experiments into a comparison table.

Usage:
    python experiments/compare_results.py --results_dir /scratch/$USER/wm_experiments_v2
"""

import argparse
import glob
import json
import os
import sys


def load_results(results_dir: str) -> list[dict]:
    """Load all trajectory_eval.json files and return a flat list of rows."""
    pattern = os.path.join(results_dir, "*", "trajectory_eval.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No trajectory_eval.json files found in {pattern}", file=sys.stderr)
        return []

    rows = []
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        experiment = data["experiment"]
        for horizon, metrics in sorted(data["results"].items()):
            rows.append(
                {
                    "experiment": experiment,
                    "horizon": horizon,
                    "ade": metrics["ade"],
                    "fde": metrics["fde"],
                    "ground_ade": metrics["ground_ade"],
                    "flight_ade": metrics["flight_ade"],
                    "n": metrics["n_episodes"],
                }
            )
    return rows


def format_table(rows: list[dict]) -> str:
    """Format rows into a markdown table."""
    header = "| Experiment                | Horizon  |      ADE |      FDE |   Ground |   Flight |     N |"
    sep = "|" + "|".join(["-" * 27, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 7]) + "|"
    lines = [header, sep]
    for r in rows:
        line = (
            f"| {r['experiment']:<25} "
            f"| {r['horizon']:<8} "
            f"| {r['ade']:8.4f} "
            f"| {r['fde']:8.4f} "
            f"| {r['ground_ade']:8.4f} "
            f"| {r['flight_ade']:8.4f} "
            f"| {r['n']:>5} |"
        )
        lines.append(line)
    return "\n".join(lines)


def best_ade_summary(rows: list[dict]) -> str:
    """Return a summary of the best ADE per horizon."""
    best: dict[str, dict] = {}
    for r in rows:
        h = r["horizon"]
        if h not in best or r["ade"] < best[h]["ade"]:
            best[h] = r
    lines = ["\nBest ADE per Horizon:"]
    for h in sorted(best):
        b = best[h]
        lines.append(f"  {h}: {b['ade']:.4f}  ({b['experiment']})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare trajectory evaluation results across experiments."
    )
    parser.add_argument(
        "--results_dir", required=True, help="Root directory containing experiment subdirectories."
    )
    parser.add_argument(
        "--output", default=None, help="Optional path to save the markdown table to a file."
    )
    args = parser.parse_args()

    rows = load_results(args.results_dir)
    if not rows:
        sys.exit(1)

    table = format_table(rows)
    summary = best_ade_summary(rows)

    print(table)
    print(summary)

    if args.output:
        with open(args.output, "w") as f:
            f.write("# World Model Experiment Comparison\n\n")
            f.write(table + "\n")
            f.write(summary + "\n")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
