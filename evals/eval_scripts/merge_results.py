"""Merge per-test-case eval CSVs into a single summary."""

import argparse
import glob
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Merge eval results")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    frames = []
    csv_pattern = os.path.join(args.results_dir, "*_results.csv")
    for csv_path in sorted(glob.glob(csv_pattern)):
        df = pd.read_csv(csv_path)
        frames.append(df)

    if not frames:
        print("No result CSVs found.")
        return

    merged = pd.concat(frames, ignore_index=True)
    expected_cols = ["index", "category", "issue_summary", "result", "details"]
    merged = merged[[c for c in expected_cols if c in merged.columns]]
    out = args.output or os.path.join(args.results_dir, "eval_summary.csv")
    merged.to_csv(out, index=False)

    total = len(merged)
    passed = (merged["result"] == "PASS").sum()
    print(f"Summary: {passed}/{total} PASS  |  Written to {out}")


if __name__ == "__main__":
    main()
