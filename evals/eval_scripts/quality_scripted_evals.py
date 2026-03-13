"""Pythonic quality evals.

Compares generated perf report CSVs against reference CSVs
with numeric tolerance.
"""

import argparse
import csv
import os
import re
import sys

import pandas as pd

CSV_COLUMNS = ["index", "category", "issue_summary", "result", "details"]
NUMERIC_TOLERANCE = 0.01
ABS_TOLERANCE = 0.05

_NUMPY_TYPE_RE = re.compile(r"np\.\w+\(([^)]+)\)")


def _normalize_numpy_reprs(s: str) -> str:
    """Strip numpy type wrappers so np.int64(135) becomes 135, etc."""
    return _NUMPY_TYPE_RE.sub(r"\1", s)


def _check_csv_alignment(output_dir: str, reference_dir: str) -> tuple[str, str]:
    gen_dir = os.path.join(output_dir, "perf_report_csvs")
    ref_dir = os.path.join(reference_dir, "perf_report_csvs")
    if not os.path.isdir(ref_dir):
        return "FAIL", "Reference perf_report_csvs/ not found"
    if not os.path.isdir(gen_dir):
        return "FAIL", "Generated perf_report_csvs/ not found"

    ref_files = {f for f in os.listdir(ref_dir) if f.endswith(".csv")}
    if not ref_files:
        return "FAIL", "No reference CSVs found"

    mismatches = []
    for fname in sorted(ref_files):
        gen_path = os.path.join(gen_dir, fname)
        if not os.path.isfile(gen_path):
            mismatches.append(f"{fname}: missing")
            continue

        ref_df = pd.read_csv(os.path.join(ref_dir, fname))
        gen_df = pd.read_csv(gen_path)

        missing_cols = set(ref_df.columns) - set(gen_df.columns)
        if missing_cols:
            mismatches.append(f"{fname}: missing required columns: {missing_cols}")
            continue
        gen_df = gen_df[ref_df.columns]
        if len(ref_df) != len(gen_df):
            mismatches.append(f"{fname}: row count {len(gen_df)} vs ref {len(ref_df)}")
            continue

        for col in ref_df.columns:
            if pd.api.types.is_numeric_dtype(ref_df[col]):
                if not ref_df[col].equals(gen_df[col]):
                    diff = (ref_df[col] - gen_df[col]).abs()
                    denom = ref_df[col].abs().replace(0, 1)
                    rel_diff = (diff / denom).max()
                    abs_diff = diff.max()
                    if rel_diff > NUMERIC_TOLERANCE and abs_diff > ABS_TOLERANCE:
                        mismatches.append(
                            f"{fname}:{col} max relative diff {rel_diff:.4f}"
                        )
            else:
                ref_norm = (
                    ref_df[col].fillna("").astype(str).map(_normalize_numpy_reprs)
                )
                gen_norm = (
                    gen_df[col].fillna("").astype(str).map(_normalize_numpy_reprs)
                )
                mask = ref_norm != gen_norm
                if mask.any():
                    rows = list(mask[mask].index[:3])
                    mismatches.append(f"{fname}:{col} differs at rows {rows}")

    if mismatches:
        return "FAIL", "; ".join(mismatches[:5])
    return "PASS", ""


EVAL_REGISTRY = [
    ("TraceLens Perf report CSVs alignment", _check_csv_alignment),
]


def run(output_dir: str, reference_dir: str, results_path: str) -> list[dict]:
    rows = []
    for i, (summary, func) in enumerate(EVAL_REGISTRY, start=1):
        result, details = func(output_dir, reference_dir)
        rows.append(
            {
                "index": f"quality_eval_{i}",
                "category": "Quality",
                "issue_summary": summary,
                "result": result,
                "details": details,
            }
        )
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Pythonic quality evals")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    rows = run(args.output_dir, args.reference_dir, args.results)
    passed = sum(1 for r in rows if r["result"] == "PASS")
    sys.exit(0 if passed == len(rows) else 1)


if __name__ == "__main__":
    main()
