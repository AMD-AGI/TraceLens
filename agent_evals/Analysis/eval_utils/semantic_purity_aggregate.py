###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Aggregate the semantic-purity regression gate across repeated runs.

semantic_partition_scripted_evals.py records per-run LCA-partition purity
metrics but does not gate on them (see that module's docstring) -- the
semantic-bucketing method has real run-to-run variance (observed on
Qwen3-30B-A3B), so a single run is not a reliable regression signal.

This script is the actual gate. For each semantic-purity test case it:
  1. Scans results_root/<test_id>/run_*/semantic_purity_results.csv,
     parsing out each run's strict_forward value (the "strict forward
     consistency" metric from compare_lca_partitions.py: the fraction of
     gold groups that map, in their entirety, to a single candidate LCA).
     strict_forward is used as the sole gating metric -- it is more
     discriminating than forward_purity because it does not give partial
     credit to a majority-only match.
  2. Averages strict_forward across however many runs were found (intended:
     NUM_REPEATS=3; a smaller N is honored but is a weaker estimate and is
     flagged as such).
  3. Compares that average against MIN_STRICT_FORWARD_AVG[test_id] -- a
     floor derived directly from currently-observed performance (mean of
     all independently-verified runs collected during method validation),
     minus a 5% buffer, sanity-checked to stay far above the random-shuffle
     baseline (Baseline 1 in compare_lca_partitions.py, ~0.04-0.06 for
     both models). The intent is regression detection ("don't get worse
     than today"), not an absolute quality bar.

Writes one verdict row (7-column eval schema) per test id to
results_root/<test_id>/semantic_purity_aggregate_verdict.csv, and prints a
summary table.

Usage: python3 semantic_purity_aggregate.py --results-root <RESULTS_ROOT>
"""

import argparse
import csv
import glob
import os
import re
import statistics
import sys

CSV_COLUMNS = [
    "index",
    "category",
    "issue_summary",
    "result",
    "details",
    "root_cause",
    "recommended_fix",
]

# Floor = mean(observed strict_forward across all independently-verified
# runs collected during method validation) * 0.95, rounded down slightly.
# Observed data (via compare_lca_partitions.py against the committed gold
# reference):
#   deepseek_r1:        0.9748, 0.9748, 0.9780 -> mean 0.9759 -> floor 0.92
#   qwen3_30b_a3b:       0.5671, 0.4677, 0.7304, 0.5691, 0.5677
#                        -> mean 0.5804 -> floor 0.55
# Both floors sit far above the random-shuffle baseline (Baseline 1) of
# ~0.04-0.06 strict_forward for both models -- i.e. nowhere near "no better
# than chance". Revisit these floors if the method is intentionally
# improved (raise them) -- do not lower them without a documented reason.
MIN_STRICT_FORWARD_AVG = {
    "semantic_purity_deepseek_r1": 0.92,
    "semantic_purity_qwen3_30b_a3b": 0.55,
}

DETAILS_RE = re.compile(r"strict_forward=([0-9.]+)")
FORWARD_RE = re.compile(r"forward_purity=([0-9.]+)")


def find_run_csvs(results_root: str, test_id: str) -> list[str]:
    pattern = os.path.join(results_root, test_id, "run_*", "semantic_purity_results.csv")
    return sorted(glob.glob(pattern))


def parse_run(csv_path: str):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None  # self-gated no-op for this test id (shouldn't happen here)
    row = rows[0]
    details = row.get("details", "")
    m_sfwd = DETAILS_RE.search(details)
    m_fwd = FORWARD_RE.search(details)
    if not m_sfwd:
        return None  # pipeline failure row, no metrics to parse
    return {
        "strict_forward": float(m_sfwd.group(1)),
        "forward_purity": float(m_fwd.group(1)) if m_fwd else None,
        "source": csv_path,
    }


def aggregate_one(results_root: str, test_id: str) -> dict:
    run_csvs = find_run_csvs(results_root, test_id)
    parsed = [p for p in (parse_run(c) for c in run_csvs) if p is not None]

    if not parsed:
        return {
            "index": "semantic_purity_aggregate",
            "category": "Quality",
            "issue_summary": f"Semantic-purity regression gate ({test_id})",
            "result": "FAIL",
            "details": f"No usable runs found under {results_root}/{test_id}/run_*/semantic_purity_results.csv",
            "root_cause": "pipeline",
            "recommended_fix": "Ensure Phase 1+2 completed for this test id before aggregating",
        }

    values = [p["strict_forward"] for p in parsed]
    avg = statistics.mean(values)
    floor = MIN_STRICT_FORWARD_AVG.get(test_id)
    if floor is None:
        return {
            "index": "semantic_purity_aggregate",
            "category": "Quality",
            "issue_summary": f"Semantic-purity regression gate ({test_id})",
            "result": "FAIL",
            "details": f"No floor configured for test id {test_id}; add one to MIN_STRICT_FORWARD_AVG",
            "root_cause": "config",
            "recommended_fix": "Add this test id to MIN_STRICT_FORWARD_AVG in semantic_purity_aggregate.py",
        }

    passed = avg >= floor
    n_note = "" if len(parsed) >= 3 else f" (WARNING: only {len(parsed)} run(s) found; intended N=3, this is a weaker estimate)"
    details = (
        f"n_runs={len(parsed)} strict_forward_values={[round(v, 4) for v in values]} "
        f"avg_strict_forward={avg:.4f} floor={floor}{n_note}"
    )
    return {
        "index": "semantic_purity_aggregate",
        "category": "Quality",
        "issue_summary": f"Semantic-purity regression gate ({test_id})",
        "result": "PASS" if passed else "FAIL",
        "details": details,
        "root_cause": "" if passed else "quality",
        "recommended_fix": "" if passed else "Semantic-bucketing quality regressed below the observed-performance floor; investigate the clustering change",
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate semantic-purity gate across repeats")
    parser.add_argument("--results-root", required=True)
    parser.add_argument(
        "--test-ids",
        default=",".join(MIN_STRICT_FORWARD_AVG.keys()),
        help="Comma-separated test ids to aggregate (default: all known semantic-purity test ids)",
    )
    args = parser.parse_args()

    test_ids = [t.strip() for t in args.test_ids.split(",") if t.strip()]
    overall_pass = True
    for test_id in test_ids:
        case_dir = os.path.join(args.results_root, test_id)
        if not os.path.isdir(case_dir):
            print(f"[{test_id}] skipped (no results directory found)")
            continue
        result = aggregate_one(args.results_root, test_id)
        out_path = os.path.join(case_dir, "semantic_purity_aggregate_verdict.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerow(result)
        print(f"[{test_id}] {result['result']}: {result['details']}")
        print(f"  -> {out_path}")
        if result["result"] != "PASS":
            overall_pass = False

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
