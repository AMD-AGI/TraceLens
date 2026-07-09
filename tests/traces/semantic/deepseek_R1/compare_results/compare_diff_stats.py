#!/usr/bin/env python3
"""Compare LCA assignments between the "with capture" (gold standard) and
"no capture" comparative analyses of the same DeepSeek-R1 decode traces.

Both analyses emit a ``diff_stats.csv`` in which every row is a single GPU
kernel (identified by ``source`` = trace1/trace2 and ``gpu_op_uid``) tagged with
a ``lowest_common_ancestor_id`` (LCA). LCA ids are opaque cluster labels: their
numeric value is only meaningful within a single analysis, so we never compare
ids across analyses directly. Instead we measure how well the two LCA
*partitions* of the shared kernel set agree, using bidirectional cluster purity.

Forward purity (gold -> no-capture):
    For every gold LCA group, take the majority no-capture LCA among its
    kernels and count how many kernels in the group carry that majority label.
    Sum across groups and divide by the number of matched kernels.

Reverse purity (no-capture -> gold): the same metric with the roles swapped.

Reporting both catches degenerate collapses: if the no-capture analysis dumped
every kernel into one LCA, forward purity stays high (each gold group trivially
agrees with the single label) but reverse purity collapses.
"""

import argparse
from pathlib import Path

import pandas as pd

KEY_COLS = ["source", "gpu_op_uid"]
LCA_COL = "lowest_common_ancestor_id"


def load(path: Path, lca_out: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in KEY_COLS + [LCA_COL, "name"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df = df.copy()
    df["key"] = df["source"].astype(str) + ":" + df["gpu_op_uid"].astype(str)
    if df["key"].duplicated().any():
        n = int(df["key"].duplicated().sum())
        raise ValueError(f"{path}: {n} duplicate (source, gpu_op_uid) keys; not a unique kernel id")
    return df[["key", "name", LCA_COL]].rename(columns={LCA_COL: lca_out})


def purity(df: pd.DataFrame, group_col: str, label_col: str):
    """Sum over each group of the majority-label count within that group.

    Returns (matched_count, total, fraction, per_group_records)."""
    total = len(df)
    matched = 0
    records = []
    for gid, grp in df.groupby(group_col, sort=True):
        counts = grp[label_col].value_counts()  # sorted desc by count
        majority_label = counts.index[0]
        majority_count = int(counts.iloc[0])
        matched += majority_count
        records.append(
            {
                "group_id": gid,
                "group_size": len(grp),
                "majority_label": majority_label,
                "majority_count": majority_count,
                "group_purity": majority_count / len(grp),
            }
        )
    frac = matched / total if total else float("nan")
    return matched, total, frac, records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("with_capture", type=Path, help="path to with-capture (GOLD) diff_stats.csv")
    ap.add_argument("no_capture", type=Path, help="path to no-capture diff_stats.csv")
    args = ap.parse_args()

    gold = load(args.with_capture, "lca_gold")
    ncap = load(args.no_capture, "lca_nc")

    print("=" * 78)
    print("LCA assignment agreement: with-capture (GOLD) vs no-capture")
    print("=" * 78)
    print(f"with-capture kernels : {len(gold)}  | distinct gold LCAs: {gold['lca_gold'].nunique()}")
    print(f"no-capture   kernels : {len(ncap)}  | distinct nc   LCAs: {ncap['lca_nc'].nunique()}")

    # Match kernels on identity (source, gpu_op_uid); keep only the intersection.
    merged = gold.merge(ncap[["key", "lca_nc", "name"]], on="key", suffixes=("_gold", "_nc"))
    name_agree = (merged["name_gold"] == merged["name_nc"]).mean() if len(merged) else float("nan")

    gold_only = len(gold) - len(merged)
    nc_only = len(ncap) - len(merged)
    print("-" * 78)
    print(f"matched kernels (in both)   : {len(merged)}")
    print(f"gold-only (unmatched)       : {gold_only}")
    print(f"no-capture-only (unmatched) : {nc_only}")
    print(f"kernel-name agreement on matched keys: {name_agree:.4f}  (sanity: should be 1.0)")

    df = merged[["key", "lca_gold", "lca_nc"]]
    print("-" * 78)
    print(f"matched-set distinct gold LCAs: {df['lca_gold'].nunique()}")
    print(f"matched-set distinct nc   LCAs: {df['lca_nc'].nunique()}")

    # Forward purity: gold groups -> majority no-capture LCA.
    f_matched, f_total, f_frac, f_recs = purity(df, "lca_gold", "lca_nc")
    # Reverse purity: no-capture groups -> majority gold LCA.
    r_matched, r_total, r_frac, r_recs = purity(df, "lca_nc", "lca_gold")

    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(
        f"Forward purity (gold group -> majority no-capture LCA): "
        f"{f_matched}/{f_total} = {f_frac:.4f}"
    )
    print(
        f"Reverse purity (no-capture group -> majority gold LCA): "
        f"{r_matched}/{r_total} = {r_frac:.4f}"
    )
    print("=" * 78)

    # Extra context: the largest no-capture groups (where collapse would show up).
    top_nc = (
        pd.DataFrame(r_recs)
        .sort_values("group_size", ascending=False)
        .head(5)
    )
    print("Largest no-capture LCA groups (matched set):")
    print(top_nc.to_string(index=False))


if __name__ == "__main__":
    main()
