#!/usr/bin/env python3
"""Random baselines for the no-capture LCA partition, evaluated with the same
bidirectional purity metric used in ``compare_diff_stats.py``.

Both real and baseline purities are computed over the SAME 636 matched kernels
(kernels present in both the with-capture/GOLD and no-capture analyses, matched
by identity ``(source, gpu_op_uid)``). The baselines preserve the no-capture
bucket sizes *as observed on the matched set*, so the total number of matched
kernels (636) and each bucket's size are identical to the real no-capture
partition -- only the kernel->bucket assignment changes.

Baseline 1 (random): randomly shuffle which matched kernels fall into each
    bucket, preserving bucket sizes. Reported as mean +/- std over many seeds.

Baseline 2 (sequential blocks, key string order): order buckets from most to
    least popular, then walk the matched kernels ordered by the lexicographic key
    string ("source:uid"), assigning the first k1 to bucket 1 (largest), the next
    k2 to bucket 2, and so on.

Baseline 3 (sequential blocks, integer uid order): identical block assignment,
    but kernels are ordered by (source, integer gpu_op_uid). This respects the
    numeric ordering of gpu_op_uid (e.g. 999 before 1000) that the string sort in
    Baseline 2 breaks.

Purpose: show what forward/reverse purity a semantics-free assignment achieves,
so the real no-capture purity can be judged against that floor.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LCA_ID = "lowest_common_ancestor_id"
LCA_NAME = "lowest_common_ancestor_name"
N_TRIALS = 2000
SEED = 0


def load(path: Path):
    df = pd.read_csv(path)
    df = df.copy()
    df["key"] = df["source"].astype(str) + ":" + df["gpu_op_uid"].astype(str)
    return df


def purity_frac(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Fraction of items whose A-group carries the A-group's majority B-label.

    Sum over groups of A of the modal B-count, divided by the total item count.
    """
    df = pd.DataFrame({"a": labels_a, "b": labels_b})
    matched = 0
    for _, grp in df.groupby("a", sort=False):
        matched += int(grp["b"].value_counts().iloc[0])
    return matched / len(df)


def both_purities(gold: np.ndarray, nc: np.ndarray):
    """Return (forward, reverse).

    forward = gold group -> majority nc label
    reverse = nc   group -> majority gold label
    """
    return purity_frac(gold, nc), purity_frac(nc, gold)


def run(merged: pd.DataFrame, title: str) -> None:
    """Compute real + baseline purities for a given matched-kernel frame.

    ``merged`` must have columns: lca_gold, lca_nc, nc_name, source, gpu_op_uid.
    Bucket sizes are recomputed from ``merged`` so the "total matched preserved"
    property holds for whatever subset is passed in.
    """
    merged = merged.sort_values("key").reset_index(drop=True)
    n = len(merged)

    gold = merged["lca_gold"].to_numpy()
    nc_real = merged["lca_nc"].to_numpy()

    # No-capture bucket sizes on this (sub)set (sum == n).
    sizes = merged["lca_nc"].value_counts()  # sorted desc by count
    bucket_ids = sizes.index.to_numpy()
    bucket_sizes = sizes.to_numpy()
    names = (
        merged.drop_duplicates("lca_nc").set_index("lca_nc")["nc_name"].to_dict()
    )
    assert bucket_sizes.sum() == n, (bucket_sizes.sum(), n)

    # Label pool: bucket id repeated by its size (length == n).
    pool = np.repeat(bucket_ids, bucket_sizes)
    assert len(pool) == n

    # ---- Real no-capture ----
    real_fwd, real_rev = both_purities(gold, nc_real)

    # ---- Baseline 1: random shuffle of the pool over matched kernels ----
    rng = np.random.default_rng(SEED)
    fwds = np.empty(N_TRIALS)
    revs = np.empty(N_TRIALS)
    for t in range(N_TRIALS):
        shuffled = pool.copy()
        rng.shuffle(shuffled)
        fwds[t], revs[t] = both_purities(gold, shuffled)
    b1_fwd_m, b1_fwd_s = fwds.mean(), fwds.std()
    b1_rev_m, b1_rev_s = revs.mean(), revs.std()

    # ---- Baseline 2: sequential blocks, buckets ordered most->least popular ----
    # Kernels ordered by key (lexicographic string: source then string uid).
    seq = pool.copy()
    b2_fwd, b2_rev = both_purities(gold, seq)

    # ---- Baseline 3: sequential blocks, kernels ordered by INT gpu_op_uid ----
    order3 = np.lexsort((merged["gpu_op_uid"].to_numpy(), merged["source"].to_numpy()))
    gold3 = gold[order3]
    b3_fwd, b3_rev = both_purities(gold3, pool)

    # ---------------- report ----------------
    print("=" * 74)
    print(f"{title}")
    print("Matched kernels:", n, "| no-capture buckets:", len(bucket_ids))
    print("Distinct gold LCAs:", merged["lca_gold"].nunique())
    print("=" * 74)
    print("No-capture bucket sizes (preserved by all baselines):")
    for bid, sz in zip(bucket_ids, bucket_sizes):
        print(f"  LCA {bid:>3}  {names.get(bid, '?'):<18} {sz:>4}")
    print("-" * 74)
    hdr = f"{'partition':<26}{'forward (gold->nc)':>24}{'reverse (nc->gold)':>24}"
    print(hdr)
    print("-" * 74)
    print(f"{'Real no-capture':<26}{real_fwd:>24.4f}{real_rev:>24.4f}")
    print(
        f"{'Baseline 1 (random)':<26}"
        f"{f'{b1_fwd_m:.4f} +/- {b1_fwd_s:.4f}':>24}"
        f"{f'{b1_rev_m:.4f} +/- {b1_rev_s:.4f}':>24}"
    )
    print(f"{'Baseline 2 (seq, key str)':<26}{b2_fwd:>24.4f}{b2_rev:>24.4f}")
    print(f"{'Baseline 3 (seq, int uid)':<26}{b3_fwd:>24.4f}{b3_rev:>24.4f}")
    print("=" * 74)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("with_capture", type=Path, help="path to with-capture (GOLD) diff_stats.csv")
    ap.add_argument("no_capture", type=Path, help="path to no-capture diff_stats.csv")
    args = ap.parse_args()

    gold_df = load(args.with_capture)
    nc_df = load(args.no_capture)

    # Matched set: identity present in both. Deterministic order by key.
    merged = gold_df[["key", LCA_ID]].rename(columns={LCA_ID: "lca_gold"}).merge(
        nc_df[["key", "source", "gpu_op_uid", LCA_ID, LCA_NAME]].rename(
            columns={LCA_ID: "lca_nc", LCA_NAME: "nc_name"}
        ),
        on="key",
    )

    run(merged, "ALL MATCHED KERNELS")

    # Exclude singleton gold groups: gold LCAs with exactly one matched kernel.
    gsz = merged.groupby("lca_gold")["key"].transform("size")
    merged_ns = merged[gsz > 1].copy()
    n_dropped = len(merged) - len(merged_ns)
    print()
    run(
        merged_ns,
        f"EXCLUDING SINGLETON GOLD GROUPS (dropped {n_dropped} kernels)",
    )
    print(f"(Baseline 1 over {N_TRIALS} random seeds; Baselines 2 & 3 deterministic.)")


if __name__ == "__main__":
    main()
