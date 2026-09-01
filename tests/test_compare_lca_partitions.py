###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fast, no-agent unit tests for agent_evals/Analysis/eval_utils/compare_lca_partitions.py.

Uses small synthetic diff_stats.csv fixtures (no traces, no LLM) to pin down
the purity/strict-consistency metric semantics.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "agent_evals",
        "Analysis",
        "eval_utils",
    ),
)
from compare_lca_partitions import both_purities, both_strict, load  # noqa: E402

COLUMNS = [
    "source",
    "gpu_op_uid",
    "name",
    "lowest_common_ancestor_id",
    "lowest_common_ancestor_name",
]


def _write_csv(path, rows):
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)


def _row(source, uid, name, lca_id):
    return [source, uid, name, lca_id, f"lca_{lca_id}"]


def test_load_requires_key_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load(path)


def test_load_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "dup.csv"
    _write_csv(
        path,
        [
            _row("trace1", 1, "kernelA", 0),
            _row("trace1", 1, "kernelA", 0),
        ],
    )
    with pytest.raises(ValueError, match="duplicate"):
        load(path)


def test_perfect_agreement_gives_purity_one():
    gold = ["g0", "g0", "g1", "g1"]
    cand = ["c0", "c0", "c1", "c1"]
    fwd, rev = both_purities(gold, cand)
    sfwd, srev = both_strict(gold, cand)
    assert fwd == 1.0
    assert rev == 1.0
    assert sfwd == 1.0
    assert srev == 1.0


def test_candidate_collapse_hurts_reverse_purity_not_forward():
    # Every gold group maps entirely into one giant candidate bucket: forward
    # purity is trivially perfect (each gold group agrees with the sole label
    # it sees), but reverse purity collapses because that one candidate bucket
    # disagrees with most of the gold labels it swallowed.
    gold = ["g0", "g0", "g1", "g1", "g2", "g2"]
    cand = ["c0"] * 6
    fwd, rev = both_purities(gold, cand)
    assert fwd == 1.0
    assert rev == pytest.approx(2 / 6)  # majority gold label (any of the 3) covers 2/6


def test_partial_split_purity_between_zero_and_one():
    # gold group g0 (4 items) splits 3-1 across two candidate buckets.
    gold = ["g0", "g0", "g0", "g0", "g1", "g1"]
    cand = ["c0", "c0", "c0", "c1", "c2", "c2"]
    fwd, rev = both_purities(gold, cand)
    sfwd, srev = both_strict(gold, cand)
    # forward: g0 majority=3/4, g1 majority=2/2 -> (3+2)/6
    assert fwd == pytest.approx(5 / 6)
    # strict forward: g0 not fully pure (0 credit), g1 fully pure (2 credit) -> 2/6
    assert sfwd == pytest.approx(2 / 6)
    assert sfwd <= fwd
    assert srev <= rev


def test_strict_never_exceeds_matching_purity_on_random_like_partition():
    gold = ["g0"] * 5 + ["g1"] * 3 + ["g2"] * 2
    cand = ["c0", "c0", "c1", "c1", "c1", "c2", "c0", "c1", "c2", "c2"]
    fwd, rev = both_purities(gold, cand)
    sfwd, srev = both_strict(gold, cand)
    assert sfwd <= fwd
    assert srev <= rev


def test_end_to_end_via_csv_files(tmp_path):
    gold_path = tmp_path / "gold.csv"
    cand_path = tmp_path / "cand.csv"
    _write_csv(
        gold_path,
        [
            _row("trace1", 1, "kernelA", 0),
            _row("trace1", 2, "kernelB", 0),
            _row("trace2", 1, "kernelC", 1),
            _row("trace2", 2, "kernelD", 1),
        ],
    )
    _write_csv(
        cand_path,
        [
            _row("trace1", 1, "kernelA", 10),
            _row("trace1", 2, "kernelB", 10),
            _row("trace2", 1, "kernelC", 11),
            _row("trace2", 2, "kernelD", 11),
        ],
    )
    gold = load(gold_path)
    cand = load(cand_path)
    merged = (
        gold[["key", "lowest_common_ancestor_id"]]
        .rename(columns={"lowest_common_ancestor_id": "lca_gold"})
        .merge(
            cand[["key", "lowest_common_ancestor_id"]].rename(
                columns={"lowest_common_ancestor_id": "lca_cand"}
            ),
            on="key",
        )
    )
    assert len(merged) == 4
    fwd, rev = both_purities(
        merged["lca_gold"].to_numpy(), merged["lca_cand"].to_numpy()
    )
    assert fwd == 1.0 and rev == 1.0
