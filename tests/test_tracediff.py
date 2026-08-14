###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

<<<<<<< Updated upstream
=======
import os
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
>>>>>>> Stashed changes
from TraceLens.TraceDiff.trace_diff import (
    _disambiguate_same_name_candidates,
)


def _make_uid2node(entries):
    """entries: list of (uid, name, children_uids, non_gpu_path=False)"""
    return {
        uid: {"name": name, "children": children, "non_gpu_path": gpu_off}
        for uid, name, children, gpu_off in entries
    }


def _match_pairs(ops):
    return {(i, j) for op, i, j in ops if op == "match"}


def _delete_idxs(ops):
    return {i for op, i, j in ops if op == "delete"}


def _insert_idxs(ops):
    return {j for op, i, j in ops if op == "insert"}


# ---------------------------------------------------------------------------
# _disambiguate_same_name_candidates unit tests
# ---------------------------------------------------------------------------


class TestDisambiguateSameNameCandidates:

    def test_no_ambiguity_returns_ops_unchanged(self):
        """No extra same-named nodes → ops returned as-is."""
        uid2node = _make_uid2node(
            [
                ("u0", "opA", [], False),
                ("u1", "opB", [], False),
                ("v0", "opA", [], False),
                ("v1", "opC", [], False),
            ]
        )
        children1 = ["u0", "u1"]
        children2 = ["v0", "v1"]
        ops = [("match", 0, 0), ("match", 1, 1)]
        result = _disambiguate_same_name_candidates(
            ops, children1, children2, uid2node, uid2node
        )
        assert result == ops

    def test_unique_survivor_trace2_triggers_reassignment(self):
        """WF matched trace1[0] with trace2[0], but trace2[1] has the same name
        and identical subtree children → trace2[1] should win because it better
        matches trace1[0]'s children."""
        uid2node = _make_uid2node(
            [
                # trace1[0]=u0 has child "kernel_X"
                ("u0", "opA", ["u0c"], False),
                ("u0c", "kernel_X", [], False),
                # trace2[0]=v0 has child "kernel_Y" (different → evicted)
                ("v0", "opA", ["v0c"], False),
                ("v0c", "kernel_Y", [], False),
                # trace2[1]=v1 has child "kernel_X" (same → survives)
                ("v1", "opA", ["v1c"], False),
                ("v1c", "kernel_X", [], False),
            ]
        )
        children1 = ["u0"]
        children2 = ["v0", "v1"]
        # WF matched u0↔v0, v1 is an insert
        ops = [("match", 0, 0), ("insert", None, 1)]
        result = _disambiguate_same_name_candidates(
            ops, children1, children2, uid2node, uid2node
        )
        pairs = _match_pairs(result)
        assert (0, 1) in pairs, "Should reassign to trace2[1] (better subtree match)"
        assert (0, 0) not in pairs

    def test_multiple_survivors_keeps_original_wf_match(self):
        """When >1 survivors remain, WF's choice is kept."""
        uid2node = _make_uid2node(
            [
                ("u0", "opA", [], False),
                ("v0", "opA", [], False),
                ("v1", "opA", [], False),
            ]
        )
        children1 = ["u0"]
        children2 = ["v0", "v1"]
        ops = [("match", 0, 0), ("insert", None, 1)]
        result = _disambiguate_same_name_candidates(
            ops, children1, children2, uid2node, uid2node
        )
        pairs = _match_pairs(result)
        assert (
            0,
            0,
        ) in pairs, "Original WF match should be preserved when tie can't be broken"

    def test_no_duplicate_trace2_index_in_matches(self):
        """All three trace1 events score highest with the same trace2 event (v0);
        only one can claim it — the others must fall back to different partners.

        Subtrees:
          u0, u1, u2: [kernel_X, kernel_Y]   → level-1 = (kernel_x, kernel_y)
          v0:         [kernel_X, kernel_Y]   → level-1 = (kernel_x, kernel_y)  ← best for all
          v1, v2, v3: [kernel_X]             → level-1 = (kernel_x,)           ← partial match

        WF positionally matched u0↔v1, u1↔v2, u2↔v3 (v0 is an insert).
        All three trace1 events prefer v0 (score (1,0) vs (0,1)), but only
        one can be assigned it.  The other two must take different partners,
        so no trace2 index can appear more than once.
        """
        uid2node = _make_uid2node(
            [
                ("u0", "opA", ["u0c1", "u0c2"], False),
                ("u0c1", "kernel_X", [], False),
                ("u0c2", "kernel_Y", [], False),
                ("u1", "opA", ["u1c1", "u1c2"], False),
                ("u1c1", "kernel_X", [], False),
                ("u1c2", "kernel_Y", [], False),
                ("u2", "opA", ["u2c1", "u2c2"], False),
                ("u2c1", "kernel_X", [], False),
                ("u2c2", "kernel_Y", [], False),
                ("v0", "opA", ["v0c1", "v0c2"], False),
                ("v0c1", "kernel_X", [], False),
                ("v0c2", "kernel_Y", [], False),
                ("v1", "opA", ["v1c1"], False),
                ("v1c1", "kernel_X", [], False),
                ("v2", "opA", ["v2c1"], False),
                ("v2c1", "kernel_X", [], False),
                ("v3", "opA", ["v3c1"], False),
                ("v3c1", "kernel_X", [], False),
            ]
        )
        children1 = ["u0", "u1", "u2"]
        children2 = ["v0", "v1", "v2", "v3"]
        ops = [
            ("match", 0, 1),
            ("match", 1, 2),
            ("match", 2, 3),
            ("insert", None, 0),
        ]
        result = _disambiguate_same_name_candidates(
            ops, children1, children2, uid2node, uid2node
        )
        match_j_vals = [j for op, i, j in result if op == "match"]
        assert len(match_j_vals) == len(
            set(match_j_vals)
        ), "Duplicate trace2 index in matches"
        assert (
            0 in match_j_vals
        ), "v0 (best match for all) must be assigned to exactly one event"

    def test_no_duplicate_trace1_index_in_matches(self):
        """All three trace2 events score highest with the same trace1 event (u0);
        only one can claim it — the others must fall back to different partners.

        Subtrees:
          u0:         [kernel_X, kernel_Y]   → level-1 = (kernel_x, kernel_y)  ← best for all
          u1, u2, u3: [kernel_X]             → level-1 = (kernel_x,)
          v0, v1, v2: [kernel_X, kernel_Y]   → level-1 = (kernel_x, kernel_y)

        WF positionally matched u1↔v0, u2↔v1, u3↔v2 (u0 is a delete).
        All three trace2 events prefer u0 (score (1,0) vs (0,1)), but only
        one can be assigned it.  The other two must take different partners,
        so no trace1 index can appear more than once.
        """
        uid2node = _make_uid2node(
            [
                ("u0", "opA", ["u0c1", "u0c2"], False),
                ("u0c1", "kernel_X", [], False),
                ("u0c2", "kernel_Y", [], False),
                ("u1", "opA", ["u1c1"], False),
                ("u1c1", "kernel_X", [], False),
                ("u2", "opA", ["u2c1"], False),
                ("u2c1", "kernel_X", [], False),
                ("u3", "opA", ["u3c1"], False),
                ("u3c1", "kernel_X", [], False),
                ("v0", "opA", ["v0c1", "v0c2"], False),
                ("v0c1", "kernel_X", [], False),
                ("v0c2", "kernel_Y", [], False),
                ("v1", "opA", ["v1c1", "v1c2"], False),
                ("v1c1", "kernel_X", [], False),
                ("v1c2", "kernel_Y", [], False),
                ("v2", "opA", ["v2c1", "v2c2"], False),
                ("v2c1", "kernel_X", [], False),
                ("v2c2", "kernel_Y", [], False),
            ]
        )
        children1 = ["u0", "u1", "u2", "u3"]
        children2 = ["v0", "v1", "v2"]
        ops = [
            ("match", 1, 0),
            ("match", 2, 1),
            ("match", 3, 2),
            ("delete", 0, None),
        ]
        result = _disambiguate_same_name_candidates(
            ops, children1, children2, uid2node, uid2node
        )
        match_i_vals = [i for op, i, j in result if op == "match"]
        assert len(match_i_vals) == len(
            set(match_i_vals)
        ), "Duplicate trace1 index in matches"
        assert (
            0 in match_i_vals
        ), "u0 (best match for all) must be assigned to exactly one event"

    def test_insert_and_delete_preserved_when_no_reassignment(self):
        """Unambiguous inserts and deletes should pass through unchanged."""
        uid2node = _make_uid2node(
            [
                ("u0", "opA", [], False),
                ("u1", "opB", [], False),
                ("v0", "opA", [], False),
                ("v1", "opC", [], False),
            ]
        )
        children1 = ["u0", "u1"]
        children2 = ["v0", "v1"]
        ops = [("match", 0, 0), ("delete", 1, None), ("insert", None, 1)]
        result = _disambiguate_same_name_candidates(
            ops, children1, children2, uid2node, uid2node
        )
        assert ("delete", 1, None) in result
        assert ("insert", None, 1) in result

    def test_multiple_same_name_different_subtrees_correct_matches(self):
        """Two trace1 nodes and three trace2 nodes all named 'opA', each with a
        distinct single kernel child (kernel_X, kernel_Y, kernel_Z).

        WF (positional) matches:
          trace1[0] (kX) ↔ trace2[0] (kX)  ← correct
          trace1[1] (kY) ↔ trace2[1] (kZ)  ← wrong
          trace2[2] (kY)                    ← insert

        Disambiguation should correct to:
          trace1[0] (kX) ↔ trace2[0] (kX)  ← preserved
          trace1[1] (kY) ↔ trace2[2] (kY)  ← fixed
          trace2[1] (kZ)                    ← becomes insert
        """
        uid2node = _make_uid2node(
            [
                ("u0", "opA", ["kX1"], False),
                ("kX1", "kernel_X", [], False),
                ("u1", "opA", ["kY1"], False),
                ("kY1", "kernel_Y", [], False),
                ("v0", "opA", ["kX2"], False),
                ("kX2", "kernel_X", [], False),
                ("v1", "opA", ["kZ"], False),
                ("kZ", "kernel_Z", [], False),
                ("v2", "opA", ["kY2"], False),
                ("kY2", "kernel_Y", [], False),
            ]
        )
        children1 = ["u0", "u1"]
        children2 = ["v0", "v1", "v2"]
        ops = [("match", 0, 0), ("match", 1, 1), ("insert", None, 2)]
        result = _disambiguate_same_name_candidates(
            ops, children1, children2, uid2node, uid2node
        )
        pairs = _match_pairs(result)
        assert (0, 0) in pairs, "kX↔kX match should be preserved"
        assert (1, 2) in pairs, "kY↔kY match should be corrected from (1,1) to (1,2)"
        assert (1, 1) not in pairs, "kY↔kZ wrong match should be removed"
        assert 1 in _insert_idxs(result), "v1(kZ) should become an insert"

    def test_second_match_falls_back_when_best_candidate_already_claimed(self):
        """Two trace1 nodes (both opA/kX) vs three trace2 nodes (opA/kY, opA/kX, opA/kX).

        WF produces:
          match(0,0) u0(kX) ↔ v0(kY)   ← wrong
          match(1,1) u1(kX) ↔ v1(kX)   ← correct
          insert(2)         ↔ v2(kX)   ← unmatched insert

        Both u0 and u1 would ideally pick v2(kX) from the insert pool, but:
          match(0,0) runs first: only insert candidate is v2 → claims it, corrects to (0,2).
          match(1,1) runs next: v2 is now claimed → filtered from candidates → only orig_j=v1
            remains → unique survivor = orig_j → no change, keeps (1,1).
        """
        uid2node = _make_uid2node(
            [
                ("u0", "opA", ["kX1"], False),
                ("kX1", "kernel_X", [], False),
                ("u1", "opA", ["kX2"], False),
                ("kX2", "kernel_X", [], False),
                ("v0", "opA", ["kY"], False),
                ("kY", "kernel_Y", [], False),
                ("v1", "opA", ["kX3"], False),
                ("kX3", "kernel_X", [], False),
                ("v2", "opA", ["kX4"], False),
                ("kX4", "kernel_X", [], False),
            ]
        )
        children1 = ["u0", "u1"]
        children2 = ["v0", "v1", "v2"]
        ops = [("match", 0, 0), ("match", 1, 1), ("insert", None, 2)]
        result = _disambiguate_same_name_candidates(
            ops, children1, children2, uid2node, uid2node
        )
        pairs = _match_pairs(result)
        assert (
            0,
            2,
        ) in pairs, "u0 should be corrected to v2(kX) — the only insert candidate"
        assert (1, 1) in pairs, "u1 should keep v1(kX) — v2 was already claimed by u0"
        assert (0, 0) not in pairs, "wrong u0↔v0(kY) match should be removed"
        assert 0 in _insert_idxs(result), "v0(kY) should become an insert"

    def test_swapped_matches_corrected_via_matched_candidates(self):
        """Issue 1: matched nodes must be candidates too.

        WF produces two matches with names swapped:
          match(0,0) u0(kX) ↔ v0(kY)   ← wrong
          match(1,1) u1(kY) ↔ v1(kX)   ← wrong

        Both sides have the same names but WF matched them cross-wise.
        Since all nodes are matched (no inserts/deletes), the old code would
        never consider them as candidates.  The new code should correct to:
          match(0,1) u0(kX) ↔ v1(kX)
          match(1,0) u1(kY) ↔ v0(kY)
        """
        uid2node = _make_uid2node(
            [
                ("u0", "opA", ["u0c"], False),
                ("u0c", "kernel_X", [], False),
                ("u1", "opA", ["u1c"], False),
                ("u1c", "kernel_Y", [], False),
                ("v0", "opA", ["v0c"], False),
                ("v0c", "kernel_Y", [], False),
                ("v1", "opA", ["v1c"], False),
                ("v1c", "kernel_X", [], False),
            ]
        )
        children1 = ["u0", "u1"]
        children2 = ["v0", "v1"]
        ops = [("match", 0, 0), ("match", 1, 1)]
        result = _disambiguate_same_name_candidates(
            ops, children1, children2, uid2node, uid2node
        )
        pairs = _match_pairs(result)
        assert (0, 1) in pairs, "u0(kX) should match v1(kX)"
        assert (1, 0) in pairs, "u1(kY) should match v0(kY)"

    def test_minority_side_fully_matched(self):
        """Issue 3: minority side must be fully matched.

        5 events in trace1, 3 in trace2, all same name but with distinct
        subtrees for 3 of the trace1 nodes:
          trace1: u0(kA), u1(kB), u2(kC), u3(kD), u4(kE)
          trace2: v0(kA), v1(kC), v2(kB)

        WF positional matching:
          match(0,0) u0(kA) ↔ v0(kA)  ← correct
          match(1,1) u1(kB) ↔ v1(kC)  ← wrong
          match(2,2) u2(kC) ↔ v2(kB)  ← wrong
          delete(3)  u3(kD)
          delete(4)  u4(kE)

        Since trace2 has fewer events, all 3 trace2 nodes must be matched.
        Correct result:
          u0(kA) ↔ v0(kA)
          u1(kB) ↔ v2(kB)
          u2(kC) ↔ v1(kC)
          u3, u4 are deletes
        """
        uid2node = _make_uid2node(
            [
                ("u0", "opA", ["u0c"], False),
                ("u0c", "kernel_A", [], False),
                ("u1", "opA", ["u1c"], False),
                ("u1c", "kernel_B", [], False),
                ("u2", "opA", ["u2c"], False),
                ("u2c", "kernel_C", [], False),
                ("u3", "opA", ["u3c"], False),
                ("u3c", "kernel_D", [], False),
                ("u4", "opA", ["u4c"], False),
                ("u4c", "kernel_E", [], False),
                ("v0", "opA", ["v0c"], False),
                ("v0c", "kernel_A", [], False),
                ("v1", "opA", ["v1c"], False),
                ("v1c", "kernel_C", [], False),
                ("v2", "opA", ["v2c"], False),
                ("v2c", "kernel_B", [], False),
            ]
        )
        children1 = ["u0", "u1", "u2", "u3", "u4"]
        children2 = ["v0", "v1", "v2"]
        ops = [
            ("match", 0, 0),
            ("match", 1, 1),
            ("match", 2, 2),
            ("delete", 3, None),
            ("delete", 4, None),
        ]
        result = _disambiguate_same_name_candidates(
            ops, children1, children2, uid2node, uid2node
        )
        pairs = _match_pairs(result)
        deletes = _delete_idxs(result)
        # All 3 trace2 nodes must be matched
        matched_j = {j for _, j in pairs}
        assert matched_j == {
            0,
            1,
            2,
        }, "All trace2 nodes must be matched (minority side)"
        # Correct pairings by subtree
        assert (0, 0) in pairs, "kA ↔ kA"
        assert (1, 2) in pairs, "kB ↔ kB"
        assert (2, 1) in pairs, "kC ↔ kC"
        # Leftover trace1 nodes are deletes
        assert 3 in deletes
        assert 4 in deletes

    def test_non_positional_match_across_sides(self):
        """5 trace1 children, 3 trace2 children, all same name.

        Both trace1[0] and trace1[2] have kernel_P at BFS level 1, so both
        *could* match trace2[0].  But trace1[2] is a deeper match (2 levels)
        while trace1[0] diverges at level 2.

        Subtrees (2 BFS levels):
          u0: kernel_P → sub_Y      u2: kernel_P → sub_X
          v0: kernel_P → sub_X      v2: kernel_P → sub_Y

        Scores vs v0(kP/sub_X):
          u0 ↔ v0: level-1 match (kernel_P), level-2 diverge (sub_Y≠sub_X) → score 1
          u2 ↔ v0: both levels match → score 2  ← better

        WF positional matching:
          match(0,0) u0(kP/sY) ↔ v0(kP/sX)  ← shallow match only
          match(1,1) u1(kB)    ↔ v1(kB)      ← fine
          match(2,2) u2(kP/sX) ↔ v2(kP/sY)  ← shallow match only
          delete(3)  u3(kD)
          delete(4)  u4(kE)

        Correct (deeper subtree wins):
          match(2,0) u2(kP/sX) ↔ v0(kP/sX)  ← 2-level match
          match(0,2) u0(kP/sY) ↔ v2(kP/sY)  ← 2-level match
          match(1,1) u1(kB)    ↔ v1(kB)
          delete(3), delete(4)
        """
        uid2node = _make_uid2node(
            [
                ("u0", "opA", ["u0c"], False),
                ("u0c", "kernel_P", ["u0cc"], False),
                ("u0cc", "sub_Y", [], False),
                ("u1", "opA", ["u1c"], False),
                ("u1c", "kernel_B", [], False),
                ("u2", "opA", ["u2c"], False),
                ("u2c", "kernel_P", ["u2cc"], False),
                ("u2cc", "sub_X", [], False),
                ("u3", "opA", ["u3c"], False),
                ("u3c", "kernel_D", [], False),
                ("u4", "opA", ["u4c"], False),
                ("u4c", "kernel_E", [], False),
                ("v0", "opA", ["v0c"], False),
                ("v0c", "kernel_P", ["v0cc"], False),
                ("v0cc", "sub_X", [], False),
                ("v1", "opA", ["v1c"], False),
                ("v1c", "kernel_B", [], False),
                ("v2", "opA", ["v2c"], False),
                ("v2c", "kernel_P", ["v2cc"], False),
                ("v2cc", "sub_Y", [], False),
            ]
        )
        children1 = ["u0", "u1", "u2", "u3", "u4"]
        children2 = ["v0", "v1", "v2"]
        ops = [
            ("match", 0, 0),
            ("match", 1, 1),
            ("match", 2, 2),
            ("delete", 3, None),
            ("delete", 4, None),
        ]
        result = _disambiguate_same_name_candidates(
            ops, children1, children2, uid2node, uid2node
        )
        pairs = _match_pairs(result)
        assert (
            2,
            0,
        ) in pairs, "trace1[2](kP/sX) should match trace2[0](kP/sX) — deeper match"
        assert (0, 2) in pairs, "trace1[0](kP/sY) should match trace2[2](kP/sY)"
        assert (1, 1) in pairs, "trace1[1](kB) should match trace2[1](kB)"

    def test_partial_overlap_breaks_tie_at_diverging_level(self):
        """When two events have the same full-level match score, partial overlap
        at the first diverging level acts as tiebreaker.

        trace1[0] (a0) and trace1[1] (a1) both share kernel_P with trace2[0]
        (b0) at BFS level 1, so both diverge at the same depth (full_levels=0).
        But a1 also shares kernel_Q with b0, giving it partial overlap 2 vs 1.

        Subtrees:
          a0: [kernel_P]                  → level-1 = (kernel_p,)
          a1: [kernel_P, kernel_Q]        → level-1 = (kernel_p, kernel_q)
          b0: [kernel_P, kernel_Q, kernel_S] → level-1 = (kernel_p, kernel_q, kernel_s)

        Scores:
          a0 ↔ b0: full_levels=0, overlap=1  → (0, 1)
          a1 ↔ b0: full_levels=0, overlap=2  → (0, 2)  ← wins

        WF positionally matched a0 ↔ b0 (wrong). Disambiguation should
        reassign to a1 ↔ b0.
        """
        uid2node = _make_uid2node(
            [
                ("a0", "opA", ["a0c1"], False),
                ("a0c1", "kernel_P", [], False),
                ("a1", "opA", ["a1c1", "a1c2"], False),
                ("a1c1", "kernel_P", [], False),
                ("a1c2", "kernel_Q", [], False),
                ("b0", "opA", ["b0c1", "b0c2", "b0c3"], False),
                ("b0c1", "kernel_P", [], False),
                ("b0c2", "kernel_Q", [], False),
                ("b0c3", "kernel_S", [], False),
            ]
        )
        children1 = ["a0", "a1"]
        children2 = ["b0"]
        ops = [("match", 0, 0), ("delete", 1, None)]
        result = _disambiguate_same_name_candidates(
            ops, children1, children2, uid2node, uid2node
        )
        pairs = _match_pairs(result)
        deletes = _delete_idxs(result)
        assert (1, 0) in pairs, "a1 (overlap=2) should win over a0 (overlap=1)"
        assert 0 in deletes, "a0 should become a delete"

    def test_all_ops_indices_valid(self):
        """All match/delete/insert indices in the result must be within bounds."""
        uid2node = _make_uid2node(
            [
                ("u0", "opA", ["u0c"], False),
                ("u0c", "kX", [], False),
                ("v0", "opA", ["v0c"], False),
                ("v0c", "kY", [], False),
                ("v1", "opA", ["v1c"], False),
                ("v1c", "kX", [], False),
                ("v2", "opB", [], False),
            ]
        )
        children1 = ["u0"]
        children2 = ["v0", "v1", "v2"]
        ops = [("match", 0, 0), ("insert", None, 1), ("insert", None, 2)]
        result = _disambiguate_same_name_candidates(
            ops, children1, children2, uid2node, uid2node
        )
        for op, i, j in result:
            if i is not None:
                assert 0 <= i < len(children1)
            if j is not None:
                assert 0 <= j < len(children2)
<<<<<<< Updated upstream
=======


class TestGpuPathChildNamesAtBfsLevels:
    def test_collects_gpu_path_children_by_level(self):
        uid2node = _make_uid2node(
            [
                ("root", "opA", ["gpu1"], False),
                ("gpu1", "kernel_X", ["gpu2"], False),
                ("gpu2", "kernel_Y", [], False),
            ]
        )
        levels = _gpu_path_child_names_at_bfs_levels("root", uid2node, max_depth=2)
        assert levels[0] == ("kernel_X",)
        assert levels[1] == ("kernel_Y",)

    def test_skips_missing_nodes(self):
        uid2node = _make_uid2node([("root", "opA", ["missing"], False)])
        levels = _gpu_path_child_names_at_bfs_levels("root", uid2node, max_depth=2)
        assert levels == [()]

    def test_non_gpu_path_children_excluded(self):
        uid2node = _make_uid2node(
            [
                ("root", "opA", ["off"], False),
                ("off", "skipped", [], True),
            ]
        )
        levels = _gpu_path_child_names_at_bfs_levels("root", uid2node, max_depth=1)
        assert levels == [()]


def _make_tracediff(baseline_events=None, variant_events=None):
    baseline_events = baseline_events or {}
    variant_events = variant_events or {}
    td = TraceDiff.__new__(TraceDiff)
    td.baseline = SimpleNamespace(events_by_uid=baseline_events, cpu_root_nodes=[])
    td.variant = SimpleNamespace(events_by_uid=variant_events, cpu_root_nodes=[])
    td.db1 = []
    td.db2 = []
    td.pod1 = set()
    td.pod2 = set()
    td.merged_tree = None
    td.merged_uid_map = {}
    td.diff_stats_df = pd.DataFrame()
    td.diff_stats_summary_df = pd.DataFrame()
    td.diff_stats_unique_args_summary_df = pd.DataFrame()
    td.identical_traces = False
    td.cpu_op_map_trace1 = None
    td.cpu_op_map_trace2 = None
    td.cpu_op_map = None
    td._merged_id_to_event = None
    td._uid1_to_merged_id = None
    td._uid2_to_merged_id = None
    return td


class TestTraceDiffHelpers:
    def test_get_op_name_handles_none_and_missing(self):
        td = _make_tracediff({1: {_TK.Name: "foo"}}, {})
        assert td._get_op_name(None, 1) is None
        assert td._get_op_name(99, 1) is None
        assert td._get_op_name(1, 1) == "foo"

    def test_get_op_name_falls_back_to_uid_string(self):
        td = _make_tracediff({5: {_TK.UID: 5}}, {})
        assert td._get_op_name(5, 1) == "5"

    def test_wagner_fischer_match_insert_delete_and_cache(self):
        events = {
            0: {_TK.UID: 0, _TK.Name: "a"},
            1: {_TK.UID: 1, _TK.Name: "b"},
            2: {_TK.UID: 2, _TK.Name: "c"},
        }
        td = _make_tracediff(events, events)
        cache = {}
        ops = td.wagner_fischer([0, 1], [0, 2], cache)
        assert ("match", 0, 0) in ops
        assert ("delete", 1, None) in ops
        assert ("insert", None, 1) in ops
        cached = td.wagner_fischer([0, 1], [0, 2], cache)
        assert cached is ops

    def test_wagner_fischer_strip_details(self):
        events = {
            0: {_TK.UID: 0, _TK.Name: "/proj/layer.py(1): matmul : a"},
            1: {_TK.UID: 1, _TK.Name: "/proj/layer.py(2): matmul : b"},
        }
        td = _make_tracediff(events, events)
        ops = td.wagner_fischer([0], [1], {}, strip_details=True)
        assert ops == [("match", 0, 0)]

    def test_get_diff_stats_df_empty(self, capsys):
        td = _make_tracediff()
        assert td.get_diff_stats_df() is None
        assert "diff_stats_df is empty" in capsys.readouterr().out

    def test_get_diff_stats_summary_df_empty(self, capsys):
        td = _make_tracediff()
        assert td.get_diff_stats_summary_df() is None
        assert "diff_stats_summary_df is empty" in capsys.readouterr().out

    def test_merged_id_cache_and_invalidation(self):
        td = _make_tracediff()
        merged_events = [
            {"merged_id": 0, "uid1": 1, "uid2": 2},
            {"merged_id": 1, "uid1": 3, "uid2": None},
        ]
        td.merged_tree = (merged_events, [0])
        by_id = td._get_merged_id_to_event()
        assert by_id[0]["uid1"] == 1
        uid1_map, uid2_map = td._get_uid_to_merged_id_maps()
        assert uid1_map[1] == 0
        assert uid2_map[2] == 0
        td._invalidate_merged_cache()
        assert td._merged_id_to_event is None

    def test_format_merged_subtree_merge_types(self):
        events = {
            1: {_TK.Name: "same"},
            2: {_TK.Name: "same"},
            3: {_TK.Name: "only1"},
            4: {_TK.Name: "only2"},
            5: {_TK.Name: "a"},
            6: {_TK.Name: "b"},
            7: {_TK.Name: "left"},
            8: {_TK.Name: "right"},
        }
        td = _make_tracediff(events, events)
        merged_id_to_event = {
            0: {
                "merged_id": 0,
                "uid1": 1,
                "uid2": 2,
                "merged_type": "combined",
                "children": [1, 2, 4],
            },
            1: {
                "merged_id": 1,
                "uid1": 3,
                "uid2": None,
                "merged_type": "trace1",
                "children": [],
            },
            2: {
                "merged_id": 2,
                "uid1": None,
                "uid2": 4,
                "merged_type": "trace2",
                "children": [],
            },
            3: {
                "merged_id": 3,
                "uid1": 5,
                "uid2": 6,
                "merged_type": "other",
                "children": [],
            },
            4: {
                "merged_id": 4,
                "uid1": 7,
                "uid2": 8,
                "merged_type": "combined",
                "children": [],
            },
        }
        lines = list(td._format_merged_subtree(0, merged_id_to_event))
        assert any(line.strip().endswith("same") for line in lines)
        assert any(">> trace1: only1" in line for line in lines)
        assert any("<< trace2: only2" in line for line in lines)
        assert any("combined: left | right" in line for line in lines)
        other_lines = list(td._format_merged_subtree(3, merged_id_to_event))
        assert any("other: a | b" in line for line in other_lines)

    def test_print_merged_subtree_errors(self):
        td = _make_tracediff()
        with pytest.raises(ValueError, match="At least one"):
            td.print_merged_subtree()
        td.merged_tree = ([], [])
        with pytest.raises(ValueError, match="Could not find merged node"):
            td.print_merged_subtree(uid_tree1=42)

    def test_merge_trees_requires_cpu_roots(self):
        td = _make_tracediff()
        with pytest.raises(ValueError, match="cpu_root_nodes"):
            td.merge_trees()

    def test_disambiguate_returns_ops_when_wf_already_optimal(self):
        uid2node = _make_uid2node(
            [
                ("u0", "opA", [], False),
                ("u1", "opA", [], False),
                ("v0", "opA", [], False),
                ("v1", "opA", [], False),
            ]
        )
        children1 = ["u0", "u1"]
        children2 = ["v0", "v1"]
        ops = [("match", 0, 0), ("match", 1, 1)]
        result = _disambiguate_same_name_candidates(
            ops, children1, children2, uid2node, uid2node
        )
        assert result is ops

    def test_disambiguate_skips_names_not_in_matches(self):
        uid2node = _make_uid2node(
            [
                ("u0", "opB", [], False),
                ("u1", "opA", [], False),
                ("v0", "opB", [], False),
                ("v1", "opA", [], False),
            ]
        )
        children1 = ["u0", "u1"]
        children2 = ["v0", "v1"]
        ops = [("match", 0, 0), ("delete", 1, None), ("insert", None, 1)]
        result = _disambiguate_same_name_candidates(
            ops, children1, children2, uid2node, uid2node
        )
        assert result == ops


def _mk_event(cat, name, ts, dur, pid, tid, args=None):
    return {
        "ph": "X",
        "cat": cat,
        "name": name,
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "dur": dur,
        "args": args or {},
    }


def _mk_ac2g(corr_id, pid, tid, ts, phase):
    evt = {
        "ph": phase,
        "id": corr_id,
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "cat": "ac2g",
        "name": "ac2g",
    }
    if phase == "f":
        evt["bp"] = "e"
    return evt


def _add_gpu_chain(
    events,
    cpu_op,
    corr,
    kernel_name,
    ts_launch,
    ts_kernel,
    kernel_dur=20.0,
):
    pid = cpu_op["pid"]
    tid = cpu_op["tid"]
    if cpu_op not in events:
        events.append(cpu_op)
    events.extend(
        [
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=ts_launch,
                dur=5,
                pid=pid,
                tid=tid,
                args={"correlation": corr},
            ),
            _mk_event(
                "kernel",
                kernel_name,
                ts=ts_kernel,
                dur=kernel_dur,
                pid=0,
                tid=7,
                args={"correlation": corr, "stream": 7},
            ),
            _mk_ac2g(corr, pid=0, tid=7, ts=ts_kernel, phase="s"),
            _mk_ac2g(corr, pid=0, tid=7, ts=ts_kernel, phase="f"),
        ]
    )


def _build_tree(events, add_python_func=False):
    tree = TraceToTree(deepcopy(events), prune_nongpu_paths=False)
    tree.build_tree(add_python_func=add_python_func)
    return tree


def _build_trace_from_specs(specs, base_ts=1000, add_python_func=False):
    events = []
    ts = base_ts
    corr = 100
    cpu_pid, cpu_tid = 100, 100
    for cpu_op_name, kernel_name, kernel_dur in specs:
        cpu_op = _mk_event(
            "cpu_op",
            cpu_op_name,
            ts=ts,
            dur=100,
            pid=cpu_pid,
            tid=cpu_tid,
            args={
                "Input Dims": [[32, 64]],
                "Input Strides": [[64, 1]],
                "Input type": ["float"],
                "Concrete Inputs": ["x"],
            },
        )
        _add_gpu_chain(
            events,
            cpu_op,
            corr,
            kernel_name,
            ts_launch=ts + 10,
            ts_kernel=ts + 50,
            kernel_dur=kernel_dur,
        )
        ts += 300
        corr += 1
    return _build_tree(events, add_python_func=add_python_func)


def _make_tracediff_from_specs(specs1, specs2, add_python_func=False):
    tree1 = _build_trace_from_specs(specs1, base_ts=1000, add_python_func=add_python_func)
    tree2 = _build_trace_from_specs(specs2, base_ts=2000, add_python_func=add_python_func)
    return TraceDiff(tree1, tree2)


class TestTraceDiffSyntheticIntegration:
    def test_merge_trees_builds_merged_structure(self):
        td = _make_tracediff_from_specs(
            [("aten::mm", "gemm_v1", 100.0)],
            [("aten::mm", "gemm_v2", 80.0)],
        )
        merged_events, merged_root_ids = td.merged_tree
        assert merged_events
        assert merged_root_ids
        assert td.merged_uid_map[(1, td.baseline.cpu_root_nodes[0])] == (
            td.variant.cpu_root_nodes[0]
        )

    def test_generate_diff_stats_detects_differences(self):
        td = _make_tracediff_from_specs(
            [("aten::mm", "gemm_v1", 100.0), ("aten::relu", "relu_v1", 20.0)],
            [("aten::mm", "gemm_v2", 80.0), ("aten::relu", "relu_v2", 25.0)],
        )
        df = td.generate_diff_stats()
        assert not df.empty
        assert set(df["source"]) == {"trace1", "trace2"}
        assert td.identical_traces is False

    def test_generate_diff_stats_identical_traces(self):
        specs = [("aten::mm", "gemm_same", 50.0)]
        td = _make_tracediff_from_specs(specs, specs)
        td.generate_diff_stats()
        assert td.identical_traces is True

    def test_generate_diff_stats_no_gpu_events(self, capsys):
        events = [_mk_event("cpu_op", "aten::noop", ts=0, dur=10, pid=1, tid=1)]
        td = TraceDiff(_build_tree(events), _build_tree(events))
        df = td.generate_diff_stats()
        assert df.empty
        assert td.identical_traces is True
        assert "No GPU events found" in capsys.readouterr().out

    def test_get_df_diff_stats_unique_args(self):
        td = _make_tracediff_from_specs(
            [("aten::mm", "gemm_v1", 100.0)],
            [("aten::mm", "gemm_v2", 80.0)],
        )
        td.generate_diff_stats()
        summary = td.get_df_diff_stats_unique_args(agg_metrics=["mean", "median"])
        assert summary is not None
        assert "kernel_time_sum" in summary.columns

    def test_get_df_diff_stats_unique_args_empty(self, capsys):
        td = _make_tracediff()
        assert td.get_df_diff_stats_unique_args() is None
        assert "diff_stats_df is empty" in capsys.readouterr().out

    def test_get_diff_stats_df_returns_dataframe(self):
        td = _make_tracediff_from_specs(
            [("aten::mm", "gemm_v1", 100.0)],
            [("aten::mm", "gemm_v2", 80.0)],
        )
        td.generate_diff_stats()
        assert td.get_diff_stats_df() is not None

    def test_get_cpu_op_to_kernels_json(self, capsys):
        td = _make_tracediff_from_specs(
            [("aten::mm", "gemm_v1", 100.0)],
            [("aten::mm", "gemm_v2", 80.0)],
        )
        td.generate_tracediff_report()
        assert td.cpu_op_map is not None
        assert "Kernel to CPU op mapping" in capsys.readouterr().out

    def test_get_cpu_op_to_kernels_json_empty_summary(self, capsys):
        td = _make_tracediff()
        td.get_cpu_op_to_kernels_json()
        assert "diff_stats_unique_args_summary_df is empty" in capsys.readouterr().out

    def test_generate_tracediff_report_identical(self):
        specs = [("aten::mm", "gemm_same", 50.0)]
        td = _make_tracediff_from_specs(specs, specs)
        td.generate_tracediff_report()
        assert td.identical_traces is True
        assert "source" not in td.diff_stats_df.columns

    def test_print_tracediff_report_files(self, tmp_path):
        td = _make_tracediff_from_specs(
            [("aten::mm", "gemm_v1", 100.0)],
            [("aten::mm", "gemm_v2", 80.0)],
        )
        td.generate_tracediff_report()
        out = str(tmp_path / "report")
        td.print_tracediff_report_files(output_folder=out, prune_non_gpu=True)
        assert os.path.exists(os.path.join(out, "merged_tree_output.txt"))
        assert os.path.exists(os.path.join(out, "diff_stats.csv"))

    def test_print_tracediff_report_files_empty_stats(self, tmp_path, capsys):
        td = _make_tracediff()
        td.merged_tree = ([], [])
        td.print_tracediff_report_files(output_folder=str(tmp_path / "empty"))
        captured = capsys.readouterr().out
        assert "diff_stats_df is empty" in captured
        assert "cpu_op_map_trace1 is empty" in captured

    def test_print_merged_subtree_by_uid(self, capsys):
        td = _make_tracediff_from_specs(
            [("aten::mm", "gemm_v1", 100.0)],
            [("aten::mm", "gemm_v2", 80.0)],
        )
        td.print_merged_subtree(uid_tree1=td.baseline.cpu_root_nodes[0])
        assert "aten::mm" in capsys.readouterr().out

    def test_print_merged_subtree_by_variant_uid(self, capsys):
        td = _make_tracediff_from_specs(
            [("aten::mm", "gemm_v1", 100.0)],
            [("aten::mm", "gemm_v2", 80.0)],
        )
        td.print_merged_subtree(uid_tree2=td.variant.cpu_root_nodes[0])
        assert "aten::mm" in capsys.readouterr().out

    def test_print_merged_tree_writes_file(self, tmp_path):
        td = _make_tracediff_from_specs(
            [("aten::mm", "gemm_v1", 100.0)],
            [("aten::mm", "gemm_v2", 80.0)],
        )
        out_file = str(tmp_path / "merged.txt")
        td.print_merged_tree(out_file)
        with open(out_file) as f:
            assert f.read()


class TestTraceDiffMergeEdgeCases:
    def test_get_top_level_root_collapses_python_wrapper(self):
        events = [
            _mk_event("python_function", "nn.Module: Model", ts=0, dur=1000, pid=1, tid=1),
            _mk_event("python_function", "nn.Module: Block", ts=5, dur=900, pid=1, tid=1),
        ]
        op = _mk_event("cpu_op", "aten::mm", ts=10, dur=100, pid=1, tid=1)
        _add_gpu_chain(events, op, 100, "gemm_kernel", ts_launch=20, ts_kernel=60)
        tree = _build_tree(events, add_python_func=True)
        td = _make_tracediff()
        td.baseline = tree
        root = td._get_top_level_root(tree, tree.cpu_root_nodes[0])
        root_node = tree.get_UID2event(root)
        assert tree.event_to_category(root_node) == "python_function"

    def test_get_top_level_root_walks_to_parent(self):
        events = []
        parent = _mk_event("cpu_op", "aten::parent", ts=0, dur=500, pid=1, tid=1)
        child = _mk_event("cpu_op", "aten::child", ts=10, dur=100, pid=1, tid=1)
        _add_gpu_chain(events, parent, 100, "parent_kernel", ts_launch=20, ts_kernel=60)
        events.append(child)
        tree = _build_tree(events)
        td = _make_tracediff()
        td.baseline = tree
        child_uid = next(e[_TK.UID] for e in tree.events if e[_TK.Name] == "aten::child")
        assert td._get_top_level_root(tree, child_uid) == tree.cpu_root_nodes[0]

    def test_merge_with_wrapper_reconciliation(self):
        events1 = [
            _mk_event("cpu_op", "aten::wrapper", ts=0, dur=200, pid=1, tid=1),
            _mk_event("cpu_op", "aten::mm", ts=10, dur=100, pid=1, tid=1),
        ]
        _add_gpu_chain(events1, events1[1], 100, "gemm_v1", ts_launch=20, ts_kernel=60)
        events2 = [_mk_event("cpu_op", "aten::mm", ts=0, dur=100, pid=1, tid=1)]
        _add_gpu_chain(events2, events2[0], 200, "gemm_v2", ts_launch=10, ts_kernel=50)
        td = TraceDiff(_build_tree(events1), _build_tree(events2))
        assert any(e["merged_type"] == "combined" for e in td.merged_tree[0])

    def test_merge_trees_unequal_root_counts(self):
        td = _make_tracediff_from_specs(
            [("aten::mm", "gemm_v1", 100.0), ("aten::add", "add_v1", 10.0)],
            [("aten::mm", "gemm_v2", 80.0)],
        )
        assert len(td.merged_tree[1]) == 2

    def test_generate_diff_stats_with_debug_columns(self, monkeypatch):
        import importlib

        trace_diff_module = importlib.import_module("TraceLens.TraceDiff.trace_diff")
        monkeypatch.setattr(trace_diff_module, "_TRACELENS_DEBUG", True)
        td = _make_tracediff_from_specs(
            [("aten::mm", "gemm_v1", 100.0)],
            [("aten::mm", "gemm_v2", 80.0)],
        )
        df = td.generate_diff_stats()
        assert "callstack" in df.columns

    def test_get_df_diff_stats_unique_args_unhashable_fallback(self):
        td = _make_tracediff_from_specs(
            [("aten::mm", "gemm_v1", 100.0)],
            [("aten::mm", "gemm_v2", 80.0)],
        )
        td.generate_diff_stats()
        td.diff_stats_df["list_col"] = td.diff_stats_df.apply(
            lambda row: [row["name"], row["source"]], axis=1
        )
        assert td.get_df_diff_stats_unique_args() is not None

    def test_get_cpu_op_to_kernels_json_renames_mismatched_ops(self):
        td = _make_tracediff_from_specs(
            [("aten::mm", "gemm_v1", 100.0)],
            [("aten::matmul", "gemm_v2", 80.0)],
        )
        td.generate_tracediff_report()
        assert td.cpu_op_map is not None

    def test_generate_diff_stats_trace1_only_branch(self):
        td = _make_tracediff_from_specs(
            [("aten::mm", "gemm_v1", 100.0), ("aten::add", "add_v1", 10.0)],
            [("aten::mm", "gemm_v2", 80.0)],
        )
        df = td.generate_diff_stats()
        assert not df.empty
        assert "trace1" in df["source"].values

    def test_generate_diff_stats_trace2_only_branch(self):
        td = _make_tracediff_from_specs(
            [("aten::mm", "gemm_v1", 100.0)],
            [("aten::mm", "gemm_v2", 80.0), ("aten::add", "add_v2", 10.0)],
        )
        df = td.generate_diff_stats()
        assert not df.empty
        assert "trace2" in df["source"].values

    def test_print_merged_tree_prunes_non_gpu(self, tmp_path):
        events = [_mk_event("cpu_op", "aten::noop", ts=0, dur=10, pid=1, tid=1)]
        td = TraceDiff(_build_tree(events), _build_tree(events))
        out_file = str(tmp_path / "pruned.txt")
        td.print_merged_tree(out_file, prune_non_gpu=True)
        with open(out_file) as f:
            content = f.read()
        assert "aten::noop" in content

    def test_print_tracediff_report_files_writes_json_maps(self, tmp_path):
        td = _make_tracediff_from_specs(
            [("aten::mm", "gemm_v1", 100.0)],
            [("aten::mm", "gemm_v2", 80.0)],
        )
        td.generate_tracediff_report()
        out = str(tmp_path / "maps")
        td.print_tracediff_report_files(output_folder=out)
        for name in [
            "cpu_op_map_trace1.json",
            "cpu_op_map_trace2.json",
            "cpu_op_map.json",
        ]:
            assert os.path.exists(os.path.join(out, name))

    def test_get_diff_stats_summary_df_with_data(self):
        td = _make_tracediff()
        td.diff_stats_summary_df = pd.DataFrame({"a": [1]})
        assert td.get_diff_stats_summary_df() is not None

    def test_get_cpu_op_to_kernels_json_unmatched_and_prefix_rename(self, capsys):
        td = _make_tracediff_from_specs(
            [
                ("aten::mm", "gemm_v1", 100.0),
                ("aten::add", "add_v1", 10.0),
            ],
            [
                ("aten::matmul", "gemm_v2", 80.0),
                ("aten::sum", "add_v2", 12.0),
            ],
        )
        td.generate_tracediff_report()
        captured = capsys.readouterr().out
        assert td.cpu_op_map is not None
        assert "Renaming" in captured or "Unmatched for LCA" in captured

    def test_merge_reconcile_insert_name_in_deleted_children(self):
        events1 = [
            _mk_event("cpu_op", "aten::wrapper", ts=0, dur=200, pid=1, tid=1),
            _mk_event("cpu_op", "aten::inner", ts=10, dur=100, pid=1, tid=1),
        ]
        _add_gpu_chain(events1, events1[1], 100, "gemm_v1", ts_launch=20, ts_kernel=60)
        events2 = [
            _mk_event("cpu_op", "aten::other", ts=0, dur=100, pid=1, tid=1),
            _mk_event("cpu_op", "aten::inner", ts=10, dur=100, pid=1, tid=1),
        ]
        _add_gpu_chain(events2, events2[1], 200, "gemm_v2", ts_launch=20, ts_kernel=60)
        td = TraceDiff(_build_tree(events1), _build_tree(events2))
        assert td.merged_tree is not None

    def test_merge_single_side_extra_kernel_dispatch_child(self):
        events1 = [
            _mk_event("cpu_op", "aten::mm", ts=0, dur=100, pid=1, tid=1),
        ]
        _add_gpu_chain(events1, events1[0], 100, "gemm_v1", ts_launch=10, ts_kernel=50)
        events1.append(
            _mk_event(
                "cuda_runtime",
                "hipLaunchKernel",
                ts=20,
                dur=5,
                pid=1,
                tid=1,
                args={"correlation": 101},
            )
        )
        events2 = [
            _mk_event("cpu_op", "aten::mm", ts=0, dur=100, pid=1, tid=1),
        ]
        _add_gpu_chain(events2, events2[0], 200, "gemm_v2", ts_launch=10, ts_kernel=50)
        td = TraceDiff(_build_tree(events1), _build_tree(events2))
        assert td.merged_tree is not None

    def test_gpu_path_bfs_skips_missing_frontier_node(self):
        uid2node = _make_uid2node([("root", "opA", ["ghost"], False)])
        levels = _gpu_path_child_names_at_bfs_levels("root", uid2node, max_depth=1)
        assert levels == [()]

    def test_merge_reconcile_wrapper_to_inner_op(self):
        events1 = [
            _mk_event("cpu_op", "aten::root", ts=0, dur=500, pid=1, tid=1),
            _mk_event("cpu_op", "aten::wrapper", ts=10, dur=200, pid=1, tid=1),
            _mk_event("cpu_op", "aten::extra", ts=300, dur=100, pid=1, tid=1),
        ]
        inner = _mk_event("cpu_op", "aten::mm", ts=20, dur=100, pid=1, tid=1)
        _add_gpu_chain(events1, inner, 100, "gemm_v1", ts_launch=30, ts_kernel=70)
        _add_gpu_chain(events1, events1[2], 101, "extra_k", ts_launch=310, ts_kernel=350)
        events2 = [
            _mk_event("cpu_op", "aten::root", ts=0, dur=500, pid=1, tid=1),
        ]
        direct = _mk_event("cpu_op", "aten::mm", ts=10, dur=100, pid=1, tid=1)
        _add_gpu_chain(events2, direct, 200, "gemm_v2", ts_launch=20, ts_kernel=60)
        td = TraceDiff(_build_tree(events1), _build_tree(events2))
        merged_events, _ = td.merged_tree
        assert any(e["merged_type"] == "combined" for e in merged_events)

    def test_merge_cuda_graph_launch_not_treated_as_runtime_mismatch(self):
        events = [
            _mk_event("cpu_op", "aten::graph", ts=0, dur=200, pid=1, tid=1),
            _mk_event(
                "cuda_runtime",
                "cudaGraphLaunch",
                ts=10,
                dur=5,
                pid=1,
                tid=1,
                args={"correlation": 100},
            ),
            _mk_event(
                "kernel",
                "graph_kernel",
                ts=50,
                dur=20,
                pid=0,
                tid=7,
                args={"correlation": 100, "stream": 7},
            ),
            _mk_ac2g(100, pid=0, tid=7, ts=50, phase="s"),
            _mk_ac2g(100, pid=0, tid=7, ts=50, phase="f"),
        ]
        td = TraceDiff(_build_tree(events), _build_tree(events))
        assert td.merged_tree is not None

    def test_merge_duplicate_python_functions_strip_details_pass2(self):
        events1 = [
            _mk_event(
                "python_function",
                "/a/model.py(1): forward : x",
                ts=0,
                dur=500,
                pid=1,
                tid=1,
            ),
            _mk_event("cpu_op", "aten::mm", ts=10, dur=100, pid=1, tid=1),
        ]
        _add_gpu_chain(events1, events1[1], 100, "gemm_v1", ts_launch=20, ts_kernel=60)
        events2 = [
            _mk_event(
                "python_function",
                "/b/model.py(9): forward : y",
                ts=0,
                dur=500,
                pid=1,
                tid=1,
            ),
            _mk_event("cpu_op", "aten::add", ts=10, dur=100, pid=1, tid=1),
        ]
        _add_gpu_chain(events2, events2[1], 200, "add_v2", ts_launch=20, ts_kernel=60)
        td = TraceDiff(
            _build_tree(events1, add_python_func=True),
            _build_tree(events2, add_python_func=True),
        )
        assert td.merged_tree is not None

    def test_get_cpu_op_to_kernels_json_one_to_one_module_match(self, capsys):
        events1 = [
            _mk_event(
                "cpu_op",
                "aten::linear",
                ts=0,
                dur=100,
                pid=1,
                tid=1,
                args={"Input Dims": [[32, 64]]},
            ),
        ]
        _add_gpu_chain(events1, events1[0], 100, "gemm_v1", ts_launch=10, ts_kernel=50)
        events1[0]["nn_module_stack"] = ["Linear"]
        events2 = [
            _mk_event(
                "cpu_op",
                "aten::linear",
                ts=0,
                dur=100,
                pid=1,
                tid=1,
                args={"Input Dims": [[32, 64]]},
            ),
        ]
        _add_gpu_chain(events2, events2[0], 200, "gemm_v2", ts_launch=10, ts_kernel=50)
        events2[0]["nn_module_stack"] = ["Linear"]
        td = TraceDiff(_build_tree(events1), _build_tree(events2))
        td.generate_tracediff_report()
        assert td.cpu_op_map is not None

    def test_print_merged_subtree_not_initialized(self):
        td = _make_tracediff()
        with pytest.raises(ValueError, match="merged_tree is not initialized"):
            td.print_merged_tree("out.txt")

    def test_merge_collapse_single_gpu_child_branch(self):
        events1 = [
            _mk_event("cpu_op", "aten::root", ts=0, dur=500, pid=1, tid=1),
            _mk_event("cpu_op", "aten::wrap1", ts=10, dur=200, pid=1, tid=1),
            _mk_event("cpu_op", "aten::wrap2", ts=20, dur=150, pid=1, tid=1),
        ]
        inner = _mk_event("cpu_op", "aten::mm", ts=30, dur=100, pid=1, tid=1)
        _add_gpu_chain(events1, inner, 100, "gemm_v1", ts_launch=40, ts_kernel=80)
        events2 = [
            _mk_event("cpu_op", "aten::root", ts=0, dur=500, pid=1, tid=1),
            _mk_event("cpu_op", "aten::mm", ts=10, dur=100, pid=1, tid=1),
            _mk_event("cpu_op", "aten::add", ts=200, dur=100, pid=1, tid=1),
        ]
        _add_gpu_chain(events2, events2[1], 200, "gemm_v2", ts_launch=20, ts_kernel=60)
        _add_gpu_chain(events2, events2[2], 201, "add_v2", ts_launch=210, ts_kernel=250)
        td = TraceDiff(_build_tree(events1), _build_tree(events2))
        assert td.merged_tree is not None

    def test_merge_phase3_pinned_matches_after_reconcile(self):
        events1 = [
            _mk_event("cpu_op", "aten::root", ts=0, dur=800, pid=1, tid=1),
            _mk_event("cpu_op", "aten::wrap", ts=10, dur=200, pid=1, tid=1),
            _mk_event("cpu_op", "aten::shared", ts=300, dur=100, pid=1, tid=1),
        ]
        inner = _mk_event("cpu_op", "aten::mm", ts=20, dur=100, pid=1, tid=1)
        _add_gpu_chain(events1, inner, 100, "gemm_v1", ts_launch=30, ts_kernel=70)
        _add_gpu_chain(events1, events1[2], 101, "shared_k1", ts_launch=310, ts_kernel=350)
        events2 = [
            _mk_event("cpu_op", "aten::root", ts=0, dur=800, pid=1, tid=1),
            _mk_event("cpu_op", "aten::mm", ts=10, dur=100, pid=1, tid=1),
            _mk_event("cpu_op", "aten::shared", ts=300, dur=100, pid=1, tid=1),
        ]
        _add_gpu_chain(events2, events2[1], 200, "gemm_v2", ts_launch=20, ts_kernel=60)
        _add_gpu_chain(events2, events2[2], 201, "shared_k2", ts_launch=310, ts_kernel=350)
        td = TraceDiff(_build_tree(events1), _build_tree(events2))
        assert td.merged_tree is not None

    def test_merge_phase4_strip_details_pass2(self):
        events1 = [
            _mk_event(
                "python_function",
                "/proj/a.py(1): op : left",
                ts=0,
                dur=500,
                pid=1,
                tid=1,
            ),
            _mk_event("cpu_op", "aten::custom", ts=10, dur=100, pid=1, tid=1),
        ]
        _add_gpu_chain(events1, events1[1], 100, "k1", ts_launch=20, ts_kernel=60)
        events2 = [
            _mk_event(
                "python_function",
                "/proj/b.py(9): op : right",
                ts=0,
                dur=500,
                pid=1,
                tid=1,
            ),
            _mk_event("cpu_op", "aten::other", ts=10, dur=100, pid=1, tid=1),
        ]
        _add_gpu_chain(events2, events2[1], 200, "k2", ts_launch=20, ts_kernel=60)
        td = TraceDiff(
            _build_tree(events1, add_python_func=True),
            _build_tree(events2, add_python_func=True),
        )
        assert td.merged_tree is not None

    def test_get_children_with_missing_adds_non_gpu_path_sibling(self):
        events1 = [
            _mk_event("cpu_op", "aten::root", ts=0, dur=500, pid=1, tid=1),
            _mk_event("cpu_op", "aten::visible", ts=10, dur=100, pid=1, tid=1),
        ]
        _add_gpu_chain(events1, events1[1], 100, "k1", ts_launch=20, ts_kernel=60)
        events2 = [
            _mk_event("cpu_op", "aten::root", ts=0, dur=500, pid=1, tid=1),
            _mk_event("cpu_op", "aten::hidden", ts=10, dur=100, pid=1, tid=1, args={}),
            _mk_event("cpu_op", "aten::visible", ts=200, dur=100, pid=1, tid=1),
        ]
        _add_gpu_chain(events2, events2[2], 200, "k2", ts_launch=210, ts_kernel=250)
        tree2 = _build_tree(events2)
        hidden = next(e for e in tree2.events if e[_TK.Name] == "aten::hidden")
        hidden["non_gpu_path"] = True
        td = TraceDiff(_build_tree(events1), tree2)
        assert td.merged_tree is not None

    def test_get_cpu_op_to_kernels_json_different_length_name_lists(self, capsys):
        td = _make_tracediff_from_specs(
            [("aten::mm", "k1", 10.0), ("aten::add", "k2", 10.0)],
            [("aten::matmul", "k3", 10.0)],
        )
        td.generate_tracediff_report()
        captured = capsys.readouterr().out
        assert td.cpu_op_map is not None
        assert "Renaming" in captured or "Unmatched for LCA" in captured

    def test_generate_diff_stats_kernel_as_combined_node(self):
        events = [
            _mk_event("cpu_op", "aten::root", ts=0, dur=200, pid=1, tid=1),
        ]
        _add_gpu_chain(events, events[0], 100, "direct_kernel", ts_launch=10, ts_kernel=50)
        tree = _build_tree(events)
        td = TraceDiff(tree, tree)
        df = td.generate_diff_stats()
        assert not df.empty

    def test_get_cpu_op_to_kernels_json_rename_branches_direct(self, capsys):
        td = _make_tracediff()
        td.diff_stats_df = pd.DataFrame(
            [
                {
                    "name": "k1",
                    "cpu_op_name": "aten::mm",
                    "source": "trace1",
                    "kernel_time": 10.0,
                    "lowest_common_ancestor_id": 1,
                    "nn_module_parent": "Linear",
                    "nn_module_stack": "root",
                },
                {
                    "name": "k2",
                    "cpu_op_name": "aten::add",
                    "source": "trace1",
                    "kernel_time": 5.0,
                    "lowest_common_ancestor_id": 1,
                    "nn_module_parent": "Linear",
                    "nn_module_stack": "root",
                },
                {
                    "name": "k3",
                    "cpu_op_name": "aten::matmul",
                    "source": "trace2",
                    "kernel_time": 8.0,
                    "lowest_common_ancestor_id": 1,
                    "nn_module_parent": "Linear",
                    "nn_module_stack": "root",
                },
            ]
        )
        td.diff_stats_unique_args_summary_df = td.diff_stats_df.copy()
        td.get_cpu_op_to_kernels_json()
        captured = capsys.readouterr().out
        assert td.cpu_op_map is not None
        assert "Renaming" in captured or "Unmatched for LCA" in captured

    def test_get_cpu_op_to_kernels_json_prefix_and_module_rename(self, capsys):
        td = _make_tracediff()
        td.diff_stats_df = pd.DataFrame(
            [
                {
                    "name": "k1",
                    "cpu_op_name": "torch::long_name_a",
                    "source": "trace1",
                    "kernel_time": 10.0,
                    "lowest_common_ancestor_id": 2,
                    "nn_module_parent": " Block ",
                    "nn_module_stack": "root",
                },
                {
                    "name": "k2",
                    "cpu_op_name": "torch::long_name_a_extra",
                    "source": "trace2",
                    "kernel_time": 8.0,
                    "lowest_common_ancestor_id": 2,
                    "nn_module_parent": " Block ",
                    "nn_module_stack": "root",
                },
            ]
        )
        td.diff_stats_unique_args_summary_df = td.diff_stats_df.copy()
        td.get_cpu_op_to_kernels_json()
        assert "Renaming" in capsys.readouterr().out

    def test_print_merged_subtree_uninitialized(self):
        td = _make_tracediff()
        with pytest.raises(ValueError, match="merged_tree is not initialized"):
            td.print_merged_subtree(uid_tree1=1)

    def test_get_df_diff_stats_unique_args_with_op_name_filter(self):
        td = _make_tracediff()
        td.diff_stats_df = pd.DataFrame(
            {
                "name": ["k1", "k2"],
                "cpu_op_name": ["op1", "op2"],
                "source": ["trace1", "trace2"],
                "kernel_time": [1.0, 2.0],
                "lowest_common_ancestor_id": [0, 0],
                "gpu_op_uid": [1, 2],
                "nn_module_stack": ["r", "r"],
                "nn_module_parent": ["r", "r"],
            }
        )
        summary = td.get_df_diff_stats_unique_args(op_name="k1")
        assert summary is not None
        assert all(summary["name"] == "k1")

    def test_get_children_with_missing_same_name_off_gpu_path(self):
        events1 = [
            _mk_event("cpu_op", "aten::root", ts=0, dur=500, pid=1, tid=1),
            _mk_event("cpu_op", "aten::mm", ts=10, dur=100, pid=1, tid=1),
        ]
        _add_gpu_chain(events1, events1[1], 100, "k1", ts_launch=20, ts_kernel=60)
        events2 = [
            _mk_event("cpu_op", "aten::root", ts=0, dur=500, pid=1, tid=1),
            _mk_event("cpu_op", "aten::mm", ts=10, dur=100, pid=1, tid=1),
            _mk_event("cpu_op", "aten::mm", ts=200, dur=100, pid=1, tid=1),
        ]
        _add_gpu_chain(events2, events2[2], 200, "k2", ts_launch=210, ts_kernel=250)
        tree2 = _build_tree(events2)
        for e in tree2.events:
            if e[_TK.Name] == "aten::mm" and e[_TK.UID] == 1:
                e["non_gpu_path"] = True
        td = TraceDiff(_build_tree(events1), tree2)
        assert td.merged_tree is not None

    def test_generate_diff_stats_non_combined_kernel_children(self):
        td = _make_tracediff_from_specs(
            [("aten::mm", "gemm_v1", 100.0)],
            [("aten::mm", "gemm_v2", 80.0), ("aten::add", "add_v2", 10.0)],
        )
        df = td.generate_diff_stats()
        assert not df.empty
        assert set(df["source"]) == {"trace1", "trace2"}

    def test_get_cpu_op_to_kernels_json_equal_name_pairing(self, capsys):
        td = _make_tracediff()
        td.diff_stats_df = pd.DataFrame(
            [
                {
                    "name": "k1",
                    "cpu_op_name": "aten::same",
                    "source": "trace1",
                    "kernel_time": 10.0,
                    "lowest_common_ancestor_id": 3,
                    "nn_module_parent": "Mod",
                    "nn_module_stack": "root",
                },
                {
                    "name": "k2",
                    "cpu_op_name": "aten::same",
                    "source": "trace2",
                    "kernel_time": 8.0,
                    "lowest_common_ancestor_id": 3,
                    "nn_module_parent": "Mod",
                    "nn_module_stack": "root",
                },
            ]
        )
        td.diff_stats_unique_args_summary_df = td.diff_stats_df.copy()
        td.get_cpu_op_to_kernels_json()
        assert td.cpu_op_map is not None

    def test_merge_phase1_collapse_single_child_side(self):
        events1 = [
            _mk_event("cpu_op", "aten::root", ts=0, dur=800, pid=1, tid=1),
            _mk_event("cpu_op", "aten::wrap", ts=10, dur=300, pid=1, tid=1),
        ]
        inner = _mk_event("cpu_op", "aten::mm", ts=20, dur=100, pid=1, tid=1)
        _add_gpu_chain(events1, inner, 100, "gemm_v1", ts_launch=30, ts_kernel=70)
        events2 = [
            _mk_event("cpu_op", "aten::root", ts=0, dur=800, pid=1, tid=1),
            _mk_event("cpu_op", "aten::mm", ts=10, dur=100, pid=1, tid=1),
            _mk_event("cpu_op", "aten::add", ts=200, dur=100, pid=1, tid=1),
        ]
        _add_gpu_chain(events2, events2[1], 200, "gemm_v2", ts_launch=20, ts_kernel=60)
        _add_gpu_chain(events2, events2[2], 201, "add_v2", ts_launch=210, ts_kernel=250)
        td = TraceDiff(_build_tree(events1), _build_tree(events2))
        assert td.merged_tree is not None

    def test_merge_phase3_free_only_inserts_and_deletes(self):
        events1 = [
            _mk_event("cpu_op", "aten::root", ts=0, dur=800, pid=1, tid=1),
            _mk_event("cpu_op", "aten::wrap", ts=10, dur=200, pid=1, tid=1),
            _mk_event("cpu_op", "aten::only1", ts=300, dur=100, pid=1, tid=1),
        ]
        inner = _mk_event("cpu_op", "aten::mm", ts=20, dur=100, pid=1, tid=1)
        _add_gpu_chain(events1, inner, 100, "gemm_v1", ts_launch=30, ts_kernel=70)
        _add_gpu_chain(events1, events1[2], 101, "only_k1", ts_launch=310, ts_kernel=350)
        events2 = [
            _mk_event("cpu_op", "aten::root", ts=0, dur=800, pid=1, tid=1),
            _mk_event("cpu_op", "aten::mm", ts=10, dur=100, pid=1, tid=1),
            _mk_event("cpu_op", "aten::only2", ts=300, dur=100, pid=1, tid=1),
        ]
        _add_gpu_chain(events2, events2[1], 200, "gemm_v2", ts_launch=20, ts_kernel=60)
        _add_gpu_chain(events2, events2[2], 201, "only_k2", ts_launch=310, ts_kernel=350)
        td = TraceDiff(_build_tree(events1), _build_tree(events2))
        assert td.merged_tree is not None

    def test_get_cpu_op_to_kernels_json_one_to_many_kernel_mapping(self, capsys):
        td = _make_tracediff()
        td.diff_stats_df = pd.DataFrame(
            [
                {
                    "name": "shared_kernel",
                    "cpu_op_name": "aten::op_a",
                    "source": "trace1",
                    "kernel_time": 10.0,
                    "lowest_common_ancestor_id": 4,
                    "nn_module_parent": "M",
                    "nn_module_stack": "root",
                },
                {
                    "name": "shared_kernel",
                    "cpu_op_name": "aten::op_b",
                    "source": "trace1",
                    "kernel_time": 12.0,
                    "lowest_common_ancestor_id": 4,
                    "nn_module_parent": "M",
                    "nn_module_stack": "root",
                },
            ]
        )
        td.diff_stats_unique_args_summary_df = td.diff_stats_df.copy()
        td.get_cpu_op_to_kernels_json()
        assert "Kernel to CPU op mapping" in capsys.readouterr().out

    def test_get_cpu_op_to_kernels_json_exact_name_match_in_else_branch(self):
        td = _make_tracediff()
        td.diff_stats_df = pd.DataFrame(
            [
                {
                    "name": "k1",
                    "cpu_op_name": "aten::exact",
                    "source": "trace1",
                    "kernel_time": 10.0,
                    "lowest_common_ancestor_id": 5,
                    "nn_module_parent": "M",
                    "nn_module_stack": "root",
                },
                {
                    "name": "k2",
                    "cpu_op_name": "aten::other",
                    "source": "trace1",
                    "kernel_time": 5.0,
                    "lowest_common_ancestor_id": 5,
                    "nn_module_parent": "M",
                    "nn_module_stack": "root",
                },
                {
                    "name": "k3",
                    "cpu_op_name": "aten::exact",
                    "source": "trace2",
                    "kernel_time": 8.0,
                    "lowest_common_ancestor_id": 5,
                    "nn_module_parent": "M",
                    "nn_module_stack": "root",
                },
            ]
        )
        td.diff_stats_unique_args_summary_df = td.diff_stats_df.copy()
        td.get_cpu_op_to_kernels_json()
        assert td.cpu_op_map is not None

    def test_get_cpu_op_to_kernels_json_find_common_name_branches(self, capsys):
        td = _make_tracediff()
        td.diff_stats_df = pd.DataFrame(
            [
                {
                    "name": "k1",
                    "cpu_op_name": "aten::abcdef",
                    "source": "trace1",
                    "kernel_time": 10.0,
                    "lowest_common_ancestor_id": 6,
                    "nn_module_parent": "Block",
                    "nn_module_stack": "root",
                },
                {
                    "name": "k2",
                    "cpu_op_name": "aten::abc",
                    "source": "trace2",
                    "kernel_time": 8.0,
                    "lowest_common_ancestor_id": 6,
                    "nn_module_parent": "Block",
                    "nn_module_stack": "root",
                },
                {
                    "name": "k3",
                    "cpu_op_name": "aten::prefix1234567890",
                    "source": "trace1",
                    "kernel_time": 5.0,
                    "lowest_common_ancestor_id": 7,
                    "nn_module_parent": "Block",
                    "nn_module_stack": "root",
                },
                {
                    "name": "k4",
                    "cpu_op_name": "aten::prefix1234567899",
                    "source": "trace2",
                    "kernel_time": 4.0,
                    "lowest_common_ancestor_id": 7,
                    "nn_module_parent": "Block",
                    "nn_module_stack": "root",
                },
            ]
        )
        td.diff_stats_unique_args_summary_df = td.diff_stats_df.copy()
        td.get_cpu_op_to_kernels_json()
        assert "Renaming" in capsys.readouterr().out

    def test_get_cpu_op_to_kernels_json_trace2_one_to_many(self, capsys):
        td = _make_tracediff()
        td.diff_stats_df = pd.DataFrame(
            [
                {
                    "name": "shared_kernel",
                    "cpu_op_name": "aten::op_x",
                    "source": "trace2",
                    "kernel_time": 10.0,
                    "lowest_common_ancestor_id": 8,
                    "nn_module_parent": "M",
                    "nn_module_stack": "root",
                },
                {
                    "name": "shared_kernel",
                    "cpu_op_name": "aten::op_y",
                    "source": "trace2",
                    "kernel_time": 12.0,
                    "lowest_common_ancestor_id": 8,
                    "nn_module_parent": "M",
                    "nn_module_stack": "root",
                },
            ]
        )
        td.diff_stats_unique_args_summary_df = td.diff_stats_df.copy()
        td.get_cpu_op_to_kernels_json()
        assert "Kernel to CPU op mapping" in capsys.readouterr().out
>>>>>>> Stashed changes
