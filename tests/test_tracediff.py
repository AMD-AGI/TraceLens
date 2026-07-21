###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
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
        uid2node = _make_uid2node([
            ("u0", "opA", [], False),
            ("u1", "opB", [], False),
            ("v0", "opA", [], False),
            ("v1", "opC", [], False),
        ])
        children1 = ["u0", "u1"]
        children2 = ["v0", "v1"]
        ops = [("match", 0, 0), ("match", 1, 1)]
        result = _disambiguate_same_name_candidates(ops, children1, children2, uid2node, uid2node)
        assert result == ops

    def test_unique_survivor_trace2_triggers_reassignment(self):
        """WF matched trace1[0] with trace2[0], but trace2[1] has the same name
        and identical subtree children → trace2[1] should win because it better
        matches trace1[0]'s children."""
        uid2node = _make_uid2node([
            # trace1[0]=u0 has child "kernel_X"
            ("u0", "opA", ["u0c"], False),
            ("u0c", "kernel_X", [], False),
            # trace2[0]=v0 has child "kernel_Y" (different → evicted)
            ("v0", "opA", ["v0c"], False),
            ("v0c", "kernel_Y", [], False),
            # trace2[1]=v1 has child "kernel_X" (same → survives)
            ("v1", "opA", ["v1c"], False),
            ("v1c", "kernel_X", [], False),
        ])
        children1 = ["u0"]
        children2 = ["v0", "v1"]
        # WF matched u0↔v0, v1 is an insert
        ops = [("match", 0, 0), ("insert", None, 1)]
        result = _disambiguate_same_name_candidates(ops, children1, children2, uid2node, uid2node)
        pairs = _match_pairs(result)
        assert (0, 1) in pairs, "Should reassign to trace2[1] (better subtree match)"
        assert (0, 0) not in pairs

    def test_multiple_survivors_keeps_original_wf_match(self):
        """When >1 survivors remain, WF's choice is kept."""
        uid2node = _make_uid2node([
            ("u0", "opA", [], False),
            ("v0", "opA", [], False),
            ("v1", "opA", [], False),
        ])
        children1 = ["u0"]
        children2 = ["v0", "v1"]
        ops = [("match", 0, 0), ("insert", None, 1)]
        result = _disambiguate_same_name_candidates(ops, children1, children2, uid2node, uid2node)
        pairs = _match_pairs(result)
        assert (0, 0) in pairs, "Original WF match should be preserved when tie can't be broken"

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
        uid2node = _make_uid2node([
            ("u0", "opA", ["u0c1", "u0c2"], False),
            ("u0c1", "kernel_X", [], False), ("u0c2", "kernel_Y", [], False),
            ("u1", "opA", ["u1c1", "u1c2"], False),
            ("u1c1", "kernel_X", [], False), ("u1c2", "kernel_Y", [], False),
            ("u2", "opA", ["u2c1", "u2c2"], False),
            ("u2c1", "kernel_X", [], False), ("u2c2", "kernel_Y", [], False),
            ("v0", "opA", ["v0c1", "v0c2"], False),
            ("v0c1", "kernel_X", [], False), ("v0c2", "kernel_Y", [], False),
            ("v1", "opA", ["v1c1"], False), ("v1c1", "kernel_X", [], False),
            ("v2", "opA", ["v2c1"], False), ("v2c1", "kernel_X", [], False),
            ("v3", "opA", ["v3c1"], False), ("v3c1", "kernel_X", [], False),
        ])
        children1 = ["u0", "u1", "u2"]
        children2 = ["v0", "v1", "v2", "v3"]
        ops = [
            ("match", 0, 1),
            ("match", 1, 2),
            ("match", 2, 3),
            ("insert", None, 0),
        ]
        result = _disambiguate_same_name_candidates(ops, children1, children2, uid2node, uid2node)
        match_j_vals = [j for op, i, j in result if op == "match"]
        assert len(match_j_vals) == len(set(match_j_vals)), "Duplicate trace2 index in matches"
        assert 0 in match_j_vals, "v0 (best match for all) must be assigned to exactly one event"

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
        uid2node = _make_uid2node([
            ("u0", "opA", ["u0c1", "u0c2"], False),
            ("u0c1", "kernel_X", [], False), ("u0c2", "kernel_Y", [], False),
            ("u1", "opA", ["u1c1"], False), ("u1c1", "kernel_X", [], False),
            ("u2", "opA", ["u2c1"], False), ("u2c1", "kernel_X", [], False),
            ("u3", "opA", ["u3c1"], False), ("u3c1", "kernel_X", [], False),
            ("v0", "opA", ["v0c1", "v0c2"], False),
            ("v0c1", "kernel_X", [], False), ("v0c2", "kernel_Y", [], False),
            ("v1", "opA", ["v1c1", "v1c2"], False),
            ("v1c1", "kernel_X", [], False), ("v1c2", "kernel_Y", [], False),
            ("v2", "opA", ["v2c1", "v2c2"], False),
            ("v2c1", "kernel_X", [], False), ("v2c2", "kernel_Y", [], False),
        ])
        children1 = ["u0", "u1", "u2", "u3"]
        children2 = ["v0", "v1", "v2"]
        ops = [
            ("match", 1, 0),
            ("match", 2, 1),
            ("match", 3, 2),
            ("delete", 0, None),
        ]
        result = _disambiguate_same_name_candidates(ops, children1, children2, uid2node, uid2node)
        match_i_vals = [i for op, i, j in result if op == "match"]
        assert len(match_i_vals) == len(set(match_i_vals)), "Duplicate trace1 index in matches"
        assert 0 in match_i_vals, "u0 (best match for all) must be assigned to exactly one event"

    def test_insert_and_delete_preserved_when_no_reassignment(self):
        """Unambiguous inserts and deletes should pass through unchanged."""
        uid2node = _make_uid2node([
            ("u0", "opA", [], False),
            ("u1", "opB", [], False),
            ("v0", "opA", [], False),
            ("v1", "opC", [], False),
        ])
        children1 = ["u0", "u1"]
        children2 = ["v0", "v1"]
        ops = [("match", 0, 0), ("delete", 1, None), ("insert", None, 1)]
        result = _disambiguate_same_name_candidates(ops, children1, children2, uid2node, uid2node)
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
        uid2node = _make_uid2node([
            ("u0", "opA", ["kX1"], False), ("kX1", "kernel_X", [], False),
            ("u1", "opA", ["kY1"], False), ("kY1", "kernel_Y", [], False),
            ("v0", "opA", ["kX2"], False), ("kX2", "kernel_X", [], False),
            ("v1", "opA", ["kZ"],  False), ("kZ",  "kernel_Z", [], False),
            ("v2", "opA", ["kY2"], False), ("kY2", "kernel_Y", [], False),
        ])
        children1 = ["u0", "u1"]
        children2 = ["v0", "v1", "v2"]
        ops = [("match", 0, 0), ("match", 1, 1), ("insert", None, 2)]
        result = _disambiguate_same_name_candidates(ops, children1, children2, uid2node, uid2node)
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
        uid2node = _make_uid2node([
            ("u0", "opA", ["kX1"], False), ("kX1", "kernel_X", [], False),
            ("u1", "opA", ["kX2"], False), ("kX2", "kernel_X", [], False),
            ("v0", "opA", ["kY"],  False), ("kY",  "kernel_Y", [], False),
            ("v1", "opA", ["kX3"], False), ("kX3", "kernel_X", [], False),
            ("v2", "opA", ["kX4"], False), ("kX4", "kernel_X", [], False),
        ])
        children1 = ["u0", "u1"]
        children2 = ["v0", "v1", "v2"]
        ops = [("match", 0, 0), ("match", 1, 1), ("insert", None, 2)]
        result = _disambiguate_same_name_candidates(ops, children1, children2, uid2node, uid2node)
        pairs = _match_pairs(result)
        assert (0, 2) in pairs, "u0 should be corrected to v2(kX) — the only insert candidate"
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
        uid2node = _make_uid2node([
            ("u0", "opA", ["u0c"], False), ("u0c", "kernel_X", [], False),
            ("u1", "opA", ["u1c"], False), ("u1c", "kernel_Y", [], False),
            ("v0", "opA", ["v0c"], False), ("v0c", "kernel_Y", [], False),
            ("v1", "opA", ["v1c"], False), ("v1c", "kernel_X", [], False),
        ])
        children1 = ["u0", "u1"]
        children2 = ["v0", "v1"]
        ops = [("match", 0, 0), ("match", 1, 1)]
        result = _disambiguate_same_name_candidates(ops, children1, children2, uid2node, uid2node)
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
        uid2node = _make_uid2node([
            ("u0", "opA", ["u0c"], False), ("u0c", "kernel_A", [], False),
            ("u1", "opA", ["u1c"], False), ("u1c", "kernel_B", [], False),
            ("u2", "opA", ["u2c"], False), ("u2c", "kernel_C", [], False),
            ("u3", "opA", ["u3c"], False), ("u3c", "kernel_D", [], False),
            ("u4", "opA", ["u4c"], False), ("u4c", "kernel_E", [], False),
            ("v0", "opA", ["v0c"], False), ("v0c", "kernel_A", [], False),
            ("v1", "opA", ["v1c"], False), ("v1c", "kernel_C", [], False),
            ("v2", "opA", ["v2c"], False), ("v2c", "kernel_B", [], False),
        ])
        children1 = ["u0", "u1", "u2", "u3", "u4"]
        children2 = ["v0", "v1", "v2"]
        ops = [
            ("match", 0, 0),
            ("match", 1, 1),
            ("match", 2, 2),
            ("delete", 3, None),
            ("delete", 4, None),
        ]
        result = _disambiguate_same_name_candidates(ops, children1, children2, uid2node, uid2node)
        pairs = _match_pairs(result)
        deletes = _delete_idxs(result)
        # All 3 trace2 nodes must be matched
        matched_j = {j for _, j in pairs}
        assert matched_j == {0, 1, 2}, "All trace2 nodes must be matched (minority side)"
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
        uid2node = _make_uid2node([
            ("u0", "opA", ["u0c"], False),
            ("u0c", "kernel_P", ["u0cc"], False),
            ("u0cc", "sub_Y", [], False),
            ("u1", "opA", ["u1c"], False), ("u1c", "kernel_B", [], False),
            ("u2", "opA", ["u2c"], False),
            ("u2c", "kernel_P", ["u2cc"], False),
            ("u2cc", "sub_X", [], False),
            ("u3", "opA", ["u3c"], False), ("u3c", "kernel_D", [], False),
            ("u4", "opA", ["u4c"], False), ("u4c", "kernel_E", [], False),
            ("v0", "opA", ["v0c"], False),
            ("v0c", "kernel_P", ["v0cc"], False),
            ("v0cc", "sub_X", [], False),
            ("v1", "opA", ["v1c"], False), ("v1c", "kernel_B", [], False),
            ("v2", "opA", ["v2c"], False),
            ("v2c", "kernel_P", ["v2cc"], False),
            ("v2cc", "sub_Y", [], False),
        ])
        children1 = ["u0", "u1", "u2", "u3", "u4"]
        children2 = ["v0", "v1", "v2"]
        ops = [
            ("match", 0, 0),
            ("match", 1, 1),
            ("match", 2, 2),
            ("delete", 3, None),
            ("delete", 4, None),
        ]
        result = _disambiguate_same_name_candidates(ops, children1, children2, uid2node, uid2node)
        pairs = _match_pairs(result)
        assert (2, 0) in pairs, "trace1[2](kP/sX) should match trace2[0](kP/sX) — deeper match"
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
        uid2node = _make_uid2node([
            ("a0", "opA", ["a0c1"], False),
            ("a0c1", "kernel_P", [], False),
            ("a1", "opA", ["a1c1", "a1c2"], False),
            ("a1c1", "kernel_P", [], False),
            ("a1c2", "kernel_Q", [], False),
            ("b0", "opA", ["b0c1", "b0c2", "b0c3"], False),
            ("b0c1", "kernel_P", [], False),
            ("b0c2", "kernel_Q", [], False),
            ("b0c3", "kernel_S", [], False),
        ])
        children1 = ["a0", "a1"]
        children2 = ["b0"]
        ops = [("match", 0, 0), ("delete", 1, None)]
        result = _disambiguate_same_name_candidates(ops, children1, children2, uid2node, uid2node)
        pairs = _match_pairs(result)
        deletes = _delete_idxs(result)
        assert (1, 0) in pairs, "a1 (overlap=2) should win over a0 (overlap=1)"
        assert 0 in deletes, "a0 should become a delete"

    def test_all_ops_indices_valid(self):
        """All match/delete/insert indices in the result must be within bounds."""
        uid2node = _make_uid2node([
            ("u0", "opA", ["u0c"], False),
            ("u0c", "kX", [], False),
            ("v0", "opA", ["v0c"], False),
            ("v0c", "kY", [], False),
            ("v1", "opA", ["v1c"], False),
            ("v1c", "kX", [], False),
            ("v2", "opB", [], False),
        ])
        children1 = ["u0"]
        children2 = ["v0", "v1", "v2"]
        ops = [("match", 0, 0), ("insert", None, 1), ("insert", None, 2)]
        result = _disambiguate_same_name_candidates(ops, children1, children2, uid2node, uid2node)
        for op, i, j in result:
            if i is not None:
                assert 0 <= i < len(children1)
            if j is not None:
                assert 0 <= j < len(children2)
