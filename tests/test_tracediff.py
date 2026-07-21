###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
from TraceLens import TraceDiff
from TraceLens.TraceDiff.trace_diff import (
    _disambiguate_same_name_candidates,
    _normalize_name_for_comparison,
    _gpu_path_child_names_at_bfs_levels,
)
from TraceLens.util import TraceEventUtils

_UID = TraceEventUtils.TraceKeys.UID


# ---------------------------------------------------------------------------
# Minimal synthetic tree used by merge_trees tests
# ---------------------------------------------------------------------------

class FakeTree:
    """Minimal stand-in for TraceToTree that satisfies TraceDiff.merge_trees."""

    def __init__(self, root_uids, uid2node):
        self.cpu_root_nodes = root_uids
        self.events_by_uid = uid2node

    def get_UID2event(self, uid):
        return self.events_by_uid[uid]

    def label_non_gpu_paths(self):
        pass

    def event_to_category(self, node):
        return node.get("cat", "")


def _node(uid, name, cat, children, ts=0, parent=None, non_gpu_path=False):
    return {
        _UID: uid,
        "name": name,
        "cat": cat,
        "children": children,
        "ts": ts,
        "non_gpu_path": non_gpu_path,
        "nn_module_stack": "",
        "parent": parent,
    }


def _merged_types(td):
    events, _ = td.merged_tree
    return [e["merged_type"] for e in events]


def _events(td):
    events, _ = td.merged_tree
    return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        """[A,A,A] vs [A,A,A,A] — no trace2 index should appear in more than one match."""
        uid2node = _make_uid2node([
            ("u0", "opA", [], False),
            ("u1", "opA", [], False),
            ("u2", "opA", [], False),
            ("v0", "opA", [], False),
            ("v1", "opA", [], False),
            ("v2", "opA", [], False),
            ("v3", "opA", [], False),
        ])
        children1 = ["u0", "u1", "u2"]
        children2 = ["v0", "v1", "v2", "v3"]
        # WF produces 3 matches + 1 insert (all same name, no subtree diff)
        ops = [
            ("match", 0, 0),
            ("match", 1, 1),
            ("match", 2, 2),
            ("insert", None, 3),
        ]
        result = _disambiguate_same_name_candidates(ops, children1, children2, uid2node, uid2node)
        match_j_vals = [j for op, i, j in result if op == "match"]
        assert len(match_j_vals) == len(set(match_j_vals)), "Duplicate trace2 index in matches"

    def test_no_duplicate_trace1_index_in_matches(self):
        """[A,A,A,A] vs [A,A,A] — no trace1 index should appear in more than one match."""
        uid2node = _make_uid2node([
            ("u0", "opA", [], False),
            ("u1", "opA", [], False),
            ("u2", "opA", [], False),
            ("u3", "opA", [], False),
            ("v0", "opA", [], False),
            ("v1", "opA", [], False),
            ("v2", "opA", [], False),
        ])
        children1 = ["u0", "u1", "u2", "u3"]
        children2 = ["v0", "v1", "v2"]
        ops = [
            ("match", 0, 0),
            ("match", 1, 1),
            ("match", 2, 2),
            ("delete", 3, None),
        ]
        result = _disambiguate_same_name_candidates(ops, children1, children2, uid2node, uid2node)
        match_i_vals = [i for op, i, j in result if op == "match"]
        assert len(match_i_vals) == len(set(match_i_vals)), "Duplicate trace1 index in matches"

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
        """Issue 2: minority side must be fully matched.

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


# ---------------------------------------------------------------------------
# _normalize_name_for_comparison
# ---------------------------------------------------------------------------

class TestNormalizeNameForComparison:

    def test_none_returns_none(self):
        assert _normalize_name_for_comparison(None) is None

    def test_plain_name_unchanged(self):
        assert _normalize_name_for_comparison("aten::linear") == "aten::linear"

    def test_hex_address_replaced(self):
        result = _normalize_name_for_comparison("op_0xDEADBEEF_fwd")
        assert "0xXXXX" in result
        assert "0xDEADBEEF" not in result

    def test_multiple_hex_addresses_all_replaced(self):
        result = _normalize_name_for_comparison("op_0xAAA_and_0xBBB")
        assert result.count("0xXXXX") == 2

    def test_line_number_stripped(self):
        result = _normalize_name_for_comparison("train.py(42): forward")
        assert "(42)" not in result
        assert "train.py:" in result

    def test_kernel_launch_remapped_hip(self):
        assert _normalize_name_for_comparison("hipModuleLaunchKernel") == "__kernel_launch__"

    def test_kernel_launch_remapped_cuda(self):
        assert _normalize_name_for_comparison("cuLaunchKernel") == "__kernel_launch__"

    def test_strip_details_removes_function_name(self):
        result = _normalize_name_for_comparison("/path/to/train.py(10): forward", strip_details=True)
        assert "forward" not in result

    def test_strip_details_removes_directory(self):
        result = _normalize_name_for_comparison("/path/to/train.py(10): forward", strip_details=True)
        assert "/path/to/" not in result
        assert "train.py" in result

    def test_strip_details_false_preserves_function(self):
        result = _normalize_name_for_comparison("/path/to/train.py(10): forward", strip_details=False)
        assert "forward" in result


# ---------------------------------------------------------------------------
# _gpu_path_child_names_at_bfs_levels
# ---------------------------------------------------------------------------

def _noop_normalize(name):
    return name


def _make_get_name(uid2node):
    def get_name(uid, tree_num):
        node = uid2node.get(uid)
        return node.get("name") if node else None
    return get_name


class TestGpuPathChildNamesAtBfsLevels:

    def _run(self, uid, uid2node, max_depth=4):
        get_name = _make_get_name(uid2node)
        return _gpu_path_child_names_at_bfs_levels(
            uid, uid2node, 1, max_depth, _noop_normalize, get_name
        )

    def _make_nodes(self, entries):
        """entries: list of (uid, name, children, non_gpu_path)"""
        return {
            uid: {"name": name, "children": children, "non_gpu_path": ngp}
            for uid, name, children, ngp in entries
        }

    def test_leaf_node_returns_empty_names_at_level1(self):
        """A leaf with no children produces one level containing an empty tuple."""
        uid2node = self._make_nodes([("root", "opA", [], False)])
        result = self._run("root", uid2node)
        assert result == [()]

    def test_single_gpu_child_at_level_1(self):
        uid2node = self._make_nodes([
            ("root", "opA", ["c1"], False),
            ("c1", "kernel_X", [], False),
        ])
        result = self._run("root", uid2node)
        assert result[0] == ("kernel_X",)

    def test_non_gpu_path_child_excluded(self):
        uid2node = self._make_nodes([
            ("root", "opA", ["c1", "c2"], False),
            ("c1", "kernel_X", [], False),
            ("c2", "cpu_op_Y", [], True),  # non_gpu_path — should be excluded
        ])
        result = self._run("root", uid2node)
        assert "cpu_op_Y" not in result[0]
        assert "kernel_X" in result[0]

    def test_level_names_are_sorted(self):
        uid2node = self._make_nodes([
            ("root", "opA", ["c1", "c2"], False),
            ("c1", "zzz", [], False),
            ("c2", "aaa", [], False),
        ])
        result = self._run("root", uid2node)
        assert result[0] == ("aaa", "zzz")

    def test_multi_level_bfs(self):
        uid2node = self._make_nodes([
            ("root", "opA", ["c1"], False),
            ("c1", "mid", ["gc1"], False),
            ("gc1", "kernel_X", [], False),
        ])
        result = self._run("root", uid2node)
        assert result[0] == ("mid",)
        assert result[1] == ("kernel_X",)

    def test_max_depth_caps_traversal(self):
        uid2node = self._make_nodes([
            ("root", "opA", ["c1"], False),
            ("c1", "lvl1", ["c2"], False),
            ("c2", "lvl2", ["c3"], False),
            ("c3", "lvl3", [], False),
        ])
        result = self._run("root", uid2node, max_depth=2)
        assert len(result) == 2
        assert result[0] == ("lvl1",)
        assert result[1] == ("lvl2",)

    def test_missing_uid_in_uid2node_does_not_crash(self):
        uid2node = self._make_nodes([
            ("root", "opA", ["missing_uid"], False),
        ])
        result = self._run("root", uid2node)
        # missing child is silently skipped; level 1 has no valid children
        assert result == [()]

    def test_unknown_root_returns_empty_level(self):
        """An unknown root UID yields one empty level (node not found, no children)."""
        uid2node = self._make_nodes([("root", "opA", [], False)])
        result = self._run("nonexistent", uid2node)
        assert result == [()]


# ---------------------------------------------------------------------------
# Synthetic trees used by merge_trees / reconcile / phase tests
# ---------------------------------------------------------------------------
#
# Tree layout used by most tests:
#
#   forward (python_function)
#   ├── aten::linear (cpu_op)  [ts=10]
#   │   └── sgemm (kernel)    [ts=15]
#   └── aten::relu  (cpu_op)  [ts=50]   ← trace2 may differ here
#

def _make_tree1():
    nodes = {
        0: _node(0, "forward",       "python_function", [1, 2], parent=None),
        1: _node(1, "aten::linear",  "cpu_op",          [3],    ts=10, parent=0),
        2: _node(2, "aten::relu",    "cpu_op",          [],     ts=50, parent=0),
        3: _node(3, "sgemm",         "kernel",          [],     ts=15, parent=1),
    }
    return FakeTree([0], nodes)


def _make_tree2_identical():
    """Identical to tree1 — self-diff should be all-combined."""
    nodes = {
        0: _node(0, "forward",       "python_function", [1, 2], parent=None),
        1: _node(1, "aten::linear",  "cpu_op",          [3],    ts=10, parent=0),
        2: _node(2, "aten::relu",    "cpu_op",          [],     ts=50, parent=0),
        3: _node(3, "sgemm",         "kernel",          [],     ts=15, parent=1),
    }
    return FakeTree([0], nodes)


def _make_tree2_extra_child():
    """trace2 has one extra cpu_op child → WF produces a trace2-only node."""
    nodes = {
        0: _node(0, "forward",        "python_function", [1, 2, 4], parent=None),
        1: _node(1, "aten::linear",   "cpu_op",          [3],       ts=10, parent=0),
        2: _node(2, "aten::relu",     "cpu_op",          [],        ts=50, parent=0),
        3: _node(3, "sgemm",          "kernel",          [],        ts=15, parent=1),
        4: _node(4, "aten::dropout",  "cpu_op",          [],        ts=80, parent=0),
    }
    return FakeTree([0], nodes)


def _make_tree2_wrapper():
    """trace2 wraps aten::linear inside an extra cpu_op layer.
    reconcile_unmatched should unwrap it and still match the inner nodes."""
    nodes = {
        0: _node(0, "forward",        "python_function", [1, 2], parent=None),
        1: _node(1, "wrapper_op",     "cpu_op",          [5],    ts=10, parent=0),
        2: _node(2, "aten::relu",     "cpu_op",          [],     ts=50, parent=0),
        5: _node(5, "aten::linear",   "cpu_op",          [3],    ts=11, parent=1),
        3: _node(3, "sgemm",          "kernel",          [],     ts=15, parent=5),
    }
    return FakeTree([0], nodes)


def _make_tree1_with_wrapper():
    """trace1 has a wrapper around [linear, relu]; trace2 has them flat.
    Phase 1 should collapse the wrapper so children counts match."""
    nodes = {
        0:  _node(0,  "forward",       "python_function", [10],    parent=None),
        10: _node(10, "wrapper",       "cpu_op",          [1, 2],  ts=5,  parent=0),
        1:  _node(1,  "aten::linear",  "cpu_op",          [3],     ts=10, parent=10),
        2:  _node(2,  "aten::relu",    "cpu_op",          [],      ts=50, parent=10),
        3:  _node(3,  "sgemm",         "kernel",          [],      ts=15, parent=1),
    }
    return FakeTree([0], nodes)


def _make_tree2_flat():
    """Flat layout matching the inner children of _make_tree1_with_wrapper."""
    nodes = {
        0: _node(0, "forward",       "python_function", [1, 2], parent=None),
        1: _node(1, "aten::linear",  "cpu_op",          [3],    ts=10, parent=0),
        2: _node(2, "aten::relu",    "cpu_op",          [],     ts=50, parent=0),
        3: _node(3, "sgemm",         "kernel",          [],     ts=15, parent=1),
    }
    return FakeTree([0], nodes)


# ---------------------------------------------------------------------------
# TestReconcileUnmatched
# ---------------------------------------------------------------------------

class TestReconcileUnmatched:

    def test_merged_tree_produced(self):
        """merge_trees must always return a non-empty event list and root list."""
        td = TraceDiff(_make_tree1(), _make_tree2_identical())
        events, root_ids = td.merged_tree
        assert len(events) > 0
        assert len(root_ids) > 0

    def test_child_merged_ids_all_valid(self):
        """Every child merged_id in any event must reference an existing event."""
        td = TraceDiff(_make_tree1(), _make_tree2_extra_child())
        events, _ = td.merged_tree
        all_ids = {e["merged_id"] for e in events}
        for e in events:
            for cid in e["children"]:
                assert cid in all_ids

    def test_combined_nodes_have_both_uids(self):
        """combined events must have non-None uid1 and uid2."""
        td = TraceDiff(_make_tree1(), _make_tree2_identical())
        for e in _events(td):
            if e["merged_type"] == "combined":
                assert e["uid1"] is not None
                assert e["uid2"] is not None

    def test_trace1_only_nodes_have_no_uid2(self):
        """trace1 events must have uid2=None."""
        td = TraceDiff(_make_tree1(), _make_tree2_extra_child())
        for e in _events(td):
            if e["merged_type"] == "trace1":
                assert e["uid2"] is None

    def test_trace2_only_nodes_have_no_uid1(self):
        """trace2 events must have uid1=None."""
        td = TraceDiff(_make_tree1(), _make_tree2_extra_child())
        for e in _events(td):
            if e["merged_type"] == "trace2":
                assert e["uid1"] is None

    def test_all_merged_types_valid(self):
        td = TraceDiff(_make_tree1(), _make_tree2_extra_child())
        valid = {"combined", "trace1", "trace2"}
        for e in _events(td):
            assert e["merged_type"] in valid

    def test_extra_trace2_child_produces_trace2_only_node(self):
        """aten::dropout exists only in trace2 → exactly one trace2-only event."""
        td = TraceDiff(_make_tree1(), _make_tree2_extra_child())
        trace2_only = [e for e in _events(td) if e["merged_type"] == "trace2"]
        assert len(trace2_only) >= 1

    def test_identical_trees_produce_no_unmatched_nodes(self):
        """Self-diff must produce zero trace1-only or trace2-only events."""
        td = TraceDiff(_make_tree1(), _make_tree2_identical())
        types = _merged_types(td)
        assert "trace1" not in types
        assert "trace2" not in types


# ---------------------------------------------------------------------------
# TestTraverseAndMergePhases
# ---------------------------------------------------------------------------

class TestTraverseAndMergePhases:

    def test_phase1_skipped_equal_children_all_combined(self):
        """Phase 1 (single-child collapse) is skipped when both sides have equal
        child counts. Identical trees → every event is combined."""
        td = TraceDiff(_make_tree1(), _make_tree2_identical())
        assert all(e["merged_type"] == "combined" for e in _events(td))

    def test_phase1_fires_collapses_single_child_wrapper(self):
        """Phase 1 fires when one side has 1 GPU child and the other has >1.
        trace1 has forward→wrapper→[linear,relu]; trace2 has forward→[linear,relu].
        After collapse, linear, relu, and sgemm must all be combined."""
        td = TraceDiff(_make_tree1_with_wrapper(), _make_tree2_flat())
        combined_uid1s = {e["uid1"] for e in _events(td) if e["merged_type"] == "combined"}
        assert 1 in combined_uid1s, "aten::linear should be combined after phase 1 collapse"
        assert 2 in combined_uid1s, "aten::relu should be combined after phase 1 collapse"
        assert 3 in combined_uid1s, "sgemm should be combined after phase 1 collapse"

    def test_phase2_positional_match_on_equal_length_no_cuda_runtime(self):
        """Phase 2 fast path: equal-length, no cuda_runtime → positional (i,i)
        matches. Identical trees guarantee all nodes are combined."""
        td = TraceDiff(_make_tree1(), _make_tree2_identical())
        events = _events(td)
        combined = [e for e in events if e["merged_type"] == "combined"]
        assert len(combined) == len(events)

    def test_phase3_reconcile_unmatched_unequal_children(self):
        """Phase 3 fires when there are unmatched ops after WF. The shared ops
        (forward, aten::linear, sgemm) must still be combined."""
        td = TraceDiff(_make_tree1(), _make_tree2_extra_child())
        combined = [e for e in _events(td) if e["merged_type"] == "combined"]
        combined_uid1s = {e["uid1"] for e in combined}
        # forward(0), aten::linear(1), sgemm(3) are in both traces identically
        assert 0 in combined_uid1s
        assert 1 in combined_uid1s
        assert 3 in combined_uid1s

    def test_phase4_collapse_matches_renamed_wrapper(self):
        """Phase 4 collapses single-GPU-child wrappers to find matches.
        tree2 wraps aten::linear in wrapper_op; after collapse, aten::linear
        and sgemm should still appear as combined."""
        td = TraceDiff(_make_tree1(), _make_tree2_wrapper())
        combined = [e for e in _events(td) if e["merged_type"] == "combined"]
        combined_uid1s = {e["uid1"] for e in combined}
        assert 1 in combined_uid1s  # aten::linear matched through wrapper
        assert 3 in combined_uid1s  # sgemm matched

    def test_no_duplicate_merged_ids(self):
        """merged_id must be unique across all events."""
        td = TraceDiff(_make_tree1(), _make_tree2_extra_child())
        ids = [e["merged_id"] for e in _events(td)]
        assert len(ids) == len(set(ids))

    def test_root_ids_are_valid_merged_ids(self):
        """Root merged_ids must reference existing events."""
        td = TraceDiff(_make_tree1(), _make_tree2_extra_child())
        events, root_ids = td.merged_tree
        all_ids = {e["merged_id"] for e in events}
        for rid in root_ids:
            assert rid in all_ids


