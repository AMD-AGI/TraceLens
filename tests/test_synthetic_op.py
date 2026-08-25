###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit test for synthetic-op vs normal-op classification in the unified perf table.

Builds a small hand-crafted trace tree matching this shape::

    cpu_op_E                         (leaf cpu_op, NO perf model)
    ├── cpu_op_B  (HAS perf model)   -> runtime -> k_B
    ├── cpu_op_C  (NO perf model)    -> runtime -> k_C
    └── runtime -> k_own             (E's own kernel, no cpu_op between)

Expected unified-perf rows:

  * cpu_op_B  -> a normal row owning k_B      (collected via its perf model)
  * cpu_op_C  -> a normal row owning k_C      (fully represents its subtree)
  * cpu_op_E  -> a synthetic op row "cpu_op_E->k_own (Synthetic Op)"
                 owning only k_own            (E is only partially represented:
                 k_B and k_C are owned by the finer ops B and C)

``(Synthetic Op)`` must be reserved for a partially-represented cpu_op (E), not
for a cpu_op that fully owns its kernels (C).
"""

from copy import deepcopy

import pytest

from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

# cpu_op_B must be an op name that has a registered perf model.
B_NAME = "aten::mm"
_DIMS = {"Input Dims": [[32, 64], [64, 128]], "Input type": ["fp16", "fp16"]}


def _ev(uid, ts, dur, cat, name, pid=100, tid=100, args=None):
    e = {
        "ph": "X",
        "UID": uid,
        "ts": ts,
        "dur": dur,
        "cat": cat,
        "name": name,
        "pid": pid,
        "tid": tid,
    }
    if args is not None:
        e["args"] = args
    return e


def _ac2g(corr, pid, tid, ts, phase):
    e = {
        "ph": phase,
        "id": corr,
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "cat": "ac2g",
        "name": "ac2g",
    }
    if phase == "f":
        e["bp"] = "e"
    return e


def _launcher_and_kernel(corr, kernel_name, rt_ts, k_ts):
    """A cuda_runtime launcher + its GPU kernel, linked via ac2g flow events."""
    return [
        _ev(f"rt{corr}", rt_ts, 5, "cuda_runtime", "hipLaunchKernel",
            args={"correlation": corr}),
        _ev(f"k{corr}", k_ts, 30, "kernel", kernel_name, pid=0, tid=7,
            args={"correlation": corr, "stream": 7}),
        _ac2g(corr, 0, 7, k_ts, "s"),
        _ac2g(corr, 0, 7, k_ts + 30, "f"),
    ]


def _build_example_tree_rows(add_python_func=False):
    # Host nesting is by timestamp: rt/cpu_op children must fall inside their
    # parent's [ts, ts+dur) and share pid/tid.
    events = [
        # E: outermost leaf cpu_op, no perf model  -> [1000, 2000]
        _ev("E", 1000, 1000, "cpu_op", "cpu_op_E", args=_DIMS),
        # B: perf-modeled cpu_op inside E          -> [1050, 1150]
        _ev("B", 1050, 100, "cpu_op", B_NAME, args=_DIMS),
        # C: non-modeled cpu_op inside E           -> [1200, 1300]
        _ev("C", 1200, 100, "cpu_op", "cpu_op_C", args=_DIMS),
    ]
    # B's launcher/kernel nest inside B; C's inside C; E's own launcher inside E
    # (but outside B and C).
    events += _launcher_and_kernel(1, "k_B", rt_ts=1060, k_ts=1500)
    events += _launcher_and_kernel(2, "k_C", rt_ts=1210, k_ts=1600)
    events += _launcher_and_kernel(3, "k_own", rt_ts=1400, k_ts=1700)

    tree = TraceToTree(deepcopy(events), prune_nongpu_paths=False)
    tree.build_tree(add_python_func=add_python_func)
    analyzer = TreePerfAnalyzer(
        tree, add_python_func=add_python_func, rebuild_tree=False
    )
    df = analyzer.build_df_unified_perf_table(include_perf_metrics=False)
    return df, analyzer


def _kernel_names(row):
    return [k.get("name") for k in (row.get("kernel_details") or [])]


_GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def _tree_kernel_uids(analyzer):
    return {
        e["UID"]
        for e in analyzer.tree.events
        if analyzer.event_to_category(e) in _GPU_CATS
    }


def _row_kernel_uids(df):
    """Every gpu-op UID referenced across all rows' kernel_details (as a list, so
    duplicates -- a kernel counted by two rows -- are detectable)."""
    uids = []
    for _, row in df.iterrows():
        for kd in row.get("kernel_details") or []:
            uids.append(kd.get("gpu_op_uid"))
    return uids


def _assert_every_kernel_accounted_once(df, analyzer):
    """Core invariant: each GPU kernel is owned by exactly one row (mirrors the
    report's 'Kernels accounted: N/N' sanity check)."""
    row_uids = _row_kernel_uids(df)
    assert len(row_uids) == len(set(row_uids)), (
        f"a kernel is attributed to more than one row: {row_uids}"
    )
    assert set(row_uids) == _tree_kernel_uids(analyzer), (
        "kernels are dropped or duplicated: "
        f"rows={sorted(set(row_uids))} tree={sorted(_tree_kernel_uids(analyzer))}"
    )


def _call_stack(row):
    """The per-row call stack: last kernel's frame chain (kd['call_stack'] +
    kernel name). Requires add_python_func=True."""
    kd = (row.get("kernel_details") or [])
    if not kd:
        return []
    return list(kd[-1].get("call_stack", [])) + [kd[-1].get("name")]


def test_synthetic_op_row_classification():
    df, analyzer = _build_example_tree_rows()
    _assert_every_kernel_accounted_once(df, analyzer)
    rows = {r["name"]: r for _, r in df.iterrows()}

    # cpu_op_B: normal row (collected via its perf model), owns only k_B.
    assert B_NAME in rows, f"expected a '{B_NAME}' row; got {list(rows)}"
    assert "(Synthetic Op)" not in B_NAME
    assert rows[B_NAME]["has_perf_model"] is True
    assert _kernel_names(rows[B_NAME]) == ["k_B"]

    # cpu_op_C: fully represents its subtree -> a NORMAL row, not a synthetic.
    assert "cpu_op_C" in rows, f"expected a normal 'cpu_op_C' row; got {list(rows)}"
    assert _kernel_names(rows["cpu_op_C"]) == ["k_C"]
    assert not any(
        n.startswith("cpu_op_C->") or "cpu_op_C (Synthetic Op)" in n for n in rows
    ), f"cpu_op_C should not be a synthetic op; got {list(rows)}"

    # cpu_op_E: only partially represented (k_B/k_C live in B/C) -> its own kernel
    # k_own becomes a synthetic op row.
    syn_name = "cpu_op_E->k_own (Synthetic Op)"
    assert syn_name in rows, f"expected '{syn_name}'; got {list(rows)}"
    assert _kernel_names(rows[syn_name]) == ["k_own"]

    # And E is never emitted as a plain/normal row (it is not fully represented).
    assert "cpu_op_E" not in rows

    # Exactly these three rows, nothing else.
    assert set(rows) == {B_NAME, "cpu_op_C", syn_name}, f"unexpected rows: {list(rows)}"


def _build_orphan_tree_rows(add_python_func=True):
    """A tree with NO cpu_op anywhere -- both kernels are orphans::

        py_func_A                     (python_function, root)
        ├── py_func_B -> hipLaunchKernel -> kernel_B
        └── hipLaunchKernel -> kernel_A
    """
    # Frame names contain "/" so they register in the call stack (the running
    # call-stack filter keeps module-like frames and cpu_ops).
    events = [
        _ev("A", 1000, 1000, "python_function", "pkg/a.py(1): py_func_A"),
        _ev("B", 1050, 200, "python_function", "pkg/b.py(2): py_func_B"),
    ]
    # kernel_B: launcher nested inside py_func_B
    events += _launcher_and_kernel(10, "kernel_B", rt_ts=1060, k_ts=1500)
    # kernel_A: launcher nested directly inside py_func_A (after py_func_B)
    events += _launcher_and_kernel(11, "kernel_A", rt_ts=1400, k_ts=1700)

    tree = TraceToTree(deepcopy(events), prune_nongpu_paths=False)
    tree.build_tree(add_python_func=add_python_func)
    analyzer = TreePerfAnalyzer(
        tree, add_python_func=add_python_func, rebuild_tree=False
    )
    df = analyzer.build_df_unified_perf_table(include_perf_metrics=False)
    return df, analyzer


@pytest.mark.parametrize("add_python_func", [True, False])
def test_orphan_launcher_synthetic_ops(add_python_func):
    # No cpu_op exists, so each kernel is launched straight from a runtime event
    # with no cpu_op ancestor -> each becomes its own synthetic launcher op.
    # Runs with add_python_func both on and off: with it off there is no python
    # root and no call stack, but the unified-roots traversal must still reach the
    # bare runtime launchers in-pass and produce the same rows.
    df, analyzer = _build_orphan_tree_rows(add_python_func=add_python_func)
    _assert_every_kernel_accounted_once(df, analyzer)
    rows = {r["name"]: r for _, r in df.iterrows()}

    syn_a = "hipLaunchKernel->kernel_A (Synthetic Op)"
    syn_b = "hipLaunchKernel->kernel_B (Synthetic Op)"

    assert syn_a in rows, f"expected '{syn_a}'; got {list(rows)}"
    assert _kernel_names(rows[syn_a]) == ["kernel_A"]

    assert syn_b in rows, f"expected '{syn_b}'; got {list(rows)}"
    assert _kernel_names(rows[syn_b]) == ["kernel_B"]

    # Exactly the two synthetic launcher rows, nothing else.
    assert set(rows) == {syn_a, syn_b}, f"unexpected rows: {list(rows)}"

    # With call stacks on, each orphan synthetic carries a full chain
    # (root python frames -> launcher -> kernel), ending in the kernel name.
    if add_python_func:
        cs_a = _call_stack(rows[syn_a])
        assert cs_a[-1] == "kernel_A", cs_a
        assert any("py_func_A" in f for f in cs_a), cs_a
        cs_b = _call_stack(rows[syn_b])
        assert cs_b[-1] == "kernel_B", cs_b
        assert any("py_func_A" in f for f in cs_b), cs_b
        assert any("py_func_B" in f for f in cs_b), cs_b


def test_synthetic_op_full_call_stack():
    # With call stacks on, the E->k_own synthetic op carries the enclosing cpu_op
    # frame and ends at the kernel -- i.e. it is NOT a bare [kernel] stub.
    df, analyzer = _build_example_tree_rows(add_python_func=True)
    _assert_every_kernel_accounted_once(df, analyzer)
    rows = {r["name"]: r for _, r in df.iterrows()}
    cs = _call_stack(rows["cpu_op_E->k_own (Synthetic Op)"])
    assert cs[-1] == "k_own", cs
    assert "cpu_op_E" in cs, cs


def test_multi_kernel_launcher_synthetic_ops():
    # A graph launch fires MULTIPLE kernels from one runtime event. Exit 4 must
    # emit one synthetic per launched kernel (not only the 1:1 case).
    events = [
        _ev("g", 1000, 200, "cuda_runtime", "hipGraphLaunch",
            args={"correlation": 20}),
        _ev("k1", 1050, 30, "kernel", "kernel_G1", pid=0, tid=7,
            args={"correlation": 20, "stream": 7}),
        _ev("k2", 1100, 30, "kernel", "kernel_G2", pid=0, tid=7,
            args={"correlation": 20, "stream": 7}),
    ]
    tree = TraceToTree(deepcopy(events), prune_nongpu_paths=False)
    tree.build_tree(add_python_func=True)
    analyzer = TreePerfAnalyzer(tree, add_python_func=True, rebuild_tree=False)
    df = analyzer.build_df_unified_perf_table(include_perf_metrics=False)
    _assert_every_kernel_accounted_once(df, analyzer)
    rows = {r["name"]: r for _, r in df.iterrows()}

    syn1 = "hipGraphLaunch->kernel_G1 (Synthetic Op)"
    syn2 = "hipGraphLaunch->kernel_G2 (Synthetic Op)"
    assert set(rows) == {syn1, syn2}, f"unexpected rows: {list(rows)}"
    assert _kernel_names(rows[syn1]) == ["kernel_G1"]
    assert _kernel_names(rows[syn2]) == ["kernel_G2"]


def test_strict_leaf_nested_cpu_op_splits():
    # cpu_op_X directly launches kernel_X AND contains a nested NON-perf-model
    # cpu_op_Y that launches kernel_Y. Under the strict-leaf rule, X is not a
    # leaf (it has a nested kernel-launching cpu_op), so it recurses:
    #   - cpu_op_Y  -> its own normal row (kernel_Y)
    #   - cpu_op_X's own kernel_X -> a synthetic op
    # Under the old perf-model-only recursion, X would instead have been a single
    # row owning BOTH kernels (cpu_op_Y subsumed).
    events = [
        _ev("X", 1000, 1000, "cpu_op", "cpu_op_X", args=_DIMS),
        _ev("Y", 1200, 200, "cpu_op", "cpu_op_Y", args=_DIMS),  # inside X
    ]
    events += _launcher_and_kernel(1, "kernel_X", rt_ts=1050, k_ts=1500)  # X's own
    events += _launcher_and_kernel(2, "kernel_Y", rt_ts=1210, k_ts=1600)  # under Y

    tree = TraceToTree(deepcopy(events), prune_nongpu_paths=False)
    tree.build_tree(add_python_func=False)
    analyzer = TreePerfAnalyzer(tree, add_python_func=False, rebuild_tree=False)
    df = analyzer.build_df_unified_perf_table(include_perf_metrics=False)
    _assert_every_kernel_accounted_once(df, analyzer)
    rows = {r["name"]: r for _, r in df.iterrows()}

    # cpu_op_Y is a normal row owning kernel_Y -- NOT subsumed into cpu_op_X.
    assert "cpu_op_Y" in rows, f"expected a normal cpu_op_Y row; got {list(rows)}"
    assert _kernel_names(rows["cpu_op_Y"]) == ["kernel_Y"]

    # cpu_op_X is only partially represented -> its own kernel is a synthetic op,
    # and it never appears as a plain row.
    syn_x = "cpu_op_X->kernel_X (Synthetic Op)"
    assert syn_x in rows, f"expected '{syn_x}'; got {list(rows)}"
    assert _kernel_names(rows[syn_x]) == ["kernel_X"]
    assert "cpu_op_X" not in rows

    assert set(rows) == {"cpu_op_Y", syn_x}, f"unexpected rows: {list(rows)}"


def test_perf_model_op_owns_whole_subtree():
    # A perf-modeled cpu_op owns its ENTIRE subtree (Exit 1): the model predicts
    # the combined time of every kernel beneath it, so the traversal collects it
    # and stops -- it never descends. Even a nested perf-modeled cpu_op and a
    # nested non-modeled cpu_op are subsumed, and the op's own directly-launched
    # kernel produces NO synthetic op.
    #
    #   aten::addmm (A, perf model)
    #   |__ aten::mm  (B, perf model)     -> runtime -> k_B
    #   |__ cpu_op_C  (no perf model)     -> runtime -> k_C
    #   |__ runtime -> k_own              (A's own kernel)
    A_NAME = "aten::addmm"  # a distinct op name that also has a perf model
    events = [
        _ev("A", 1000, 1000, "cpu_op", A_NAME, args=_DIMS),
        _ev("B", 1050, 100, "cpu_op", B_NAME, args=_DIMS),  # inside A
        _ev("C", 1200, 100, "cpu_op", "cpu_op_C", args=_DIMS),  # inside A
    ]
    events += _launcher_and_kernel(1, "k_B", rt_ts=1060, k_ts=1500)  # under B
    events += _launcher_and_kernel(2, "k_C", rt_ts=1210, k_ts=1600)  # under C
    events += _launcher_and_kernel(3, "k_own", rt_ts=1400, k_ts=1700)  # A's own

    tree = TraceToTree(deepcopy(events), prune_nongpu_paths=False)
    tree.build_tree(add_python_func=False)
    analyzer = TreePerfAnalyzer(tree, add_python_func=False, rebuild_tree=False)
    df = analyzer.build_df_unified_perf_table(include_perf_metrics=False)
    _assert_every_kernel_accounted_once(df, analyzer)
    rows = {r["name"]: r for _, r in df.iterrows()}

    # Exactly one row: the perf-modeled root, owning ALL three kernels.
    assert set(rows) == {A_NAME}, f"expected only '{A_NAME}'; got {list(rows)}"
    assert rows[A_NAME]["has_perf_model"] is True
    assert sorted(_kernel_names(rows[A_NAME])) == ["k_B", "k_C", "k_own"]

    # The nested perf-modeled op is subsumed -- it does NOT get its own row.
    assert B_NAME not in rows
    # No synthetic op is emitted for A's own kernel (the traversal never recurses).
    assert not any("(Synthetic Op)" in n for n in rows), f"unexpected synthetic rows: {list(rows)}"
