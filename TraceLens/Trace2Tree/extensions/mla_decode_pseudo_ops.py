###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import re
import logging
from collections import defaultdict

from .pseudo_ops_utils import inject_pseudo_op_wrap_children

logger = logging.getLogger(__name__)

MLA_DECODE_FWD_PATTERN = re.compile(r"aiter/mla.py\(\d+\): mla_decode_fwd")
STAGE1_KERNEL_NAME = "aiter::mla_decode_stage1_asm_fwd"


def _is_mla_decode_fwd_python(name: str) -> bool:
    return (
        "mla.py" in name
        and "mla_decode_fwd" in name
        and bool(MLA_DECODE_FWD_PATTERN.search(name))
    )


def create_pseudo_ops_mla_decode(trace_tree):
    """
    Create pseudo ops for MLA decode forward operations.

    Finds python_function events matching 'aiter/mla.py<line>: mla_decode_fwd',
    verifies they have a child op 'aiter::mla_decode_stage1_asm_fwd' with GPU
    kernels, then injects 'pseudo_mla_decode_fwd' wrapping all children.
    Shapes are inherited from the stage1 event.
    """

    mla_python_funcs = _find_mla_decode_python_funcs(trace_tree)

    if not mla_python_funcs:
        logger.warning(
            "No python_function events matching mla_decode_fwd pattern found"
        )
        return

    logger.info(f"Processing {len(mla_python_funcs)} MLA decode operations")

    stage1_index = _build_mla_decode_stage1_index(trace_tree)

    # One O(R) snapshot; per-wrap root updates go through a set so we do not
    # rebuild ``cpu_root_nodes`` (length R) on every MLA op (O(M×R) total).
    cpu_roots_acc = set(trace_tree.cpu_root_nodes)

    for py_func_evt in mla_python_funcs:
        _create_pseudo_op_mla_decode(
            trace_tree, py_func_evt, stage1_index, cpu_roots_acc
        )

    trace_tree.cpu_root_nodes[:] = sorted(
        cpu_roots_acc,
        key=lambda u: (trace_tree.get_UID2event(u).get("ts", 0), u),
    )


def _find_mla_decode_python_funcs(trace_tree):
    """Find all python_function events matching the mla_decode_fwd pattern."""

    matched = []
    for name, uids in trace_tree.name2event_uids.items():
        if "mla.py" not in name or "mla_decode_fwd" not in name:
            continue
        if not MLA_DECODE_FWD_PATTERN.search(name):
            continue
        for uid in uids:
            evt = trace_tree.get_UID2event(uid)
            if evt.get("cat") == "python_function":
                matched.append(evt)
    return matched


def _build_mla_decode_stage1_index(trace_tree):
    """Map each mla_decode_fwd python_function UID → stage1 events under it.

    One O(#stage1 × depth) pass. Avoids O(#decode × #stage1 × depth) when both
    name buckets are large (``add_python_func=True`` traces).

    Mirrors legacy ``_find_stage1_child`` parent walks: walk upward from the
    stage1 event itself, then its ancestors, and attach the stage1 to every
    matching ``mla_decode_fwd`` python_function seen on that chain (so nested
    wrappers behave the same as repeated full scans).
    """

    index = defaultdict(list)
    candidates = trace_tree.name2event_uids.get(STAGE1_KERNEL_NAME, [])
    for uid in candidates:
        evt = trace_tree.get_UID2event(uid)
        cur = evt
        while cur is not None:
            if cur.get("cat") == "python_function" and _is_mla_decode_fwd_python(
                cur.get("name", "")
            ):
                index[cur["UID"]].append(evt)
            cur = trace_tree.get_parent_event(cur)
    return index


def _find_stage1_child(parent_evt, stage1_index):
    """
    Pick earliest-by-ts ``aiter::mla_decode_stage1_asm_fwd`` under *parent_evt*.

    *stage1_index* maps mla_decode_fwd python_function UID → list of stage1
    events on any descendant path below that op (same events the legacy
    parent-walk-from-all-stage1 approach would attach).
    """

    py_uid = parent_evt["UID"]
    under = stage1_index.get(py_uid)
    if not under:
        return None
    return min(under, key=lambda e: e.get("ts", 0))


def _create_pseudo_op_mla_decode(trace_tree, py_func_evt, stage1_index, cpu_roots_acc):
    """Create a single pseudo op for one MLA decode python_function event."""

    stage1_evt = _find_stage1_child(py_func_evt, stage1_index)

    if stage1_evt is None:
        logger.warning(
            f"No {STAGE1_KERNEL_NAME} child found for "
            f"'{py_func_evt['name']}' (UID {py_func_evt['UID']}), skipping"
        )
        return

    gpu_event_ids = py_func_evt.get("gpu_events", [])
    if not gpu_event_ids:
        logger.warning(
            f"No GPU events for MLA decode python_function UID {py_func_evt['UID']}"
        )
        return

    inject_pseudo_op_wrap_children(
        trace_tree,
        py_func_evt,
        "pseudo_mla_decode_fwd",
        shape_donor_evt=stage1_evt,
        cpu_roots_acc=cpu_roots_acc,
    )
