###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Generic call-tree helpers over :class:`TraceToTree`.

Entry-root discovery, worker-thread reattachment, and descendant GPU-time
accounting -- tree operations that stand on their own, independent of any one
analysis that consumes them.
"""

from bisect import bisect_right
from collections import Counter
from typing import List, Optional, Sequence

from ..util import GPU_KERNEL_CATEGORIES
from .trace_to_tree import TraceToTree

# The python_function event category, used when reuniting worker threads and
# when scanning a thread's frames.
PYTHON_TIER = "python_function"


def _entry_roots(tree: TraceToTree) -> List[dict]:
    """Parentless per-thread entry nodes, deduped, honoring current parents."""
    seen: set = set()
    roots: List[dict] = []
    for uid in tree.cpu_root_nodes:
        event = tree.get_UID2event(uid)
        while True:
            parent = tree.get_parent_event(event)
            if parent is None:
                break
            event = parent
        if event["UID"] not in seen:
            seen.add(event["UID"])
            roots.append(event)
    return roots


def _reattach_worker_threads(tree: TraceToTree) -> TraceToTree:
    """Fold roots living on a worker thread under the host frame that, in time,
    encloses them.

    An autograd-engine (or similar dispatch) thread runs work the host thread is
    blocked waiting on, so each of its roots sits entirely inside one host leaf,
    yet the profiler records no parent link across the thread boundary. Rebuilding
    that link reunites e.g. ``backward_step`` with the kernels it triggered and
    lifts the worker roots off the top level, where they otherwise swamp the
    repeating-pattern search.
    """
    pyf: Counter = Counter()
    for e in tree.events_by_uid.values():
        if e.get("cat") == PYTHON_TIER:
            pyf[(e.get("pid"), e.get("tid"))] += 1
    if not pyf:
        return tree
    host = pyf.most_common(1)[0][0]

    host_nodes = sorted(
        (
            e
            for e in tree.events_by_uid.values()
            if (e.get("pid"), e.get("tid")) == host
            and e.get("ts") is not None
            and e.get("t_end") is not None
        ),
        key=lambda e: e["ts"],
    )
    if not host_nodes:
        return tree
    starts = [e["ts"] for e in host_nodes]

    def deepest_container(lo: float, hi: float) -> Optional[dict]:
        # Host frames nest, so among those starting at/before ``lo`` the latest
        # one still ending at/after ``hi`` is the innermost enclosing frame.
        i = bisect_right(starts, lo) - 1
        while i >= 0:
            node = host_nodes[i]
            if node["t_end"] >= hi:
                return node
            i -= 1
        return None

    reattached = 0
    for root in _entry_roots(tree):
        if (root.get("pid"), root.get("tid")) == host:
            continue
        lo, hi = root.get("ts"), root.get("t_end")
        if lo is None or hi is None:
            continue
        host_node = deepest_container(lo, hi)
        if host_node is None:
            continue
        root["parent"] = host_node["UID"]
        host_node.setdefault("children", []).append(root["UID"])
        gpu_uids = root.get("gpu_events", [])
        ancestor: Optional[dict] = host_node
        while ancestor is not None and ancestor.get("non_gpu_path", False):
            if gpu_uids:
                ancestor.setdefault("gpu_events", []).extend(gpu_uids)
            ancestor.pop("non_gpu_path", None)
            ancestor = tree.get_parent_event(ancestor)
        reattached += 1
    return tree


def _descendant_gpu_time(tree: TraceToTree, nodes: Sequence[dict]) -> float:
    """Total GPU time under ``nodes`` in the tree, each kernel counted once."""
    seen: set = set()
    total = 0.0
    stack = [n["UID"] for n in nodes]
    while stack:
        uid = stack.pop()
        if uid in seen:
            continue
        seen.add(uid)
        event = tree.get_UID2event(uid)
        if event.get("cat") in GPU_KERNEL_CATEGORIES:
            total += event.get("dur", 0)
        children = event.get("children")
        if children:
            stack.extend(children)
    return total
