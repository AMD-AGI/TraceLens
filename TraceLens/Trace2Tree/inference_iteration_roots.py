###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Generic iteration-root detection via TraceToTree call-tree traversal."""

from typing import List, Optional, Tuple

from .trace_to_tree import TraceToTree


def _find_repeating_period(
    names: List[str], min_repeats: int = 3
) -> Tuple[Optional[int], Optional[List[str]], Optional[int]]:
    """Find the shortest repeating name sequence anywhere in ``names``.

    Slides a start offset forward to skip any non-repeating prefix (setup
    events before the loop body). Returns ``(period, pattern, start_offset)``
    where ``start_offset`` is the index in ``names`` where the first block
    begins. Returns ``(None, None, None)`` if no qualifying period is found.

    Requires at least ``min_repeats`` consecutive repetitions covering more
    than half of the suffix starting at ``start_offset``.
    """
    n = len(names)
    for start in range(n):
        suffix = names[start:]
        m = len(suffix)
        for p in range(1, m // 2 + 1):
            pattern = suffix[:p]
            count = 0
            i = 0
            while i + p <= m and suffix[i : i + p] == pattern:
                count += 1
                i += p
            if count >= min_repeats and count * p > m * 0.5:
                return p, pattern, start
    return None, None, None


def _detect_iteration_roots_from_tree(tree: TraceToTree, roots) -> Optional[List[dict]]:
    """BFS down the tree from one or more root nodes to find and return synthetic
    iteration-root events.

    ``roots`` may be a single event dict or a list of event dicts — all are
    seeded into the BFS at depth 0 so they are explored level-by-level together.

    Pattern detection uses all children (not just GPU-path ones) so that
    leading CPU-only events (e.g. ``next`` in the OWL pipeline) are included
    as part of the iteration anchor. A minimum child count guards against false
    positives from short utility-function child lists.

    Returns a list of synthetic root events, one per detected iteration, where
    each event's ``dur`` spans from the first to the last child of the block.
    """
    from collections import deque

    if isinstance(roots, dict):
        roots = [roots]

    queue = deque((node, 0) for node in roots)
    while queue:
        current, depth = queue.popleft()
        children = tree.get_children_events(current)
        if not children:
            continue

        # Only recurse into GPU-bearing subtrees.
        if not any(c.get("gpu_events") for c in children):
            continue

        p, _, start = _find_repeating_period([c.get("name", "") for c in children])
        if p is None:
            for child in children:
                if child.get("gpu_events"):
                    queue.append((child, depth + 1))
            continue

        print(
            f"Generic fallback: repeating pattern found under '{current.get('name')}' at depth {depth}"
        )
        print(f"Generic fallback: period={p}")

        # Anchor each iteration between the Nth occurrence of the first and last
        # events in the detected pattern. Using all-children anchors means
        # CPU-only leading/trailing events are included naturally.
        first_anchor_name = children[start]["name"]
        last_anchor_name = children[start + p - 1]["name"]

        first_anchors = [
            i
            for i, c in enumerate(children)
            if i >= start and c.get("name") == first_anchor_name
        ]
        last_anchors = [
            i
            for i, c in enumerate(children)
            if i >= start and c.get("name") == last_anchor_name
        ]

        iteration_roots = []
        for n in range(min(len(first_anchors), len(last_anchors))):
            block_start = first_anchors[n]
            block_end = last_anchors[n]
            if block_end < block_start:
                break
            block = children[block_start : block_end + 1]
            first, last = block[0], block[-1]
            root_event = dict(first)
            root_event["dur"] = (last["ts"] + last.get("dur", 0)) - first["ts"]
            iteration_roots.append(root_event)

        print(f"Generic fallback: identified {len(iteration_roots)} iterations.")
        return iteration_roots if iteration_roots else None

    return None


def find_iteration_roots_generic(events: List[dict]) -> Optional[List[dict]]:
    """Fallback: detect iteration roots by finding a repeating child pattern in
    the call tree, using TraceToTree for parent/child relationships.

    Works for any workload (diffusion, training, etc.) where the iteration loop
    body is a repeating sequence of top-level calls under a common parent.
    """
    try:
        tree = TraceToTree(events, prune_nongpu_paths=False)
        tree.build_tree(add_python_func=True)
    except Exception as e:
        print(f"Generic fallback: TraceToTree build failed ({e}), skipping.")
        return None

    # Walk every cpu_root_node upward through python_function parents until
    # reaching a parentless node — these are the true per-thread entry points.
    seen_roots: set = set()
    trace_roots = []
    for uid in tree.cpu_root_nodes:
        e = tree.get_UID2event(uid)
        while True:
            parent = tree.get_parent_event(e)
            if parent is None:
                break
            e = parent
        if id(e) not in seen_roots:
            seen_roots.add(id(e))
            trace_roots.append(e)

    if not trace_roots:
        print("Generic fallback: no root nodes found.")
        return None

    roots = _detect_iteration_roots_from_tree(tree, trace_roots)
    if roots is None:
        print("Generic fallback: no repeating child pattern found.")
    return roots
