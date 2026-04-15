###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Utilities for merging capture trace information into graph replay traces.

This module provides functionality to extract events from a capture tree,
validate them against corresponding graph events, and merge them into an
augmented graph tree. The augmented tree combines both capture-time
information (actual kernel launches, memory operations) and graph replay
information (optimized execution paths), producing a complete execution
model suitable for performance analysis.
"""

import sys
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import re
import warnings
import TraceLens

UID = TraceLens.util.TraceEventUtils.TraceKeys.UID
from .trace_to_tree import TraceToTree

EXECUTE_CONTEXT_PATTERN = re.compile(
    r"execute_\d+_context_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)_generation_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)"
)


def get_subtree_events(tree, event, cat_filter=None, name_filter=None):
    list_events = []
    list_filtered_events = []
    queue = []
    queue.append(event)
    while len(queue) > 0:
        node = queue.pop(0)
        list_events.append(node)
        if cat_filter is not None:
            if tree.event_to_category(node) in cat_filter:
                if name_filter is not None:
                    if any([nf.lower() in node["name"].lower() for nf in name_filter]):
                        list_filtered_events.append(node)
                else:
                    list_filtered_events.append(node)
        for child in tree.get_children_events(node):
            queue.append(child)
    list_events.sort(key=lambda x: x["ts"])
    list_filtered_events.sort(key=lambda x: x["ts"])
    return list_events, list_filtered_events


def verify_subtree_events(capture_events, graph_events):
    if len(capture_events) != len(graph_events):
        print(
            "Mismatch in number of events: Capture {}, Graph {}".format(
                len(capture_events), len(graph_events)
            )
        )
        return 0
    else:
        # print("=========matching ========")
        for j, i in zip(capture_events, graph_events):
            if "kernel" not in j.get("args", {}).keys():
                if "hipMemcpy" in j["name"]:
                    continue
                warnings.warn(
                    "Kernel name missing in capture event args, "
                    "alignment has not been verified",
                    stacklevel=2,
                )
                continue
            if i["name"] != j.get("args", {}).get("kernel", j["name"]):
                print(
                    "Mismatch in kernel name: {} vs {}".format(
                        i["name"], j.get("args", {}).get("kernel", j["name"])
                    )
                )
                return 0
    # print(
    #    "Subtree events match successfully with {} events".format(len(capture_events))
    # )
    return 1


def update_subtree_uids_and_timestamps(
    capture_tree,
    subtree_events,
    subtree_filtered_events,
    start_uid,
    new_start_ts,
    c_root,
    g_root_dur,
):
    cpu_root_nodes = []
    uid_mapping = {}
    for idx, event in enumerate(subtree_events):
        old_uid = event[UID]
        new_uid = start_uid + idx
        uid_mapping[old_uid] = new_uid
        event[UID] = new_uid
    # Second pass to update parents and children
    for event in subtree_events:

        if "children" in event:
            new_children = []
            for child_uid in event["children"]:
                if child_uid in uid_mapping:
                    new_children.append(uid_mapping[child_uid])
                else:
                    print(
                        "Warning: child UID {} not found in mapping".format(child_uid)
                    )
            event["children"] = new_children
        if "parent" in event:
            parent_uid = event["parent"]
            if parent_uid in uid_mapping:
                event["parent"] = uid_mapping[parent_uid]
            else:
                if event["name"] != c_root["name"]:
                    print(
                        "Warning: parent UID {} not found in mapping".format(parent_uid)
                    )
    # Update timestamps
    original_start_ts = subtree_events[0]["ts"]
    ts_offset = new_start_ts - original_start_ts
    # for event in subtree_filtered_events:
    #    event[UID]=uid_mapping[event[UID]]
    for event in subtree_events:
        event["ts"] += ts_offset
        event["ts"] = new_start_ts
        event["dur"] = g_root_dur
    for i in capture_tree.cpu_root_nodes:
        if i in uid_mapping:
            cpu_root_nodes.append(uid_mapping[i])
    return subtree_events, subtree_filtered_events, cpu_root_nodes


def append_subtree_to_event(tree, subtree_events, parent_event, cpu_root_nodes):
    # Append the subtree events to the tree's events list
    tree.events.extend(subtree_events)
    # Update the parent event's children to include the root of the subtree
    parent_event["children"] = [subtree_events[0][UID]]
    for event in subtree_events:
        if event[UID] in tree.events_by_uid:
            print("Warning: UID {} already exists in events_by_uid".format(event[UID]))
        tree.events_by_uid[event[UID]] = event
        tree.name2event_uids[
            event[TraceLens.util.TraceEventUtils.TraceKeys.Name]
        ].append(event[TraceLens.util.TraceEventUtils.TraceKeys.UID])
    tree.cpu_root_nodes.extend(cpu_root_nodes)
    return tree


def make_connections(graph_tree, graph_filtered_events, capture_filtered_events):
    events_by_uid = graph_tree.events_by_uid

    # Accumulate gpu_event UIDs per ancestor, then batch-apply with extend.
    # This avoids repeated setdefault+append on the same ancestor across pairs.
    ancestor_gpu_events: dict = {}

    # Once the first pair with a non-empty parent chain is processed, we record
    # its ancestor path so subsequent pairs can skip the shared suffix.
    # We use an explicit boolean rather than checking `if not ancestor_chain`
    # to correctly handle the edge case where the first c_event has no parents
    # (which would leave ancestor_chain=[] and cause every pair to re-enter the
    # "first pair" branch).
    chain_built: bool = False
    ancestor_chain: list = (
        []
    )  # ancestor UIDs bottom→top, grows as new ancestors are seen
    ancestor_chain_idx: dict = (
        {}
    )  # uid → index in ancestor_chain for O(1) merge detection

    for g_event, c_event in zip(graph_filtered_events, capture_filtered_events):
        g_uid = g_event[UID]

        graph_tree.events[c_event[UID]]["children"] = [g_uid]
        graph_tree.events[g_uid]["parent"] = c_event[UID]
        graph_tree.events[g_uid]["args"]["correlation"] = c_event["args"].get(
            "correlation", None
        )
        c_event.setdefault("gpu_events", []).append(g_uid)

        parent = graph_tree.get_parent_event(c_event)

        if not chain_built:
            # Walk to root and record every ancestor UID for future short-circuits.
            while parent:
                p_uid = parent[UID]
                ancestor_chain.append(p_uid)
                ancestor_chain_idx[p_uid] = len(ancestor_chain) - 1
                ancestor_gpu_events.setdefault(p_uid, []).append(g_uid)
                parent = graph_tree.get_parent_event(parent)
            chain_built = True
        else:
            # Walk the private prefix until we merge with the recorded chain,
            # then index directly into it — skipping the shared suffix walk.
            # Private ancestors are added to ancestor_gpu_events but NOT to
            # ancestor_chain: ancestor_chain must stay a single linear path
            # (bottom→top) so that ancestor_chain[idx:] always yields exactly
            # the ancestors above the merge point. Mixing in private ancestors
            # from different paths would corrupt the slice semantics.
            while parent:
                p_uid = parent[UID]
                if p_uid in ancestor_chain_idx:
                    # Merge point: batch-add g_uid to this ancestor and everything above.
                    for shared_uid in ancestor_chain[ancestor_chain_idx[p_uid] :]:
                        ancestor_gpu_events.setdefault(shared_uid, []).append(g_uid)
                    break
                ancestor_gpu_events.setdefault(p_uid, []).append(g_uid)
                parent = graph_tree.get_parent_event(parent)

    # Single pass to flush all accumulated gpu_event lists onto ancestor events.
    for uid, g_uids in ancestor_gpu_events.items():
        events_by_uid[uid].setdefault("gpu_events", []).extend(g_uids)

    return graph_tree


def finalize_non_gpu_paths(graph_tree):
    """Set/clear ``non_gpu_path`` on every non-GPU event in *graph_tree*.

    Called once after all ``make_connections`` calls are complete, rather than
    scanning the full event list inside every ``make_connections`` invocation.
    """
    gpu_cats = {"kernel", "gpu_memset", "gpu_memcpy"}
    for event in graph_tree.events:
        if event.get("cat") in gpu_cats:
            continue
        if "gpu_events" not in event:
            event["non_gpu_path"] = True
        else:
            event.pop("non_gpu_path", None)


_CAPTURE_TREE_CACHE_MAX_SIZE = 8
_capture_tree_cache: OrderedDict = OrderedDict()


def _get_cached_capture_tree(key, filepath, TreePerfAnalyzer):
    """Load a capture tree, returning a cached copy when *key* has been seen.

    Uses LRU eviction with at most ``_CAPTURE_TREE_CACHE_MAX_SIZE`` entries.
    Callers must deep-copy any events they intend to mutate so the cached tree
    stays clean for subsequent look-ups.
    """
    if key in _capture_tree_cache:
        _capture_tree_cache.move_to_end(key)
        print("Cache hit for capture tree (key={})".format(key))
        return _capture_tree_cache[key]

    print("Loading capture trace: {} (key={})".format(filepath, key))
    capture_perf_analyzer = TreePerfAnalyzer.from_file(filepath, add_python_func=True)
    capture_tree = capture_perf_analyzer.tree
    capture_roots = find_capture_roots(capture_tree)

    capture_root_data = []
    for c_root in capture_roots:
        capture_events, capture_filtered_events = get_subtree_events(
            capture_tree,
            c_root,
            cat_filter=["cuda_runtime", "cuda_driver"],
            name_filter=["Launch", "Memcpy", "Memset"],
        )
        filtered_uids = {e[UID] for e in capture_filtered_events}
        capture_root_data.append((capture_events, filtered_uids))

    _capture_tree_cache[key] = (capture_tree, capture_roots, capture_root_data)
    if len(_capture_tree_cache) > _CAPTURE_TREE_CACHE_MAX_SIZE:
        evicted_key, _ = _capture_tree_cache.popitem(last=False)
        print("Evicted capture tree cache entry (key={})".format(evicted_key))

    return capture_tree, capture_roots, capture_root_data


def find_capture_roots(capture_tree):
    """Find capture roots by pairing StreamBeginCapture / StreamEndCapture events."""
    capture_roots = []
    capture_begin = [
        event
        for event in capture_tree.events
        if "StreamBeginCapture" in event.get("name", "")
        and event.get("cat", "") == "cuda_runtime"
    ]
    capture_begin_ts = [event["ts"] + event["dur"] for event in capture_begin]
    capture_end = [
        event
        for event in capture_tree.events
        if "StreamEndCapture" in event.get("name", "")
        and event.get("cat", "") == "cuda_runtime"
    ]
    capture_end_ts = [event["ts"] for event in capture_end]
    for ts, te in zip(capture_begin_ts, capture_end_ts):
        filtered = [
            e
            for e in capture_tree.events
            if e["ts"] >= ts and e["ts"] + e.get("dur", 0) <= te
        ]
        filtered_uids = {e[UID] for e in filtered}
        root_events = [
            e
            for e in filtered
            if e.get("parent", None) not in filtered_uids
            and e.get("parent", None) is not None
            and (e.get("cat", "") == "cpu_op" or e.get("cat", "") == "python_function")
        ]
        new_uid = max(capture_tree.events_by_uid.keys()) + 1
        dummy = {
            UID: new_uid,
            "name": "CaptureRoot",
            "ts": ts,
            "dur": te - ts,
            "cat": "cuda_runtime",
            "children": [e[UID] for e in root_events],
            "args": {},
        }
        capture_tree.events_by_uid[new_uid] = dummy
        capture_roots.append(dummy)
        capture_tree.events.append(dummy)
        for e in root_events:
            e["parent"] = new_uid
    return capture_roots


def find_execution_roots(graph_tree):
    """Find execution root events matching ``execute_context_*`` in the graph tree."""
    roots = [
        event
        for event in graph_tree.events
        if EXECUTE_CONTEXT_PATTERN.match(event.get("name", ""))
        and event.get("cat") == "user_annotation"
    ]
    roots.sort(key=lambda x: x.get("ts", 0))
    return roots


def find_graph_roots_under_execution(execution_root, graphlaunch_events):
    """Return all ``graphlaunch`` events that fall within *execution_root*'s time range.

    Accepts a pre-collected list of graphlaunch events so callers can avoid
    rescanning the full event list for every execution root.  No tree traversal
    is needed: graphlaunch events are identified directly by name, so filtering
    by timestamp is sufficient.
    """
    exec_ts = execution_root.get("ts", 0)
    exec_te = exec_ts + execution_root.get("dur", 0)
    graph_roots = [
        e
        for e in graphlaunch_events
        if exec_ts <= e.get("ts", 0) and e.get("ts", 0) + e.get("dur", 0) <= exec_te
    ]
    graph_roots.sort(key=lambda x: x.get("ts", 0))
    return graph_roots


def build_execution_graph_root_map(graph_tree):
    """Build a list of ``(execution_root, [graph_roots])`` for the graph tree."""
    execution_roots = find_execution_roots(graph_tree)
    print("Found {} execution roots in graph tree".format(len(execution_roots)))

    # Collect graphlaunch events once; reused for every execution root.
    graphlaunch_events = [
        e for e in graph_tree.events if "graphlaunch" in e.get("name", "").lower()
    ]

    result = []
    for exec_root in execution_roots:
        g_roots = find_graph_roots_under_execution(exec_root, graphlaunch_events)
        print(
            "  Execution root '{}': {} graph roots".format(
                exec_root["name"], len(g_roots)
            )
        )
        result.append((exec_root, g_roots))
    return result


def load_capture_folder(
    capture_folder: str,
    metadata_json_path: str,
) -> Tuple[Dict[str, List[Tuple[Any, List]]], List[int]]:
    """Load capture traces from a folder and group by ``{batch_size}_{mode}``.

    Args:
        capture_folder: Directory containing ``graph_capture_rank_0*`` trace files.
        metadata_json_path: Path to a JSON file — a list of objects each with
            ``file``, ``batch_size``, and ``mode`` keys.

    Returns:
        Dictionary keyed by ``"{batch_size}_{mode}"`` whose values are lists of
        ``(capture_tree, capture_roots)`` tuples.
    """
    from ..TreePerf.tree_perf import TreePerfAnalyzer

    with open(metadata_json_path, "r") as f:
        metadata_list = json.load(f)

    result: Dict[str, List[Tuple[Any, List]]] = {}
    batch_sizes = []
    for entry in metadata_list:
        filename = entry["file"]
        batch_size = entry["batch_size"]
        mode = entry["mode"]
        if not (mode in ["FULL", "PIECEWISE"] and isinstance(batch_size, int)):
            print("Warning: invalid batch size or mode, skipping: {}".format(entry))
            continue
        key = "{}_{}".format(batch_size, mode)
        batch_sizes.append(int(batch_size))
        filepath = os.path.join(capture_folder, filename)
        if not os.path.isfile(filepath):
            print("Warning: capture file not found, skipping: {}".format(filepath))
            continue

        result[key] = filepath

    print(
        "\nCapture folder loaded: {} unique batch_size_mode keys".format(
            len(result.keys())
        )
    )

    return result, batch_sizes


def find_closest_batch_size(
    batch_size: int, capture_batch_sizes: List[int]
) -> Optional[int]:
    """Return the smallest value in *capture_batch_sizes* that is >= *batch_size*.

    Returns ``None`` when *batch_size* exceeds every entry in
    *capture_batch_sizes* (i.e. no valid round-up exists).
    """
    candidates = [bs for bs in capture_batch_sizes if bs >= batch_size]
    if not candidates:
        return None
    return min(candidates)


def find_execution_details(execution_root):
    name = execution_root["name"].split("_")[1]
    return name


def merge_capture_trace_into_graph(
    capture_folder: str,
    metadata_json_path: str,
    graph_tree_filepath: str,
) -> "TraceToTree":
    """
    Merge capture trace information into a graph replay trace.

    Extracts matching subtrees from capture and graph trees, validates alignment,
    remaps UIDs/timestamps, and integrates capture events into the graph tree.
    The result is a single augmented TraceToTree that combines both traces,
    suitable for standard performance analysis APIs.

    Args:
        capture_tree: TraceToTree from actual/capture execution
        graph_tree: TraceToTree from graph/replay execution
        add_python_func: If True, include python function events in tree structure (default: False)

    Returns:
        Augmented graph_tree with capture information merged in
    """
    # Lazy import to avoid circular dependency
    from ..TreePerf.tree_perf import TreePerfAnalyzer

    graph_perf_analyzer = TreePerfAnalyzer.from_file(
        graph_tree_filepath, add_python_func=True
    )

    graph_tree = graph_perf_analyzer.tree
    print("Loaded graph tree with {} events".format(len(graph_tree.events)))
    ##Use cuda graph APIs to find the root node for capture subtrees
    execution_graph_root_map = build_execution_graph_root_map(graph_tree)
    capture_map, capture_batch_sizes = load_capture_folder(
        capture_folder, metadata_json_path
    )
    for execution_root, graph_roots in execution_graph_root_map:
        print("Processing execution root: {}".format(execution_root["name"]))
        if len(graph_roots) == 0:
            print(
                "No graph roots found for execution root {}".format(
                    execution_root["name"]
                )
            )
            continue
        batch_size = find_execution_details(execution_root)
        closest_batch_size = find_closest_batch_size(
            int(batch_size), capture_batch_sizes
        )
        if closest_batch_size is None:
            print(
                "Warning: no capture batch size found for batch size {}".format(
                    batch_size
                )
            )
            continue
        num_graph_roots = len(graph_roots)
        if num_graph_roots != 1:
            mode = "PIECEWISE"
        else:
            mode = "FULL"
        key = "{}_{}".format(closest_batch_size, mode)
        filepath = capture_map[key]
        capture_tree, capture_roots, capture_root_data = _get_cached_capture_tree(
            key, filepath, TreePerfAnalyzer
        )

        print(
            "Found {} capture roots and {} graph roots".format(
                len(capture_roots), len(graph_roots)
            )
        )
        for (c_root, g_root), (cached_events, filtered_uids) in zip(
            zip(capture_roots, graph_roots), capture_root_data
        ):
            # Shallow-copy each event dict: every mutation downstream replaces
            # top-level keys (UID, ts, dur, parent, children) or adds new ones
            # (gpu_events). No nested structure is ever mutated in-place, so a
            # full deepcopy is unnecessary.
            capture_events = [{**e} for e in cached_events]

            capture_filtered_events = [
                e for e in capture_events if e[UID] in filtered_uids
            ]

            graph_events, graph_filtered_events = get_subtree_events(
                graph_tree, g_root, cat_filter=["kernel", "gpu_memset", "gpu_memcpy"]
            )

            verify_success = verify_subtree_events(
                capture_filtered_events, graph_filtered_events
            )

            if verify_success == 0:
                print(
                    "Warning: subtree events verification failed for capture root {} and graph root {}".format(
                        c_root["name"], g_root["name"]
                    )
                )
                continue

            start_uid = graph_tree.events[-1][UID] + 1
            capture_events, _, cpu_root_nodes = update_subtree_uids_and_timestamps(
                capture_tree,
                capture_events,
                capture_filtered_events,
                start_uid,
                g_root["ts"],
                c_root,
                g_root["dur"],
            )

            capture_events[0]["parent"] = g_root[UID]
            g_root["children"].append(capture_events[0][UID])
            graph_tree = append_subtree_to_event(
                graph_tree, capture_events, g_root, cpu_root_nodes
            )

            graph_tree = make_connections(
                graph_tree, graph_filtered_events, capture_filtered_events
            )
    finalize_non_gpu_paths(graph_tree)
    return graph_tree
