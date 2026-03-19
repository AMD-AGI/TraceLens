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

from typing import Any, Dict, List, Optional, Tuple

import TraceLens

UID = TraceLens.util.TraceEventUtils.TraceKeys.UID
from .trace_to_tree import TraceToTree


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
                    if not any(
                        [nf.lower() in node["name"].lower() for nf in name_filter]
                    ):
                        continue
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
    else:
        # print("=========matching ========")
        for j, i in zip(capture_events, graph_events):
            if "kernel" not in j.get("args", {}).keys():
                print(
                    "Kernel name missing in capture event args for event {}, alignment has not been verified".format(
                        j["name"]
                    )
                )
                return
            if i["name"] != j.get("args", {}).get("kernel", j["name"]):
                print(
                    "Mismatch in kernel name: {} vs {}".format(
                        i["name"], j.get("args", {}).get("kernel", j["name"])
                    )
                )
                return
    print(
        "Subtree events match successfully with {} events".format(len(capture_events))
    )
    return


def update_subtree_uids_and_timestamps(
    capture_tree, subtree_events, subtree_filtered_events, start_uid, new_start_ts
):
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
                print("Warning: parent UID {} not found in mapping".format(parent_uid))
    # Update timestamps
    original_start_ts = subtree_events[0]["ts"]
    ts_offset = new_start_ts - original_start_ts
    # for event in subtree_filtered_events:
    #    event[UID]=uid_mapping[event[UID]]
    for event in subtree_events:
        event["ts"] += ts_offset
    return subtree_events, subtree_filtered_events


def append_subtree_to_event(tree, subtree_events, parent_event):
    # Append the subtree events to the tree's events list
    tree.events.extend(subtree_events)
    # Update the parent event's children to include the root of the subtree
    parent_event["children"] = [subtree_events[0][UID]]
    for event in subtree_events:
        if event[UID] in tree.events_by_uid:
            print("Warning: UID {} already exists in events_by_uid".format(event[UID]))
        tree.events_by_uid[event[UID]] = event
    return tree


def make_connections(graph_tree, graph_filtered_events, capture_filtered_events):
    for g_event, c_event in zip(graph_filtered_events, capture_filtered_events):
        graph_tree.events[c_event[UID]]["children"] = [g_event[UID]]
        graph_tree.events[g_event[UID]]["parent"] = c_event[UID]
        graph_tree.events[g_event[UID]]["args"]["correlation"] = c_event["args"].get(
            "correlation", None
        )
        c_event.setdefault("gpu_events", []).append(
            g_event[TraceLens.util.TraceEventUtils.TraceKeys.UID]
        )

        parent = graph_tree.get_parent_event(c_event)
        while parent:
            parent.setdefault("gpu_events", []).append(
                g_event[TraceLens.util.TraceEventUtils.TraceKeys.UID]
            )
            parent = graph_tree.get_parent_event(parent)
    for event in graph_tree.events:
        # Skip GPU events
        cat = event.get("cat")
        if cat in {"kernel", "gpu_memset", "gpu_memcpy"}:
            continue
        # Now, we are dealing with non-GPU events
        if "gpu_events" not in event:
            event["non_gpu_path"] = True
        else:
            event.pop("non_gpu_path", None)
    return graph_tree


def merge_capture_trace_into_graph(
    capture_tree_filepath: str,
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

    capture_perf_analyzer = TreePerfAnalyzer.from_file(
        capture_tree_filepath, add_python_func=True
    )
    capture_tree = capture_perf_analyzer.tree

    graph_perf_analyzer = TreePerfAnalyzer.from_file(
        graph_tree_filepath, add_python_func=True
    )
    graph_tree = graph_perf_analyzer.tree

    print("Loaded capture tree with {} events".format(len(capture_tree.events)))
    print("Loaded graph tree with {} events".format(len(graph_tree.events)))

    ##Use cuda graph APIs to find the root node for capture subtrees
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
        filtered = [e for e in capture_tree.events if e["ts"] >= ts and e["ts"] <= te]
        capture_roots.append(max(filtered, key=lambda e: e.get("dur", 0)))

    graph_roots = [
        event
        for event in graph_tree.events
        if "graphlaunch" in event.get("name", "").lower()
    ]
    print(
        "Found {} capture roots and {} graph roots".format(
            len(capture_roots), len(graph_roots)
        )
    )
    for c_root, g_root in zip(capture_roots, graph_roots):
        capture_events, capture_filtered_events = get_subtree_events(
            capture_tree,
            c_root,
            cat_filter=["cuda_runtime", "cuda_driver"],
            name_filter=[
                "Launch",
                "Memcpy",
            ],
        )
        graph_events, graph_filtered_events = get_subtree_events(
            graph_tree, g_root, cat_filter=["kernel", "gpu_memset", "gpu_memcpy"]
        )
        print(
            "Verifying subtree events for capture root {} and graph root {}".format(
                c_root["name"], g_root["name"]
            )
        )
        verify_subtree_events(capture_filtered_events, graph_filtered_events)
        start_uid = graph_tree.events[-1][UID] + 1
        capture_events, _ = update_subtree_uids_and_timestamps(
            capture_tree,
            capture_events,
            capture_filtered_events,
            start_uid,
            g_root["ts"],
        )
        capture_events[0]["parent"] = g_root[UID]
        g_root["children"].append(capture_events[0][UID])
        graph_tree = append_subtree_to_event(graph_tree, capture_events, g_root)
        graph_tree = make_connections(
            graph_tree, graph_filtered_events, capture_filtered_events
        )

    return graph_tree
