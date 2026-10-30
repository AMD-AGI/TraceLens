###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

from collections import Counter, OrderedDict, deque, defaultdict
from typing import Dict, List, Optional, Tuple
import json
import os
import re
import warnings
import TraceLens

_SGLANG_CAPTURE_PATTERN = re.compile(r"^\w*CudaGraphRunner_bs_\d+_")

UID = TraceLens.util.TraceEventUtils.TraceKeys.UID
from .trace_to_tree import TraceToTree
from ..util import GPU_KERNEL_CATEGORIES, extract_rank
from ..TraceUtils.utils.annotation_utils import (
    SPEC_ROLE_TARGET,
    find_known_annotations,
    find_spec_decode_annotations,
    parse_annotation,
    spec_decode_batch_sizes,
    spec_decode_role,
)


def get_subtree_events(tree, event, cat_filter=None, name_filter=None):
    list_events = []
    list_filtered_events = []
    queue = deque()
    queue.append(event)
    while queue:
        node = queue.popleft()
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


def _capture_kernel_name(event):
    return event.get("args", {}).get("kernel", event["name"])


def _align_capture_to_graph(capture_events, graph_events):
    """
    Robust greedy alignment of capture dispatch events to graph kernel
    events in case of minor mismatch due to profiler error.

    For each graph kernel, scan forward through capture_events until a
    dispatch whose args.kernel matches is found, skipping any unmatched
    dispatches along the way.  Unmatched capture events at the tail are
    discarded.  Returns the aligned capture list (same length as
    graph_events), or None if any graph kernel cannot be matched.
    """

    aligned = []
    ci = 0
    for g_event in graph_events:
        g_name = g_event["name"]
        matched = False
        while ci < len(capture_events):
            c_event = capture_events[ci]
            ci += 1
            c_name = _capture_kernel_name(c_event)
            if ("Memcpy" in c_event["name"] and "Memcpy" in g_name) or (
                "Memset" in c_event["name"]
                and ("Memset" in g_name or "fillBuffer" in g_name)
            ):
                aligned.append(c_event)
                matched = True
                break
            if c_name == g_name:
                aligned.append(c_event)
                matched = True
                break
        if not matched:
            return None
    return aligned


def _align_graph_to_capture_by_group(capture_events, graph_events):
    """
    Align capture dispatch events to graph kernel events by grouping on kernel
    name and matching positionally within each group.
    Returns the graph_events reordered to match capture_events order, or None
    if any kernel name has a count mismatch between capture and graph.
    """
    capture_groups = defaultdict(list)
    for e in capture_events:
        capture_groups[_capture_kernel_name(e)].append(e)

    graph_groups = defaultdict(list)
    for e in graph_events:
        graph_groups[e["name"]].append(e)

    # Verify per-name counts match
    all_names = set(capture_groups) | set(graph_groups)
    for name in all_names:
        if len(capture_groups[name]) != len(graph_groups[name]):
            print(
                "Group alignment: count mismatch for kernel '{}': "
                "capture={} graph={}".format(
                    name, len(capture_groups[name]), len(graph_groups[name])
                )
            )
            return None

    # Reassemble graph events in capture order using per-group position
    group_idx = defaultdict(int)
    aligned = []
    for c_event in capture_events:
        name = _capture_kernel_name(c_event)
        idx = group_idx[name]
        aligned.append(graph_groups[name][idx])
        group_idx[name] += 1

    return aligned


def _stream_of(event):
    """Return the CUDA stream id a graph kernel executed on, or None."""
    return event.get("args", {}).get("stream")


def _names_match(c_event, g_name):
    """Pair a capture dispatch to a graph kernel by name (Memcpy/Memset carry no kernel arg)."""
    if ("Memcpy" in c_event["name"] and "Memcpy" in g_name) or (
        "Memset" in c_event["name"] and ("Memset" in g_name or "fillBuffer" in g_name)
    ):
        return True
    return _capture_kernel_name(c_event) == g_name


def is_multistream(graph_filtered_events):
    """True when the graph root's kernels span more than one CUDA stream."""
    streams = set()
    for e in graph_filtered_events:
        streams.add(_stream_of(e))
        if len(streams) > 1:
            return True
    return False


def capture_has_kernel_names(capture_filtered_events):
    """False if any non-Memcpy/Memset capture dispatch lacks args.kernel."""
    for e in capture_filtered_events:
        if "Memcpy" in e["name"] or "Memset" in e["name"]:
            continue
        if "kernel" not in e.get("args", {}):
            return False
    return True


def _best_stream_by_run(candidates, stream_queues, capture_events, ci):
    """Tie-break: pick the stream matching the longest forward run of capture dispatches from ci."""
    best_stream = candidates[0]
    best_run = -1
    for s in candidates:
        q = stream_queues[s]
        run = 0
        while (
            run < len(q)
            and ci + run < len(capture_events)
            and _names_match(capture_events[ci + run], q[run]["name"])
        ):
            run += 1
        if run > best_run:
            best_run = run
            best_stream = s
    return best_stream


def align_streams(graph_filtered_events, capture_filtered_events):
    """Pair capture dispatches (CPU program order) to graph kernels (GPU order across
    streams) via per-stream FIFO queues. Returns capture reordered into graph order, or
    None if any graph kernel is unmatched."""
    stream_queues = OrderedDict()
    for g_event in graph_filtered_events:
        stream_queues.setdefault(_stream_of(g_event), deque()).append(g_event)

    mapping = {}
    for ci, c_event in enumerate(capture_filtered_events):
        candidates = [
            s
            for s, q in stream_queues.items()
            if q and _names_match(c_event, q[0]["name"])
        ]
        if not candidates:
            continue
        if len(candidates) > 1:
            chosen = _best_stream_by_run(
                candidates, stream_queues, capture_filtered_events, ci
            )
        else:
            chosen = candidates[0]
        g_event = stream_queues[chosen].popleft()
        mapping[id(g_event)] = c_event

    if any(id(g) not in mapping for g in graph_filtered_events):
        return None
    return [mapping[id(g)] for g in graph_filtered_events]


def verify_subtree_events(capture_events, graph_events):
    """
    Verify that capture dispatch events align 1-to-1 with graph kernel events.

    Alignment is attempted in three stages:
      1. Direct positional match — succeeds when both sequences are already
         in the same order (return code 1).
      2. Greedy forward-scan by name — handles count mismatches by discarding
         extra capture dispatches (return code 2).
      3. Group-by-name positional match — handles cases graph replay events
         and capture events are in different order (return code 3).

    Returns:
        (success, aligned_capture_events, aligned_graph_events) where success
        is 1, 2, or 3 on success and 0 on failure.
    """
    if len(capture_events) != len(graph_events):
        print(
            "Mismatch in number of events: Capture {}, Graph {}".format(
                len(capture_events), len(graph_events)
            )
        )
        aligned = _align_capture_to_graph(capture_events, graph_events)
        if aligned is not None:
            print(
                "Greedy alignment succeeded: matched {} capture events to {} graph kernels "
                "({} capture events discarded).".format(
                    len(aligned),
                    len(graph_events),
                    len(capture_events) - len(aligned),
                )
            )
            return 2, aligned, graph_events
        return 0, capture_events, graph_events

    for j, i in zip(capture_events, graph_events):
        if "kernel" not in j.get("args", {}).keys():
            if "Memcpy" in j["name"] or "Memset" in j["name"]:
                continue
            warnings.warn(
                "Kernel name missing in capture event args, "
                "alignment has not been verified",
                stacklevel=2,
            )
            continue
        if i["name"] != _capture_kernel_name(j):
            # Counts match but order differs
            aligned_graph = _align_graph_to_capture_by_group(
                capture_events, graph_events
            )
            if aligned_graph is not None:
                print("Group-by-name alignment succeeded.")
                return 3, capture_events, aligned_graph
            return 0, capture_events, graph_events

    return 1, capture_events, graph_events


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
    # UID remapping handled when rebuilding the merged capture subtree.
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


def _load_trace_tree_from_file(
    profile_filepath: str, add_python_func: bool = False
) -> TraceToTree:
    """Load a trace file into a built TraceToTree without TreePerfAnalyzer."""
    from ..util import DataLoader

    data = DataLoader.load_data(profile_filepath)
    trace_metadata = {key: value for key, value in data.items() if key != "traceEvents"}
    events = data["traceEvents"]
    tree = TraceToTree(
        events,
        event_to_category=TraceToTree.default_categorizer,
        trace_metadata=trace_metadata,
    )
    tree.build_tree(add_python_func=add_python_func)
    return tree


def _get_cached_capture_tree(key, filepath):
    """Load a capture tree, returning a cached copy when *key* has been seen.

    Uses LRU eviction with at most ``_CAPTURE_TREE_CACHE_MAX_SIZE`` entries.
    Callers must deep-copy any events they intend to mutate so the cached tree
    stays clean for subsequent look-ups.

    Returns:
        (capture_tree, capture_roots, capture_root_data) where
        *capture_root_data* is a list of ``(capture_events, filtered_uids)``
        tuples — one per capture root — pre-computed from
        :func:`get_subtree_events`.
    """
    if key in _capture_tree_cache:
        _capture_tree_cache.move_to_end(key)
        print("Cache hit for capture tree (key={})".format(key[0]))
        return _capture_tree_cache[key]

    print("Loading capture trace: {} (key={})".format(filepath, key[0]))
    capture_tree = _load_trace_tree_from_file(filepath, add_python_func=True)
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
        print("Evicted capture tree cache entry (key={})".format(evicted_key[0]))

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
    """Find iteration-annotation root events (vLLM / SGLang / ATOM) in the graph tree.

    Primary (detailed) patterns are tried first, and native (backup) patterns are used only when no primary root is found.
    Spec-decode roots (verify, draft, draft_extend) are always added.
    """
    roots = {
        e[UID]: e
        for e in find_known_annotations(graph_tree.events)
        + find_spec_decode_annotations(graph_tree.events)
    }
    return sorted(roots.values(), key=lambda e: e.get("ts", 0))


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
    replay_rank: Optional[int] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[int]]]:
    """Load capture traces from a folder and group by ``{role}_{batch_size}_{mode}``.

    Args:
        capture_folder: Directory containing ``graph_capture_rank_0*`` trace files.
        metadata_json_path: Path to a JSON file — a list of objects each with
            ``file``, ``batch_size``, ``mode`` and optional ``capture_type``
            (``SPEC_ROLE_*``, default target) keys.
        replay_rank: When set and some captures belong to this rank, only
            those captures are used.

    Returns:
        Dictionary keyed by ``"{role}_{batch_size}_{mode}"`` whose values are
        lists of capture file paths, and the captured batch sizes per role.
    """

    with open(metadata_json_path, "r") as f:
        metadata_list = json.load(f)
    if any(extract_rank(e["file"]) == replay_rank for e in metadata_list):
        metadata_list = [
            e for e in metadata_list if extract_rank(e["file"]) == replay_rank
        ]

    result: Dict[str, List[str]] = defaultdict(list)
    batch_sizes: Dict[str, List[int]] = defaultdict(list)
    request_bs_multiplier: Dict[str, int] = {}
    for entry in metadata_list:
        filename = entry["file"]
        batch_size = entry["batch_size"]
        mode = entry["mode"]
        role = entry.get("capture_type", SPEC_ROLE_TARGET)
        if not (mode in ["FULL", "PIECEWISE"] and isinstance(batch_size, int)):
            print("Warning: invalid batch size or mode, skipping: {}".format(entry))
            continue
        key = "{}_{}_{}".format(role, batch_size, mode)
        batch_sizes[role].append(int(batch_size))
        rbs = entry.get("request_bs")
        if rbs is not None and rbs > 0 and _SGLANG_CAPTURE_PATTERN.match(filename):
            request_bs_multiplier.setdefault(role, int(batch_size) // int(rbs))
        filepath = os.path.join(capture_folder, filename)
        if not os.path.isfile(filepath):
            print("Warning: capture file not found, skipping: {}".format(filepath))
            continue

        if filepath not in result[key]:
            result[key].append(filepath)

    print(
        "\nCapture folder loaded: {} unique batch_size_mode keys".format(
            len(result.keys())
        )
    )

    return dict(result), batch_sizes, request_bs_multiplier


def _load_all_capture_sources(capture_map, role, batch_sizes):
    """Load all capture files for *role* across FULL and PIECEWISE modes.

    *batch_sizes* is an iterable of batch sizes to try (e.g. the resolved
    spec-decode batch size and the tensor batch size from the annotation).

    Returns a list of ``(capture_tree, capture_roots, capture_root_data, mode)``
    tuples — one per distinct capture file.
    """
    sources = []
    for bs in batch_sizes:
        if bs is None:
            continue
        for mode in ("PIECEWISE", "FULL"):
            key = "{}_{}_{}".format(role, bs, mode)
            for filepath in capture_map.get(key, []):
                cache_key = (key, os.path.abspath(filepath))
                tree, roots, root_data = _get_cached_capture_tree(cache_key, filepath)
                sources.append((tree, roots, root_data, mode))
    return sources


def _find_closest_capture(kernel_count, all_counts):
    """Find the pool index whose dispatch count is closest to *kernel_count*.

    Exact matches are preferred.  When no exact match exists, the nearest
    count wins (handles off-by-one mismatches common in profiler traces).
    """
    if kernel_count in all_counts:
        return all_counts[kernel_count][0]
    best_idx = None
    best_diff = float("inf")
    for dispatch_count, indices in all_counts.items():
        diff = abs(dispatch_count - kernel_count)
        if diff < best_diff:
            best_diff = diff
            best_idx = indices[0]
    return best_idx


def _match_graph_roots_to_captures(graph_tree, graph_roots, all_sources):
    """Match graph roots to capture roots from multiple capture sources by count.

    Each graph root is matched to the capture root whose dispatch count is
    closest to the graph root's kernel count.  Graph roots are classified as
    PIECEWISE (kernel count below the dominant count) or FULL (at or above),
    and matching is constrained to capture roots of the same mode.  Falls back
    to unconstrained matching when the constrained pool is empty.

    Returns ``(pairs, data, trees)`` — lists of ``(capture_root, graph_root)``,
    corresponding ``capture_root_data`` entries, and the ``capture_tree`` each
    root came from — or ``None`` when no captures are available.
    """
    pool = []
    for src_tree, cap_roots, cap_root_data, mode in all_sources:
        for capture_root, (events, filtered_uids) in zip(cap_roots, cap_root_data):
            dispatch_count = len(filtered_uids)
            pool.append(
                (src_tree, capture_root, (events, filtered_uids), dispatch_count, mode)
            )

    if not pool:
        return None

    kernel_counts = []
    for gr in graph_roots:
        _, filtered = get_subtree_events(
            graph_tree, gr, cat_filter=list(GPU_KERNEL_CATEGORIES)
        )
        kernel_counts.append(len(filtered))

    freq = Counter(kernel_counts)
    dominant = max(freq, key=lambda c: (freq[c], c))

    def _build_counts_index(entries):
        counts: Dict[int, List[int]] = defaultdict(list)
        for idx, (_, _, _, dc, _) in enumerate(entries):
            counts[dc].append(idx)
        return counts

    full_pool = [e for e in pool if e[4] == "FULL"]
    pw_pool = [e for e in pool if e[4] == "PIECEWISE"]
    full_counts = _build_counts_index(full_pool)
    pw_counts = _build_counts_index(pw_pool)
    all_counts = _build_counts_index(pool)

    pairs = []
    data = []
    trees = []
    for gr, kc in zip(graph_roots, kernel_counts):
        is_piecewise = kc < dominant
        if is_piecewise and pw_counts:
            target_pool, target_counts = pw_pool, pw_counts
        elif not is_piecewise and full_counts:
            target_pool, target_counts = full_pool, full_counts
        else:
            target_pool, target_counts = pool, all_counts
        pool_idx = _find_closest_capture(kc, target_counts)
        src_tree, capture_root, crd, _, _ = target_pool[pool_idx]
        pairs.append((capture_root, gr))
        data.append(crd)
        trees.append(src_tree)

    return pairs, data, trees


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


def find_execution_details(execution_root) -> Optional[str]:
    """Effective batch size for the iteration annotation root.

    Returns ``None`` when no integer batch size can be determined.
    """
    name = execution_root["name"]
    ann = parse_annotation(name)
    if ann.matched and ann.batch_size is not None:
        return str(ann.batch_size)
    parts = name.split("_")
    if len(parts) > 1 and parts[1].lstrip("-").isdigit():
        return parts[1]
    return None


def merge_capture_trace_into_graph(
    capture_folder: str = None,
    metadata_json_path: str = None,
    graph_tree_filepath: str = None,
    graph_tree: "TraceToTree" = None,
) -> "TraceToTree":
    """Merge capture trace information into a graph replay trace.

    Args:
        capture_folder: Directory containing capture trace files.
        metadata_json_path: Path to ``execution_details.json`` metadata
            index.
        graph_tree_filepath: Path to the graph-replay timing trace.
        graph_tree: Optional pre-built :class:`TraceToTree`.  When provided
            the expensive file load is skipped and this tree is used
            directly (after ensuring it is built with ``add_python_func``).

    Returns:
        Augmented graph_tree with capture information merged in, or ``None``
        if the merge failed.
    """
    if graph_tree is None:
        graph_tree = _load_trace_tree_from_file(
            graph_tree_filepath, add_python_func=True
        )
    else:
        graph_tree.build_tree(add_python_func=True)
    print("Loaded graph tree with {} events".format(len(graph_tree.events)))

    execution_graph_root_map = build_execution_graph_root_map(graph_tree)

    capture_map, capture_batch_sizes, request_bs_multiplier = load_capture_folder(
        capture_folder,
        metadata_json_path,
        extract_rank(os.path.basename(graph_tree_filepath)),
    )
    spec_batch_sizes = spec_decode_batch_sizes(
        [root for root, _ in execution_graph_root_map]
    )

    # ── Per-execution-root merge loop ──
    merge_failed = False
    for (execution_root, graph_roots), spec_batch_size in zip(
        execution_graph_root_map, spec_batch_sizes
    ):
        print("Processing execution root: {}".format(execution_root["name"]))
        if len(graph_roots) == 0:
            print(
                "No graph roots found for execution root {}".format(
                    execution_root["name"]
                )
            )
            continue

        # ── Resolve capture tree + roots for this execution root ──
        role = spec_decode_role(execution_root["name"]) or SPEC_ROLE_TARGET
        # spec_batch_size is a tuple of candidate batch sizes.  For vLLM
        # drafter roots this contains both the decode batch size
        # (num_requests) and the prefill batch size
        # (num_requests * (num_spec_tokens + 1)), because the drafter
        # uses separate CUDA graph managers for each phase.
        batch_size = spec_batch_size or find_execution_details(execution_root)
        if batch_size is None:
            print(
                "Warning: could not determine batch size for execution root {}".format(
                    execution_root["name"]
                )
            )
            continue
        if not isinstance(batch_size, tuple):
            batch_size = (batch_size,)
        candidate_batch_sizes = set()
        for bs in batch_size:
            bs_int = int(bs) * request_bs_multiplier.get(role, 1)
            cbs = find_closest_batch_size(bs_int, capture_batch_sizes[role])
            if cbs is not None:
                candidate_batch_sizes.add(cbs)
        if not candidate_batch_sizes:
            print(
                "Warning: no {} capture batch size found for execution root {}".format(
                    role, execution_root["name"]
                )
            )
            continue

        all_sources = _load_all_capture_sources(
            capture_map, role, sorted(candidate_batch_sizes)
        )
        # When a single source has the right root count, pair positionally
        # (preserves the old 1:1 and 1:N behaviour and avoids mis-pairing
        # roots that share the same dispatch count).
        result = None
        for src_tree, cap_roots, cap_root_data, _mode in all_sources:
            if len(cap_roots) == 1:
                result = (
                    [(cap_roots[0], g) for g in graph_roots],
                    [cap_root_data[0]] * len(graph_roots),
                    [src_tree] * len(graph_roots),
                )
                break
            if len(cap_roots) == len(graph_roots):
                result = (
                    list(zip(cap_roots, graph_roots)),
                    cap_root_data,
                    [src_tree] * len(graph_roots),
                )
                break
        if result is None:
            result = _match_graph_roots_to_captures(
                graph_tree, graph_roots, all_sources
            )
        if result is None:
            print(
                "Warning: could not match capture roots to graph roots "
                "for execution root {}".format(execution_root["name"])
            )
            continue
        pairs, data, capture_trees = result

        print(
            "Found {} capture roots and {} graph roots".format(
                len(pairs), len(graph_roots)
            )
        )
        for (c_root, g_root), (cached_events, filtered_uids), capture_tree in zip(
            pairs, data, capture_trees
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

            if len(graph_filtered_events) == 0:
                print(
                    "Warning: no kernel events in graph root {}; skipping merge".format(
                        g_root["name"]
                    )
                )
                continue

            if is_multistream(graph_filtered_events):
                if not capture_has_kernel_names(capture_filtered_events):
                    print(
                        "Graph capture merge failed: multistream graph root {} but "
                        "capture kernel names are absent".format(g_root["name"])
                    )
                    merge_failed = True
                    continue
                aligned = align_streams(graph_filtered_events, capture_filtered_events)
                if aligned is None:
                    print(
                        "Warning: multistream alignment failed for capture root {} and graph root {}".format(
                            c_root["name"], g_root["name"]
                        )
                    )
                    continue
                capture_filtered_events = aligned
            else:
                (
                    verify_success,
                    capture_filtered_events,
                    graph_filtered_events,
                ) = verify_subtree_events(
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
    if merge_failed:
        print(
            "Graph capture merge failed; returning None (report will be unaugmented)."
        )
        return None
    finalize_non_gpu_paths(graph_tree)
    return graph_tree
