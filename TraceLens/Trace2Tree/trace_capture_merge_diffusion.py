###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Merge shape metadata from a torch.compile shape trace into a graph-replay
timing trace at the tree level.

Analogous to ``trace_capture_merge_experimental.py`` (which merges vLLM/SGLang
CUDA graph capture traces into graph-replay traces), but for the torch.compile
path used by diffusion models (xDiT, FLUX.1, etc.).

Two-pass profiling:
  1. **Shape trace**: ``torch.compile(mode="max-autotune-no-cudagraphs")`` —
     each compiled kernel dispatches individually, so the profiler records
     per-kernel ``cpu_op`` events with ``Input Dims``, ``Concrete Inputs``,
     and ``Input type``.
  2. **Timing trace**: ``torch.compile(mode="max-autotune")`` — kernels
     replay via ``hipGraphLaunch`` for representative timing.

This module merges the shape data from (1) into the tree built from (2),
re-parenting GPU kernel nodes under synthetic cpu_op events that carry the
shape metadata.  Unlike the raw-event-level merge in ``run_with_profiling.py``,
this operates after ``TraceToTree`` has been built, so the tree's
``parent``/``children``/``gpu_events`` fields are updated correctly.

Usage:
    from TraceLens.Trace2Tree.trace_capture_merge_diffusion import (
        merge_diffusion_shape_trace,
    )

    # graph_tree: TraceToTree built from the timing trace
    # shape_trace_path: path to the shape trace (.json.gz)
    augmented_tree = merge_diffusion_shape_trace(shape_trace_path, graph_tree)
"""

import copy
import gzip
import json
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import TraceLens.util

UID = TraceLens.util.TraceEventUtils.TraceKeys.UID
Name = TraceLens.util.TraceEventUtils.TraceKeys.Name
Args = TraceLens.util.TraceEventUtils.TraceKeys.Args


def _build_kernel_shape_map(
    shape_trace_path: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build a mapping from GPU kernel name to its cpu_op parent's shape metadata.

    Reads the shape trace, finds each GPU kernel event, looks up its parent
    cpu_op via ``External id``, and extracts ``Input Dims``, ``Concrete Inputs``,
    ``Input type``, ``Input Strides``, and the cpu_op name.

    Returns:
        Dict mapping kernel name -> list of shape metadata dicts (one per
        occurrence, to handle kernels that appear multiple times with
        different shapes).
    """
    with gzip.open(shape_trace_path, "rt") as f:
        shape_data = json.load(f)

    events = shape_data.get("traceEvents", shape_data)

    # Index cpu_ops with shape metadata by External id
    cpu_by_ext_id: Dict[Any, dict] = {}
    for e in events:
        if (
            e.get("cat") == "cpu_op"
            and e.get("ph") == "X"
            and e.get("args", {}).get("Input Dims")
        ):
            ext_id = e.get("args", {}).get("External id")
            if ext_id is not None:
                cpu_by_ext_id[ext_id] = e

    # Map kernel name -> list of shape info dicts
    kernel_shapes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in events:
        if e.get("cat") == "kernel" and e.get("ph") == "X":
            ext_id = e.get("args", {}).get("External id")
            cpu_e = cpu_by_ext_id.get(ext_id)
            if cpu_e is not None:
                kernel_shapes[e["name"]].append(
                    {
                        "cpu_op_name": cpu_e["name"],
                        "Input Dims": cpu_e["args"].get("Input Dims"),
                        "Input type": cpu_e["args"].get("Input type"),
                        "Input Strides": cpu_e["args"].get("Input Strides"),
                        "Concrete Inputs": cpu_e["args"].get("Concrete Inputs"),
                    }
                )

    return dict(kernel_shapes)


def _find_gpu_events_without_shapes(graph_tree) -> List[dict]:
    """Find all GPU kernel events that lack a cpu_op ancestor with shape data.

    This includes kernels launched via hipGraphLaunch (graph replay),
    hipModuleLaunchKernel (AITER/CK kernels), or any other path where
    the profiler did not record Input Dims on a cpu_op parent.
    """
    gpu_cats = {"kernel", "gpu_memset", "gpu_memcpy"}
    result = []
    for event in graph_tree.events:
        if event.get("cat") not in gpu_cats:
            continue
        # Walk parent chain looking for a cpu_op with Input Dims
        has_shape_ancestor = False
        parent_uid = event.get("parent")
        while parent_uid is not None:
            parent = graph_tree.events_by_uid.get(parent_uid)
            if parent is None:
                break
            if (
                parent.get("cat") == "cpu_op"
                and parent.get("args", {}).get("Input Dims")
            ):
                has_shape_ancestor = True
                break
            parent_uid = parent.get("parent")
        if not has_shape_ancestor:
            result.append(event)
    return result


def merge_diffusion_shape_trace(
    shape_trace_path: str,
    graph_tree_filepath: str = None,
    graph_tree=None,
) -> Any:
    """Merge shape metadata into a graph-replay timing tree.

    For each GPU kernel in the timing tree that was launched via
    ``hipGraphLaunch``, this function:

    1. Looks up the kernel's name in the shape trace's kernel→cpu_op map.
    2. Creates a synthetic ``cpu_op`` event in the timing tree with the
       shape metadata (``Input Dims``, ``Concrete Inputs``, ``Input type``).
    3. Re-parents the GPU kernel under the synthetic cpu_op.
    4. Propagates ``gpu_events`` up the tree so ancestors (including the
       ``Torch-Compiled Region``) correctly aggregate GPU kernel time.

    Args:
        shape_trace_path: Path to the shape trace (.json.gz).
        graph_tree_filepath: Path to the timing trace. If provided, a new
            tree is built from this file.
        graph_tree: A pre-built ``TraceToTree`` from the timing trace.
            Ignored if ``graph_tree_filepath`` is provided.

    Returns:
        The modified tree with shape metadata injected.
    """
    if graph_tree_filepath is not None:
        from ..TreePerf.tree_perf import TreePerfAnalyzer
        perf = TreePerfAnalyzer.from_file(graph_tree_filepath, add_python_func=False)
        graph_tree = perf.tree
    kernel_shapes = _build_kernel_shape_map(shape_trace_path)
    if not kernel_shapes:
        warnings.warn(
            "No kernel shape data found in shape trace: {}".format(shape_trace_path),
            stacklevel=2,
        )
        return graph_tree

    # Find GPU events that lack shape-bearing cpu_op ancestors
    graph_gpu_events = _find_gpu_events_without_shapes(graph_tree)
    if not graph_gpu_events:
        warnings.warn(
            "No GPU events without shape data found in timing tree — "
            "nothing to merge",
            stacklevel=2,
        )
        return graph_tree

    # Track per-kernel-name occurrence index for kernels with multiple shapes
    kernel_occurrence: Dict[str, int] = defaultdict(int)

    # Next UID for synthetic events
    next_uid = max(graph_tree.events_by_uid.keys()) + 1

    matched = 0
    unmatched = 0

    for gpu_event in graph_gpu_events:
        kname = gpu_event.get("name", "")
        shape_list = kernel_shapes.get(kname)
        if not shape_list:
            unmatched += 1
            continue

        # Use occurrence index to handle repeated kernels with same name
        occ_idx = kernel_occurrence[kname]
        kernel_occurrence[kname] += 1
        shape_info = shape_list[occ_idx % len(shape_list)]

        matched += 1
        gpu_uid = gpu_event[UID]

        # Remember the original parent (hipGraphLaunch runtime event)
        original_parent_uid = gpu_event.get("parent")

        # Create synthetic cpu_op event with shape metadata
        synthetic_event = {
            UID: next_uid,
            "ph": "X",
            "cat": "cpu_op",
            "name": shape_info["cpu_op_name"],
            "pid": gpu_event.get("pid", 0),
            "tid": gpu_event.get("tid", 0),
            "ts": gpu_event["ts"],
            "dur": gpu_event.get("dur", 0),
            "args": {},
            "children": [gpu_uid],
            "gpu_events": [gpu_uid],
            "tree": True,
        }

        # Add shape metadata to args
        for key in ("Input Dims", "Input type", "Input Strides", "Concrete Inputs"):
            val = shape_info.get(key)
            if val is not None:
                synthetic_event["args"][key] = val

        # Insert synthetic event into the tree
        graph_tree.events.append(synthetic_event)
        graph_tree.events_by_uid[next_uid] = synthetic_event
        graph_tree.name2event_uids[shape_info["cpu_op_name"]].append(next_uid)

        # Re-parent: GPU kernel -> synthetic cpu_op -> original parent
        gpu_event["parent"] = next_uid

        if original_parent_uid is not None:
            synthetic_event["parent"] = original_parent_uid
            original_parent = graph_tree.events_by_uid.get(original_parent_uid)
            if original_parent is not None:
                # Replace the GPU kernel in the original parent's children
                # with the synthetic event
                children = original_parent.get("children", [])
                if gpu_uid in children:
                    idx = children.index(gpu_uid)
                    children[idx] = next_uid
                else:
                    children.append(next_uid)
                original_parent["children"] = children

        # Propagate gpu_events up the parent chain
        parent_uid = synthetic_event.get("parent")
        while parent_uid is not None:
            parent = graph_tree.events_by_uid.get(parent_uid)
            if parent is None:
                break
            parent.setdefault("gpu_events", []).append(gpu_uid)
            parent_uid = parent.get("parent")

        next_uid += 1

    print(
        "Diffusion shape merge: {} GPU kernels matched, {} unmatched "
        "(out of {} graph-launched kernels)".format(
            matched, unmatched, len(graph_gpu_events)
        )
    )

    # Re-label non-GPU paths now that new cpu_op events exist
    gpu_cats = {"kernel", "gpu_memset", "gpu_memcpy"}
    for event in graph_tree.events:
        if event.get("cat") in gpu_cats:
            continue
        if "gpu_events" not in event:
            event["non_gpu_path"] = True
        else:
            event.pop("non_gpu_path", None)

    return graph_tree
