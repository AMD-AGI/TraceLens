#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Step 1+3: Load a Chrome trace JSON and extract structured data.

Outputs a JSON with:
  - ordered kernel list (name, duration, timestamp, gpu_op_uid)
  - metadata (categories, graph mode detection, total kernel time)

vLLM traces with annotation iterations are auto-detected and split into
per-region subdirectories.  Pass ``--no-split`` to force single-file mode.

Usage:
    python extract_trace_data.py <trace.json> -o output_dir/
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter

from trace_split_adapter import split_vllm_trace, get_steady_state_key
from annotation_metadata import gather_metadata
from _helpers import load_json

from TraceLens import GPUEventAnalyser

logger = logging.getLogger(__name__)


def load_trace(path_or_data):
    """Load trace from path (str) or use dict directly."""
    if isinstance(path_or_data, dict):
        data = path_or_data
    else:
        data = load_json(path_or_data)
    events = data.get("traceEvents", [])
    by_cat = {}
    for e in events:
        if not isinstance(e, dict):
            continue
        cat = e.get("cat", "unknown")
        by_cat.setdefault(cat, []).append(e)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda e: e.get("ts", 0))
    return data, by_cat


def _stamp_raw_uid(data):
    """Tag each event with its position in the raw traceEvents array.

    This mirrors the UID scheme TraceToTree/TreePerfAnalyzer assign
    (enumerate() over the full, unfiltered traceEvents list), so a kernel's
    ``_gpu_op_uid`` here lines up with the same kernel's UID in a perf report
    built from the same trace file -- without ever building a tree. Only
    valid for a single, non-split trace file; do not call this on a
    per-region slice produced by vLLM trace splitting, since a region's
    traceEvents subset does not share the original file's indexing.
    """
    for i, e in enumerate(data.get("traceEvents", [])):
        if isinstance(e, dict):
            e["_gpu_op_uid"] = i


def get_stream_id(event):
    """Get stream from args['stream'] or tid (fallback for magic-trace style)."""
    stream = event.get("args", {}).get("stream")
    if stream is not None:
        try:
            return int(stream)
        except (TypeError, ValueError):
            pass
    tid = event.get("tid")
    if tid is not None:
        try:
            return int(tid)
        except (TypeError, ValueError):
            pass
    return None


def filter_to_primary_stream(by_cat):
    """If multiple streams in kernel events, keep only primary (most kernels).

    Skips filtering when secondary streams carry significant compute (>5%
    of total kernel time), since some runtimes schedule MoE / communication
    kernels on secondary CUDA streams.
    """
    kernels = by_cat.get("kernel", [])
    if not kernels:
        return
    stream_counts = Counter(
        get_stream_id(k) for k in kernels if get_stream_id(k) is not None
    )
    if len(stream_counts) <= 1:
        return
    total_time = sum(k.get("dur", 0) for k in kernels)
    primary = max(stream_counts, key=stream_counts.get)
    secondary_time = sum(
        k.get("dur", 0)
        for k in kernels
        if get_stream_id(k) != primary and get_stream_id(k) is not None
    )
    if total_time > 0 and secondary_time / total_time > 0.05:
        logger.info(
            "Keeping all %d streams: secondary streams have %.1f%% of kernel time",
            len(stream_counts),
            100 * secondary_time / total_time,
        )
        return
    by_cat["kernel"] = [k for k in kernels if get_stream_id(k) == primary]


def extract_kernel_sequence(by_cat):
    kernels = by_cat.get("kernel", [])
    memcpy = by_cat.get("gpu_memcpy", [])
    combined = sorted(kernels + memcpy, key=lambda e: e["ts"])
    return [
        {
            "name": k["name"],
            "cat": k.get("cat", "kernel"),
            "dur": k["dur"],
            "ts": k["ts"],
            "args": k.get("args", {}),
            "stream_id": get_stream_id(k),
            # Single-trace path: _stamp_raw_uid() sets "_gpu_op_uid" in-memory.
            # Split path: split_inference_trace_annotation.py (invoked with
            # --emit-gpu-op-uid) already persisted "gpu_op_uid" onto the event
            # before writing the per-region file, so it survives the subprocess
            # boundary as literal JSON content.
            "gpu_op_uid": k.get("_gpu_op_uid", k.get("gpu_op_uid")),
        }
        for k in combined
    ]


def detect_graph_mode(by_cat):
    rt = by_cat.get("cuda_runtime", [])
    graph_launches = [e for e in rt if "GraphLaunch" in e.get("name", "")]
    return len(graph_launches) > 0, graph_launches


def run_assertions(data, by_cat, kernels, is_graph_mode, strict=True):
    errors = []

    if "traceEvents" not in data:
        errors.append("A1.1 FAIL: Missing traceEvents key")

    required_cats = {"kernel", "cpu_op"} if strict else {"kernel"}
    missing = required_cats - set(by_cat.keys())
    if missing:
        errors.append(f"A1.2 FAIL: Missing categories: {missing}")

    if len(kernels) == 0:
        errors.append("A1.3 FAIL: No GPU kernels found")

    for i, k in enumerate(kernels):
        if k["dur"] <= 0:
            errors.append(
                f"A3.2 FAIL: Kernel {i} ({k['name'][:50]}) has non-positive duration {k['dur']}"
            )
            break

    timestamps = [k["ts"] for k in kernels]
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            errors.append(f"A3.1 FAIL: Kernel timestamps not monotonic at index {i}")
            break

    total_time = sum(k["dur"] for k in kernels)
    if total_time <= 0:
        errors.append("A1.5 FAIL: Zero total kernel time")

    return errors


def compute_gpu_timeline_metrics(events):  # pragma: no cover
    """
    Run GPUEventAnalyser on events and return gpu_timeline dict for metadata.
    Returns None on failure.
    """
    try:
        # GPUEventAnalyser needs a unique UID per event for overlap
        # computation. Raw trace events don't carry one, so assign a contiguous
        # 0..N-1. If an event already has a "UID", our assumption is broken and
        # uniqueness is no longer guaranteed -- fail loudly rather than proceed.
        for i, e in enumerate(events):
            if "UID" not in e:
                e["UID"] = i
            else:
                raise ValueError(
                    "Event unexpectedly already carries a 'UID'; "
                    "cannot guarantee unique ids for GPUEventAnalyser"
                )
        analyzer = GPUEventAnalyser(events)
        metrics = analyzer.compute_metrics()
        total = metrics.get("total_time", 0)
        busy = metrics.get("busy_time", 0)
        idle = metrics.get("idle_time", 0)
        if total <= 0:
            return None
        return {
            "busy_time_us": metrics.get("busy_time", 0),
            "idle_time_us": metrics.get("idle_time", 0),
            "total_time_us": total,
            "computation_time_us": metrics.get("computation_time", 0),
            "exposed_comm_time_us": metrics.get("exposed_comm_time", 0),
            "exposed_memcpy_time_us": metrics.get("exposed_memcpy_time", 0),
            "idle_pct": 100 * idle / total,
            "busy_pct": 100 * busy / total,
        }
    except Exception as e:
        logger.warning("GPUEventAnalyser failed: %s", e)
        return None


def extract_and_build_result(data, by_cat, source_file, region_metadata=None):
    """Build extraction result dict."""
    kernels = extract_kernel_sequence(by_cat)
    is_graph_mode, graph_launches = detect_graph_mode(by_cat)
    total_kernel_time = sum(k["dur"] for k in kernels)
    categories_found = sorted(by_cat.keys())
    result = {
        "source_file": source_file,
        "metadata": {
            "total_kernels": len(kernels),
            "total_kernel_time_us": round(total_kernel_time, 2),
            "is_graph_mode": is_graph_mode,
            "graph_launch_count": len(graph_launches),
            "categories": categories_found,
        },
        "kernels": kernels,
    }
    if region_metadata:
        result["region_metadata"] = region_metadata
    return result, kernels


def _write_split_regions(split_result, trace_path, output_dir):  # pragma: no cover
    """Write per-region extracted.json + metadata.json files."""
    os.makedirs(output_dir, exist_ok=True)
    for trace_dict, region_meta in split_result:
        key = get_steady_state_key(region_meta)
        region_dir = os.path.join(output_dir, key)
        os.makedirs(region_dir, exist_ok=True)
        merged_meta = gather_metadata(
            trace_path,
            trace_dict.get("traceEvents", []),
            annotation_meta=region_meta,
        )
        gpu_timeline = compute_gpu_timeline_metrics(trace_dict.get("traceEvents", []))
        if gpu_timeline:
            merged_meta["gpu_timeline"] = gpu_timeline
            region_meta = {**region_meta, "gpu_timeline": gpu_timeline}
        data, by_cat = load_trace(trace_dict)
        filter_to_primary_stream(by_cat)
        kernels_tmp = extract_kernel_sequence(by_cat)
        is_graph_tmp, _ = detect_graph_mode(by_cat)
        errors = run_assertions(data, by_cat, kernels_tmp, is_graph_tmp, strict=False)
        if errors:
            print(f"Skipping {key}: {'; '.join(errors)}", file=sys.stderr)
            continue
        result, kernels = extract_and_build_result(
            data, by_cat, trace_path, region_metadata=region_meta
        )
        extracted_path = os.path.join(region_dir, "extracted.json")
        meta_path = os.path.join(region_dir, "metadata.json")
        with open(extracted_path, "w") as f:
            json.dump(result, f, indent=2)
        with open(meta_path, "w") as f:
            json.dump(merged_meta, f, indent=2)
        print(f"Wrote {extracted_path} ({len(kernels)} kernels)", file=sys.stderr)


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Extract structured data from a Chrome trace JSON"
    )
    parser.add_argument("trace", help="Path to trace JSON file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory. Single-trace writes extracted.json inside "
        "it; multi-region (vLLM) writes per-region subdirectories.",
    )
    parser.add_argument(
        "--split-vllm",
        action="store_true",
        help="(deprecated, now auto-detected) kept for backward compatibility",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Skip annotation auto-detection; always extract as one trace",
    )
    args = parser.parse_args()

    # --- Auto-detect vLLM annotation regions (unless suppressed) -----------
    if not args.no_split:
        split_result = split_vllm_trace(args.trace)
        if split_result:
            output_dir = args.output or "."
            _write_split_regions(split_result, args.trace, output_dir)
            return

    # --- Single-trace extraction -------------------------------------------
    data, by_cat = load_trace(args.trace)
    _stamp_raw_uid(data)
    kernels = extract_kernel_sequence(by_cat)
    is_graph_mode, graph_launches = detect_graph_mode(by_cat)

    errors = run_assertions(data, by_cat, kernels, is_graph_mode)
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    result, _ = extract_and_build_result(data, by_cat, args.trace)
    output = json.dumps(result, indent=2)

    if args.output:
        out_path = args.output
        if os.path.isdir(out_path) or out_path.endswith("/"):
            os.makedirs(out_path, exist_ok=True)
            out_path = os.path.join(out_path, "extracted.json")
        with open(out_path, "w") as f:
            f.write(output)
        print(
            f"Wrote {out_path} ({len(kernels)} kernels, "
            f"{sum(k['dur'] for k in kernels):.1f}us total)",
            file=sys.stderr,
        )
    else:
        print(output)


if __name__ == "__main__":  # pragma: no cover
    main()
