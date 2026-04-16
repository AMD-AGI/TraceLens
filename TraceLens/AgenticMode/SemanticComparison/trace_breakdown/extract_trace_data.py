#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Step 1+3: Load a Chrome trace JSON and extract structured data.

Outputs a JSON with:
  - ordered kernel list (name, duration, timestamp)
  - python call stack (nested)
  - metadata (categories, graph mode detection, total kernel time)

Usage:
    python extract_trace_data.py <trace.json> [-o output.json]
    python extract_trace_data.py <trace.json> --split-vllm -o output_dir/
"""

import argparse
import gzip
import json
import logging
import os
import sys
from collections import Counter

from trace_split_adapter import split_vllm_trace, get_steady_state_key
from annotation_metadata import gather_metadata

try:
    from TraceLens import GPUEventAnalyser
except ImportError:
    GPUEventAnalyser = None

logger = logging.getLogger(__name__)


def load_trace(path_or_data):
    """Load trace from path (str) or use dict directly."""
    if isinstance(path_or_data, dict):
        data = path_or_data
    else:
        opener = gzip.open if path_or_data.endswith(".gz") else open
        with opener(path_or_data, "rt") as f:
            data = json.load(f)
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
        }
        for k in combined
    ]


def detect_graph_mode(by_cat):
    rt = by_cat.get("cuda_runtime", [])
    graph_launches = [e for e in rt if "GraphLaunch" in e.get("name", "")]
    return len(graph_launches) > 0, graph_launches


def extract_python_callstack(by_cat):
    pyfuncs = sorted(by_cat.get("python_function", []), key=lambda e: e["ts"])
    for p in pyfuncs:
        p["_end"] = p["ts"] + p["dur"]
    stack = []
    result = []
    for p in pyfuncs:
        while stack and stack[-1]["_end"] <= p["ts"]:
            stack.pop()
        result.append({"name": p["name"], "dur": p["dur"], "depth": len(stack)})
        stack.append(p)
    return result


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


def compute_gpu_timeline_metrics(events):
    """
    Run GPUEventAnalyser on events and return gpu_timeline dict for metadata.
    Returns None if GPUEventAnalyser unavailable or fails.
    """
    if GPUEventAnalyser is None:
        return None
    try:
        # Ensure events have UID (required by GPUEventAnalyser for overlap computation)
        for i, e in enumerate(events):
            if "UID" not in e:
                e["UID"] = i
        analyzer = GPUEventAnalyser(events)
        metrics = analyzer.compute_metrics()
        total = metrics.get("total_time", 0)
        busy = metrics.get("busy_time", 0)
        idle = metrics.get("idle_time", 0)
        if total <= 0:
            return None
        return {
            "busy_time_us": round(metrics.get("busy_time", 0), 2),
            "idle_time_us": round(metrics.get("idle_time", 0), 2),
            "total_time_us": round(total, 2),
            "computation_time_us": round(metrics.get("computation_time", 0), 2),
            "exposed_comm_time_us": round(metrics.get("exposed_comm_time", 0), 2),
            "exposed_memcpy_time_us": round(metrics.get("exposed_memcpy_time", 0), 2),
            "idle_pct": round(100 * idle / total, 1),
            "busy_pct": round(100 * busy / total, 1),
        }
    except Exception as e:
        logger.warning("GPUEventAnalyser failed: %s", e)
        return None


def extract_and_build_result(data, by_cat, source_file, region_metadata=None):
    """Build extraction result dict."""
    kernels = extract_kernel_sequence(by_cat)
    is_graph_mode, graph_launches = detect_graph_mode(by_cat)
    callstack = extract_python_callstack(by_cat)
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
            "has_python_stack": len(callstack) > 0,
        },
        "kernels": kernels,
        "python_callstack": callstack,
    }
    if region_metadata:
        result["region_metadata"] = region_metadata
    return result, kernels


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured data from a Chrome trace JSON"
    )
    parser.add_argument("trace", help="Path to trace JSON file")
    parser.add_argument(
        "-o", "--output", help="Output JSON path or directory (with --split-vllm)"
    )
    parser.add_argument(
        "--split-vllm",
        action="store_true",
        help="Split by gpu_user_annotation, extract per steady-state region",
    )
    args = parser.parse_args()

    if args.split_vllm:
        split_result = split_vllm_trace(args.trace)
        if not split_result:
            print(
                "No gpu_user_annotation events found, falling back to full trace",
                file=sys.stderr,
            )
            data, by_cat = load_trace(args.trace)
            result, kernels = extract_and_build_result(data, by_cat, args.trace)
            out_path = args.output or "extracted.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Wrote {out_path} ({len(kernels)} kernels)", file=sys.stderr)
        else:
            output_dir = args.output or "."
            os.makedirs(output_dir, exist_ok=True)
            for trace_dict, region_meta in split_result:
                key = get_steady_state_key(region_meta)
                region_dir = os.path.join(output_dir, key)
                os.makedirs(region_dir, exist_ok=True)
                merged_meta = gather_metadata(
                    args.trace,
                    trace_dict.get("traceEvents", []),
                    annotation_meta=region_meta,
                )
                gpu_timeline = compute_gpu_timeline_metrics(
                    trace_dict.get("traceEvents", [])
                )
                if gpu_timeline:
                    merged_meta["gpu_timeline"] = gpu_timeline
                    region_meta = {**region_meta, "gpu_timeline": gpu_timeline}
                data, by_cat = load_trace(trace_dict)
                filter_to_primary_stream(by_cat)
                kernels_tmp = extract_kernel_sequence(by_cat)
                is_graph_tmp, _ = detect_graph_mode(by_cat)
                errors = run_assertions(
                    data, by_cat, kernels_tmp, is_graph_tmp, strict=False
                )
                if errors:
                    print(f"Skipping {key}: {'; '.join(errors)}", file=sys.stderr)
                    continue
                result, kernels = extract_and_build_result(
                    data, by_cat, args.trace, region_metadata=region_meta
                )
                extracted_path = os.path.join(region_dir, "extracted.json")
                meta_path = os.path.join(region_dir, "metadata.json")
                with open(extracted_path, "w") as f:
                    json.dump(result, f, indent=2)
                with open(meta_path, "w") as f:
                    json.dump(merged_meta, f, indent=2)
                print(
                    f"Wrote {extracted_path} ({len(kernels)} kernels)", file=sys.stderr
                )
        return

    data, by_cat = load_trace(args.trace)
    kernels = extract_kernel_sequence(by_cat)
    is_graph_mode, graph_launches = detect_graph_mode(by_cat)
    callstack = extract_python_callstack(by_cat)

    errors = run_assertions(data, by_cat, kernels, is_graph_mode)
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    result, _ = extract_and_build_result(data, by_cat, args.trace)
    output = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(
            f"Wrote {args.output} ({len(kernels)} kernels, {sum(k['dur'] for k in kernels):.1f}us total)",
            file=sys.stderr,
        )
    else:
        print(output)


if __name__ == "__main__":
    main()
