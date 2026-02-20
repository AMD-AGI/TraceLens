###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Classify graph capture trace files by batch size and compilation mode.

Reads all files starting with ``graph_capture_rank_0`` in an input folder,
determines the batch size and compilation mode (full / piecewise) for each,
and prints the resulting dictionary.

Usage:
    python classify_graph_captures.py <input_folder> [--json-out <path>]

Approach:
    1. For each trace file, find root events using two strategies:
       a) ``_dummy_run`` pattern in event names.
       b) ``capture_<batch>_<mode>`` user-annotation pattern.
    2. If both strategies yield the same number of roots, prefer the
       annotation-based roots (they carry batch size & mode directly).
    3. Otherwise, fall back to counting ``StreamBeginCapture`` events
       (1 → full, >1 → piecewise) and infer batch size from the most
       common first dimension in ``cpu_op`` events.
"""

import argparse
import collections
import gzip
import json
import os
import re
import sys
import zipfile
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
DUMMY_RUN_PATTERN = re.compile(
    r"vllm/v1/worker/gpu_model_runner\.py\(\d+\): _dummy_run"
)
ANNOTATION_PATTERN = re.compile(r"capture_(\d+)_(.*)")

# ---------------------------------------------------------------------------
# Trace loading (supports .json, .json.gz, .zip)
# ---------------------------------------------------------------------------

def load_trace(filepath: str) -> dict:
    """Load a Chrome-trace JSON from *filepath*."""
    if filepath.endswith(".zip"):
        with zipfile.ZipFile(filepath, "r") as zf:
            json_files = [f for f in zf.namelist() if f.endswith(".json")]
            if not json_files:
                raise ValueError(f"No .json file found inside {filepath}")
            with zf.open(json_files[0]) as f:
                return json.load(f)
    elif filepath.endswith(".json.gz"):
        with gzip.open(filepath, "rt") as f:
            return json.load(f)
    else:
        with open(filepath, "r") as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Root-finding helpers
# ---------------------------------------------------------------------------

def find_dummy_run_roots(events: List[dict]) -> List[dict]:
    """Return events whose name matches the ``_dummy_run`` pattern."""
    roots = [e for e in events if DUMMY_RUN_PATTERN.match(e.get("name", ""))]
    roots.sort(key=lambda x: x.get("ts", 0))
    return roots


def find_annotation_roots(events: List[dict]) -> List[dict]:
    """Return user-annotation events matching ``capture_<batch>_<mode>``."""
    roots = [
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and ANNOTATION_PATTERN.match(e.get("name", ""))
    ]
    roots.sort(key=lambda x: x.get("ts", 0))
    return roots


def parse_annotation(name: str) -> Tuple[int, str]:
    """Extract (batch_size, mode) from an annotation name like ``capture_8192_full``."""
    m = ANNOTATION_PATTERN.match(name)
    if not m:
        raise ValueError(f"Annotation name does not match expected pattern: {name}")
    batch_size = int(m.group(1))
    mode = m.group(2)
    return batch_size, mode


# ---------------------------------------------------------------------------
# Fallback inference (no usable annotations)
# ---------------------------------------------------------------------------

def count_stream_begin_captures(events: List[dict]) -> int:
    """Count ``StreamBeginCapture`` cuda_runtime events."""
    return sum(
        1
        for e in events
        if e.get("name", "").startswith("StreamBeginCapture")
        and e.get("cat") == "cuda_runtime"
    )


def infer_batch_size_from_cpu_ops(events: List[dict]) -> Optional[int]:
    """Return the most common first dimension across ``cpu_op`` Input Dims."""
    first_dims: List[int] = []
    for e in events:
        if e.get("cat") != "cpu_op":
            continue
        input_dims = e.get("args", {}).get("Input Dims")
        if not input_dims:
            continue
        for dim_list in input_dims:
            if isinstance(dim_list, list) and len(dim_list) > 0:
                first_dims.append(dim_list[0])
    if not first_dims:
        return None
    counter = collections.Counter(first_dims)
    return counter.most_common(1)[0][0]


def infer_mode_from_captures(num_captures: int) -> str:
    if num_captures <= 1:
        return "full"
    return "piecewise"


# ---------------------------------------------------------------------------
# Per-file classification
# ---------------------------------------------------------------------------

def classify_trace(filepath: str) -> Dict:
    """Return ``{file, batch_size, mode}`` for a single trace file."""
    trace_json = load_trace(filepath)
    events = trace_json.get("traceEvents", [])

    dummy_roots = find_dummy_run_roots(events)
    annotation_roots = find_annotation_roots(events)

    basename = os.path.basename(filepath)

    if annotation_roots and len(annotation_roots) == len(dummy_roots):
        batch_size, mode = parse_annotation(annotation_roots[0]["name"])
        return {
            "file": basename,
            "batch_size": batch_size,
            "mode": mode,
        }

    num_captures = count_stream_begin_captures(events)
    mode = infer_mode_from_captures(num_captures)
    batch_size = infer_batch_size_from_cpu_ops(events)

    return {
        "file": basename,
        "batch_size": batch_size,
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Classify graph-capture traces by batch size and compilation mode."
    )
    parser.add_argument("input_folder", help="Folder containing graph_capture_rank_0* trace files")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the results as JSON",
    )
    args = parser.parse_args()

    input_folder = args.input_folder
    if not os.path.isdir(input_folder):
        print(f"Error: {input_folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    trace_files = sorted(
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.startswith("graph_capture_rank_0")
    )

    if not trace_files:
        print(f"No files starting with 'graph_capture_rank_0' found in {input_folder}")
        sys.exit(0)

    print(f"Found {len(trace_files)} graph-capture trace file(s) in {input_folder}\n")

    results = []
    for filepath in trace_files:
        print(f"Processing: {os.path.basename(filepath)} ...")
        try:
            info = classify_trace(filepath)
            results.append(info)
            print(f"  batch_size={info['batch_size']}, mode={info['mode']}")
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            results.append({
                "file": os.path.basename(filepath),
                "batch_size": None,
                "mode": None,
                "error": str(exc),
            })

    print("\n=== Results ===")
    for r in results:
        print(r)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.json_out}")


if __name__ == "__main__":
    main()
