###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Split a vLLM trace into per-iteration or per-dummy-run traces.

Usage:
    python split_vllm_trace_simple.py <trace_path> -o <output_dir> [--iterations START:END] [--dummy START:END]

Examples:
    # Extract all iterations and dummy runs (default if no args)
    python split_vllm_trace_simple.py trace.json.gz -o ./output
    
    # Extract all iterations
    python split_vllm_trace_simple.py trace.json.gz -o ./iterations --iterations all
    
    # Extract iterations 10-20
    python split_vllm_trace_simple.py trace.json.gz -o ./iterations --iterations 10:20
    
    # Extract dummy runs 0-2
    python split_vllm_trace_simple.py trace.json.gz -o ./dummy_runs --dummy 0:2
    
    # Extract both
    python split_vllm_trace_simple.py trace.json.gz -o ./output --iterations 50:55 --dummy 0:2
"""

import argparse
import gzip
import json
import os
import re
import sys
import zipfile
from typing import List, Set, Tuple, Optional

# Iteration marker patterns
EXECUTE_MODEL_PATTERN = re.compile(r"execute_\d+_context_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)_generation_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)")
## Use this for default vLLM
## EXECUTE_MODEL_PATTERN = re.compile(r"execute_context_\d+\(\d+\)_generation_\d+\(\d+\)")

DUMMY_RUN_PATTERN = re.compile(r"vllm/v1/worker/gpu_model_runner\.py\(\d+\): _dummy_run")
GPU_EVENT_CATEGORIES = ["kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"]

def load_trace(filepath: str) -> dict:
    """Load trace JSON from file (.json, .json.gz, or .zip)."""
    print(f"Loading trace: {filepath}")
    if filepath.endswith(".zip"):
        with zipfile.ZipFile(filepath, "r") as zf:
            # Find the JSON file inside the zip
            json_files = [f for f in zf.namelist() if f.endswith(".json")]
            if not json_files:
                raise ValueError(f"No .json file found in {filepath}")
            json_file = json_files[0]
            print(f"  Reading {json_file} from zip...")
            with zf.open(json_file) as f:
                return json.load(f)
    elif filepath.endswith(".json.gz"):
        with gzip.open(filepath, "rt") as f:
            return json.load(f)
    else:
        with open(filepath, "r") as f:
            return json.load(f)


def find_events_by_pattern(events: List[dict], pattern: re.Pattern, name: str, cat: str = None) -> List[dict]:
    """Find events matching a regex pattern."""
    matches = [e for e in events if pattern.match(e.get("name", ""))]
    if cat is not None:
        matches = [e for e in matches if e.get("cat") == cat]
    matches.sort(key=lambda x: x.get("ts", 0))
    print(f"Found {len(matches)} {name} events")
    return matches


def extract_iteration(
    iteration_root: dict,
    events: List[dict],
    trace_json: dict,
) -> dict:
    """Extract a single iteration trace."""
    iter_tid = iteration_root.get("tid")
    iter_pid = iteration_root.get("pid")
    iter_ts = iteration_root.get("ts", 0)
    iter_end = iter_ts + iteration_root.get("dur", 0)
    
    filtered_events = []
    correlation_ids: Set[int] = set()
    flow_events = []  # Collect flow events separately
    gpu_events = []  # Collect GPU events separately

    for e in events:
        ts = e.get("ts")
        
        # Metadata events (no timestamp) - always keep
        if ts is None:
            filtered_events.append(e)
            continue
        
        ph = e.get("ph")
        
        # Flow events - collect for later filtering
        if ph in ("s", "f"):
            flow_events.append(e)
            continue

        cat = e.get("cat")
        if cat in GPU_EVENT_CATEGORIES:
            gpu_events.append(e)
            continue
        
        dur = e.get("dur")
        if dur is None:
            continue
        
        e_end = ts + dur
        
        # Skip if outside iteration time range
        if ts < iter_ts or e_end > iter_end:
            continue
        
        e_tid = e.get("tid")
        e_pid = e.get("pid")
        
        # CPU events: same tid/pid
        if e_tid == iter_tid and e_pid == iter_pid:
            filtered_events.append(e)
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                correlation_ids.add(corr)
    
    # Add matching flow events
    for e in flow_events:
        if e.get("id") in correlation_ids:
            filtered_events.append(e)
    
    # Add matching GPU events (use args.correlation, not id)
    for e in gpu_events:
        if e.get("args", {}).get("correlation") in correlation_ids:
            filtered_events.append(e)
    
    # Create output trace
    output = trace_json.copy()
    output["traceEvents"] = filtered_events
    return output


def parse_range(range_str: str, max_len: int) -> Tuple[int, int]:
    """Parse a range string like '10:20' or 'all'."""
    if range_str == "all":
        return 0, max_len
    parts = range_str.split(":")
    start = int(parts[0])
    end = int(parts[1]) if len(parts) > 1 else start + 1
    return start, min(end, max_len)


def extract_and_save(
    roots: List[dict],
    events: List[dict],
    trace_json: dict,
    output_dir: str,
    base_name: str,
    prefix: str,
    start: int,
    end: int,
):
    """Extract and save a range of iterations/dummy runs."""
    selected = roots[start:end]
    indices = range(start, end)
    
    for idx, root in zip(indices, selected):
        iter_trace = extract_iteration(root, events, trace_json)
        name_append=root["name"].replace("(", "_").replace(")", "")
        out_path = os.path.join(output_dir, f"{base_name}_{prefix}_{idx}_{name_append}.json.gz")
        with gzip.open(out_path, "wt") as f:
            json.dump(iter_trace, f)
        
        print(f"  {prefix} {idx}: {len(iter_trace['traceEvents'])} events -> {out_path}")
    
    return len(selected)


def main():
    parser = argparse.ArgumentParser(
        description="Split vLLM trace into per-iteration or per-dummy-run traces"
    )
    parser.add_argument("trace_path", help="Path to trace file (.json or .json.gz)")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--iterations", "-i",
        help="Iteration range: 'all', single index '50', or range '10:20'"
    )
    parser.add_argument(
        "--dummy", "-d",
        help="Dummy run range: 'all', single index '0', or range '0:2'"
    )
    args = parser.parse_args()
    
    # Load trace
    trace_json = load_trace(args.trace_path)
    events = trace_json.get("traceEvents", [])
    print(f"Loaded {len(events)} events")
    
    # Find iterations and dummy runs
    iteration_roots = find_events_by_pattern(events, EXECUTE_MODEL_PATTERN, "execute_model (iteration)", cat="user_annotation")
    dummy_roots = find_events_by_pattern(events, DUMMY_RUN_PATTERN, "_dummy_run")
    
    # Default behavior: extract all if nothing specified
    if not args.iterations and not args.dummy:
        if iteration_roots:
            args.iterations = "all"
        if dummy_roots:
            args.dummy = "all"
        if not iteration_roots and not dummy_roots:
            print("Error: No iterations or dummy runs found in trace")
            sys.exit(1)
    
    # Check if requested events exist
    if args.iterations and not iteration_roots:
        print("Error: --iterations specified but no execute_model events found in trace")
        sys.exit(1)
    
    if args.dummy and not dummy_roots:
        print("Error: --dummy specified but no _dummy_run events found in trace")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_name = os.path.basename(args.trace_path)
    base_name = base_name.replace(".json.gz", "").replace(".json", "")
    
    total_extracted = 0
    
    # Extract iterations
    if args.iterations:
        start, end = parse_range(args.iterations, len(iteration_roots))
        print(f"\nExtracting iterations {start} to {end-1}...")
        count = extract_and_save(
            iteration_roots, events, trace_json, args.output_dir,
            base_name, "iteration", start, end
        )
        total_extracted += count
    
    # Extract dummy runs
    if args.dummy:
        start, end = parse_range(args.dummy, len(dummy_roots))
        print(f"\nExtracting dummy runs {start} to {end-1}...")
        count = extract_and_save(
            dummy_roots, events, trace_json, args.output_dir,
            base_name, "dummy", start, end
        )
        total_extracted += count
    
    print(f"\nDone! Extracted {total_extracted} traces to {args.output_dir}")


if __name__ == "__main__":
    main()
