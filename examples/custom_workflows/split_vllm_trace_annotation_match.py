#!/usr/bin/env python3
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
from dataclasses import dataclass, field
import csv
from TraceLens.util import DataLoader
# Try to use faster JSON parser (orjson is 2-10x faster than json)

# Iteration marker patterns
EXECUTE_MODEL_PATTERN = re.compile(r"execute_context_\d+\(\d+_\d+\)_generation_\d+\(\d+\)")

DUMMY_RUN_PATTERN = re.compile(r"vllm/v1/worker/gpu_model_runner\.py\(\d+\): _dummy_run")
GPU_EVENT_CATEGORIES = ["kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"]

def get_filename(filepath: str) -> dict:
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
            return json_file
    return filepath

def find_events_by_pattern(events: List[dict], pattern: re.Pattern, name: str, cat: str = None) -> List[dict]:
    """Find events matching a regex pattern."""
    matches = [e for e in events if pattern.match(e.get("name", ""))]
    if cat is not None:
        matches = [e for e in matches if e.get("cat") == cat]
    matches.sort(key=lambda x: x.get("ts", 0))
    print(f"Found {len(matches)} {name} events")
    for m in matches:
        print(m["name"])
    return matches


def extract_iteration(
    iteration_roots: List[dict],
    events: List[dict],
    trace_json: dict,
) -> dict:
    """Extract a single iteration trace."""
    filtered_events = []

    gpu_dur=0
    for iteration_root in iteration_roots:
        start_time=[]
        end_time=[]
        iter_tid = iteration_root.get("tid")
        iter_pid = iteration_root.get("pid")
        iter_ts = iteration_root.get("ts", 0)
        iter_end = iter_ts + iteration_root.get("dur", 0)
        
        
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
                start_time.append(e.get("ts"))
                end_time.append(e.get("ts")+e.get("dur"))
        gpu_dur+=max(end_time)-min(start_time) if start_time else 0
    # Create output trace
    output = trace_json.copy()
    output["traceEvents"] = filtered_events
    return output,gpu_dur

def parse_range(range_str: str, max_len: int) -> Tuple[int, int]:
    """Parse a range string like '10:20' or 'all'."""
    if range_str == "all":
        return 0, max_len
    parts = range_str.split(":")
    start = int(parts[0])
    end = int(parts[1]) if len(parts) > 1 else start + 1
    return start, min(end, max_len)

def get_iter_details_from_name(name: str)-> dict:
    iter_details=name.replace("(","_").replace(")","_").split("_")
    ctx_req,ctx_sum,gen_req,gen_sum=iter_details[2],iter_details[3],iter_details[7],iter_details[8]
    return {
        "batch_size": int(ctx_sum)+int(gen_sum),
        "num_requests": int(ctx_req)+int(gen_req),
        "context_requests": int(ctx_req),
        "context_sum": int(ctx_sum),
        "generation_requests": int(gen_req),
        "generation_sum": int(gen_sum),  
    }
def extract_and_save(
    roots: List[List[dict]],
    events: List[dict],
    trace_json: dict,
    output_dir: str,
    base_name: str,
    prefix: str,
):
    """Extract and save a range of iterations/dummy runs."""
    extraction_summary= []
    
    for idx, root in enumerate(roots):
        iter_trace,gpu_dur = extract_iteration(root, events, trace_json)
        if base_name is None:
            name_append=root[0]["name"].replace("(", "_").replace(")", "")
        else:
            name_append=f"{base_name}"
        out_path = os.path.join(output_dir, f"{prefix}_{idx}_{name_append}.json.gz")
        # Use binary mode and optimized compression for faster writing
        #with gzip.open(out_path, "wb") as f:
        #    f.write(json.dumps(iter_trace).encode('utf-8'))
        
        print(f"  {prefix} {idx}: {len(iter_trace['traceEvents'])} events -> {out_path}")
        extraction_summary.append({
            "iteration": idx,
            "output_path": out_path,
            "event_count": len(iter_trace['traceEvents']),
            "gpu_duration": gpu_dur,
            "steps": [get_iter_details_from_name(r["name"]) for r in root]
        })
    return extraction_summary
 
def find_similar_sequences(list1, list2, similarity_threshold=0.95):
    """
    Find contiguous sequences where:
    - First element is the same
    - Rest of elements are similar (within threshold)
    
    Args:
        list1, list2: Lists of tuples
        similarity_threshold: How similar numbers should be (0-1)
    
    Returns:
        List of (start_idx1, end_idx1, start_idx2, end_idx2) for matching sequences
    """
    matches = []
    i, j = 0, 0
    
    while i < len(list1) and j < len(list2):
        print(i,j)
        # Check if first elements match
        if list1[i][0] == list2[j][0]:
            print("match found")
            seq_start_i, seq_start_j = i, j
            seq_len = 0
            
            # Extend while first elements match and rest are similar
            #while (i < len(list1) and j < len(list2) and 
            #       list1[i][0] == list2[j][0] and
            #       _are_similar(list1[i][1:], list2[j][1:], similarity_threshold)):
            while (i < len(list1) and j < len(list2) and 
                   list1[i][0] == list2[j][0]):
                seq_len += 1
                i += 1
                j += 1
            
            if seq_len > 0:
                matches.append({
                    'list1_range': (seq_start_i, seq_start_i + seq_len),
                    'list2_range': (seq_start_j, seq_start_j + seq_len),
                    'length': seq_len,
                    'key': list1[seq_start_i][0]
                })
        else:
            # Move pointer of smaller first element
            if i<j:
                i += 1
            else:
                j += 1
    
    return matches

def _are_similar(vals1, vals2, threshold):
    """Check if tuples of numbers are similar within threshold."""
    if len(vals1) != len(vals2):
        return False
    
    for v1, v2 in zip(vals1, vals2):
        if abs(v1 - v2) / max(abs(v1), abs(v2), 1) > (1 - threshold):
            return False
    return True


def find_windows_with_high_batch(
    iteration_roots: List[dict],
    window_size: int = 3,
    tol: float = 0.1,             # within 10% of the max by default
    min_required: int = 1,        # at least one entry >= 1
    relative_to: str = "window",  # "window" or "global"
) -> List[Tuple[int, List[Tuple[int, ...]]]]:
    n = len(iteration_roots)
    if n < window_size:
        return []
    iter_details=[get_iter_details_from_name(r["name"]) for r in iteration_roots]
    global_max = max(t["num_requests"] for t in iter_details)
    matches = []
    for i in range(n - window_size + 1):
        window = iter_details[i : i + window_size]
        roots= iteration_roots[i : i + window_size]
        bs = [t["num_requests"] for t in window]
        ref = max(bs) if relative_to == "window" else global_max
        # all batch sizes must be within (1-tol)*ref
        if min(bs) < (1 - tol) * ref:
            continue
        # at least one entry meets the minimum requirement
        if not any(x >= min_required for x in bs):
            continue
        matches.append((i, roots))
        print(f"Window starting at iteration {i} matches: batch sizes = {window}")
    return matches
def main():
    parser = argparse.ArgumentParser(
        description="Split vLLM trace into per-iteration or per-dummy-run traces"
    )
    parser.add_argument("--trace-path", required=False, help="Path to trace file (.json or .json.gz)")
    parser.add_argument("--ref-trace-path", required=False, help="Path to reference trace file (.json or .json.gz)")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory")
    parser.add_argument("--ref-output-dir", required=False, help="Reference Output directory")
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

    trace_json = DataLoader.load_data(get_filename(args.trace_path))
    events = trace_json.get("traceEvents", [])
    print(f"Loaded {len(events)} events")
    
    # Find iterations and dummy runs
    iteration_roots = find_events_by_pattern(events, EXECUTE_MODEL_PATTERN, "execute_model (iteration)", cat="user_annotation")
    if not iteration_roots:
            print("Error: No iterations or dummy runs found in trace")
            sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    execution_details = extract_and_save(
        [[root] for root in iteration_roots], events, trace_json, args.output_dir, None, "single_iteration"
    )

    # Save execution details as JSON
    if execution_details:
        json_path = os.path.join(args.output_dir, "execution_details.json")
        with open(json_path, "w") as f:
            json.dump(execution_details, f, indent=2)
        print(f"Wrote execution details JSON to {json_path}")
    
    
    ## Find chunks of interest
    
    windows=find_windows_with_high_batch(iteration_roots, window_size=3, tol=0.25, min_required=1, relative_to="global")
    
    print(f"Found {len(windows)} windows with high batch sizes")
    execution_details = extract_and_save(
        [match[1] for match in windows], events, trace_json, args.output_dir, "mix", "high_batch_window"
    )
    if execution_details:
        json_path = os.path.join(args.output_dir, "high_match_execution_details.json")
        with open(json_path, "w") as f:
            json.dump(execution_details, f, indent=2)
        print(f"Wrote execution details JSON to {json_path}")


    if args.ref_trace_path is not None:
        trace_json = DataLoader.load_data(get_filename(args.ref_trace))
        ref_events = trace_json.get("traceEvents", [])
        ref_iteration_roots = find_events_by_pattern(ref_events, EXECUTE_MODEL_PATTERN, "execute_model (iteration)", cat="user_annotation")


    
    total_extracted = 0
    if args.ref_trace_path is not None:
        base_iters=get_iter_details(iteration_roots)
        ref_iters=get_iter_details(ref_iteration_roots)
        for i,j in zip(base_iters,ref_iters):
            print(i,j)
        print("Done extracting")
        matches = find_similar_sequences(base_iters, ref_iters, similarity_threshold=0.75)
        match_id=1
        for match_id,match in enumerate(matches):
            b_start=match["list1_range"][0]
            r_start=match["list2_range"][0]
            print(f"==== Match {match_id+1} ====\n")
            print("Main (batch_size, ctx_requests, ctx_sum, gen_requests, gen_sum) -> Reference (batch_size, ctx_requests, ctx_sum, gen_requests, gen_sum)")
            for i in range(match["length"]):
                print(f"{base_iters[b_start]} -> {ref_iters[r_start]}" )
                b_start+=1
                r_start+=1
            print("\n\n")

    #sys.exit()
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
    
    
    print(f"\nDone! Extracted {total_extracted} traces to {args.output_dir}")


if __name__ == "__main__":
    main()
