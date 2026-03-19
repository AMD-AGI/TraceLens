###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
🚀 vLLM/SGLang Trace Splitting and Analysis Tool

This script splits large vLLM and SGLang inference traces into smaller, analyzable components:
- Individual execution iterations
- Steady-state regions (representative execution windows)
- Per-phase traces (prefill-decode vs decode-only)
- Dummy run traces (graph capture phases)

This enables efficient performance analysis and comparison without processing massive tracefiles.

═══════════════════════════════════════════════════════════════════════════════

📋 BASIC USAGE
───────────────────────────────────────────────────────────────────────────────
    python split_vllm_trace_annotation.py <trace_path> -o <output_dir> [OPTIONS]

🔧 REQUIRED ARGUMENTS
───────────────────────────────────────────────────────────────────────────────
    trace_path              Path to input trace file (.json, .json.gz, or .zip)
    -o, --output-dir        Directory where split traces will be saved

📊 OPTIONAL ARGUMENTS
───────────────────────────────────────────────────────────────────────────────    
    -i, --iterations        Iteration range to extract (default: 'all'):
                           'START:END'  - Extract iterations START through END-1
    
    -d, --dummy             Dummy run range (default: 'all'):
                           'START:END'  - Extract dummy runs START through END-1
    
    --store-single-iteration  Store each iteration/dummy run as individual file
                             
    --find-steady-state      Automatically detect and extract steady-state region:
                            - Combines specified number of iterations from steady-state window
                            - Extracts separate prefill-decode phase trace
                            - Extracts separate decode-only phase trace
                            (Requires --iterations all # Default value)

    --num-steps             Number of iterations to extract for steady-state (default: 32)
💡 QUICK EXAMPLES
───────────────────────────────────────────────────────────────────────────────

1. DEFAULT: Extract all iterations separately
   
   $ python split_vllm_trace_annotation.py trace.json.gz -o ./output --store-single-iteration
   
   → Individual trace files for each iteration (e.g. trace_annotation )

─────────────────────────────────────────────────────────────────────────────

2. EXTRACT SPECIFIC ITERATION RANGE (combined)
   
   $ python split_vllm_trace_annotation.py trace.json.gz \\
     -o ./output \\
     --iterations 10:20
   
   → Single trace file containing iterations 10-19 with prefill-decode and decode-only phases separated to two different tracefiles
   

─────────────────────────────────────────────────────────────────────────────

3. FIND AND EXTRACT STEADY STATE REGION ⭐ RECOMMENDED
   
   $ python split_vllm_trace_annotation.py trace.json.gz \\
     -o ./steady_state_analysis \\
     --find-steady-state
   
   This automatically:
   • Analyzes all iterations to find steady-state region
   • Extracts the specified number  (default: 32) representative iterations
   • Splits into phase-specific traces:
     - Combined steady-state trace
     - Prefill-decode phase trace
     - Decode-only phase trace
     - execution_details.json with metadata
─────────────────────────────────────────────────────────────────────────────

4. EXTRACT DUMMY RUNS (Graph Capture Phases)
   
   $ python split_vllm_trace_annotation.py trace.json.gz \\
     -o ./dummy_runs --store-single-iteration
   
   → Tracefile per specified dummy run

─────────────────────────────────────────────────────────────────────────────

───────────────────────────────────────────────────────────────────────────────

Generated outputs:

  ✓ Individual .json.gz trace files in output directory
  ✓ execution_details.json - Metadata about extracted traces

Example file structure:
  output/
  ├── trace_annotation_iteration_0_prefilldecode_10_bs32_conc18.json.gz
  ├── trace_decode_3_bs32_conc18.json.gz
  ├── trace_prefilldecode_10_bs32_conc18.json.gz
  └── execution_details.json

Example execution_details.json entry:
{
  "idx": 0,
  "output_path": "./output/trace_annotation_iteration_0.json.gz",
  "event_count": 45230,
  "num_gpu_events": 1250,
  "gpu_duration": 2300000,
  "gpu_busy_duration": 1000000,
  "phase": {
    "num_prefill": 5,
    "num_prefilldecode": 10,
    "num_decode": 3,
    "avg_bs": 32,
    "avg_conc": 18
  }
}

🔗 RELATED TOOLS
───────────────────────────────────────────────────────────────────────────────

After splitting traces, analyze them with:

• generate_perf_report_pytorch_vllm.py - Performance analysis
• TraceDiff - Compare two traces

═══════════════════════════════════════════════════════════════════════════════
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
from statistics import mean
from TraceLens.util import DataLoader
import pandas as pd

# Try to use faster JSON parser (orjson is 2-10x faster than json)
import orjson
from tqdm import tqdm

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


def find_events_by_pattern(
    events: List[dict], patterns, name: str, cat: str = None
) -> List[dict]:
    """Find events matching a regex pattern."""
    matches = []
    for pattern in patterns:
        matches.extend([e for e in events if pattern.match(e.get("name", ""))])
        if cat is not None:
            matches = [e for e in matches if e.get("cat") == cat]
    matches.sort(key=lambda x: x.get("ts", 0))
    print(f"Found {len(matches)} {name} events")
    for m in matches:
        print(m["name"])
    if len(matches) == 0:
        return None
    return matches


def preprocess_trace(events: List[dict]):
    gpu_corr_map = {}
    flow_corr_map = {}
    meta_events = []
    for e in tqdm(events):
        ts = e.get("ts")
        ph = e.get("ph")
        cat = e.get("cat")
        if ts is None:
            meta_events.append(e)
            continue
        if ph in ("s", "f"):
            corr = e.get("id")
            if corr is not None:
                flow_corr_map.setdefault(corr, []).append(e)
            continue
        if cat in GPU_EVENT_CATEGORIES:
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                gpu_corr_map.setdefault(corr, []).append(e)
            continue
    return gpu_corr_map, flow_corr_map, meta_events


def extract_iteration(
    iteration_roots: List[dict],
    events: List[dict],
    trace_json: dict,
    gpu_corr_map: dict,
    flow_corr_map: dict,
    meta_events: List[dict],
) -> dict:
    """Extract a single iteration trace."""

    filtered_events = []
    gpu_dur = 0
    gpu_busy = 0
    num_gpu_events = 0
    batch_list = []

    # Pre-index GPU and flow events by correlation id

    # Compute the global time window for all iteration roots
    if not iteration_roots:
        return trace_json.copy(), [], 0, 0
    min_iter_ts = min(root.get("ts", 0) for root in iteration_roots)
    max_iter_end = max(
        root.get("ts", 0) + root.get("dur", 0) for root in iteration_roots
    )
    # Collect all relevant tid/pid pairs
    tid_pid_set = set((root.get("tid"), root.get("pid")) for root in iteration_roots)

    # Pre-filter all CPU events in the global window and by tid/pid
    cpu_events = []
    for e in events:
        ts = e.get("ts")
        if ts is None:
            continue
        dur = e.get("dur")
        if dur is None:
            continue
        e_end = ts + dur
        e_tid = e.get("tid")
        e_pid = e.get("pid")
        if (e_tid, e_pid) in tid_pid_set and (
            min_iter_ts <= ts and e_end <= max_iter_end
        ):
            cpu_events.append(e)

    # For each iteration root, filter CPU events and collect correlation ids
    for iteration_root in tqdm(iteration_roots):
        batch = 0
        start_time = []
        end_time = []
        iter_tid = iteration_root.get("tid")
        iter_pid = iteration_root.get("pid")
        iter_ts = iteration_root.get("ts", 0)
        iter_end = iter_ts + iteration_root.get("dur", 0)

        correlation_ids: Set[int] = set()

        # CPU events: filter from pre-filtered list
        for e in cpu_events:
            ts = e.get("ts")
            dur = e.get("dur")
            e_end = ts + dur
            e_tid = e.get("tid")
            e_pid = e.get("pid")
            if (
                e_tid == iter_tid
                and e_pid == iter_pid
                and (iter_ts <= ts and e_end <= iter_end)
            ):
                filtered_events.append(e)
                corr = e.get("args", {}).get("correlation")
                if corr is not None:
                    correlation_ids.add(corr)

        # Add matching flow events
        for corr in correlation_ids:
            for e in flow_corr_map.get(corr, []):
                filtered_events.append(e)
        # Add matching GPU events
        for corr in correlation_ids:
            for e in gpu_corr_map.get(corr, []):
                filtered_events.append(e)
                start_time.append(e.get("ts"))
                end_time.append(e.get("ts") + e.get("dur"))
                gpu_busy += e.get("dur")
                num_gpu_events += 1
        gpu_dur += max(end_time) - min(start_time) if start_time else 0

    # Add all meta events (no timestamp)
    filtered_events.extend(meta_events)

    for e in tqdm(filtered_events):
        if "vllm::unified_attention_with_output" in e.get(
            "name", ""
        ) or "sgl_kernel::sgl_per_token_group_quant_8bit" in e.get("name", ""):
            dims = e.get("args", {}).get("Input Dims")
            if dims and len(dims) > 0 and len(dims[0]) > 0:
                batch_list.append(dims[0][0])
    # Create output trace
    output = trace_json.copy()
    output["traceEvents"] = filtered_events
    return output, list(set(batch_list)), num_gpu_events, gpu_dur, gpu_busy


def parse_range(range_str: str, max_len: int) -> Tuple[int, int]:
    """Parse a range string like '10:20' or 'all'."""
    if range_str == "all":
        return 0, max_len
    parts = range_str.split(":")
    start = int(parts[0])
    end = int(parts[1]) if len(parts) > 1 else start + 1
    return start, min(end, max_len)


def get_iter_details_from_name(name: str, prefix: str = "annotation_iteration") -> dict:

    name = name.replace("(", "_").replace(")", "_")
    if not "annotation_iteration" in prefix:
        return {"batch_size": 0}
    iter_details = re.sub(r"[sqk]+", "_", name).split("_")
    if len(iter_details) < 10:
        ctx_req, ctx_sum, gen_req, gen_sum = (
            iter_details[2],
            iter_details[3],
            iter_details[6],
            iter_details[7],
        )
    elif len(iter_details) < 12:
        ctx_req, ctx_sum, gen_req, gen_sum = (
            iter_details[2],
            iter_details[3],
            iter_details[7],
            iter_details[8],
        )
    else:
        ctx_req, ctx_sum, gen_req, gen_sum = (
            iter_details[3],
            iter_details[5],
            iter_details[11],
            iter_details[13],
        )
    # print(name,iter_details,ctx_req,ctx_sum,gen_req,gen_sum)
    return {
        "batch_size": int(ctx_sum) + int(gen_sum),
        "num_requests": int(ctx_req) + int(gen_req),
        "context_requests": int(ctx_req),
        "context_sum": int(ctx_sum),
        "generation_requests": int(gen_req),
        "generation_sum": int(gen_sum),
    }


def find_phase_from_window(iter_details: List[dict]) -> dict:

    num_prefill = len(
        [
            d
            for d in iter_details
            if d.get("context_requests", 0) > 0 and d.get("generation_requests", 0) == 0
        ]
    )
    num_prefilldecode = len(
        [
            d
            for d in iter_details
            if d.get("context_requests", 0) > 0 and d.get("generation_requests", 0) > 0
        ]
    )

    num_decode = len(
        [
            d
            for d in iter_details
            if d.get("generation_requests", 0) > 0 and d.get("context_requests", 0) == 0
        ]
    )
    avg_batch_size = int(
        sum(d.get("batch_size", 0) for d in iter_details) / len(iter_details)
    )
    avg_concurrency = int(
        sum(d.get("num_requests", 0) for d in iter_details) / len(iter_details)
    )

    return {
        "num_prefill": num_prefill,
        "num_prefilldecode": num_prefilldecode,
        "num_decode": num_decode,
        "avg_bs": avg_batch_size,
        "avg_conc": avg_concurrency,
    }


def extract_and_save(
    roots: List[List[dict]],
    events: List[dict],
    trace_json: dict,
    output_dir: str,
    base_name: str,
    prefix: str,
    start: int,
    end: int,
    gpu_corr_map: dict,
    flow_corr_map: dict,
    meta_events: List[dict],
):
    """Extract and save a range of iterations/dummy runs."""
    extraction_summary = []
    selected = roots[start:end]
    indices = range(start, end)
    if len(selected) == 0:
        print(f"No {prefix} events found in the specified range, skipping extraction")
        return extraction_summary
    for idx, root in zip(indices, selected):
        iter_details = [get_iter_details_from_name(r["name"], prefix) for r in root]
        iter_trace, batch_list, num_gpu_events, gpu_dur, gpu_busy = extract_iteration(
            root, events, trace_json, gpu_corr_map, flow_corr_map, meta_events
        )
        if "annotation_iteration" in prefix and len(root) == 1:
            name_append = root[0]["name"].replace("(", "_").replace(")", "")
        else:
            if len(batch_list) == len(iter_details):
                name_append = f"batch{int(sum(batch_list)/len(batch_list))}_gpu{prefix}"
                for bs, iteration in zip(batch_list, iter_details):
                    iteration["batch_size"] = bs
            else:
                name_append = f"batch_NA_gpu{prefix}"

        phase_details = find_phase_from_window(iter_details)
        if "annotation_iteration" in prefix and len(root) > 1:
            name_append = f"prefilldecode_{phase_details['num_prefilldecode']}_decode_{phase_details['num_decode']}_bs{phase_details['avg_bs']}_conc{phase_details['avg_conc']}"

        out_path = os.path.join(
            output_dir, f"{base_name}_{prefix}_{idx}_{name_append}.json.gz"
        )
        with gzip.open(out_path, "wb") as f:
            f.write(json.dumps(iter_trace).encode("utf-8"))

        print(
            f"  {prefix} {idx}: {len(iter_trace['traceEvents'])} events -> {out_path}"
        )
        extraction_summary.append(
            {
                "idx": idx,
                "output_path": out_path,
                "event_count": len(iter_trace["traceEvents"]),
                "num_gpu_events": num_gpu_events,
                "gpu_duration": gpu_dur,
                "gpu_busy_duration": gpu_busy,
                "steps": iter_details,
                "phase": phase_details,
            }
        )
    return extraction_summary


def extract_phases_and_save(
    roots: List[List[dict]],
    events: List[dict],
    trace_json: dict,
    output_dir: str,
    base_name: str,
    prefix: str,
    start: int,
    end: int,
    gpu_corr_map: dict,
    flow_corr_map: dict,
    meta_events: List[dict],
):
    """Extract and save a range of iterations/dummy runs."""
    extraction_summary = []

    if "annotation_iteration" not in prefix:
        print("phase extraction only supported for annotation iterations, skipping")
        return extraction_summary
    for root in roots:
        iter_details = [get_iter_details_from_name(r["name"], prefix) for r in root]
        phase_details = find_phase_from_window(iter_details)
        prefilldecode_steps = [
            r for r, i in zip(root, iter_details) if i.get("context_requests", 0) > 0
        ]
        decode_steps = [
            r
            for r, i in zip(root, iter_details)
            if i.get("generation_requests", 0) > 0 and i.get("context_requests", 0) == 0
        ]

        if len(prefilldecode_steps) > 0:
            iter_details = [
                get_iter_details_from_name(r["name"], prefix)
                for r in prefilldecode_steps
            ]
            phase_details = find_phase_from_window(iter_details)

            iter_trace, batch_list, num_gpu_events, gpu_dur, gpu_busy = (
                extract_iteration(
                    prefilldecode_steps,
                    events,
                    trace_json,
                    gpu_corr_map,
                    flow_corr_map,
                    meta_events,
                )
            )
            name_append = f"prefilldecode_{phase_details['num_prefilldecode']}_bs{phase_details['avg_bs']}_conc{phase_details['avg_conc']}"

            out_path = os.path.join(output_dir, f"{name_append}_{base_name}.json.gz")
            with gzip.open(out_path, "wb") as f:
                f.write(json.dumps(iter_trace).encode("utf-8"))

            print(f"  {prefix}: {len(iter_trace['traceEvents'])} events -> {out_path}")
            extraction_summary.append(
                {
                    "idx": 0,
                    "output_path": out_path,
                    "event_count": len(iter_trace["traceEvents"]),
                    "num_gpu_events": num_gpu_events,
                    "gpu_duration": gpu_dur,
                    "gpu_busy_duration": gpu_busy,
                    "steps": iter_details,
                    "phase": phase_details,
                }
            )

        iter_details = [
            get_iter_details_from_name(r["name"], prefix) for r in decode_steps
        ]
        phase_details = find_phase_from_window(iter_details)
        iter_trace, batch_list, num_gpu_events, gpu_dur, gpu_busy = extract_iteration(
            decode_steps, events, trace_json, gpu_corr_map, flow_corr_map, meta_events
        )
        name_append = f"decode_{phase_details['num_decode']}_bs{phase_details['avg_bs']}_conc{phase_details['avg_conc']}"

        out_path = os.path.join(output_dir, f"{name_append}_{base_name}.json.gz")
        with gzip.open(out_path, "wb") as f:
            f.write(json.dumps(iter_trace).encode("utf-8"))

        print(f"  {prefix}: {len(iter_trace['traceEvents'])} events -> {out_path}")
        extraction_summary.append(
            {
                "idx": 0,
                "output_path": out_path,
                "event_count": len(iter_trace["traceEvents"]),
                "num_gpu_events": num_gpu_events,
                "gpu_duration": gpu_dur,
                "gpu_busy_duration": gpu_busy,
                "steps": iter_details,
                "phase": phase_details,
            }
        )
    return extraction_summary


def find_steady_state_iterations(
    iteration_roots: List[dict], num_steps: int = 5, decode_only: bool = False
) -> List[dict]:
    n = len(iteration_roots)
    thresh = 0.1
    if n < num_steps:
        print(
            f"Not enough iterations ({n}) to find steady state with {num_steps} steps"
        )
        thresh = 0.2
    iter_details = [get_iter_details_from_name(r["name"]) for r in iteration_roots]
    global_max = max(t["num_requests"] for t in iter_details)
    # Find indices where num_requests is close to global average
    steady_state_started = False
    steady_state_ended = False
    prev_events_in_steady = 0
    regions = []
    for i, t in enumerate(iter_details):
        # print(t["num_requests"],global_max,abs(t["num_requests"] - global_max))

        if abs(t["num_requests"] - global_max) <= max(1, thresh * global_max):
            if not steady_state_started:
                print(
                    f"iteration {i} with num_requests {t['num_requests']} is within steady state threshold of global max {global_max}"
                )
                prev_events_in_steady += 1
        else:
            if steady_state_started:
                prev_events_in_steady -= 1
        if prev_events_in_steady > 5 and not steady_state_started:
            print("steady state started at index", i - 5)
            steady_state_started = True
            start_index = i - prev_events_in_steady + 1
        if (
            prev_events_in_steady <= 0
            and steady_state_started
            and not steady_state_ended
        ):
            print("steady state ended at index", i)
            steady_state_ended = True
            end_index = i
            regions.append((start_index, end_index))
            steady_state_started = False
            steady_state_ended = False
            prev_events_in_steady = 0
    if steady_state_started and not steady_state_ended:
        regions.append((start_index, i))
    print("steady state regions:", regions)

    if len(regions) == 0:
        delta = max(8, num_steps - n)
        regions = [(delta // 2, n - delta // 2)]
        print(
            "warning: no steady state region found, discarding initial and final iterations and selecting middle region"
        )
    sub_regions = []
    divider = min(int(num_steps/2),10)
    for s, e in regions:
        if (e - s) > num_steps:
            for s1 in range(s, e, num_steps // divider):
                region = iter_details[s1 : s1 + num_steps]
                sub_regions.append(
                    [
                        s1,
                        s1 + num_steps,
                        len([t for t in region if t["context_requests"] > 0]),
                        mean([t["num_requests"] for t in region]),
                    ]
                )
        else:
            sub_regions.append(
                [
                    s,
                    e,
                    len([t for t in iter_details[s:e] if t["context_requests"] > 0]),
                    mean([t["num_requests"] for t in iter_details[s:e]]),
                ]
            )
    if not decode_only:
        sub_regions_tmp = [t for t in sub_regions if t[2]>0]
        if len(sub_regions_tmp)==0:
            print("prefilldecode step not found, selecting decode-only region")
        else:
            sub_regions = sub_regions_tmp
    best_window = sorted(sub_regions, key=lambda x: x[3], reverse=True)[0]
    print("Selected steady state window:", best_window)
    return iteration_roots[best_window[0] : best_window[1]]


def main():
    parser = argparse.ArgumentParser(
        description="Split vLLM trace into per-iteration or per-dummy-run traces"
    )
    parser.add_argument("trace_path", help="Path to trace file (.json or .json.gz)")

    parser.add_argument("-o", "--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--iterations",
        "-i",
        default="all",
        help="Iteration range: 'all', single index '50', or range '10:20'",
    )
    parser.add_argument(
        "--dummy",
        "-d",
        default="all",
        help="Dummy run range: 'all', single index '0', or range '0:2'",
    )
    parser.add_argument(
        "--store-single-iteration",
        action="store_true",
        default=False,
        help="Store each iteration separately",
    )
    parser.add_argument(
        "--find-steady-state",
        action="store_true",
        default=False,
        help="For iterations, find steady state region and extract from there instead of sequential iterations",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=32,
        help="Number of iterations to extract for steady state (default: 32)",
    )
    parser.add_argument(
        "--decode-only",
        action="store_true",
        default=False,
        help="Extract only decode phase",
    )
    args = parser.parse_args()
    execution_details = []
    # Iteration marker patterns
    ANNOTATION_PATTERN = [
        re.compile(
            r"execute_\d+_context_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)_generation_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)"
        ),
        re.compile(r"execute_context_\d+\(\d+\)_generation_\d+\(\d+\)"),
        re.compile(r"execute_new_\d+_cached_\d+"),
        re.compile(r"execute_context_\d+\(\d+_\d+\)_generation_\d+\(\d+\)"),
    ]
    RUNTIME_EVENT_PATTERN = [
        re.compile(r"vllm/v1/worker/gpu_model_runner\.py\(\d+\): _dummy_run"),
        ## re.compile(r"/sgl-workspace/sglang/python/sglang/srt/managers/scheduler\.py\(\d+\): run_batch"),
        re.compile(
            r"/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py\(\d+\): _capture_graph"
        ),
    ]

    # Load trace
    trace_json = DataLoader.load_data(get_filename(args.trace_path))
    events = trace_json.get("traceEvents", [])
    gpu_corr_map, flow_corr_map, meta_events = preprocess_trace(events)
    print(f"Loaded {len(events)} events")

    # Find iterations and dummy runs
    iteration_roots = find_events_by_pattern(
        events, ANNOTATION_PATTERN, "execution steps (iteration)", cat="user_annotation"
    )
    dummy_roots = find_events_by_pattern(events, RUNTIME_EVENT_PATTERN, "_dummy_run")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.basename(args.trace_path)
    base_name = (
        base_name.replace(".pt.trace", "").replace(".json.gz", "").replace(".json", "")
    )

    # Extract iterations
    if args.iterations and iteration_roots:
        start, end = parse_range(args.iterations, len(iteration_roots))
        if args.store_single_iteration:
            print(f"\nExtracting iterations {start} to {end-1}...")
            temp_execution_details = extract_and_save(
                [[root] for root in iteration_roots],
                events,
                trace_json,
                args.output_dir,
                base_name,
                "annotation_iteration",
                start,
                end,
                gpu_corr_map,
                flow_corr_map,
                meta_events,
            )
            execution_details.extend(temp_execution_details)

        if args.iterations != "all" or args.find_steady_state:
            if args.iterations != "all":
                iteration_roots_subset = iteration_roots[start:end]
            else:
                iteration_roots_subset = find_steady_state_iterations(
                    iteration_roots,
                    num_steps=args.num_steps,
                    decode_only=args.decode_only,
                )
            print(f"\nExtracting iterations {start} to {end-1}...")
            temp_execution_details = extract_and_save(
                [iteration_roots_subset],
                events,
                trace_json,
                args.output_dir,
                base_name,
                "annotation_iteration",
                0,
                1,
                gpu_corr_map,
                flow_corr_map,
                meta_events,
            )
            execution_details.extend(temp_execution_details)
            print("starting phase extraction...")
            temp_execution_details = extract_phases_and_save(
                [iteration_roots_subset],
                events,
                trace_json,
                args.output_dir,
                base_name,
                "annotation_iteration",
                0,
                1,
                gpu_corr_map,
                flow_corr_map,
                meta_events,
            )
            execution_details.extend(temp_execution_details)
    # Extract dummy runs
    if args.dummy and dummy_roots:
        start, end = parse_range(args.dummy, len(dummy_roots))
        if args.store_single_iteration:
            print(f"\nExtracting dummy runs {start} to {end-1}...")
            temp_execution_details = extract_and_save(
                [[root] for root in dummy_roots],
                events,
                trace_json,
                args.output_dir,
                base_name,
                "run_iteration",
                start,
                end,
                gpu_corr_map,
                flow_corr_map,
                meta_events,
            )
            execution_details.extend(temp_execution_details)
        if args.dummy != "all":
            print(f"\nExtracting dummy runs {start} to {end-1}...")
            temp_execution_details = extract_and_save(
                [dummy_roots[start:end]],
                events,
                trace_json,
                args.output_dir,
                base_name,
                "run_iteration",
                0,
                1,
                gpu_corr_map,
                flow_corr_map,
                meta_events,
            )
            execution_details.extend(temp_execution_details)
        if args.find_steady_state:
            print(f"finding steady state without annotations not supported")

    print(f"\nDone! Extracted {len(execution_details)} traces to {args.output_dir}")
    if len(execution_details) > 0:
        json_path = os.path.join(args.output_dir, "execution_details.json")
        with open(json_path, "w") as f:
            json.dump(execution_details, f, indent=2)
        print(f"Wrote execution details JSON to {json_path}")

        rows = []
        for entry in execution_details:
            row = {k: v for k, v in entry.items() if k not in ("steps", "phase")}
            if "phase" in entry and entry["phase"]:
                for pk, pv in entry["phase"].items():
                    row[f"phase_{pk}"] = pv
            row["num_steps"] = len(entry.get("steps", []))
            row["gpu_busy_duration"] = entry.get("gpu_busy_duration", 0)
            row["gpu_duration"] = entry.get("gpu_duration", 0)
            row["num_gpu_events"] = entry.get("num_gpu_events", 0)
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(args.output_dir, "execution_details.csv")
        df.to_csv(csv_path, index=False, float_format="%.2f")
        print(f"Wrote execution details CSV to {csv_path}")


if __name__ == "__main__":
    main()
