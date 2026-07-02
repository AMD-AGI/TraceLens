###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
vLLM/SGLang/ATOM Trace Splitting and Analysis Tool

This script splits large vLLM and SGLang inference traces into smaller, analyzable components:
- Individual execution iterations
- Steady-state regions (representative execution windows)
- Per-phase traces (prefill-decode vs decode-only)
- Dummy run traces (graph capture phases)

This enables efficient performance analysis and comparison without processing massive tracefiles.

═══════════════════════════════════════════════════════════════════════════════

BASIC USAGE
───────────────────────────────────────────────────────────────────────────────
    python split_inference_trace_annotation.py <trace_path> -o <output_dir> [OPTIONS]

REQUIRED ARGUMENTS
───────────────────────────────────────────────────────────────────────────────
    trace_path              Path to input trace file (.json, .json.gz, or .zip)
    -o, --output-dir        Directory where split traces will be saved

OPTIONAL ARGUMENTS
───────────────────────────────────────────────────────────────────────────────
    -i, --iterations        Iteration range to extract (default: 'all'):
                            'all'        - All iterations
                            'N'          - Single iteration N
                            'START:END'  - Iterations START through END-1

    -d, --dummy             Dummy run range (default: 'all'):
                            'all'        - All dummy runs
                            'N'          - Single dummy run N
                            'START:END'  - Dummy runs START through END-1

    --store-single-iteration  Store each iteration/dummy run as an individual file

    --find-steady-state      Automatically detect steady-state and extract three
                             representative contiguous windows (no idle gaps):
                             - mixed_steady_state_*        : representative DO:PD mix
                             - decode_only_steady_state_*  : fewest prefill-decode steps
                             - prefilldecode_steady_state_*: most prefill-decode steps

    --divide-phases          Find all steady-state regions and store each individual
                             step into phase-specific sub-folders:
                             output_dir/prefilldecodemix/ and output_dir/decode_only/.
                             Each step is written as a separate trace file.

    --num-steps             Number of iterations to extract for steady-state (default: 32)

    --CONC                  Expected peak concurrency (number of concurrent requests).
                            A warning is printed if the trace peak differs from this value.

    --OSL                   Average output sequence length (decode tokens per request).
                            Used with --R to compute the ideal PD ratio for mixed-window
                            selection under --find-steady-state.

    --R                     OSL window ratio in [0, 1]. OSL per request is sampled from
                            [R*OSL, OSL], giving mean OSL = OSL*(1+R)/2.
                            R=0 means all requests have exactly OSL tokens;
                            R=1 means OSL is uniform in [0, OSL].

QUICK EXAMPLES
───────────────────────────────────────────────────────────────────────────────

1. EXTRACT ALL ITERATIONS SEPARATELY

   $ python split_inference_trace_annotation.py trace.json.gz -o ./output --store-single-iteration

   → One trace file per iteration in ./output/

─────────────────────────────────────────────────────────────────────────────

2. EXTRACT SPECIFIC ITERATION RANGE (combined)

   $ python split_inference_trace_annotation.py trace.json.gz \\
     -o ./output \\
     --iterations 10:20

   → Single combined trace file containing iterations 10-19

─────────────────────────────────────────────────────────────────────────────

3. FIND AND EXTRACT STEADY STATE REGION (recommended)

   $ python split_inference_trace_annotation.py trace.json.gz \\
     -o ./steady_state_analysis \\
     --find-steady-state

   This automatically:
   • Identifies all steady-state regions across the trace
   • Computes the PD/total ratio for every region and derives a reference
     ratio (largest region, cross-checked against the median of all regions)
   • Extracts THREE separate contiguous windows — no idle gaps:
     - mixed_steady_state_*        : representative DO:PD mix
     - decode_only_steady_state_*  : fewest prefill-decode steps
     - prefilldecode_steady_state_*: most prefill-decode steps

─────────────────────────────────────────────────────────────────────────────

4. SPLIT STEADY-STATE STEPS BY PHASE

   $ python split_inference_trace_annotation.py trace.json.gz \\
     -o ./phase_split \\
     --divide-phases

   → Writes each steady-state step into phase-specific sub-folders:
       ./phase_split/prefilldecodemix/
       ./phase_split/decode_only/

─────────────────────────────────────────────────────────────────────────────

5. EXTRACT DUMMY RUNS (Graph Capture Phases)

   $ python split_inference_trace_annotation.py trace.json.gz \\
     -o ./dummy_runs --store-single-iteration

   → One trace file per dummy run in ./dummy_runs/

─────────────────────────────────────────────────────────────────────────────

Generated outputs:

  ✓ Individual .json.gz trace files in output directory
  ✓ execution_details.json - Metadata about extracted traces
  ✓ execution_details.csv  - Flat CSV version of the same metadata

Example file structure (--find-steady-state):
  output/
  ├── mixed_steady_state_prefilldecode_5_decode_27_bs32_conc18_{base}.json.gz
  ├── decode_only_steady_state_prefilldecode_0_decode_32_bs30_conc16_{base}.json.gz
  ├── prefilldecode_steady_state_prefilldecode_12_decode_20_bs48_conc20_{base}.json.gz
  ├── execution_details.json
  └── execution_details.csv

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

RELATED TOOLS
───────────────────────────────────────────────────────────────────────────────

After splitting traces, analyze them with:

• generate_perf_report_pytorch_vllm.py - Performance analysis
• TraceDiff - Compare two traces

═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import gzip
import json
import math
import os
import re
import sys
import zipfile
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass, field
import csv
from statistics import mean
from TraceLens.util import DataLoader
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
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


# Iteration marker patterns. The primary pattern is preferred; the backup
# patterns are only consulted when the primary matches nothing (see
# find_iteration_roots). The execute_new_<n>_cached_<n> shape is intentionally
# excluded because get_iter_details_from_name cannot parse it.
ANNOTATION_PATTERN = [
    re.compile(
        r"execute_\d+_context_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)_generation_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)"
    ),
]
ANNOTATION_PATTERN_BACKUP = [
    re.compile(r"execute_context_\d+\(\d+\)_generation_\d+\(\d+\)"),
    re.compile(r"execute_context_\d+\(\d+_\d+\)_generation_\d+\(\d+\)"),
    # SGLang profiler per-step annotations, e.g.
    #   "step[EXTEND bs=1 toks=862]"  (prefill / extend batch)
    #   "step[DECODE bs=25]"          (decode batch)
    re.compile(r"step\[(?:EXTEND|DECODE|MIXED)\b.*\]"),
]


def find_iteration_roots(events: List[dict]) -> Optional[List[dict]]:
    """Return iteration-root events.

    Tries the primary annotation pattern first, then backup patterns, then
    falls back to generic call-tree traversal via Trace2Tree.
    """
    roots = find_events_by_pattern(
        events, ANNOTATION_PATTERN, "execution steps (iteration)", cat="user_annotation"
    )
    if roots is None:
        print("No primary annotations found; falling back to backup patterns...")
        roots = find_events_by_pattern(
            events,
            ANNOTATION_PATTERN_BACKUP,
            "execution steps (iteration, backup)",
            cat="user_annotation",
        )
    if roots is None:
        print("No annotation patterns found; trying generic call-tree traversal...")
        roots = find_iteration_roots_generic(events)
    return roots


def _find_repeating_period(
    names: List[str], min_repeats: int = 3
) -> Tuple[Optional[int], Optional[List[str]], Optional[int]]:
    """Find the shortest repeating name sequence anywhere in ``names``.

    Slides a start offset forward to skip any non-repeating prefix (setup
    events before the loop body). Returns ``(period, pattern, start_offset)``
    where ``start_offset`` is the index in ``names`` where the first block
    begins. Returns ``(None, None, None)`` if no qualifying period is found.

    Requires at least ``min_repeats`` consecutive repetitions covering more
    than half of the suffix starting at ``start_offset``.
    """
    n = len(names)
    for start in range(n):
        suffix = names[start:]
        m = len(suffix)
        for p in range(1, m // 2 + 1):
            pattern = suffix[:p]
            count = 0
            i = 0
            while i + p <= m and suffix[i : i + p] == pattern:
                count += 1
                i += p
            if count >= min_repeats and count * p > m * 0.5:
                return p, pattern, start
    return None, None, None


def _detect_iteration_roots_from_tree(tree: TraceToTree, roots) -> Optional[List[dict]]:
    """BFS down the tree from one or more root nodes to find and return synthetic
    iteration-root events.

    ``roots`` may be a single event dict or a list of event dicts — all are
    seeded into the BFS at depth 0 so they are explored level-by-level together.

    Pattern detection uses all children (not just GPU-path ones) so that
    leading CPU-only events (e.g. ``next`` in the OWL pipeline) are included
    as part of the iteration anchor. A minimum child count guards against false
    positives from short utility-function child lists.

    Returns a list of synthetic root events, one per detected iteration, where
    each event's ``dur`` spans from the first to the last child of the block.
    """
    from collections import deque

    if isinstance(roots, dict):
        roots = [roots]

    queue = deque((node, 0) for node in roots)
    while queue:
        current, depth = queue.popleft()
        children = tree.get_children_events(current)
        if not children:
            continue

        # Only recurse into GPU-bearing subtrees.
        if not any(c.get("gpu_events") for c in children):
            continue

        p, _, start = _find_repeating_period([c.get("name", "") for c in children])
        if p is None:
            for child in children:
                if child.get("gpu_events"):
                    queue.append((child, depth + 1))
            continue

        print(
            f"Generic fallback: repeating pattern found under '{current.get('name')}' at depth {depth}"
        )
        print(f"Generic fallback: period={p}")

        # Anchor each iteration between the Nth occurrence of the first and last
        # events in the detected pattern. Using all-children anchors means
        # CPU-only leading/trailing events are included naturally.
        first_anchor_name = children[start]["name"]
        last_anchor_name = children[start + p - 1]["name"]

        first_anchors = [
            i
            for i, c in enumerate(children)
            if i >= start and c.get("name") == first_anchor_name
        ]
        last_anchors = [
            i
            for i, c in enumerate(children)
            if i >= start and c.get("name") == last_anchor_name
        ]

        iteration_roots = []
        for n in range(min(len(first_anchors), len(last_anchors))):
            block_start = first_anchors[n]
            block_end = last_anchors[n]
            if block_end < block_start:
                break
            block = children[block_start : block_end + 1]
            first, last = block[0], block[-1]
            root_event = dict(first)
            root_event["dur"] = (last["ts"] + last.get("dur", 0)) - first["ts"]
            iteration_roots.append(root_event)

        print(f"Generic fallback: identified {len(iteration_roots)} iterations.")
        return iteration_roots if iteration_roots else None

    return None


def find_iteration_roots_generic(events: List[dict]) -> Optional[List[dict]]:
    """Fallback: detect iteration roots by finding a repeating child pattern in
    the call tree, using Trace2Tree for parent/child relationships.

    Works for any workload (diffusion, training, etc.) where the iteration loop
    body is a repeating sequence of top-level calls under a common parent.
    """
    try:
        tree = TraceToTree(events, prune_nongpu_paths=False)
        tree.build_tree(add_python_func=True)
    except Exception as e:
        print(f"Generic fallback: Trace2Tree build failed ({e}), skipping.")
        return None

    # Walk every cpu_root_node upward through python_function parents until
    # reaching a parentless node — these are the true per-thread entry points.
    seen_roots: set = set()
    trace_roots = []
    for uid in tree.cpu_root_nodes:
        e = tree.get_UID2event(uid)
        while True:
            parent = tree.get_parent_event(e)
            if parent is None:
                break
            e = parent
        if id(e) not in seen_roots:
            seen_roots.add(id(e))
            trace_roots.append(e)

    if not trace_roots:
        print("Generic fallback: no root nodes found.")
        return None

    roots = _detect_iteration_roots_from_tree(tree, trace_roots)
    if roots is None:
        print("Generic fallback: no repeating child pattern found.")
    return roots


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

    # SGLang profiler step annotations, e.g. "step[EXTEND bs=1 toks=862]" (prefill)
    # or "step[DECODE bs=25]" (decode). Map to the context/generation request and
    # token counts that the steady-state logic downstream expects.
    sglang_step = re.match(r"step\[(\w+)\s+bs=(\d+)(?:\s+toks=(\d+))?\]", name)
    if sglang_step:
        kind, bs = sglang_step.group(1), int(sglang_step.group(2))
        toks = int(sglang_step.group(3) or 0)
        if kind == "DECODE":
            ctx_req, ctx_sum, gen_req, gen_sum, batch_size = 0, 0, bs, bs, bs
        else:  # EXTEND / MIXED treated as prefill; toks = total prompt tokens.
            ctx_req, ctx_sum, gen_req, gen_sum, batch_size = (
                bs,
                toks,
                0,
                0,
                (toks or bs),
            )
        return {
            "batch_size": batch_size,
            "num_requests": ctx_req + gen_req,
            "context_requests": ctx_req,
            "context_sum": ctx_sum,
            "generation_requests": gen_req,
            "generation_sum": gen_sum,
        }

    if "annotation_iteration" not in prefix:
        # Generic workload (e.g. diffusion): treat every iteration as one
        # decode-equivalent step so steady-state detection sees a flat line.
        return {
            "batch_size": 1,
            "num_requests": 1,
            "context_requests": 0,
            "context_sum": 0,
            "generation_requests": 1,
            "generation_sum": 1,
        }

    # vLLM execute_..._context_..._generation_... annotations: strip parens and the
    # sq/sk shape letters to a flat token list, then pick counts by index.
    try:
        parts = re.sub(r"[sqk]+", "_", name.replace("(", "_").replace(")", "_")).split(
            "_"
        )
        if len(parts) < 10:
            idx = (2, 3, 6, 7)
        elif len(parts) < 12:
            idx = (2, 3, 7, 8)
        else:
            idx = (3, 5, 11, 13)
        ctx_req, ctx_sum, gen_req, gen_sum = (int(parts[i]) for i in idx)
        return {
            "batch_size": ctx_sum + gen_sum,
            "num_requests": ctx_req + gen_req,
            "context_requests": ctx_req,
            "context_sum": ctx_sum,
            "generation_requests": gen_req,
            "generation_sum": gen_sum,
        }
    except (ValueError, IndexError):
        return {
            "batch_size": 1,
            "num_requests": 1,
            "context_requests": 0,
            "context_sum": 0,
            "generation_requests": 1,
            "generation_sum": 1,
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
    output_label: Optional[str] = None,
):
    """Extract and save a range of iterations/dummy runs.

    If ``output_label`` is provided the output filename becomes
    ``{output_label}_{name_append}_{base_name}.json.gz`` instead of the
    default ``{base_name}_{prefix}_{idx}_{name_append}.json.gz``.
    """
    extraction_summary = []
    # print(f"roots: {roots}")
    if len(roots) == 0 or len(roots[0]) == 0:
        print(f"No {prefix} events found in the specified range, skipping extraction")
        return extraction_summary
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
        is_annotation = "annotation_iteration" in prefix
        # Use the structured phase-aware name for any annotation extraction
        # produced by the steady-state code paths (output_label is set), and
        # for any multi-step annotation window. Single-step annotations from
        # --store-single-iteration keep their literal step name.
        is_structured = is_annotation and (output_label is not None or len(root) > 1)

        if is_structured or not is_annotation:
            if len(batch_list) == len(iter_details):
                for bs, iteration in zip(batch_list, iter_details):
                    iteration["batch_size"] = bs

        phase_details = find_phase_from_window(iter_details)

        if is_structured:
            name_append = (
                f"prefill_{phase_details['num_prefill']}"
                f"_prefilldecode_{phase_details['num_prefilldecode']}"
                f"_decode_{phase_details['num_decode']}"
                f"_bs{phase_details['avg_bs']}_conc{phase_details['avg_conc']}"
            )
        elif is_annotation and len(root) == 1:
            root_name = root[0]["name"]
            is_known_annotation = any(
                pat.match(root_name)
                for pat in ANNOTATION_PATTERN + ANNOTATION_PATTERN_BACKUP
            )
            if is_known_annotation:
                name_append = (
                    root_name.replace("/", "_")
                    .replace("(", "_")
                    .replace(")", "")
                    .replace(":", "")
                    .replace(" ", "_")
                )
            else:
                name_append = ""
        else:
            if len(batch_list) == len(iter_details):
                name_append = f"batch{int(sum(batch_list)/len(batch_list))}_gpu{prefix}"
            else:
                name_append = f"batch_NA_gpu{prefix}"

        if output_label is not None:
            out_path = os.path.join(
                output_dir, f"{output_label}_{name_append}_{base_name}.json.gz"
            )
        else:
            suffix = f"_{name_append}" if name_append else ""
            out_path = os.path.join(
                output_dir, f"{base_name}_{prefix}_{idx}{suffix}.json.gz"
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
        if len(decode_steps) > 0:
            iter_details = [
                get_iter_details_from_name(r["name"], prefix) for r in decode_steps
            ]
            phase_details = find_phase_from_window(iter_details)
            iter_trace, batch_list, num_gpu_events, gpu_dur, gpu_busy = (
                extract_iteration(
                    decode_steps,
                    events,
                    trace_json,
                    gpu_corr_map,
                    flow_corr_map,
                    meta_events,
                )
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


def identify_steady_state_regions(
    iter_details: List[dict], num_steps: int
) -> Tuple[List[Tuple[int, int]], int]:
    """Detect contiguous steady-state regions based on num_requests proximity to global max.

    Returns ``(regions, global_max)`` where ``regions`` is a list of
    ``(start, end)`` index pairs and ``global_max`` is the peak concurrency
    observed across all iterations.
    """
    n = len(iter_details)
    thresh = 0.1 if n >= num_steps else 0.2
    global_max = max(t["num_requests"] for t in iter_details)

    steady_state_started = False
    steady_state_ended = False
    prev_events_in_steady = 0
    start_index = 0
    regions = []

    for i, t in enumerate(iter_details):
        if abs(t["num_requests"] - global_max) <= max(1, thresh * global_max):
            if not steady_state_started:
                prev_events_in_steady += 1
        else:
            if steady_state_started:
                prev_events_in_steady -= 1

        if prev_events_in_steady > 5 and not steady_state_started:
            print(f"Steady state started at index {i - 5}")
            steady_state_started = True
            start_index = i - prev_events_in_steady + 1

        if (
            prev_events_in_steady <= 0
            and steady_state_started
            and not steady_state_ended
        ):
            print(f"Steady state ended at index {i}")
            steady_state_ended = True
            regions.append((start_index, i))
            steady_state_started = False
            steady_state_ended = False
            prev_events_in_steady = 0

    if steady_state_started and not steady_state_ended:
        regions.append((start_index, i))

    print(f"Steady state regions: {regions}")

    if len(regions) == 0:
        delta = min(n, max(8, num_steps - n))
        start = max(0, delta // 2)
        end = max(start + 1, min(n, n - delta // 2))
        regions = [(start, end)]
        print(
            "Warning: no steady state region found; discarding initial/final iterations "
            "and selecting middle region"
        )

    return regions, global_max


def compute_reference_pd_ratio(
    regions: List[Tuple[int, int]], iter_details: List[dict]
) -> Tuple[Tuple[int, int], float, float]:
    """
    Return the largest steady-state region, a reference PD ratio, and the
    median PD ratio across all regions.

    The reference ratio starts as the PD/total ratio of the largest region.  A
    sanity check compares it to the median ratio across ALL regions.  If the
    largest region deviates by more than 50 % relative from the median, the
    median is used instead and a warning is printed.
    """
    region_stats = []
    total_steps = 0
    total_pd_steps = 0
    for s, e in regions:
        window = iter_details[s:e]
        total = len(window)
        total_steps += total
        pd_count = sum(1 for t in window if t.get("context_requests", 0) > 0)
        total_pd_steps += pd_count
        ratio = pd_count / total if total > 0 else 0.0
        region_stats.append({"start": s, "end": e, "size": total, "pd_ratio": ratio})
        print(
            f"  Region [{s}, {e}): size={total}, "
            f"prefilldecodemix_steps={pd_count}, prefilldecodemix_to_totalsteps_ratio={ratio:.3f}"
        )

    largest = max(region_stats, key=lambda x: x["size"])
    average_ratio = total_pd_steps / total_steps if total_steps > 0 else 0.0
    largest_window_ratio = largest["pd_ratio"]
    print(
        f"Reference prefilldecodemix_to_totalsteps_ratio={largest_window_ratio:.3f} (largest region [{largest['start']}, {largest['end']}), Average across all regions={average_ratio:.3f})"
    )

    return (largest["start"], largest["end"]), average_ratio, largest_window_ratio


def divide_phases_and_save(
    iteration_roots: List[dict],
    events: List[dict],
    trace_json: dict,
    output_dir: str,
    base_name: str,
    gpu_corr_map: dict,
    flow_corr_map: dict,
    meta_events: List[dict],
    steady_state_regions: List[Tuple[int, int]],
) -> List[dict]:
    """
    Group contiguous steps of the same phase within steady-state regions and
    save each contiguous run as a single trace file into one of two sub-folders:

    - ``{output_dir}/prefilldecodemix/`` — runs where every step has ``context_requests > 0``
    - ``{output_dir}/decode_only/``      — runs where every step has ``context_requests == 0``
                                           and ``generation_requests > 0``

    A phase transition (PD → DO or DO → PD) always starts a new file.

    Parameters
    ----------
    steady_state_regions
        Pre-computed steady-state region list as ``(start, end)`` index pairs.
        Pass ``[(0, len(iteration_roots))]`` to treat the entire slice as steady state.
    """
    iter_details = [get_iter_details_from_name(r["name"]) for r in iteration_roots]
    regions = steady_state_regions
    print(f"[divide-phases] Steady-state regions: {regions}")

    # Build an ordered list of (phase_label, root) for all steady-state steps
    steady_steps: List[Tuple[str, dict]] = []
    for s, e in regions:
        for idx in range(s, e):
            detail = iter_details[idx]
            root = iteration_roots[idx]
            if detail.get("context_requests", 0) > 0:
                steady_steps.append(("prefilldecodemix", root))
            elif detail.get("generation_requests", 0) > 0:
                steady_steps.append(("decode_only", root))
            # steps that are neither (e.g. idle) are skipped

    # Group into contiguous runs of the same phase
    runs: List[Tuple[str, List[dict]]] = []  # (phase, [roots])
    for phase, root in steady_steps:
        if runs and runs[-1][0] == phase:
            runs[-1][1].append(root)
        else:
            runs.append((phase, [root]))

    pd_count = sum(1 for p, _ in runs if p == "prefilldecodemix")
    do_count = sum(1 for p, _ in runs if p == "decode_only")
    total_pd_steps = sum(len(r) for p, r in runs if p == "prefilldecodemix")
    total_do_steps = sum(len(r) for p, r in runs if p == "decode_only")
    print(
        f"\n[divide-phases] {pd_count} prefilldecodemix runs ({total_pd_steps} steps) and "
        f"{do_count} decode_only runs ({total_do_steps} steps) across all steady-state regions."
    )

    pd_dir = os.path.join(output_dir, "prefilldecodemix")
    do_dir = os.path.join(output_dir, "decode_only")
    if pd_count:
        os.makedirs(pd_dir, exist_ok=True)
    if do_count:
        os.makedirs(do_dir, exist_ok=True)

    extraction_summary = []
    pd_chunk_idx = 0
    do_chunk_idx = 0

    for phase, chunk_roots in runs:
        if phase == "prefilldecodemix":
            out_dir = pd_dir
            chunk_idx = pd_chunk_idx
            pd_chunk_idx += 1
        else:
            out_dir = do_dir
            chunk_idx = do_chunk_idx
            do_chunk_idx += 1

        phase_details = find_phase_from_window(
            [get_iter_details_from_name(r["name"]) for r in chunk_roots]
        )
        name_append = (
            f"chunk{chunk_idx}_"
            f"steps{len(chunk_roots)}_"
            f"bs{phase_details['avg_bs']}_"
            f"conc{phase_details['avg_conc']}"
        )
        extraction_summary.extend(
            extract_and_save(
                [chunk_roots],
                events,
                trace_json,
                out_dir,
                base_name,
                "annotation_iteration",
                0,
                1,
                gpu_corr_map,
                flow_corr_map,
                meta_events,
                output_label=f"{phase}_{name_append}",
            )
        )

    return extraction_summary


def find_steady_state_window(
    iteration_roots: List[dict],
    num_steps: int,
    steady_state_regions: List[Tuple[int, int]],
    mode: str = "mixed",
    CONC: Optional[int] = None,
    OSL: Optional[float] = None,
    R: Optional[float] = None,
) -> List[dict]:
    """
    Find the best contiguous window of up to ``num_steps`` iterations.

    Parameters
    ----------
    iteration_roots : list of iteration-root events
    num_steps : requested window size
    steady_state_regions : pre-computed steady-state region list as ``(start, end)``
        index pairs.  Pass ``[(0, len(iteration_roots))]`` to treat the entire
        slice as steady state.
    mode : one of ``"mixed"``, ``"decode_only"``, ``"max_prefilldecode"``
    CONC : expected peak concurrency (number of concurrent requests).
        If provided, a warning is printed when the observed peak in the trace
        differs from this value.
    OSL : average output sequence length (decode tokens per request).
        Combined with ``R`` to derive the ideal PD ratio.
    R : OSL window ratio in [0, 1]. The actual OSL per request is sampled from
        ``[R * OSL, OSL]``, giving mean OSL = OSL * (1 + R) / 2.

    When ``CONC``, ``OSL``, and ``R`` are all provided the ideal PD ratio is

        ideal_pd_ratio = (CONC * 2) / (OSL * (1 + R))

    and ``num_steps`` is automatically raised to ``ceil(1 / ideal_pd_ratio)``
    if it is too small to capture the true DO/PD distribution.

    Modes
    -----
    ``"mixed"``
        Pick the sub-window whose pd_ratio is closest to the reference ratio
        (ideal when available, otherwise largest-region / median sanity-checked).
        Ties broken by highest average num_requests.
    ``"decode_only"``
        Fewest-PD window: sub-window with lowest pd_ratio.
    ``"max_prefilldecode"``
        Most-PD window: sub-window with highest pd_ratio.
    """
    iter_details = [get_iter_details_from_name(r["name"]) for r in iteration_roots]
    regions = steady_state_regions
    global_max = max(t["num_requests"] for t in iter_details)

    (largest_start, largest_end), reference_ratio, largest_window_ratio = (
        compute_reference_pd_ratio(regions, iter_details)
    )

    # --- Optional: CONC / OSL / R validation and ideal ratio override ----------
    ideal_pd_ratio: Optional[float] = None

    if CONC is not None and global_max != CONC:
        print(
            f"Warning: expected peak concurrency CONC={CONC} but the trace peak is "
            f"global_max={global_max}. The trace may not contain requests at the "
            f"intended concurrency level."
        )

    if CONC is not None and OSL is not None and R is not None:
        if not (0.0 <= R <= 1.0):
            print(f"Warning: R={R} is outside [0, 1]; clamping to valid range.")
            R = max(0.0, min(1.0, R))
        mean_osl = OSL * (1.0 + R) / 2.0
        ideal_pd_ratio = (CONC * 2.0) / (OSL * (1.0 + R))
        print(
            f"Ideal prefilldecodemix_to_totalsteps_ratio = (CONC={CONC} * 2) / (OSL={OSL} * (1 + R={R})) "
            f"= {ideal_pd_ratio:.4f}  [mean OSL = {mean_osl:.1f}]"
        )

        min_steps_for_ratio = math.ceil(1.0 / ideal_pd_ratio)
        if num_steps < min_steps_for_ratio:
            print(
                f"Warning: --num-steps={num_steps} is too small to capture the true "
                f"decode_only/prefilldecodemix distribution. At prefilldecodemix_to_totalsteps_ratio={ideal_pd_ratio:.4f} you need at "
                f"least {min_steps_for_ratio} steps to see a representative mix. "
                f"Raising num_steps to {min_steps_for_ratio}."
            )
            num_steps = min_steps_for_ratio
        else:
            print(f"num_steps={num_steps} >= min required {min_steps_for_ratio} — OK.")

        # Ideal ratio overrides the empirical reference for the mixed mode
        reference_ratio = ideal_pd_ratio
        print(
            f"Using ideal prefilldecodemix_to_totalsteps_ratio={ideal_pd_ratio:.4f} as reference (overrides empirical {reference_ratio:.4f})"
        )
    print(f"\n --------------------------------")
    # ---------------------------------------------------------------------------

    divider = max(1, min(int(num_steps / 2), 10))
    step = max(1, num_steps // divider)

    # Build candidate sub-windows from the largest region
    candidates = []
    s, e = largest_start, largest_end

    def _count_mixed(window: List[dict]) -> int:
        """Count truly-mixed steps (both context and generation requests > 0)."""
        return sum(
            1
            for t in window
            if t.get("context_requests", 0) > 0 and t.get("generation_requests", 0) > 0
        )

    if (e - s) >= num_steps:
        for s1 in range(s, e - num_steps + 1, step):
            window = iter_details[s1 : s1 + num_steps]
            pd_count = sum(1 for t in window if t.get("context_requests", 0) > 0)
            candidates.append(
                {
                    "start": s1,
                    "end": s1 + num_steps,
                    "pd_count": pd_count,
                    "pd_ratio": pd_count / num_steps,
                    "mixed_count": _count_mixed(window),
                    "avg_requests": mean(t["num_requests"] for t in window),
                }
            )
    else:
        # Region is smaller than num_steps — use the whole region
        window = iter_details[s:e]
        pd_count = sum(1 for t in window if t.get("context_requests", 0) > 0)
        candidates.append(
            {
                "start": s,
                "end": e,
                "pd_count": pd_count,
                "pd_ratio": pd_count / len(window) if window else 0.0,
                "mixed_count": _count_mixed(window),
                "avg_requests": (
                    mean(t["num_requests"] for t in window) if window else 0
                ),
            }
        )

    if mode == "mixed":
        # Prefer candidate windows that contain at least one prefill-bearing
        # step (pure prefill OR truly mixed, i.e. context_requests > 0). Fall
        # back to all candidates only when no window contains any prefill
        # activity at all.
        pd_candidates = [c for c in candidates if c["pd_count"] > 0]
        if pd_candidates:
            print(
                f"[mixed] Filtering to {len(pd_candidates)}/{len(candidates)} "
                f"candidate windows that contain at least one prefill or "
                f"prefill-decode step."
            )
            selection_pool = pd_candidates
        else:
            print(
                "[mixed] No candidate window contains a prefill or prefill-decode "
                "step; falling back to the full candidate set."
            )
            selection_pool = candidates

        best = min(
            selection_pool,
            key=lambda c: (abs(c["pd_ratio"] - reference_ratio), -c["avg_requests"]),
        )
        print(
            f"[mixed] Selected window [{best['start']}, {best['end']}): "
            f"prefilldecodemix_to_totalsteps_ratio={best['pd_ratio']:.3f} (target={reference_ratio:.3f}), "
            f"avg_requests={best['avg_requests']:.1f}, "
            f"pd_count={best['pd_count']}, mixed_count={best['mixed_count']}"
        )

    elif mode == "decode_only":
        # Find the longest contiguous run of pure decode-only steps (active
        # generation with no context requests) in the largest steady-state
        # region, capped at num_steps.
        do_runs: List[Tuple[int, int]] = []  # (start, end) in iter_details coords
        run_start: Optional[int] = None
        for idx in range(largest_start, largest_end):
            step_info = iter_details[idx]
            is_do = (
                step_info.get("generation_requests", 0) > 0
                and step_info.get("context_requests", 0) == 0
            )
            if is_do:
                if run_start is None:
                    run_start = idx
            else:
                if run_start is not None:
                    do_runs.append((run_start, idx))
                    run_start = None
        if run_start is not None:
            do_runs.append((run_start, largest_end))

        if do_runs:
            longest = max(do_runs, key=lambda r: r[1] - r[0])
            run_s, run_e = longest
            win_s = run_s
            win_e = min(run_e, run_s + num_steps)
            print(
                f"[decode_only] Longest pure decode-only run: [{run_s}, {run_e}) "
                f"({run_e - run_s} steps). "
                f"Selected [{win_s}, {win_e}) ({win_e - win_s} steps, "
                f"capped at num_steps={num_steps})."
            )
            return iteration_roots[win_s:win_e]
        else:
            print(
                "[decode_only] No pure decode-only run found in steady-state region; "
            )
            return []

    elif mode == "max_prefilldecode":
        # Find the longest contiguous run of pure PD steps (no decode-only) in
        # the largest steady-state region, capped at num_steps.
        pd_runs: List[Tuple[int, int]] = []  # (start, end) in iter_details coords
        run_start: Optional[int] = None
        for idx in range(largest_start, largest_end):
            is_pd = iter_details[idx].get("context_requests", 0) > 0
            if is_pd:
                if run_start is None:
                    run_start = idx
            else:
                if run_start is not None:
                    pd_runs.append((run_start, idx))
                    run_start = None
        if run_start is not None:
            pd_runs.append((run_start, largest_end))

        if pd_runs:
            # Pick the longest pure-PD run
            longest = max(pd_runs, key=lambda r: r[1] - r[0])
            run_s, run_e = longest
            # Cap to num_steps from the start of the run
            win_s = run_s
            win_e = min(run_e, run_s + num_steps)
            print(
                f"[max_prefilldecode] Longest pure prefilldecodemix run: [{run_s}, {run_e}) "
                f"({run_e - run_s} steps). "
                f"Selected [{win_s}, {win_e}) ({win_e - win_s} steps, "
                f"capped at num_steps={num_steps})."
            )
            return iteration_roots[win_s:win_e]
        else:
            print(
                "[max_prefilldecode] No pure prefilldecodemix run found in steady-state "
            )
            return []

    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. Use 'mixed', 'decode_only', or 'max_prefilldecode'."
        )

    return iteration_roots[best["start"] : best["end"]]


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
        "--CONC",
        type=int,
        default=None,
        help=(
            "Expected peak concurrency (number of concurrent requests). "
            "A warning is printed if the trace peak differs from this value."
        ),
    )
    parser.add_argument(
        "--OSL",
        type=float,
        default=None,
        help=(
            "Maximum output sequence length (decode tokens per request). "
            "Used with --R to compute the ideal PD ratio for mixed-window selection."
        ),
    )
    parser.add_argument(
        "--R",
        type=float,
        default=None,
        help=(
            "OSL window ratio in [0, 1]. OSL per request is sampled from "
            "[R*OSL, OSL], giving mean OSL = OSL*(1+R)/2. "
            "R=0 means all requests have exactly OSL tokens; "
            "R=1 means OSL is uniform in [0, OSL]."
        ),
    )
    parser.add_argument(
        "--divide-phases",
        action="store_true",
        default=False,
        help=(
            "Find all steady-state regions and store each individual step into "
            "phase-specific sub-folders: output_dir/prefilldecodemix/ and "
            "output_dir/decode_only/. Each step is a separate trace file."
        ),
    )
    args = parser.parse_args()
    execution_details = []
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
    iteration_roots = find_iteration_roots(events)
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
            print(f"\nExtracting iterations {start} to {end - 1} individually...")
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

        # Determine the working set and compute steady-state regions once,
        # shared across all downstream calls.
        if args.iterations != "all":
            working_roots = iteration_roots[start:end]
            steady_state_regions: List[Tuple[int, int]] = [(0, end - start)]
            print(
                f"\nUsing explicit iteration range [{start}, {end}) as the working region."
            )
        else:
            working_roots = iteration_roots
            if args.find_steady_state or args.divide_phases:
                _iter_details = [
                    get_iter_details_from_name(r["name"]) for r in working_roots
                ]
                steady_state_regions, _ = identify_steady_state_regions(
                    _iter_details, args.num_steps
                )

        _extract_args = (
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

        if args.divide_phases:
            print("\n--- Dividing steady-state steps by phase ---")
            temp_execution_details = divide_phases_and_save(
                working_roots,
                events,
                trace_json,
                args.output_dir,
                base_name,
                gpu_corr_map,
                flow_corr_map,
                meta_events,
                steady_state_regions=steady_state_regions,
            )
            execution_details.extend(temp_execution_details)

        elif args.find_steady_state:
            # Three separate contiguous windows — no phase-splitting, no idle gaps
            print("\n--- Finding mixed steady-state window ---")
            mixed_roots = find_steady_state_window(
                working_roots,
                num_steps=args.num_steps,
                steady_state_regions=steady_state_regions,
                mode="mixed",
                CONC=args.CONC,
                OSL=args.OSL,
                R=args.R,
            )
            temp_execution_details = extract_and_save(
                [mixed_roots], *_extract_args, output_label="mixed_steady_state"
            )
            execution_details.extend(temp_execution_details)

            print("\n--- Finding decode-only steady-state window ---")
            do_roots = find_steady_state_window(
                working_roots,
                num_steps=args.num_steps,
                steady_state_regions=steady_state_regions,
                mode="decode_only",
            )
            temp_execution_details = extract_and_save(
                [do_roots], *_extract_args, output_label="decode_only_steady_state"
            )
            execution_details.extend(temp_execution_details)

            print("\n--- Finding biggest prefill-decode steady-state window ---")
            pd_roots = find_steady_state_window(
                working_roots,
                num_steps=args.num_steps,
                steady_state_regions=steady_state_regions,
                mode="max_prefilldecode",
            )
            temp_execution_details = extract_and_save(
                [pd_roots], *_extract_args, output_label="prefilldecode_steady_state"
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
