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

    --store-single-iteration  Store each iteration as an individual file

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
import json
import os

import pandas as pd

from ..util import DataLoader
from .annotation_utils import (
    ITERATION_BACKUP_PATTERNS,  # noqa: F401
    ITERATION_PATTERNS,  # noqa: F401
    iteration_details,
)

# Re-exports for tests and downstream callers.
from .split_inference import (  # noqa: F401
    compute_reference_pd_ratio,
    divide_phases_and_save,
    extract_and_save,
    extract_iteration,
    extract_phases_and_save,
    find_iteration_roots,
    find_steady_state_window,
    get_filename,
    identify_steady_state_regions,
    parse_range,
    preprocess_trace,
)


def main():
    parser = argparse.ArgumentParser(
        description="Split vLLM trace into per-iteration traces"
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
    parser.add_argument(
        "--emit-gpu-op-uid",
        action="store_true",
        default=False,
        help=(
            "Tag each event with a 'gpu_op_uid' field equal to its index in "
            "the original (unfiltered) traceEvents array before splitting, "
            "so downstream consumers can recover each extracted event's "
            "position in the source trace without re-loading it. Off by "
            "default to keep existing output byte-for-byte unchanged."
        ),
    )
    args = parser.parse_args()
    execution_details = []

    # Load trace
    trace_json = DataLoader.load_data(get_filename(args.trace_path))
    events = trace_json.get("traceEvents", [])
    if args.emit_gpu_op_uid:
        for i, e in enumerate(events):
            if isinstance(e, dict):
                e["gpu_op_uid"] = i
    gpu_corr_map, flow_corr_map, meta_events = preprocess_trace(events)
    print(f"Loaded {len(events)} events")

    iteration_roots = find_iteration_roots(events)

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
        steady_state_regions: list[tuple[int, int]] = []
        if args.iterations != "all":
            working_roots = iteration_roots[start:end]
            steady_state_regions: list[tuple[int, int]] = [(0, end - start)]
            print(
                f"\nUsing explicit iteration range [{start}, {end}) as the working region."
            )
        else:
            working_roots = iteration_roots
            if args.find_steady_state or args.divide_phases:
                _iter_details = iteration_details(working_roots)
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

    print(f"\nDone! Extracted {len(execution_details)} traces to {args.output_dir}")
    if len(execution_details) > 0:
        json_path = os.path.join(args.output_dir, "execution_details.json")
        with open(json_path, "w") as f:
            json.dump(execution_details, f, indent=2)
        print(f"Wrote execution details JSON to {json_path}")

        rows = []
        for entry in execution_details:
            row = {k: v for k, v in entry.items() if k not in ("steps", "phase")}
            if entry.get("phase"):
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
