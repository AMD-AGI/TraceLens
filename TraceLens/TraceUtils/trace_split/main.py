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
    python -m TraceLens.TraceUtils.trace_split.main <trace_path> -o <output_dir> [OPTIONS]

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

   $ python -m TraceLens.TraceUtils.trace_split.main trace.json.gz -o ./output --store-single-iteration

   → One trace file per iteration in ./output/

─────────────────────────────────────────────────────────────────────────────

2. EXTRACT SPECIFIC ITERATION RANGE (combined)

   $ python -m TraceLens.TraceUtils.trace_split.main trace.json.gz \\
     -o ./output \\
     --iterations 10:20

   → Single combined trace file containing iterations 10-19

─────────────────────────────────────────────────────────────────────────────

3. FIND AND EXTRACT STEADY STATE REGION (recommended)

   $ python -m TraceLens.TraceUtils.trace_split.main trace.json.gz \\
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

   $ python -m TraceLens.TraceUtils.trace_split.main trace.json.gz \\
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
  "output_path": "./output/trace_iteration_0.json.gz",
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

from ...util import DataLoader
from ..utils.annotation_utils import (
    ITERATION_BACKUP_PATTERNS,  # noqa: F401
    ITERATION_PATTERNS,  # noqa: F401
    IterationAnnotation,
)

SERVING_KINDS = {
    "vllm_detailed",
    "vllm_native",
    "sglang_detailed",
    "sglang_native",
    "atom_detailed",
    "atom_native",
}

# Re-exports for tests and downstream callers.
from . import (  # noqa: F401
    DetectStatus,
    ExtractContext,
    TraceData,
    EventIndex,
    build_cpu_event_index,
    build_root_tiles,
    classify_phases_from_batch_sizes,
    collect_ancestor_events,
    divide_phases_and_save,
    extract_and_save_single_trace,
    extract_and_save_split,
    extract_iteration,
    find_iteration_roots,
    find_steady_state,
    find_steady_state_generic,
    find_steady_state_inference,
    find_steady_state_inference_from_shapes,
    get_filename,
    infer_batch_sizes_from_shapes,
    parse_range,
)
from ...util import GPU_KERNEL_CATEGORIES

MANIFEST_NAME = "split_manifest.json"


def _write_manifest(output_dir: str, manifest: dict) -> None:
    """Record how the split was decided, next to the slices it produced.

    A split is only trustworthy if its quality is written down, so this is
    emitted even when nothing was extracted.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, MANIFEST_NAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote split manifest to {path}")


def _conservation(events: list, per_iteration_details: list | None, args) -> dict:
    """Check that slicing did not lose or duplicate GPU events.

    Only the one-file-per-iteration pass can be checked this way, and only over
    the whole trace: those windows partition the timeline, so every kernel should
    land in exactly one slice. Steady-state windows deliberately re-extract the
    same kernels, so counting them too would compare a total against itself plus
    overlap.

    The tiles span the iterations, not the capture, so a healthy trace still
    leaves kernels unclaimed: warmup launched before the first root and teardown
    after the last one. Those are excluded on purpose, which is why the failure
    this reports is duplication rather than a shortfall -- extracting more than
    exists means some kernel was counted under two iterations, and that is a bug.
    The shortfall is reported as a quantity instead, since only its size is
    interesting.
    """
    kernels_in_trace = sum(1 for e in events if e.get("cat") in GPU_KERNEL_CATEGORIES)
    report = {"n_gpu_events_in_trace": kernels_in_trace}
    partitioned = (
        per_iteration_details is not None
        and args.iterations == "all"
        and not args.no_gap_fill
    )
    if not (partitioned and kernels_in_trace):
        return report

    extracted = sum(entry.get("num_gpu_events", 0) for entry in per_iteration_details)
    report["n_gpu_events_extracted"] = extracted
    report["n_gpu_events_outside_iterations"] = kernels_in_trace - extracted
    report["gpu_events_duplicated"] = extracted > kernels_in_trace
    report["gpu_event_retention"] = round(extracted / kernels_in_trace, 4)
    return report


# Bookend roots are the warmup/wrapup spans, excluded from the analysis window.
_BOOKEND_NAMES = {"warmup", "wrapup"}


def _base_name(trace_path: str) -> str:
    name = os.path.basename(trace_path)
    return name.replace(".pt.trace", "").replace(".json.gz", "").replace(".json", "")


def _load_and_detect(args):
    """Load the trace, detect iteration roots, and enforce the splittability gate.

    Returns ``(detection, trace_json, trace_index)``, or ``None`` when there is
    nothing to split: no GPU work, a NOT_SPLITTABLE result, or a DEGRADED result
    without ``--allow-degraded``.
    """
    trace_json = DataLoader.load_data(get_filename(args.trace_path))
    events = trace_json.get("traceEvents", [])
    trace_index = EventIndex(events)
    print(f"Loaded {len(events)} events")

    if sum(k.get("dur", 0) for k in trace_index.kernels) == 0:
        print("No GPU work in trace; nothing to split.")
        return None

    detection = find_iteration_roots(events, trace_index=trace_index)
    print(
        f"\nDetection: {detection.method} -> {len(detection.roots)} roots, "
        f"status={detection.status.name}, phase_confidence="
        f"{detection.phase_confidence.value}"
    )
    if detection.coverage:
        print(
            f"GPU coverage ({detection.coverage.strategy}): "
            f"{detection.coverage.covered_selected:.1%} by the selected roots, "
            f"{detection.coverage.span_share:.1%} of that inside their spans"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # SPLITTABLE always extracts; DEGRADED extracts only with --allow-degraded;
    # NOT_SPLITTABLE never extracts (the flag does not override it).
    splittable = detection.status is DetectStatus.SPLITTABLE or (
        detection.status is DetectStatus.DEGRADED and args.allow_degraded
    )
    if not splittable:
        manifest = detection.to_manifest()
        manifest["aborted"] = True
        _write_manifest(args.output_dir, manifest)
        if detection.status is DetectStatus.DEGRADED:
            print(
                "\nRefusing to split: the detected roots account for only a "
                "degraded share of the GPU's work, so per-iteration slices may be "
                f"misleading. See {MANIFEST_NAME} for the coverage breakdown, or "
                "pass --allow-degraded to split anyway."
            )
        else:
            print(
                "\nRefusing to split: no iteration roots account for enough of the "
                "GPU's work to split on. "
                f"See {MANIFEST_NAME} for the coverage breakdown."
            )
        return None

    return detection, trace_json, trace_index


def _gap_fill_tiles(iteration_roots, manifest, args):
    """Build gap-free extraction windows and record the choice in the manifest."""
    if args.no_gap_fill or not iteration_roots:
        manifest["gap_fill"] = False
        return None
    root_tiles, overlaps = build_root_tiles(iteration_roots)
    manifest["gap_fill"] = True
    manifest["n_overlapping_roots"] = overlaps
    if overlaps:
        print(
            f"Warning: {overlaps} roots overlap their successor and keep their "
            "own span; their windows are not gap-free."
        )
    return root_tiles


def _extract_iterations(detection, ctx, args, start, end):
    """Extract the requested iterations.

    Returns ``(details, per_iteration_details)`` where ``per_iteration_details``
    is populated only on the ``--store-single-iteration`` path (the one the
    kernel-conservation check can audit).
    """
    iteration_roots = detection.roots
    if args.store_single_iteration:
        print(f"\nExtracting iterations {start} to {end - 1} individually...")
        details = extract_and_save_split(
            [[root] for root in iteration_roots],
            ctx,
            "iteration",
            start,
            end,
        )
        return details, details
    if args.iterations != "all":
        details = extract_and_save_single_trace(
            iteration_roots[start:end],
            ctx,
            start,
            end,
            uid_map=detection.diagnostics.get("_events_by_uid", {}),
        )
        return details, None
    return [], None


def _working_roots(iteration_roots, args, start, end):
    """Roots to analyze for phases/steady-state, excluding warmup/wrapup bookends."""
    if args.iterations != "all":
        print(
            f"\nUsing explicit iteration range [{start}, {end}) as the working region."
        )
        source = iteration_roots[start:end]
    else:
        source = iteration_roots
    return [r for r in source if r.get("name") not in _BOOKEND_NAMES]


def _has_serving_annotations(working_roots) -> bool:
    """True when the roots carry vLLM/SGLang/ATOM serving (concurrency) semantics."""
    if not working_roots:
        return False
    ann = IterationAnnotation(working_roots[0]["name"])
    return ann.kind in SERVING_KINDS


def _derive_batch_sizes(working_roots, has_annotations, ctx, args):
    """Infer per-iteration batch size from shapes for annotation-less LLM traces.

    Returns the batch-size list, or ``None`` when the shape-based path does not
    apply or no cpu_op shapes were found (caller falls back to duration-based).
    """
    if not (args.llm_inference and not has_annotations and working_roots):
        return None
    cpu_idx = build_cpu_event_index(ctx.trace.events)
    batch_sizes = infer_batch_sizes_from_shapes(working_roots, cpu_idx, ctx.root_tiles)
    valid = [b for b in batch_sizes if b is not None]
    if not valid:
        print(
            "\n[llm-inference] No cpu_op shapes found — falling back to "
            "generic duration-based steady state."
        )
        return None
    print(
        f"\n[llm-inference] Inferred batch sizes from shapes for "
        f"{len(valid)}/{len(batch_sizes)} iterations"
        f", median={sorted(valid)[len(valid) // 2]}"
    )
    return batch_sizes


def _steady_state(working_roots, batch_sizes, has_annotations, args, mode):
    """Bind this trace's context to the steady-state dispatcher."""
    return find_steady_state(
        working_roots,
        mode,
        has_annotations=has_annotations,
        num_steps=args.num_steps,
        batch_sizes=batch_sizes,
        max_num_seq=args.max_num_seq,
        CONC=args.CONC,
        OSL=args.OSL,
        R=args.R,
    )


def _divide_phases(working_roots, batch_sizes, has_annotations, ctx, args):
    """``--divide-phases``: split steady-state steps into phase-specific folders."""
    if not has_annotations and batch_sizes is None:
        print(
            "\n--divide-phases requires LLM inference annotations or "
            "--llm-inference flag. Skipping phase division for this trace."
        )
        return []
    if has_annotations:
        print("\n--- Dividing steady-state steps by phase ---")
        phase_labels = None
    else:
        print("\n--- Dividing steady-state steps by phase (from shapes) ---")
        phase_labels = classify_phases_from_batch_sizes(
            batch_sizes, max_num_seq=args.max_num_seq
        )
    _, ss_regions = _steady_state(
        working_roots, batch_sizes, has_annotations, args, "mixed"
    )
    return divide_phases_and_save(
        working_roots,
        ctx,
        ss_regions,
        phase_labels=phase_labels,
    )


def _find_steady_state_windows(working_roots, batch_sizes, has_annotations, ctx, args):
    """``--find-steady-state``: extract steady-state window(s)."""
    if not (has_annotations or args.llm_inference):
        # Generic path: single duration-based window.
        print("\n--- Finding steady-state window by duration ---")
        ss_roots, _ = _steady_state(
            working_roots, batch_sizes, has_annotations, args, "mixed"
        )
        return extract_and_save_split(
            [ss_roots],
            ctx,
            "iteration",
            0,
            1,
            output_label="steady_state",
        )
    # Inference path: three windows (annotation-based or shape-based).
    details = []
    windows = (
        ("mixed", "mixed_steady_state", "mixed"),
        ("decode_only", "decode_only_steady_state", "decode-only"),
        ("max_prefilldecode", "prefilldecode_steady_state", "biggest prefill-decode"),
    )
    for mode, label, desc in windows:
        print(f"\n--- Finding {desc} steady-state window ---")
        roots, _ = _steady_state(
            working_roots, batch_sizes, has_annotations, args, mode
        )
        details.extend(
            extract_and_save_split(
                [roots],
                ctx,
                "iteration",
                0,
                1,
                output_label=label,
                llm_inference=True,
            )
        )
    return details


def _write_execution_details(execution_details, output_dir):
    """Write the per-extraction summary as JSON and a flattened CSV."""
    json_path = os.path.join(output_dir, "execution_details.json")
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
    csv_path = os.path.join(output_dir, "execution_details.csv")
    df.to_csv(csv_path, index=False, float_format="%.2f")
    print(f"Wrote execution details CSV to {csv_path}")


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
        "--no-gap-fill",
        action="store_true",
        default=False,
        help=(
            "Score each iteration by its own annotation span instead of extending "
            "it to the next root. Work between two roots is then dropped, as it "
            "was before gap-free extraction; use this only to reproduce old output."
        ),
    )
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        default=False,
        help=(
            "Split even when the result is DEGRADED (GPU coverage below the "
            "splittable gate but above the floor). Has no effect on NOT_SPLITTABLE "
            "traces, which are never split. The manifest records the coverage "
            "either way."
        ),
    )
    parser.add_argument(
        "--llm-inference",
        action="store_true",
        default=False,
        help=(
            "Treat as LLM inference trace. When annotations lack serving "
            "semantics (e.g. no vLLM/SGLang/ATOM annotations), derive batch "
            "size from cpu_op shapes and use it for phase classification "
            "(prefill vs decode) and steady-state identification."
        ),
    )
    parser.add_argument(
        "--max-num-seq",
        type=int,
        default=None,
        help=(
            "Maximum number of concurrent sequences (decode batch size cap). "
            "Iterations with batch size above this value are classified as "
            "prefill-bearing. When not set, a heuristic is used."
        ),
    )
    args = parser.parse_args()

    loaded = _load_and_detect(args)
    if loaded is None:
        return
    detection, trace_json, trace_index = loaded
    events = trace_json.get("traceEvents", [])
    iteration_roots = detection.roots
    manifest = detection.to_manifest()

    root_tiles = _gap_fill_tiles(iteration_roots, manifest, args)
    ctx = ExtractContext(
        trace=TraceData(
            events=events,
            trace_json=trace_json,
            gpu_corr_map=trace_index.gpu_corr_map,
            flow_corr_map=trace_index.flow_corr_map,
            meta_events=trace_index.meta_events,
        ),
        output_dir=args.output_dir,
        base_name=_base_name(args.trace_path),
        root_tiles=root_tiles,
    )

    execution_details = []
    # Only the --store-single-iteration pass can be audited for kernel conservation.
    per_iteration_details: list | None = None
    if iteration_roots:
        start, end = parse_range(args.iterations, len(iteration_roots))
        extracted, per_iteration_details = _extract_iterations(
            detection, ctx, args, start, end
        )
        execution_details.extend(extracted)

        working_roots = _working_roots(iteration_roots, args, start, end)
        has_annotations = _has_serving_annotations(working_roots)
        batch_sizes = _derive_batch_sizes(working_roots, has_annotations, ctx, args)

        if args.divide_phases:
            execution_details.extend(
                _divide_phases(working_roots, batch_sizes, has_annotations, ctx, args)
            )
        if args.find_steady_state:
            execution_details.extend(
                _find_steady_state_windows(
                    working_roots, batch_sizes, has_annotations, ctx, args
                )
            )

    print(f"\nDone! Extracted {len(execution_details)} traces to {args.output_dir}")
    manifest.update(_conservation(events, per_iteration_details, args))
    _write_manifest(args.output_dir, manifest)
    if execution_details:
        _write_execution_details(execution_details, args.output_dir)


if __name__ == "__main__":
    main()
