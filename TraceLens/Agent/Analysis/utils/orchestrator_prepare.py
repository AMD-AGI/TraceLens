#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
TraceLens Agent - Orchestrator Preparation Script
Steps 3-6: GPU Utilization, Top Ops, Multi-Kernel and Fusion Data, Category Filtering
"""

import argparse
import json
import os
import sys
import traceback

import pandas as pd

from TraceLens.Agent.Analysis.category_analyses.analysis_utils import (
    IDLE_PCT_THRESHOLD,
    PEAK_MEM_BW_KEY,
    gpu_utilization_from_timeline,
    load_perf_rows,
)
from TraceLens.Agent.Analysis.utils.arch_utils import list_platforms, load_arch
from TraceLens.Agent.Analysis.utils.classify_kernels import TAXONOMY
from TraceLens.Agent.Analysis.utils.kernel_fusion import (
    FusionCandidateBuilder,
    FusionForms,
    FusionLabeler,
)
from TraceLens.Agent.Analysis.utils.report_utils import MODEL_INFO_FIELDS
from TraceLens.Reporting.aggregate_perf_reports import NODE_COL
from TraceLens.Reporting.generate_perf_report_from_config import (
    AGGREGATED_CSVS_DIR,
    RESOLVED_CONFIG_NAME,
    load_run_config,
    plan_reports,
)
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    merge_capture_trace_into_graph,
)
from TraceLens.TreePerf import TreePerfAnalyzer
from TraceLens.TreePerf.gpu_event_analyser import GPUEventAnalyser

KERNEL_TIME_US_COL = "Kernel Time (µs)_sum"
DIRECT_KERNEL_TIME_MS_COL = "total_direct_kernel_time_ms"
MIN_COMPUTE_UTIL_PCT = 85
# A category op is a sync bottleneck when CPU duration exceeds SYNC_DURATION_RATIO
# times its kernel time. The category is flagged when the summed sync time exceeds
# SYNC_TIME_FRACTION of its GPU kernel time and SYNC_TIME_MIN_MS.
SYNC_DURATION_RATIO = 5
SYNC_TIME_FRACTION = 0.1
SYNC_TIME_MIN_MS = 1
# Lowercase event-name markers per memcpy direction, matched in this order.
MEMCPY_DIRECTIONS = (
    ("D2H", ("dtoh", "device -> host", "devicetohost")),
    ("H2D", ("htod", "host -> device", "hosttodevice")),
    ("D2D", ("dtod", "device -> device", "devicetodevice")),
)
MEMCPY_OTHER = "other"

CATEGORY_SKILL_MAP = {
    "gemm": "gemm-analyzer",
    "groupedgemm_fwd": "gemm-analyzer",
    "groupedgemm_bwd": "gemm-analyzer",
    "moe_fused": "moe-analyzer",
    "moe_unfused": "moe-analyzer",
    "moe_aux": "moe-analyzer",
    "sdpa_fwd": "sdpa-analyzer",
    "sdpa_bwd": "sdpa-analyzer",
    "inferenceattention": "sdpa-analyzer",
    "elementwise": "elementwise-analyzer",
    "groupquant": "elementwise-analyzer",
    "rope_fwd": "elementwise-analyzer",
    "rope_bwd": "elementwise-analyzer",
    "fusedrope": "elementwise-analyzer",
    "multi_tensor_apply": "elementwise-analyzer",
    "reduce": "reduce-analyzer",
    "crossentropy_fwd": "reduce-analyzer",
    "crossentropy_bwd": "reduce-analyzer",
    "triton": "triton-analyzer",
    "norm_fwd": "norm-analyzer",
    "norm_bwd": "norm-analyzer",
    "rmsnorm": "norm-analyzer",
    "conv_fwd": "convolution-analyzer",
    "conv_bwd": "convolution-analyzer",
    "other": "generic-op-analyzer",
    "mhc": "generic-op-analyzer",
    "gaussiansplat": "generic-op-analyzer",
    "ssm_fwd": "generic-op-analyzer",
    "ssm_bwd": "generic-op-analyzer",
    "indexedcopy": "generic-op-analyzer",
    "customcollective": "generic-op-analyzer",
    "record_param_comms": "generic-op-analyzer",
    "moe_comm_fwd": "generic-op-analyzer",
    "moe_comm_bwd": "generic-op-analyzer",
}


def _normalize_category(row):
    """Normalize upstream ``op category`` to a filesystem-safe key."""
    category = row.get("op category", "")
    if pd.isna(category) or category == "":
        return "other", "Other"
    category_name = category.replace(" ", "_").replace("/", "_").lower()
    return category_name, category


def _add_enhanced_category(df: pd.DataFrame) -> None:
    df["enhanced_category"], df["display_name"] = zip(
        *df.apply(_normalize_category, axis=1), strict=True
    )


def _time_col(df: pd.DataFrame) -> str | None:
    for col in (KERNEL_TIME_US_COL, DIRECT_KERNEL_TIME_MS_COL):
        if col in df.columns:
            return col
    return None


def _category_summary_csv_records(csv_dir: str) -> list:
    return json.loads(
        pd.read_csv(os.path.join(csv_dir, "ops_summary_by_category.csv")).to_json(
            orient="records"
        )
    )


def ops_summary_by_enhanced_category(csv_dir: str) -> list:
    """Roll a perf report's unified_perf_summary.csv up by enhanced category.

    Rows are sorted by GPU kernel time, descending. Categories come from
    _normalize_category. Falls back to ops_summary_by_category.csv if
    unified_perf_summary.csv is not available.
    """
    unified_path = os.path.join(csv_dir, "unified_perf_summary.csv")
    if not os.path.isfile(unified_path):
        return _category_summary_csv_records(csv_dir)

    df = pd.read_csv(unified_path)
    _add_enhanced_category(df)

    time_col = _time_col(df)
    if time_col is None:
        return _category_summary_csv_records(csv_dir)
    time_scale = 1 / 1000 if time_col == KERNEL_TIME_US_COL else 1.0  # µs → ms

    count_col = (
        "operation_count" if "operation_count" in df.columns else "enhanced_category"
    )
    grouped = (
        df.groupby("enhanced_category")
        .agg(
            Count=(count_col, "sum" if count_col == "operation_count" else "size"),
            total_direct_kernel_time_ms=(time_col, "sum"),
        )
        .reset_index()
    )
    grouped[DIRECT_KERNEL_TIME_MS_COL] *= time_scale
    total_time = grouped[DIRECT_KERNEL_TIME_MS_COL].sum()
    grouped["Percentage (%)"] = (
        (grouped[DIRECT_KERNEL_TIME_MS_COL] / total_time * 100) if total_time > 0 else 0
    )
    grouped = grouped.sort_values(DIRECT_KERNEL_TIME_MS_COL, ascending=False)
    grouped["Cumulative Percentage (%)"] = grouped["Percentage (%)"].cumsum()
    grouped = grouped.rename(columns={"enhanced_category": "op category"})

    return json.loads(grouped.to_json(orient="records"))


def _kernel_time_ms(df: pd.DataFrame) -> float:
    time_col = _time_col(df)
    if time_col is None:
        return 0
    total = df[time_col].sum()
    return total / 1000 if time_col == KERNEL_TIME_US_COL else total


def perf_report_metadata(csv_dir: str, platform: str) -> dict:
    """Peaks of ``platform`` plus the GPU utilization of the perf report in ``csv_dir``.

    The metadata ``build_operation_metrics`` reads, shared by the agent route's
    manifest `platform_specs` and any caller analyzing a perf report directly.
    """
    arch = load_arch(platform)
    gpu_timeline = pd.read_csv(os.path.join(csv_dir, "gpu_timeline.csv"))
    return {
        PEAK_MEM_BW_KEY: arch["mem_bw_gbps"] / 1000,
        "max_achievable_tflops": arch["max_achievable_tflops"],
        "gpu_utilization": gpu_utilization_from_timeline(gpu_timeline),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare category data for TraceLens analysis"
    )
    parser.add_argument("--trace-path", required=True, help="Path to trace file")
    parser.add_argument(
        "--platform",
        required=True,
        choices=list_platforms(),
        help="AMD platform (MI300X, MI325X, MI350X, MI355X, MI455X)",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--comparison-scope",
        choices=("standalone", "comparative"),
        default="standalone",
        help=(
            "standalone: read <output-dir>/perf_report_csvs (single trace). "
            "comparative: trace 1 from perf_report_trace1_csvs, trace 2 from perf_report_trace2_csvs."
        ),
    )
    parser.add_argument(
        "--platform2",
        default=None,
        help=(
            "Platform of trace 2 (comparative only). Fusion fills trace 2 "
            "Efficiency/Bound only when it equals --platform."
        ),
    )
    parser.add_argument(
        "--capture-folder",
        default=None,
        help=(
            "Graph capture folder of trace 1. Merged into trace 1's tree, as the "
            "perf report does."
        ),
    )
    parser.add_argument(
        "--disable_pseudo_ops",
        action="store_false",
        dest="enable_pseudo_ops",
        default=True,
        help="Disable pseudo-op augmentation (enabled by default).",
    )

    args = parser.parse_args()

    trace_path = args.trace_path
    platform = args.platform
    output_dir = args.output_dir
    enable_pseudo_ops = args.enable_pseudo_ops
    comparison_scope = args.comparison_scope
    # multi_trace is the wrapper config with an aggregation; trace 1 is the combined report.
    resolved_config = os.path.join(output_dir, RESOLVED_CONFIG_NAME)
    if not os.path.isfile(resolved_config):
        print(
            f"[DIAG:pipeline:STEP1_FAIL] Resolved run config missing at {resolved_config}"
        )
        sys.exit(1)
    config = load_run_config(resolved_config)
    plans = plan_reports(config) if "aggregation" in config else None
    if comparison_scope == "comparative":
        trace1_csv_dir = os.path.join(output_dir, "perf_report_trace1_csvs")
        trace2_csv_dir = os.path.join(output_dir, "perf_report_trace2_csvs")
    elif plans is not None:
        trace1_csv_dir = os.path.join(output_dir, AGGREGATED_CSVS_DIR)
        trace2_csv_dir = None
    else:
        trace1_csv_dir = os.path.join(output_dir, "perf_report_csvs")
        trace2_csv_dir = None

    try:
        for csv_dir in filter(None, [trace1_csv_dir, trace2_csv_dir]):
            if not os.path.isdir(csv_dir) or not os.listdir(csv_dir):
                raise FileNotFoundError(csv_dir)
    except FileNotFoundError as e:
        print(f"[DIAG:pipeline:STEP1_FAIL] Perf report CSVs missing at {e}")
        sys.exit(1)

    print("=" * 80)
    print("TRACELENS AGENT - ORCHESTRATOR PREPARATION")
    print("=" * 80)
    print(f"Platform: {platform}")
    print(f"Trace: {trace_path}")
    if args.capture_folder is not None:
        print(f"Capture folder: {args.capture_folder}")
    print(f"Output: {output_dir}")
    print(f"Comparison scope: {comparison_scope}")
    print(f"Trace 1 perf CSVs: {trace1_csv_dir}")
    if trace2_csv_dir is not None:
        print(f"Trace 2 perf CSVs: {trace2_csv_dir}")
    print(f"Pseudo Ops: {'Enabled' if enable_pseudo_ops else 'Disabled'}")
    print("=" * 80)

    # Create directory structure
    for d in [
        output_dir,
        f"{output_dir}/cache",
        f"{output_dir}/category_data",
        f"{output_dir}/category_findings",
        f"{output_dir}/system_findings",
    ]:
        os.makedirs(d, exist_ok=True)

    # ============================================================================
    # STEP 3: Assess GPU Utilization
    # ============================================================================
    print("\n[STEP 3] Assessing GPU Utilization...")

    trace1_metadata = perf_report_metadata(trace1_csv_dir, platform)
    gpu_utilization_metrics = trace1_metadata["gpu_utilization"]
    platform_specs = {
        PEAK_MEM_BW_KEY: trace1_metadata[PEAK_MEM_BW_KEY],
        "max_achievable_tflops": trace1_metadata["max_achievable_tflops"],
        "memory_gb": load_arch(platform)["memory_gb"],
    }
    if plans is not None:
        focus_job = plans[0].node_id
        focus_csv_dir = os.path.join(
            output_dir, os.path.basename(plans[0].output_csvs_dir)
        )
        system_utilization = gpu_utilization_metrics["nodes"][focus_job]
    else:
        focus_csv_dir = trace1_csv_dir
        system_utilization = gpu_utilization_metrics
    trace2_gpu_utilization = None
    trace2_ops_summary_by_category = None

    if comparison_scope == "comparative" and trace2_csv_dir is not None:
        trace2_gpu_utilization = gpu_utilization_from_timeline(
            pd.read_csv(os.path.join(trace2_csv_dir, "gpu_timeline.csv"))
        )
        trace2_ops_summary_by_category = ops_summary_by_enhanced_category(
            trace2_csv_dir
        )

    print("\nGPU Utilization Metrics:")
    print(f"  Total Time: {gpu_utilization_metrics['total_time_ms']:.2f} ms")
    print(f"  Computation: {gpu_utilization_metrics['computation_time_percent']:.2f}%")
    print(
        f"  Communication: {gpu_utilization_metrics['exposed_comm_time_percent']:.4f}%"
    )
    print(f"  MemCpy: {gpu_utilization_metrics['exposed_memcpy_time_percent']:.2f}%")
    idle_pct = gpu_utilization_metrics["idle_time_percent"]
    print(f"  Idle: {idle_pct:.2f}%")

    if idle_pct > IDLE_PCT_THRESHOLD:
        print(
            f"  [DIAG:trace_quality:HIGH_IDLE] GPU idle time {idle_pct:.1f}% exceeds {IDLE_PCT_THRESHOLD}% threshold"
        )
    if gpu_utilization_metrics["computation_time_percent"] < MIN_COMPUTE_UTIL_PCT:
        print(
            f"  [DIAG:trace_quality:HIGH_IDLE] Compute utilization < {MIN_COMPUTE_UTIL_PCT}%"
        )

    # ============================================================================
    # STEP 4: Identify Top Operations
    # ============================================================================
    print("\n[STEP 4] Identifying Top Operations...")

    ops_summary = pd.read_csv(os.path.join(trace1_csv_dir, "ops_summary.csv"))

    # Sort by total_direct_kernel_time_ms if available, else by time column
    if DIRECT_KERNEL_TIME_MS_COL in ops_summary.columns:
        ops_summary_sorted = ops_summary.sort_values(
            DIRECT_KERNEL_TIME_MS_COL, ascending=False
        )
        time_col = DIRECT_KERNEL_TIME_MS_COL
    elif KERNEL_TIME_US_COL in ops_summary.columns:
        ops_summary_sorted = ops_summary.sort_values(
            KERNEL_TIME_US_COL, ascending=False
        )
        time_col = KERNEL_TIME_US_COL
    else:
        print("  ⚠️  Could not determine time column")
        print(f"  Available columns: {ops_summary.columns.tolist()}")
        ops_summary_sorted = ops_summary
        time_col = None

    print("\nTop 10 Operations by GPU Time:")
    print("=" * 80)
    if time_col:
        for _, row in ops_summary_sorted.head(10).iterrows():
            op_name = row.get("name", "Unknown")
            time_val = row[time_col]
            category = row.get("op category", "N/A")
            print(f"  {op_name:50s} | {time_val:10.2f} | {category}")

    # ============================================================================
    # STEP 5: Load trace tree
    # ============================================================================
    print("\n[STEP 5] Loading trace tree...")

    merged_tree = None
    multi_kernel_data = None
    fusion_candidates = None
    try:
        print(f"  Loading trace: {trace_path}")
        print(f"  Pseudo ops: {'enabled' if enable_pseudo_ops else 'disabled'}")
        # Build the tree the perf report built: merge the capture folder (Step 1
        # wrote its execution_details.json), and fall back to the raw trace when
        # the merge fails.
        if args.capture_folder is not None:
            merged_tree = merge_capture_trace_into_graph(
                args.capture_folder,
                os.path.join(args.capture_folder, "execution_details.json"),
                trace_path,
            )
            if merged_tree is None:
                print("  Capture merge failed; using the raw trace")
        if merged_tree is not None:
            # The merge builds the tree, so it is not rebuilt.
            analyzer = TreePerfAnalyzer(
                tree=merged_tree,
                add_python_func=True,
                enable_pseudo_ops=enable_pseudo_ops,
                rebuild_tree=False,
            )
        else:
            analyzer = TreePerfAnalyzer.from_file(
                trace_path,
                add_python_func=True,
                enable_pseudo_ops=enable_pseudo_ops,
            )
        tree = analyzer.tree
        print("  ✓ Trace loaded successfully")
        print(f"  ✓ Tree has {len(tree.events)} events")

        # ====================================================================
        # STEP 5.5: Pre-compute Multi-Kernel Issue Data
        # ====================================================================
        print("\n[STEP 5.5] Pre-computing Multi-Kernel Issue Data...")

        try:
            gpu_analyser = GPUEventAnalyser(tree.events)
            event_lists = gpu_analyser.get_gpu_event_lists()
            mk_gpu_events = event_lists[GPUEventAnalyser.all_gpu_key]
            mk_comm_events = event_lists[GPUEventAnalyser.communication_key]
            mk_memcpy_events = event_lists[GPUEventAnalyser.memcpy_key]

            # Sub-classify memcpy by direction (not provided by GPUEventAnalyser)
            memcpy_by_direction = {d: [] for d, _ in MEMCPY_DIRECTIONS}
            memcpy_by_direction[MEMCPY_OTHER] = []
            for event in mk_memcpy_events:
                name = event.get("name", "").lower()
                direction = next(
                    (
                        d
                        for d, markers in MEMCPY_DIRECTIONS
                        if any(m in name for m in markers)
                    ),
                    MEMCPY_OTHER,
                )
                memcpy_by_direction[direction].append(event)

            # Build memcpy summary
            memcpy_summary = {
                "total_count": len(mk_memcpy_events),
                "total_time_us": 0,
                "by_direction": {},
            }
            for direction, events in memcpy_by_direction.items():
                if not events:
                    continue
                durations = [e.get("dur", 0) for e in events]
                sizes = [e.get("args", {}).get("bytes", 0) for e in events]
                dir_summary = {
                    "count": len(events),
                    "total_time_us": round(sum(durations), 2),
                    "avg_time_us": round(sum(durations) / len(durations), 2),
                    "max_time_us": round(max(durations), 2),
                    "total_bytes": sum(s for s in sizes if s),
                    "avg_bytes": (
                        round(sum(s for s in sizes if s) / len(events), 2)
                        if any(sizes)
                        else 0
                    ),
                }
                memcpy_summary["by_direction"][direction] = dir_summary
                memcpy_summary["total_time_us"] += dir_summary["total_time_us"]
            memcpy_summary["total_time_us"] = round(memcpy_summary["total_time_us"], 2)

            # Build NCCL/communication summary
            nccl_summary = {"total_count": len(mk_comm_events), "total_time_us": 0}
            if mk_comm_events:
                nccl_durations = [e.get("dur", 0) for e in mk_comm_events]
                nccl_summary["total_time_us"] = round(sum(nccl_durations), 2)
                nccl_summary["avg_time_us"] = round(
                    sum(nccl_durations) / len(nccl_durations), 2
                )
                nccl_summary["max_time_us"] = round(max(nccl_durations), 2)

                # Top NCCL ops by duration
                sorted_nccl = sorted(
                    mk_comm_events, key=lambda e: e.get("dur", 0), reverse=True
                )
                nccl_summary["top_ops"] = [
                    {
                        "name": e.get("name", ""),
                        "duration_us": round(e.get("dur", 0), 2),
                        "stream": e.get("args", {}).get("stream", None),
                    }
                    for e in sorted_nccl[:10]
                ]

            # Compute overlap metrics using GPUEventAnalyser
            overlap_analysis = {}
            if mk_gpu_events:
                try:
                    GPUEventAnalyser.verify_dict_gpu_event_lists(event_lists)
                    metrics = GPUEventAnalyser.compute_metrics_dict(event_lists)

                    total_time = metrics.get("total_time", 0)
                    comp_time = metrics.get("computation_time", 0)
                    total_comm_time = metrics.get("total_comm_time", 0)
                    exposed_comm_time = metrics.get("exposed_comm_time", 0)
                    total_memcpy_time = metrics.get("total_memcpy_time", 0)
                    exposed_memcpy_time = metrics.get("exposed_memcpy_time", 0)

                    overlap_analysis = {
                        "total_time_us": round(total_time, 2),
                        "computation_time_us": round(comp_time, 2),
                        "total_comm_time_us": round(total_comm_time, 2),
                        "exposed_comm_time_us": round(exposed_comm_time, 2),
                        "total_memcpy_time_us": round(total_memcpy_time, 2),
                        "exposed_memcpy_time_us": round(exposed_memcpy_time, 2),
                        "comm_overlap_ratio": (
                            round(1 - (exposed_comm_time / total_comm_time), 4)
                            if total_comm_time > 0
                            else None
                        ),
                        "memcpy_overlap_ratio": (
                            round(1 - (exposed_memcpy_time / total_memcpy_time), 4)
                            if total_memcpy_time > 0
                            else None
                        ),
                        "comm_percent_of_total": (
                            round(exposed_comm_time / total_time * 100, 2)
                            if total_time > 0
                            else 0
                        ),
                        "memcpy_percent_of_total": (
                            round(exposed_memcpy_time / total_time * 100, 2)
                            if total_time > 0
                            else 0
                        ),
                    }
                except Exception as e:
                    # Overlap metrics are optional.
                    print(f"    ⚠️  Could not compute overlap metrics: {e}")
                    overlap_analysis = {"error": str(e)}

            # Raw statistics only -- pattern detection and recommendations are
            # handled by multi_kernel_analysis.py and the multi-kernel-analyzer
            # sub-agent respectively
            multi_kernel_data = {
                "memcpy_summary": memcpy_summary,
                "nccl_summary": nccl_summary,
                "overlap_analysis": overlap_analysis,
            }

            print(
                f"  ✓ Multi-kernel data: {len(mk_memcpy_events)} memcpy events, {len(mk_comm_events)} NCCL events"
            )

        except Exception as e:
            print(f"  ⚠️  Error during multi-kernel data pre-computation: {e}")
            traceback.print_exc()
            # Empty data, written below, so downstream scripts do not fail
            multi_kernel_data = {
                "memcpy_summary": {
                    "total_count": 0,
                    "total_time_us": 0,
                    "by_direction": {},
                },
                "nccl_summary": {"total_count": 0, "total_time_us": 0},
                "overlap_analysis": {},
                "error": str(e),
            }

        multi_kernel_data_file = f"{output_dir}/category_data/multi_kernel_data.json"
        with open(multi_kernel_data_file, "w") as f:
            json.dump(multi_kernel_data, f, indent=2)

        # ====================================================================
        # STEP 5b: Extract Kernel Fusion Candidates (Experimental)
        # ====================================================================
        print("\n[STEP 5b] Extracting Kernel Fusion Candidates...")

        fusion_candidates_file = f"{output_dir}/category_data/fusion_candidates.json"

        builder = FusionCandidateBuilder(
            tree,
            analyzer,
            load_perf_rows(os.path.join(focus_csv_dir, "unified_perf_summary.csv")),
            FusionLabeler(TAXONOMY, FusionForms(TAXONOMY)),
        )
        if comparison_scope == "comparative":
            # Comparative mode: use LCA-aligned diff_stats.csv as the source
            # so candidates carry both kernels_trace1 and kernels_trace2 context.
            diff_stats_csv = os.path.join(trace1_csv_dir, "diff_stats.csv")
            if os.path.exists(diff_stats_csv):
                fusion_candidates = builder.comparative(pd.read_csv(diff_stats_csv))
                print(
                    f"  ✓ Comparative fusion: {len(fusion_candidates)} candidates "
                    f"(from diff_stats.csv)"
                )
            else:
                print(
                    "  ⚠️  diff_stats.csv not found; "
                    "comparative fusion extraction skipped"
                )
                fusion_candidates = []
        else:
            fusion_candidates = builder.standalone()

        fusion_candidates.sort(key=lambda c: c.time_us, reverse=True)

        with open(fusion_candidates_file, "w") as f:
            json.dump(
                [c.to_dict() for c in fusion_candidates], f, indent=2, default=str
            )

        print(
            f"  ✓ Written to fusion_candidates.json "
            f"({os.path.getsize(fusion_candidates_file) / 1024:.1f} KB)"
        )

    except Exception as e:
        # Report the failure and let Step 6 export whatever was built.
        print(
            f"  [DIAG:pipeline:STEP3_6_FAIL] Error loading trace tree or extracting multi-kernel/fusion data: {e}"
        )
        traceback.print_exc()

    # ============================================================================
    # STEP 6: Filter and Export Category Data
    # ============================================================================
    print("\n[STEP 6] Filtering and Exporting Category Data...")

    unified_df = pd.read_csv(os.path.join(trace1_csv_dir, "unified_perf_summary.csv"))

    # Apply enhanced categorization
    _add_enhanced_category(unified_df)

    categories = unified_df["enhanced_category"].unique()
    exported_categories = []

    for category_name in categories:
        category_df = unified_df[unified_df["enhanced_category"] == category_name]
        display_name = category_df.iloc[0]["display_name"]

        print(f"\n  Category: {display_name} ({category_name})")

        # Export filtered CSV
        csv_file = f"{output_dir}/category_data/{category_name}_ops.csv"
        category_df.to_csv(csv_file, index=False)
        print(f"    ✓ Exported CSV: {len(category_df)} ops")

        exported_categories.append(
            {
                "name": category_name,
                "display_name": display_name,
                "skill": CATEGORY_SKILL_MAP.get(category_name, "generic-op-analyzer"),
                "tier": "compute_kernel",
                "ops_count": len(category_df),
            }
        )

    # ============================================================================
    # CPU/Idle Category Creation (always created)
    # ============================================================================
    print("\n  Category: CPU/Idle Analysis (cpu_idle)")
    print(f"    Idle time: {system_utilization['idle_time_percent']:.1f}%")

    exported_categories.insert(
        0,
        {
            "name": "cpu_idle",
            "display_name": "CPU/Idle Analysis",
            "skill": "cpu-idle-analyzer",
            "tier": "system",
            "ops_count": 0,
            "priority": 0,
        },
    )

    # ============================================================================
    # Multi-Kernel System-Level Category Creation
    # ============================================================================
    if multi_kernel_data is not None and (
        multi_kernel_data["memcpy_summary"]["total_count"] > 0
        or multi_kernel_data["nccl_summary"]["total_count"] > 0
    ):
        print("\n  Category: Multi-Kernel Issues (multi_kernel)")
        print("    ℹ️  Multi-kernel data available (memcpy/NCCL events present)")

        # Add to categories list (system tier)
        exported_categories.append(
            {
                "name": "multi_kernel",
                "display_name": "Multi-Kernel Issues",
                "skill": "multi-kernel-analyzer",
                "tier": "system",
                "ops_count": 0,
            }
        )

    # ============================================================================
    # Kernel Fusion System-Level Category Creation (Experimental)
    # ============================================================================
    if fusion_candidates:
        print("\n  Category: Kernel Fusion Opportunities (kernel_fusion)")
        print(f"    ℹ️  {len(fusion_candidates)} fusion candidates available")
        exported_categories.append(
            {
                "name": "kernel_fusion",
                "display_name": "Kernel Fusion Opportunities",
                "skill": "kernel-fusion-analyzer",
                "tier": "system",
                "ops_count": len(fusion_candidates),
            }
        )

    # ============================================================================
    # STEP 6.5: Calculate Time Metric Breakdown per Category
    # ============================================================================
    print("\n[STEP 6.5] Calculating Time Metric Breakdown per Category...")

    # Calculate GPU kernel time vs CPU duration per category
    # GPU kernel time = actual GPU execution (use for bottleneck prioritization)
    # CPU duration = total operation time including sync/launch overhead
    # Sync time = operations where CPU duration >> GPU kernel time

    for cat_info in exported_categories:
        category_name = cat_info["name"]
        if category_name in ("cpu_idle", "multi_kernel"):
            continue  # Skip system-level categories - no ops CSV

        category_df = unified_df[unified_df["enhanced_category"] == category_name]

        # Calculate GPU kernel time (ms)
        gpu_kernel_time_ms = _kernel_time_ms(category_df)

        # Calculate CPU duration (ms) - total_duration_us if available
        if "total_duration_us" in category_df.columns:
            cpu_duration_ms = category_df["total_duration_us"].sum() / 1000
        elif "Duration (µs)_sum" in category_df.columns:
            cpu_duration_ms = category_df["Duration (µs)_sum"].sum() / 1000
        else:
            cpu_duration_ms = gpu_kernel_time_ms  # Fallback to kernel time

        # Calculate sync time (ops where CPU duration >> GPU kernel time)
        sync_time_ms = 0
        sync_ops_count = 0

        for _, row in category_df.iterrows():
            if KERNEL_TIME_US_COL in row and "total_duration_us" in row:
                kernel_us = row.get(KERNEL_TIME_US_COL, 0) or 0
                duration_us = row.get("total_duration_us", 0) or 0
                if kernel_us > 0 and duration_us > kernel_us * SYNC_DURATION_RATIO:
                    sync_time_ms += (duration_us - kernel_us) / 1000
                    sync_ops_count += 1

        # Add time metrics to category info
        cat_info["gpu_kernel_time_ms"] = round(gpu_kernel_time_ms, 3)
        cat_info["cpu_duration_ms"] = round(cpu_duration_ms, 3)
        cat_info["sync_time_ms"] = round(sync_time_ms, 3)
        cat_info["sync_ops_count"] = sync_ops_count
        if NODE_COL in category_df.columns:
            cat_info["nodes"] = {
                node: {
                    "gpu_kernel_time_ms": round(_kernel_time_ms(node_df), 3),
                    "ops_count": len(node_df),
                }
                for node, node_df in category_df.groupby(NODE_COL, sort=False)
            }

        # Flag sync bottleneck if significant
        if (
            sync_time_ms > SYNC_TIME_FRACTION * gpu_kernel_time_ms
            and sync_time_ms > SYNC_TIME_MIN_MS
        ):
            cat_info["has_sync_bottleneck"] = True
            print(
                f"    ⚠️  {category_name}: Sync bottleneck detected ({sync_time_ms:.2f}ms sync time)"
            )
        else:
            cat_info["has_sync_bottleneck"] = False

    print("  ✓ Time metrics calculated for all categories")

    # Save category manifest
    manifest = {
        "platform": platform,
        "platform_specs": platform_specs,
        "trace_path": trace_path,
        "comparison_scope": comparison_scope,
        "gpu_utilization": gpu_utilization_metrics,
        "focus_csv_dir": os.path.basename(focus_csv_dir),
        "capture_folder": args.capture_folder if merged_tree is not None else None,
        "categories": exported_categories,
        "time_metric_note": "Use gpu_kernel_time_ms for bottleneck prioritization. cpu_duration_ms includes sync/launch overhead.",
    }
    if comparison_scope == "comparative" and trace2_gpu_utilization is not None:
        manifest["platform2"] = args.platform2
        manifest["trace2_csv_dir"] = os.path.basename(trace2_csv_dir)
        manifest["trace2_gpu_utilization"] = trace2_gpu_utilization
        manifest["trace2_ops_summary_by_category"] = trace2_ops_summary_by_category

    if plans is not None:
        manifest["focus_job"] = focus_job

    manifest_file = f"{output_dir}/category_data/category_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    model_info_path = os.path.join(output_dir, "cache", "model_info.json")
    default_info = dict.fromkeys(MODEL_INFO_FIELDS, "Cannot be inferred from trace")
    if os.path.exists(model_info_path):
        with open(model_info_path) as f:
            existing = json.load(f)
        model_info = {**default_info, **existing}
    else:
        model_info = default_info
    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"\n{'='*80}")
    print("✓ Orchestrator Preparation Complete (Steps 3-6)")
    print(f"✓ Exported {len(exported_categories)} categories")
    print(f"✓ Manifest saved: {manifest_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
