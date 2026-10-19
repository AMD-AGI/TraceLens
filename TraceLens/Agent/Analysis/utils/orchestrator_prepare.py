#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
TraceLens Agent - Orchestrator Preparation Script
Steps 3-6: GPU Utilization, Top Ops, Tree Data Pre-computation, Category Filtering
"""

import argparse
import json
import os
import sys
import traceback

import pandas as pd

from TraceLens.Agent.Analysis.category_analyses.analysis_utils import load_perf_rows
from TraceLens.Agent.Analysis.utils.arch_utils import list_platforms, load_arch
from TraceLens.Agent.Analysis.utils.classify_kernels import TAXONOMY
from TraceLens.Agent.Analysis.utils.kernel_fusion import (
    FusionCandidateBuilder,
    FusionForms,
    FusionLabeler,
)
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    merge_capture_trace_into_graph,
)
from TraceLens.TreePerf import TreePerfAnalyzer
from TraceLens.TreePerf.gpu_event_analyser import GPUEventAnalyser

CATEGORY_SKILL_MAP = {
    "cpu_idle": "cpu-idle-analyzer",
    "gemm": "gemm-analyzer",
    "groupedgemm_fwd": "gemm-analyzer",
    "groupedgemm_bwd": "gemm-analyzer",
    "moe_fused": "moe-analyzer",
    "moe_unfused": "moe-analyzer",
    "sdpa_fwd": "sdpa-analyzer",
    "sdpa_bwd": "sdpa-analyzer",
    "inferenceattention": "sdpa-analyzer",
    "elementwise": "elementwise-analyzer",
    "reduce": "reduce-analyzer",
    "triton": "triton-analyzer",
    "norm": "norm-analyzer",
    "norm_fwd": "norm-analyzer",
    "norm_bwd": "norm-analyzer",
    "rmsnorm": "norm-analyzer",
    "convolution": "convolution-analyzer",
    "conv_fwd": "convolution-analyzer",
    "conv_bwd": "convolution-analyzer",
}


def _normalize_category(row):
    """Normalize upstream ``op category`` to a filesystem-safe key."""
    category = row.get("op category", "")
    if pd.isna(category) or category == "":
        return "other", "Other"
    category_name = category.replace(" ", "_").replace("/", "_").lower()
    return category_name, category


def _build_trace2_ops_summary_by_enhanced_category(trace2_csv_dir: str) -> list:
    """Build trace 2 ops summary grouped by enhanced category (op-name-based).

    Uses unified_perf_summary.csv and get_enhanced_category() so that trace 2
    categories match the trace 1 enhanced categories (e.g. CONV_fwd + CONV_bwd
    merge into 'convolution', NORM_fwd + NORM_bwd merge into 'norm').
    Falls back to ops_summary_by_category.csv if unified_perf_summary.csv is
    not available.
    """
    unified_path = os.path.join(trace2_csv_dir, "unified_perf_summary.csv")
    if not os.path.isfile(unified_path):
        return json.loads(
            pd.read_csv(
                os.path.join(trace2_csv_dir, "ops_summary_by_category.csv")
            ).to_json(orient="records")
        )

    df = pd.read_csv(unified_path)
    df["enhanced_category"], df["display_name"] = zip(
        *df.apply(_normalize_category, axis=1)
    )

    if "Kernel Time (µs)_sum" in df.columns:
        time_col = "Kernel Time (µs)_sum"
        time_scale = 1 / 1000  # µs → ms
    elif "total_direct_kernel_time_ms" in df.columns:
        time_col = "total_direct_kernel_time_ms"
        time_scale = 1.0
    else:
        return json.loads(
            pd.read_csv(
                os.path.join(trace2_csv_dir, "ops_summary_by_category.csv")
            ).to_json(orient="records")
        )

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
    grouped["total_direct_kernel_time_ms"] *= time_scale
    total_time = grouped["total_direct_kernel_time_ms"].sum()
    grouped["Percentage (%)"] = (
        (grouped["total_direct_kernel_time_ms"] / total_time * 100)
        if total_time > 0
        else 0
    )
    grouped = grouped.sort_values("total_direct_kernel_time_ms", ascending=False)
    grouped["Cumulative Percentage (%)"] = grouped["Percentage (%)"].cumsum()
    grouped = grouped.rename(columns={"enhanced_category": "op category"})

    return json.loads(grouped.to_json(orient="records"))


def _gpu_utilization_metrics_from_gpu_timeline_df(gpu_timeline: pd.DataFrame) -> dict:
    """Build the same gpu_utilization dict used in category_manifest from gpu_timeline.csv rows."""
    gpu_data = {}
    for _, row in gpu_timeline.iterrows():
        gpu_data[row["type"]] = {"time_ms": row["time ms"], "percent": row["percent"]}
    return {
        "total_time_ms": gpu_data.get("total_time", {}).get("time_ms", 0),
        "computation_time_percent": gpu_data.get("computation_time", {}).get(
            "percent", 0
        ),
        "exposed_comm_time_percent": gpu_data.get("exposed_comm_time", {}).get(
            "percent", 0
        ),
        "exposed_memcpy_time_percent": gpu_data.get("exposed_memcpy_time", {}).get(
            "percent", 0
        ),
        "idle_time_percent": gpu_data.get("idle_time", {}).get("percent", 0),
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
    if comparison_scope == "comparative":
        trace1_csv_dir = os.path.join(output_dir, "perf_report_trace1_csvs")
        trace2_csv_dir = os.path.join(output_dir, "perf_report_trace2_csvs")
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
        f"{output_dir}/metadata",
        f"{output_dir}/category_data",
        f"{output_dir}/category_findings",
        f"{output_dir}/system_findings",
    ]:
        os.makedirs(d, exist_ok=True)

    platform_specs = load_arch(platform)

    # ============================================================================
    # STEP 3: Assess GPU Utilization
    # ============================================================================
    print("\n[STEP 3] Assessing GPU Utilization...")

    gpu_timeline = pd.read_csv(os.path.join(trace1_csv_dir, "gpu_timeline.csv"))
    gpu_utilization_metrics = _gpu_utilization_metrics_from_gpu_timeline_df(
        gpu_timeline
    )
    trace2_gpu_utilization = None
    trace2_ops_summary_by_category = None

    if comparison_scope == "comparative" and trace2_csv_dir is not None:
        trace2_gpu_utilization = _gpu_utilization_metrics_from_gpu_timeline_df(
            pd.read_csv(os.path.join(trace2_csv_dir, "gpu_timeline.csv"))
        )
        with open(
            os.path.join(output_dir, "metadata", "trace2_gpu_utilization.json"), "w"
        ) as f:
            json.dump(trace2_gpu_utilization, f, indent=2)
        trace2_ops_summary_by_category = _build_trace2_ops_summary_by_enhanced_category(
            trace2_csv_dir
        )

    print(f"\nGPU Utilization Metrics:")
    print(f"  Total Time: {gpu_utilization_metrics['total_time_ms']:.2f} ms")
    print(f"  Computation: {gpu_utilization_metrics['computation_time_percent']:.2f}%")
    print(
        f"  Communication: {gpu_utilization_metrics['exposed_comm_time_percent']:.4f}%"
    )
    print(f"  MemCpy: {gpu_utilization_metrics['exposed_memcpy_time_percent']:.2f}%")
    idle_pct = gpu_utilization_metrics["idle_time_percent"]
    print(f"  Idle: {idle_pct:.2f}%")

    if idle_pct > 15:
        print(
            f"  [DIAG:trace_quality:HIGH_IDLE] GPU idle time {idle_pct:.1f}% exceeds 15% threshold"
        )
    if gpu_utilization_metrics["computation_time_percent"] < 85:
        print(f"  [DIAG:trace_quality:HIGH_IDLE] Compute utilization < 85%")

    # ============================================================================
    # STEP 4: Identify Top Operations
    # ============================================================================
    print("\n[STEP 4] Identifying Top Operations...")

    ops_summary = pd.read_csv(os.path.join(trace1_csv_dir, "ops_summary.csv"))

    # Sort by total_direct_kernel_time_ms if available, else by time column
    if "total_direct_kernel_time_ms" in ops_summary.columns:
        ops_summary_sorted = ops_summary.sort_values(
            "total_direct_kernel_time_ms", ascending=False
        )
        time_col = "total_direct_kernel_time_ms"
    elif "Kernel Time (µs)_sum" in ops_summary.columns:
        ops_summary_sorted = ops_summary.sort_values(
            "Kernel Time (µs)_sum", ascending=False
        )
        time_col = "Kernel Time (µs)_sum"
    else:
        print(f"  ⚠️  Could not determine time column")
        print(f"  Available columns: {ops_summary.columns.tolist()}")
        ops_summary_sorted = ops_summary
        time_col = None

    print(f"\nTop 10 Operations by GPU Time:")
    print("=" * 80)
    if time_col:
        for idx, row in ops_summary_sorted.head(10).iterrows():
            op_name = row.get("name", "Unknown")
            time_val = row[time_col]
            category = row.get("op category", "N/A")
            print(f"  {op_name:50s} | {time_val:10.2f} | {category}")

    # ============================================================================
    # STEP 5: Pre-compute Tree Data (Optimization)
    # ============================================================================
    print("\n[STEP 5] Pre-computing Tree Data for Bottleneck Operations...")

    try:
        print(f"  Loading trace: {trace_path}")
        print(f"  Pseudo ops: {'enabled' if enable_pseudo_ops else 'disabled'}")
        # Build the tree the perf report built: merge the capture folder (Step 1
        # wrote its execution_details.json), and fall back to the raw trace when
        # the merge fails.
        merged_tree = None
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
        print(f"  ✓ Trace loaded successfully")
        print(f"  ✓ Tree has {len(tree.events)} events")

        # Read unified performance summary
        unified_df = pd.read_csv(
            os.path.join(trace1_csv_dir, "unified_perf_summary.csv")
        )

        # Get unique categories
        categories = unified_df["op category"].unique()
        print(f"\n  Found {len(categories)} categories")

        # For each category, pre-compute tree data for bottlenecks
        for category in categories:
            if pd.isna(category) or category == "":
                category_name = "other"
                display_name = "Other"
            else:
                category_name = category.replace(" ", "_").replace("/", "_").lower()
                display_name = category

            # Filter operations for this category
            if pd.isna(category) or category == "":
                category_df = unified_df[
                    unified_df["op category"].isna() | (unified_df["op category"] == "")
                ]
            else:
                category_df = unified_df[unified_df["op category"] == category]

            if len(category_df) == 0:
                continue

            # Identify bottlenecks: ops with >10% of category time
            if "Kernel Time (µs)_sum" in category_df.columns:
                category_total_time = category_df["Kernel Time (µs)_sum"].sum()
                category_df = category_df.copy()
                category_df["category_percent"] = (
                    (category_df["Kernel Time (µs)_sum"] / category_total_time) * 100
                    if category_total_time > 0
                    else 0
                )

                bottleneck_ops = category_df[category_df["category_percent"] > 10]

                # If no ops > 10%, take top 5 by time
                if len(bottleneck_ops) == 0:
                    bottleneck_ops = category_df.nlargest(
                        min(5, len(category_df)), "Kernel Time (µs)_sum"
                    )

        print(f"  ✓ Identified bottleneck operations")

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
            memcpy_by_direction = {"D2H": [], "H2D": [], "D2D": [], "other": []}
            for event in mk_memcpy_events:
                name = event.get("name", "").lower()
                if "dtoh" in name or "device -> host" in name or "devicetohost" in name:
                    memcpy_by_direction["D2H"].append(event)
                elif (
                    "htod" in name or "host -> device" in name or "hosttodevice" in name
                ):
                    memcpy_by_direction["H2D"].append(event)
                elif (
                    "dtod" in name
                    or "device -> device" in name
                    or "devicetodevice" in name
                ):
                    memcpy_by_direction["D2D"].append(event)
                else:
                    memcpy_by_direction["other"].append(event)

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
                    print(f"    ⚠️  Could not compute overlap metrics: {e}")
                    overlap_analysis = {"error": str(e)}

            # Write multi_kernel_data.json (raw statistics only -- pattern
            # detection and recommendations are handled by multi_kernel_analysis.py
            # and the multi-kernel-analyzer sub-agent respectively)
            multi_kernel_data = {
                "memcpy_summary": memcpy_summary,
                "nccl_summary": nccl_summary,
                "overlap_analysis": overlap_analysis,
            }

            multi_kernel_data_file = (
                f"{output_dir}/category_data/multi_kernel_data.json"
            )
            with open(multi_kernel_data_file, "w") as f:
                json.dump(multi_kernel_data, f, indent=2)

            print(
                f"  ✓ Multi-kernel data: {len(mk_memcpy_events)} memcpy events, {len(mk_comm_events)} NCCL events"
            )

        except Exception as e:
            print(f"  ⚠️  Error during multi-kernel data pre-computation: {e}")
            traceback.print_exc()
            # Write empty data so downstream scripts don't fail
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
            multi_kernel_data_file = (
                f"{output_dir}/category_data/multi_kernel_data.json"
            )
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
            load_perf_rows(os.path.join(trace1_csv_dir, "unified_perf_summary.csv")),
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
        print(
            f"  [DIAG:pipeline:STEP3_6_FAIL] Error during tree data pre-computation: {e}"
        )
        traceback.print_exc()

    # ============================================================================
    # STEP 6: Filter and Export Category Data
    # ============================================================================
    print("\n[STEP 6] Filtering and Exporting Category Data...")

    unified_df = pd.read_csv(os.path.join(trace1_csv_dir, "unified_perf_summary.csv"))

    # Apply enhanced categorization
    unified_df["enhanced_category"], unified_df["display_name"] = zip(
        *unified_df.apply(_normalize_category, axis=1)
    )

    categories = unified_df["enhanced_category"].unique()
    exported_categories = []

    for category_name in categories:
        category_df = unified_df[unified_df["enhanced_category"] == category_name]
        display_name = category_df.iloc[0]["display_name"]

        print(f"\n  Category: {display_name} ({category_name})")

        if len(category_df) == 0:
            print(f"    No operations - skipping")
            continue

        # Export filtered CSV
        csv_file = f"{output_dir}/category_data/{category_name}_ops.csv"
        category_df.to_csv(csv_file, index=False)
        print(f"    ✓ Exported CSV: {len(category_df)} ops")

        # Create metadata JSON
        metadata = {
            "platform": platform,
            "peak_mem_bw_tbs": platform_specs["mem_bw_gbps"] / 1000,
            "max_achievable_tflops": platform_specs["max_achievable_tflops"],
            "memory_gb": platform_specs.get("memory_gb"),
            "trace_path": trace_path,
            "output_dir": output_dir,
            "category": display_name,
            "category_name": category_name,
            "gpu_utilization": gpu_utilization_metrics,
            "trace_loading_policy": "DO_NOT_LOAD_TRACE_use_precomputed_tree_data",
        }

        metadata_file = f"{output_dir}/metadata/{category_name}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"    ✓ Exported metadata")

        exported_categories.append(
            {
                "name": category_name,
                "display_name": display_name,
                "skill": CATEGORY_SKILL_MAP.get(category_name, "generic-op-analyzer"),
                "tier": "compute_kernel",
                "ops_count": len(category_df),
                "csv_file": csv_file,
                "metadata_file": metadata_file,
            }
        )

    # ============================================================================
    # CPU/Idle Category Creation (always created)
    # ============================================================================
    print(f"\n  Category: CPU/Idle Analysis (cpu_idle)")
    print(f"    Idle time: {gpu_utilization_metrics['idle_time_percent']:.1f}%")

    cpu_idle_metadata = {
        "platform": platform,
        "peak_mem_bw_tbs": platform_specs["mem_bw_gbps"] / 1000,
        "max_achievable_tflops": platform_specs["max_achievable_tflops"],
        "memory_gb": platform_specs.get("memory_gb"),
        "trace_path": trace_path,
        "output_dir": output_dir,
        "category": "CPU/Idle Analysis",
        "category_name": "cpu_idle",
        "gpu_utilization": gpu_utilization_metrics,
    }

    cpu_idle_metadata_file = f"{output_dir}/metadata/cpu_idle_metadata.json"
    with open(cpu_idle_metadata_file, "w") as f:
        json.dump(cpu_idle_metadata, f, indent=2)

    cpu_idle_csv = f"{output_dir}/category_data/cpu_idle_ops.csv"
    pd.DataFrame().to_csv(cpu_idle_csv, index=False)

    print(f"    ✓ Exported metadata")

    exported_categories.insert(
        0,
        {
            "name": "cpu_idle",
            "display_name": "CPU/Idle Analysis",
            "skill": "cpu-idle-analyzer",
            "tier": "system",
            "ops_count": 0,
            "csv_file": cpu_idle_csv,
            "metadata_file": cpu_idle_metadata_file,
            "priority": 0,
        },
    )

    # ============================================================================
    # Multi-Kernel System-Level Category Creation
    # ============================================================================
    multi_kernel_data_file = f"{output_dir}/category_data/multi_kernel_data.json"
    has_multi_kernel_events = False
    if os.path.exists(multi_kernel_data_file):
        with open(multi_kernel_data_file, "r") as f:
            mk_data = json.load(f)
        has_multi_kernel_events = (
            mk_data.get("memcpy_summary", {}).get("total_count", 0) > 0
            or mk_data.get("nccl_summary", {}).get("total_count", 0) > 0
        )

    if has_multi_kernel_events:
        print(f"\n  Category: Multi-Kernel Issues (multi_kernel)")
        print(f"    ℹ️  Multi-kernel data available (memcpy/NCCL events present)")

        # Create multi-kernel metadata
        multi_kernel_metadata = {
            "platform": platform,
            "peak_mem_bw_tbs": platform_specs["mem_bw_gbps"] / 1000,
            "max_achievable_tflops": platform_specs["max_achievable_tflops"],
            "memory_gb": platform_specs.get("memory_gb"),
            "trace_path": trace_path,
            "output_dir": output_dir,
            "category": "Multi-Kernel Issues",
            "category_name": "multi_kernel",
            "gpu_utilization": gpu_utilization_metrics,
            "tier": "system",
        }

        multi_kernel_metadata_file = f"{output_dir}/metadata/multi_kernel_metadata.json"
        with open(multi_kernel_metadata_file, "w") as f:
            json.dump(multi_kernel_metadata, f, indent=2)

        print(f"    ✓ Exported metadata")

        # Add to categories list (system tier)
        exported_categories.append(
            {
                "name": "multi_kernel",
                "display_name": "Multi-Kernel Issues",
                "skill": "multi-kernel-analyzer",
                "tier": "system",
                "ops_count": 0,
                "csv_file": None,
                "metadata_file": multi_kernel_metadata_file,
                "data_file": multi_kernel_data_file,
            }
        )

    # ============================================================================
    # Kernel Fusion System-Level Category Creation (Experimental)
    # ============================================================================
    fusion_candidates_file = f"{output_dir}/category_data/fusion_candidates.json"
    if os.path.exists(fusion_candidates_file):
        with open(fusion_candidates_file, "r") as f:
            _fc = json.load(f)
        if _fc:
            print(f"\n  Category: Kernel Fusion Opportunities (kernel_fusion)")
            print(f"    ℹ️  {len(_fc)} fusion candidates available")
            exported_categories.append(
                {
                    "name": "kernel_fusion",
                    "display_name": "Kernel Fusion Opportunities",
                    "skill": "kernel-fusion-analyzer",
                    "tier": "system",
                    "ops_count": len(_fc),
                    "csv_file": None,
                    "metadata_file": None,
                    "data_file": fusion_candidates_file,
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
        if "Kernel Time (µs)_sum" in category_df.columns:
            gpu_kernel_time_ms = category_df["Kernel Time (µs)_sum"].sum() / 1000
        elif "total_direct_kernel_time_ms" in category_df.columns:
            gpu_kernel_time_ms = category_df["total_direct_kernel_time_ms"].sum()
        else:
            gpu_kernel_time_ms = 0

        # Calculate CPU duration (ms) - total_duration_us if available
        if "total_duration_us" in category_df.columns:
            cpu_duration_ms = category_df["total_duration_us"].sum() / 1000
        elif "Duration (µs)_sum" in category_df.columns:
            cpu_duration_ms = category_df["Duration (µs)_sum"].sum() / 1000
        else:
            cpu_duration_ms = gpu_kernel_time_ms  # Fallback to kernel time

        # Calculate sync time (ops where CPU duration >> GPU kernel time)
        # Sync bottleneck = CPU duration - GPU kernel time when ratio > 5x
        sync_time_ms = 0
        sync_ops_count = 0

        for _, row in category_df.iterrows():
            if "Kernel Time (µs)_sum" in row and "total_duration_us" in row:
                kernel_us = row.get("Kernel Time (µs)_sum", 0) or 0
                duration_us = row.get("total_duration_us", 0) or 0
                if kernel_us > 0 and duration_us > kernel_us * 5:
                    sync_time_ms += (duration_us - kernel_us) / 1000
                    sync_ops_count += 1

        # Add time metrics to category info
        cat_info["gpu_kernel_time_ms"] = round(gpu_kernel_time_ms, 3)
        cat_info["cpu_duration_ms"] = round(cpu_duration_ms, 3)
        cat_info["sync_time_ms"] = round(sync_time_ms, 3)
        cat_info["sync_ops_count"] = sync_ops_count

        # Flag sync bottleneck if significant
        if sync_time_ms > 0.1 * gpu_kernel_time_ms and sync_time_ms > 1:
            cat_info["has_sync_bottleneck"] = True
            print(
                f"    ⚠️  {category_name}: Sync bottleneck detected ({sync_time_ms:.2f}ms sync time)"
            )
        else:
            cat_info["has_sync_bottleneck"] = False

        # Also update the metadata file with time breakdown
        metadata_file = cat_info.get("metadata_file")
        if metadata_file and os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            metadata["time_breakdown"] = {
                "gpu_kernel_time_ms": cat_info["gpu_kernel_time_ms"],
                "cpu_duration_ms": cat_info["cpu_duration_ms"],
                "sync_time_ms": cat_info["sync_time_ms"],
                "sync_ops_count": cat_info["sync_ops_count"],
                "has_sync_bottleneck": cat_info["has_sync_bottleneck"],
                "note": "Use gpu_kernel_time_ms for bottleneck prioritization. sync_time_ms indicates host-device sync overhead.",
            }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

    print(f"  ✓ Time metrics calculated for all categories")

    # Save category manifest
    manifest = {
        "platform": platform,
        "trace_path": trace_path,
        "output_dir": output_dir,
        "comparison_scope": comparison_scope,
        "gpu_utilization": gpu_utilization_metrics,
        "categories": exported_categories,
        "time_metric_note": "Use gpu_kernel_time_ms for bottleneck prioritization. cpu_duration_ms includes sync/launch overhead.",
    }
    if comparison_scope == "comparative" and trace2_gpu_utilization is not None:
        manifest["platform2"] = args.platform2
        manifest["trace2_gpu_utilization"] = trace2_gpu_utilization
        manifest["trace2_ops_summary_by_category"] = trace2_ops_summary_by_category

    manifest_file = f"{output_dir}/category_data/category_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    model_info_path = os.path.join(output_dir, "metadata", "model_info.json")
    default_info = {
        "model": "Cannot be inferred from trace",
        "architecture": "Cannot be inferred from trace",
        "scale": "Cannot be inferred from trace",
        "precision": "Cannot be inferred from trace",
    }
    if os.path.exists(model_info_path):
        with open(model_info_path) as f:
            existing = json.load(f)
        model_info = {**default_info, **existing}
    else:
        model_info = default_info
    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ Orchestrator Preparation Complete (Steps 3-6)")
    print(f"✓ Exported {len(exported_categories)} categories")
    print(f"✓ Manifest saved: {manifest_file}")
    if comparison_scope == "comparative":
        print(f"✓ metadata/trace2_gpu_utilization.json + manifest trace2_* fields")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
