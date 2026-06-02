###############################################################################
# Genesis extension for TraceLens — physics-sim workload analysis
###############################################################################

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Taichi / Genesis physics kernel name patterns (not ML GEMM/conv categories)
GENESIS_CATEGORIES: Dict[str, List[str]] = {
    "Rigid Body Solver": [
        "_kernel_solve_body", "_kernel_solve_one_iter", "func_solve_body",
        "func_solve_init", "_kernel_linesearch",
        "_kernel_solve_iter_post_linesearch", "_kernel_update_constraint",
        "_kernel_update_gradient", "_kernel_cg_save", "_kernel_update_search_direction",
        "kernel_compute_mass_matrix",
    ],
    "Broadphase Collision": ["func_broad_phase"],
    "Narrowphase Collision": [
        "_func_narrowphase", "_func_prepare_gjk", "_func_reset_narrowphase",
    ],
    "Contact Management": ["func_sort_contacts", "func_update_contact"],
    "Time Integration": ["kernel_step_1", "kernel_step_2", "func_update_qacc"],
    "Constraints": ["add_equality_constraints", "add_inequality_constraints"],
    "Forward Kinematics": ["kernel_forward_kinematics", "kernel_update_verts"],
    "Geometry / AABB": ["kernel_update_geom", "kernel_bit_reduction"],
    "Memory Ops (ROCm)": [
        "__amd_rocclr_copyBuffer", "__amd_rocclr_fillBuffer", "__amd_rocclr_initHeap",
    ],
    "Runtime Init": [
        "runtime_initialize", "runtime_get_memory", "runtime_allocate", "fill_ndarray",
        "kernel_init_geom", "kernel_init_vvert", "ext_arr_to_ndarray",
    ],
    "PyTorch Runtime": ["at::native::", "elementwise_kernel"],
}


def categorize_kernel(name: str) -> str:
    for cat, patterns in GENESIS_CATEGORIES.items():
        if any(p in name for p in patterns):
            return cat
    return "Other"


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _gpu_timeline_from_intervals(
    window_start_ns: int,
    window_end_ns: int,
    intervals_ns: List[Tuple[int, int]],
) -> pd.DataFrame:
    total_ns = max(window_end_ns - window_start_ns, 1)
    merged = _merge_intervals(intervals_ns)
    busy_ns = sum(e - s for s, e in merged)
    kernel_ns = busy_ns  # no separate memory intervals in this helper
    idle_ns = max(0, total_ns - busy_ns)
    rows = [
        ("total_time", total_ns / 1e6, 100.0),
        ("kernel", kernel_ns / 1e6, 100.0 * kernel_ns / total_ns),
        ("memory", 0.0, 0.0),
        ("busy_time", busy_ns / 1e6, 100.0 * busy_ns / total_ns),
        ("idle", idle_ns / 1e6, 100.0 * idle_ns / total_ns),
    ]
    return pd.DataFrame(rows, columns=["type", "time ms", "percent"])


def detect_steady_state_cutoff_ns(
    starts_ns: np.ndarray,
    ends_ns: np.ndarray,
    gap_threshold_ns: int = 1_000_000_000,
    fallback_window_ns: int = 5_000_000_000,
) -> Tuple[int, str]:
    """
    Detect steady-state region for physics sim workloads (JIT/build then burst).

    Uses the largest inter-kernel gap as phase boundary when it exceeds threshold;
    otherwise uses the last ``fallback_window_ns`` of the trace.
    """
    order = np.argsort(starts_ns)
    starts = starts_ns[order]
    ends = ends_ns[order]
    if len(starts) < 2:
        return int(starts[0]) if len(starts) else 0, "single_dispatch"

    gaps = starts[1:] - ends[:-1]
    max_idx = int(gaps.argmax())
    if gaps[max_idx] >= gap_threshold_ns:
        return int(starts[max_idx + 1]), f"after_max_gap_{gaps[max_idx]/1e6:.1f}ms"

    cutoff = int(ends[-1] - fallback_window_ns)
    return cutoff, f"last_{fallback_window_ns/1e9:.1f}s"


def _utilization_from_cutoff(
    starts_ns: np.ndarray,
    ends_ns: np.ndarray,
    cutoff_ns: int,
) -> Tuple[float, int, int, int]:
    mask = starts_ns >= cutoff_ns
    if not mask.any():
        return 0.0, 0, 0, 0
    ss_starts = starts_ns[mask]
    ss_ends = ends_ns[mask]
    window_start = int(ss_starts.min())
    window_end = int(ss_ends.max())
    intervals = list(zip(ss_starts.tolist(), ss_ends.tolist()))
    tl = _gpu_timeline_from_intervals(window_start, window_end, intervals)
    busy_pct = float(tl.loc[tl["type"] == "busy_time", "percent"].iloc[0])
    return busy_pct, window_start, window_end, int(mask.sum())


def compute_steady_state_timeline(
    kernel_trace_csv: str,
    gap_threshold_ns: int = 1_000_000_000,
    fallback_window_ns: int = 5_000_000_000,
) -> Tuple[pd.DataFrame, dict]:
    df = pd.read_csv(kernel_trace_csv)
    starts = df["Start_Timestamp"].values.astype(np.int64)
    ends = df["End_Timestamp"].values.astype(np.int64)

    cutoff_gap, gap_method = detect_steady_state_cutoff_ns(
        starts, ends, gap_threshold_ns, fallback_window_ns
    )
    cutoff_last = int(ends.max() - fallback_window_ns)

    candidates = [
        (gap_method, cutoff_gap),
        (f"last_{fallback_window_ns/1e9:.1f}s", cutoff_last),
    ]
    best_method, best_cutoff, best_pct = gap_method, cutoff_gap, -1.0
    for method, cutoff in candidates:
        pct, _, _, _ = _utilization_from_cutoff(starts, ends, cutoff)
        if pct > best_pct:
            best_pct, best_method, best_cutoff = pct, method, cutoff

    mask = starts >= best_cutoff
    ss_starts = starts[mask]
    ss_ends = ends[mask]
    window_start = int(ss_starts.min())
    window_end = int(ss_ends.max())
    timeline = _gpu_timeline_from_intervals(
        window_start, window_end, list(zip(ss_starts.tolist(), ss_ends.tolist()))
    )

    meta = {
        "method": best_method,
        "cutoff_ns": best_cutoff,
        "window_ms": (window_end - window_start) / 1e6,
        "dispatch_count": int(mask.sum()),
        "gpu_util_pct": best_pct,
        "gap_threshold_ms": gap_threshold_ns / 1e6,
    }
    return timeline, meta


def compute_genesis_category_summary(kernel_stats_csv: str) -> pd.DataFrame:
    stats = pd.read_csv(kernel_stats_csv)
    stats["Genesis_Category"] = stats["Name"].apply(categorize_kernel)
    summary = (
        stats.groupby("Genesis_Category")
        .agg(
            Kernels=("Name", "nunique"),
            Dispatches=("Calls", "sum"),
            Total_ms=("TotalDurationNs", lambda x: x.sum() / 1e6),
        )
        .sort_values("Total_ms", ascending=False)
        .reset_index()
    )
    total = summary["Total_ms"].sum()
    summary["Pct"] = (summary["Total_ms"] / total * 100).round(2) if total > 0 else 0
    return summary


def rebuild_kernel_summary_by_category(kernel_summary: pd.DataFrame) -> pd.DataFrame:
    """Rebuild rocprof kernel_summary_by_category using Genesis physics categories."""
    if kernel_summary is None or kernel_summary.empty:
        return pd.DataFrame(
            columns=[
                "op category",
                "Count",
                "total_direct_kernel_time_ms",
                "Percentage (%)",
                "Cumulative Percentage (%)",
            ]
        )
    df = kernel_summary.copy()
    if "Category" not in df.columns:
        df["Category"] = df["name"].apply(categorize_kernel)
    grouped = (
        df.groupby("Category", as_index=False)
        .agg(Count=("Count", "sum"), total_direct_kernel_time_ms=("Total Kernel Time (ms)", "sum"))
        .rename(columns={"Category": "op category"})
    )
    total = grouped["total_direct_kernel_time_ms"].sum()
    grouped["Percentage (%)"] = (
        grouped["total_direct_kernel_time_ms"] / total * 100.0 if total > 0 else 0.0
    )
    grouped["Cumulative Percentage (%)"] = grouped["Percentage (%)"].cumsum()
    return grouped.sort_values("total_direct_kernel_time_ms", ascending=False).reset_index(drop=True)


def apply_genesis_categories_to_rocprof(rocprof_reports: Dict[str, pd.DataFrame]) -> None:
    """Replace TraceLens ML categories in rocprof sheets with Genesis physics categories."""
    kernel_summary = rocprof_reports.get("kernel_summary")
    if kernel_summary is None or kernel_summary.empty:
        return
    kernel_summary = kernel_summary.copy()
    kernel_summary["Category"] = kernel_summary["name"].apply(categorize_kernel)
    rocprof_reports["kernel_summary"] = kernel_summary
    rocprof_reports["kernel_summary_by_category"] = rebuild_kernel_summary_by_category(
        kernel_summary
    )


def fix_rocprof_kernel_summary_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    TraceLens rocprof kernel_summary mis-labels ns as µs/ms when timestamps stay in ns.
    Scale Total Kernel Time (ms) and Mean Kernel Time (µs) for display.
    """
    out = df.copy()
    if "Total Kernel Time (ms)" in out.columns:
        out["Total Kernel Time (ms)"] = out["Total Kernel Time (ms)"] / 1000.0
    if "Mean Kernel Time (µs)" in out.columns:
        out["Mean Kernel Time (µs)"] = out["Mean Kernel Time (µs)"] / 1000.0
    return out
