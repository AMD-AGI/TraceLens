###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


# Categorize kernels based on name patterns
def _categorize_kernel(name: str) -> str:
    """
    Categorize kernel by name pattern.

    Args:
        name: Kernel name to categorize

    Returns:
        Category string (GEMM, Elementwise, Reduction, etc.)
    """
    name_lower = name.lower()

    # Use any() with lists for easier pattern maintenance and extensibility
    if any(x in name_lower for x in ["gemm", "matmul", "cijk"]):
        return "GEMM"
    elif any(x in name_lower for x in ["elementwise"]):
        return "Elementwise"
    elif any(x in name_lower for x in ["reduce", "sum"]):
        return "Reduction"
    elif any(x in name_lower for x in ["conv"]):
        return "Convolution"
    elif any(x in name_lower for x in ["norm", "batch_norm", "layer_norm"]):
        return "Normalization"
    elif any(x in name_lower for x in ["flash", "attn", "fmha"]):
        return "Attention"
    elif any(x in name_lower for x in ["copy", "memcpy"]):
        return "Memory"
    elif any(x in name_lower for x in ["nccl", "rccl"]):
        return "COMM"
    else:
        return "Other"


class RocprofAnalyzer:
    """Analyzer for rocprofv3 data - generates performance DataFrames"""

    def __init__(
        self,
        kernel_events: List[dict],
        memory_events: List[dict],
        api_events: List[dict],
        metadata: dict,
    ):
        self.kernel_events = kernel_events
        self.memory_events = memory_events
        self.api_events = api_events
        self.metadata = metadata

    @staticmethod
    def _merge_intervals(intervals: List[tuple]) -> List[tuple]:
        """
        Merge overlapping time intervals to calculate actual busy time.

        Args:
            intervals: List of (start, end) tuples representing time intervals

        Returns:
            List of merged non-overlapping intervals

        Example:
            >>> intervals = [(0, 10), (5, 15), (20, 30)]
            >>> RocprofAnalyzer._merge_intervals(intervals)
            [(0, 15), (20, 30)]
        """
        if not intervals:
            return []

        # Sort intervals by start time
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]

        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            # Check if current interval overlaps with the last merged interval
            if start <= last_end:
                # Merge by extending the end time
                merged[-1] = (last_start, max(last_end, end))
            else:
                # No overlap, add as new interval
                merged.append((start, end))

        return merged

    def _convert_timestamps_to_microseconds(self):
        """Convert all timestamps from nanoseconds to microseconds"""
        for event in self.kernel_events:
            event["ts"] = event["ts"] / 1000.0  # ns to us
            event["dur"] = event["dur"] / 1000.0  # ns to us

        for event in self.memory_events:
            event["ts"] = event["ts"] / 1000.0
            event["dur"] = event["dur"] / 1000.0

        for event in self.api_events:
            event["ts"] = event["ts"] / 1000.0
            event["dur"] = event["dur"] / 1000.0

    def get_df_gpu_timeline(self) -> pd.DataFrame:
        """
        Generate GPU timeline showing:
        - Total time
        - Kernel execution time (accounting for overlaps)
        - Memory operation time (accounting for overlaps)
        - Idle time

        Note: This method calculates GPU busy time by merging overlapping
        kernel and memory operation intervals, rather than simply summing
        their durations. This provides an accurate view of actual GPU
        utilization.
        """
        if not self.kernel_events:
            logger.warning("No kernel events found in rocprof data")
            return pd.DataFrame(columns=["type", "time ms", "percent"])

        # Calculate total time from metadata
        init_time = self.metadata.get("init_time", 0) / 1000.0  # ns to us
        fini_time = self.metadata.get("fini_time", 0) / 1000.0  # ns to us
        total_time_us = fini_time - init_time

        # Create intervals for kernels: (start_time, end_time) in microseconds
        kernel_intervals = [
            (event["ts"] / 1000.0, (event["ts"] + event["dur"]) / 1000.0)
            for event in self.kernel_events
        ]

        # Merge overlapping kernel intervals to get actual busy time
        merged_kernel_intervals = self._merge_intervals(kernel_intervals)
        kernel_time_us = sum(end - start for start, end in merged_kernel_intervals)

        # Create intervals for memory operations
        memory_intervals = [
            (event["ts"] / 1000.0, (event["ts"] + event["dur"]) / 1000.0)
            for event in self.memory_events
        ]

        # Merge overlapping memory intervals
        merged_memory_intervals = self._merge_intervals(memory_intervals)
        memory_time_us = sum(end - start for start, end in merged_memory_intervals)

        # Merge all GPU activity (kernels + memory) to calculate total busy time
        all_gpu_intervals = kernel_intervals + memory_intervals
        merged_all_intervals = self._merge_intervals(all_gpu_intervals)
        busy_time_us = sum(end - start for start, end in merged_all_intervals)

        # Calculate idle time
        idle_time_us = max(0, total_time_us - busy_time_us)

        # Create timeline DataFrame
        data = [
            {"type": "total_time", "time ms": total_time_us / 1000.0, "percent": 100.0},
            {
                "type": "kernel",
                "time ms": kernel_time_us / 1000.0,
                "percent": (
                    (kernel_time_us / total_time_us * 100.0) if total_time_us > 0 else 0
                ),
            },
            {
                "type": "memory",
                "time ms": memory_time_us / 1000.0,
                "percent": (
                    (memory_time_us / total_time_us * 100.0) if total_time_us > 0 else 0
                ),
            },
            {
                "type": "busy_time",
                "time ms": busy_time_us / 1000.0,
                "percent": (
                    (busy_time_us / total_time_us * 100.0) if total_time_us > 0 else 0
                ),
            },
            {
                "type": "idle",
                "time ms": idle_time_us / 1000.0,
                "percent": (
                    (idle_time_us / total_time_us * 100.0) if total_time_us > 0 else 0
                ),
            },
        ]

        return pd.DataFrame(data)

    def get_df_kernel_summary(self) -> pd.DataFrame:
        """
        Kernel summary with columns:
        - name: Kernel name
        - Count: Number of kernel invocations
        - Total Kernel Time (ms): Total execution time in milliseconds
        - Mean/Median/Std/Min/Max Kernel Time (µs): Statistics in microseconds
        - Percentage (%): Percentage of total kernel time
        - Cumulative Percentage (%): Running total percentage
        - Category: Kernel category (GEMM, Convolution, etc.)
        """
        if not self.kernel_events:
            return pd.DataFrame(
                columns=[
                    "name",
                    "Count",
                    "Mean Kernel Time (µs)",
                    "Median Kernel Time (µs)",
                    "Std Kernel Time (µs)",
                    "Min Kernel Time (µs)",
                    "Max Kernel Time (µs)",
                    "Total Kernel Time (ms)",
                    "Percentage (%)",
                    "Cumulative Percentage (%)",
                    "Category",
                ]
            )

        # Create DataFrame from kernel events
        df_kernels = pd.DataFrame(self.kernel_events)

        # Group by kernel name and aggregate
        agg_dict = {"dur": ["count", "sum", "mean", "median", "std", "min", "max"]}
        df_summary = df_kernels.groupby("name").agg(agg_dict)
        df_summary.columns = [
            "Count",
            "Total Kernel Time (µs)",
            "Mean Kernel Time (µs)",
            "Median Kernel Time (µs)",
            "Std Kernel Time (µs)",
            "Min Kernel Time (µs)",
            "Max Kernel Time (µs)",
        ]
        df_summary.reset_index(inplace=True)

        # Add percentage column
        total_time = df_summary["Total Kernel Time (µs)"].sum()
        df_summary["Percentage (%)"] = (
            (df_summary["Total Kernel Time (µs)"] / total_time * 100.0)
            if total_time > 0
            else 0
        )
        df_summary["Cumulative Percentage (%)"] = df_summary["Percentage (%)"].cumsum()

        # Sort by total time descending
        df_summary.sort_values(
            by="Total Kernel Time (µs)", ascending=False, inplace=True
        )
        df_summary.reset_index(drop=True, inplace=True)

        # Convert to milliseconds for total time column
        df_summary["Total Kernel Time (ms)"] = (
            df_summary["Total Kernel Time (µs)"] / 1000.0
        )

        # Drop the microseconds column to avoid redundancy (keep only ms)
        df_summary.drop(columns=["Total Kernel Time (µs)"], inplace=True)

        # Add a category column for each kernel in the summary
        df_summary["Category"] = df_summary["name"].apply(_categorize_kernel)

        return df_summary

    def get_df_kernel_summary_by_category(self) -> pd.DataFrame:
        """Group kernels by category (GEMM, elementwise, etc.)"""
        if not self.kernel_events:
            return pd.DataFrame(
                columns=[
                    "op category",
                    "Count",
                    "total_direct_kernel_time_ms",
                    "Percentage (%)",
                    "Cumulative Percentage (%)",
                ]
            )

        # Add category to kernel events
        df_kernels = pd.DataFrame(self.kernel_events)
        df_kernels["op category"] = df_kernels["name"].apply(_categorize_kernel)

        # Group by category
        df_summary = df_kernels.groupby("op category").agg({"dur": ["count", "sum"]})
        df_summary.columns = ["Count", "total_direct_kernel_time_us"]
        df_summary.reset_index(inplace=True)

        # Convert to milliseconds
        df_summary["total_direct_kernel_time_ms"] = (
            df_summary["total_direct_kernel_time_us"] / 1000.0
        )
        df_summary.drop(columns=["total_direct_kernel_time_us"], inplace=True)

        # Add percentages
        total_time = df_summary["total_direct_kernel_time_ms"].sum()
        df_summary["Percentage (%)"] = (
            (df_summary["total_direct_kernel_time_ms"] / total_time * 100.0)
            if total_time > 0
            else 0
        )
        df_summary["Cumulative Percentage (%)"] = df_summary["Percentage (%)"].cumsum()

        # Sort by total time descending
        df_summary.sort_values(
            by="total_direct_kernel_time_ms", ascending=False, inplace=True
        )
        df_summary.reset_index(drop=True, inplace=True)

        return df_summary

    def get_df_short_kernels(self, threshold_us: float = 10.0) -> pd.DataFrame:
        """Analyze short-duration kernels"""
        if not self.kernel_events:
            return pd.DataFrame(
                columns=[
                    "name",
                    "Short Kernel count",
                    "Short Kernel duration (µs) sum",
                    "Short Kernel duration (µs) mean",
                    "Short Kernel duration (µs) percent of total time",
                ]
            )

        df_kernels = pd.DataFrame(self.kernel_events)
        df_short = df_kernels[df_kernels["dur"] < threshold_us].copy()

        if df_short.empty:
            logger.info(f"No kernels found with duration < {threshold_us} µs")
            return pd.DataFrame(
                columns=[
                    "name",
                    "Short Kernel count",
                    "Short Kernel duration (µs) sum",
                    "Short Kernel duration (µs) mean",
                    "Short Kernel duration (µs) percent of total time",
                ]
            )

        # Group by kernel name
        df_summary = df_short.groupby("name").agg({"dur": ["count", "sum", "mean"]})
        df_summary.columns = [
            "Short Kernel count",
            "Short Kernel duration (µs) sum",
            "Short Kernel duration (µs) mean",
        ]
        df_summary.reset_index(inplace=True)

        # Calculate percentage of total time
        total_time = sum(event["dur"] for event in self.kernel_events)
        df_summary["Short Kernel duration (µs) percent of total time"] = (
            (df_summary["Short Kernel duration (µs) sum"] / total_time * 100.0)
            if total_time > 0
            else 0
        )

        # Sort by total time descending
        df_summary.sort_values(
            by="Short Kernel duration (µs) sum", ascending=False, inplace=True
        )
        df_summary.reset_index(drop=True, inplace=True)

        return df_summary

    def get_df_short_kernel_histogram(
        self, threshold_us: float = 10.0, bins: int = 100
    ) -> pd.DataFrame:
        """Generate histogram of short kernel durations"""
        if not self.kernel_events:
            return pd.DataFrame(columns=["bin_start", "bin_end", "count"])

        df_kernels = pd.DataFrame(self.kernel_events)
        df_short = df_kernels[df_kernels["dur"] < threshold_us]

        if df_short.empty:
            return pd.DataFrame(columns=["bin_start", "bin_end", "count"])

        # Create histogram
        counts, bin_edges = np.histogram(df_short["dur"].values, bins=bins)
        df_hist = pd.DataFrame(
            {"bin_start": bin_edges[:-1], "bin_end": bin_edges[1:], "count": counts}
        )

        return df_hist

    def get_df_kernel_details(self, topk: Optional[int] = None) -> pd.DataFrame:
        """
        Get detailed kernel information including grid/block dimensions

        Args:
            topk: If specified, return only top K kernels by duration
        """
        if not self.kernel_events:
            return pd.DataFrame()

        df_kernels = pd.DataFrame(self.kernel_events)

        # Expand grid and block tuples into separate columns
        df_kernels["grid_x"] = df_kernels["grid"].apply(lambda x: x[0])
        df_kernels["grid_y"] = df_kernels["grid"].apply(lambda x: x[1])
        df_kernels["grid_z"] = df_kernels["grid"].apply(lambda x: x[2])
        df_kernels["block_x"] = df_kernels["block"].apply(lambda x: x[0])
        df_kernels["block_y"] = df_kernels["block"].apply(lambda x: x[1])
        df_kernels["block_z"] = df_kernels["block"].apply(lambda x: x[2])

        # Rename duration column for clarity
        df_kernels.rename(columns={"dur": "Kernel duration (µs)"}, inplace=True)

        # Select relevant columns
        columns = [
            "name",
            "Kernel duration (µs)",
            "stream",
            "dispatch_id",
            "grid_x",
            "grid_y",
            "grid_z",
            "block_x",
            "block_y",
            "block_z",
            "agent_id",
            "thread_id",
        ]
        df_result = df_kernels[columns].copy()

        # Sort by duration descending
        df_result.sort_values(by="Kernel duration (µs)", ascending=False, inplace=True)

        if topk is not None:
            df_result = df_result.head(topk)

        df_result.reset_index(drop=True, inplace=True)

        return df_result
