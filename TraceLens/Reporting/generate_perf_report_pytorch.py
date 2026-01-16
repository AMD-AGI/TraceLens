###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from TraceLens import NcclAnalyser, TraceToTree, TreePerfAnalyzer
from TraceLens.Reporting.reporting_utils import request_install


def get_dfs_short_kernels(
    perf_analyzer, short_kernel_threshold_us=10, histogram_bins=100, topk=None
):
    """
    TODO: move this to the TreePerfAnalyzer class
    Analyze short kernel events from the performance data and return two DataFrames:
    a histogram of short kernel durations and a summary of top short kernels.

    Args:
        perf_analyzer (TreePerfAnalyzer): The performance analyzer object containing kernel data.
        short_kernel_threshold_us (int, optional): Threshold in microseconds to classify a kernel as "short". Defaults to 10.
        histogram_bins (int, optional): Number of bins for the histogram of short kernel durations. Defaults to 100.
        topk (int, optional): Number of top short kernels to include in the summary. If None, include all. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Histogram of short kernel durations with columns ['bin_start', 'bin_end', 'count'].
            - pd.DataFrame: Summary of top short kernels with detailed statistics and percentage contribution to total time.
    """
    df_kernels = perf_analyzer.get_df_kernels()
    df_filtered = df_kernels[
        df_kernels["Kernel duration (µs)"] < short_kernel_threshold_us
    ]

    # 1. get histogram of these short kernels
    if df_filtered.empty:
        df_hist = pd.DataFrame(columns=["bin_start", "bin_end", "count"])
    else:
        vals = df_filtered["Kernel duration (µs)"].values
        counts, bin_edges = np.histogram(vals, bins=histogram_bins)
        df_hist = pd.DataFrame(
            {"bin_start": bin_edges[:-1], "bin_end": bin_edges[1:], "count": counts}
        )

    # 2. get df short kernels topk by total time
    agg_dict = {
        "Kernel duration (µs)": ["sum", "count", "mean"],
    }
    # For GPU-only traces, only group by Kernel name (CPU-related columns don't exist)
    # For regular traces, group by all available columns
    if perf_analyzer.gpu_only:
        groupby_cols = ["Kernel name"]
    else:
        groupby_cols = [
            "Parent cpu_op",
            "Input dims",
            "Input strides",
            "Concrete Inputs",
            "Kernel name",
        ]

    # If dataframe is empty, return empty dataframe
    if df_filtered.empty:
        df_grouped = pd.DataFrame()
    else:
        df_grouped = df_filtered.groupby(
            groupby_cols,
            sort=False,
        ).agg(agg_dict)

    # Handle empty dataframe case
    if df_grouped.empty:
        return df_hist, df_grouped

    # Flatten multi-level column names
    df_grouped.columns = ["_".join(col).strip() for col in df_grouped.columns]

    # Rename columns for clarity
    df_grouped.rename(
        columns={
            "Kernel duration (µs)_sum": "Short Kernel duration (µs) sum",
            "Kernel duration (µs)_count": "Short Kernel count",
            "Kernel duration (µs)_mean": "Short Kernel duration (µs) mean",
        },
        inplace=True,
    )

    # Add percentage contribution to total time
    df_grouped["Short Kernel duration (µs) percent of total time"] = (
        df_grouped["Short Kernel duration (µs) sum"]
        / (perf_analyzer.total_time_ms * 1e3)
        * 100
    )

    # Sort and format
    df_grouped.sort_values(
        by="Short Kernel duration (µs) sum", ascending=False, inplace=True
    )
    df_grouped.reset_index(inplace=True)
    if topk is not None:
        df_grouped = df_grouped.head(topk)
    return df_hist, df_grouped


def apply_extension(perf_analyzer, extension_path):
    extension_path = os.path.abspath(extension_path)
    extension_name = os.path.splitext(os.path.basename(extension_path))[0]

    spec = importlib.util.spec_from_file_location(extension_name, extension_path)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)

    if hasattr(extension, "tree_postprocess_extension"):
        print(f"Applying tree postprocess extension from {extension_path}")
        tree_postprocess_extension = getattr(extension, "tree_postprocess_extension")
        tree_postprocess_extension(perf_analyzer.tree)
        perf_analyzer.tree.label_non_gpu_paths()

    if hasattr(extension, "perf_model_extension"):
        print(f"Applying perf model extension from {extension_path}")
        perf_model_extension = getattr(extension, "perf_model_extension")
        if not isinstance(perf_model_extension, dict):
            raise ValueError(
                f"Expected perf_model_extension to be a dict, got {type(perf_model_extension)}"
            )
        perf_analyzer.op_to_perf_model_class_map.update(perf_model_extension)
    if hasattr(extension, "dict_cat2names_extension"):
        print(f"Updating dict_cat2names with extension from {extension_path}")
        if not isinstance(extension.dict_cat2names_extension, dict):
            raise ValueError(
                f"Expected dict_cat2names_extension to be a dict, got {type(extension.dict_cat2names_extension)}"
            )

        # defaultdict(<class 'list'>,
        for cat, names in extension.dict_cat2names_extension.items():
            if cat not in perf_analyzer.dict_cat2names:
                perf_analyzer.dict_cat2names[cat] = []
            if not isinstance(names, list):
                raise ValueError(f"Expected names to be a list, got {type(names)}")
            perf_analyzer.dict_cat2names[cat].extend(names)


def trunc_kernel_details(row, kernel_detail_col, trunc_length=64):
    """
    Truncates the kernel details in a row to a specified length for readability.
    """
    if kernel_detail_col not in row or not row[kernel_detail_col]:
        return None  # No kernel details available

    truncated_details = []
    for detail in row[kernel_detail_col]:
        truncated_name = (
            detail["name"][:trunc_length] + "..."
            if len(detail["name"]) > trunc_length
            else detail["name"]
        )
        truncated_details.append(
            {
                "name": truncated_name,
                "stream": detail.get("stream", None),
                "mean_duration_us": round(detail.get("mean_duration_us", 0), 2),
            }
        )

    return truncated_details if truncated_details else None


def add_truncated_kernel_details(
    df: pd.DataFrame,
    source_col: str = "kernel_details",
    new_col_name: str = None,
    trunc_length: int = 64,
) -> pd.DataFrame:
    """
    Applies the truncation logic to a DataFrame column and inserts the new
    truncated column immediately after the source column for easy comparison.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        source_col (str): The name of the column containing the full kernel details.
        new_col_name (str): The name for the new truncated column.
        trunc_length (int): The character length to truncate kernel names to.

    Returns:
        pd.DataFrame: A new DataFrame with the added truncated column.
    """
    # First, ensure the source column exists. If not, do nothing.
    if source_col not in df.columns:
        warnings.warn(
            f"Source column '{source_col}' not found in DataFrame. Skipping truncation.",
            UserWarning,
        )
        return df
    if new_col_name is None:
        new_col_name = f"trunc_{source_col}"
    # 1. Create the new column's data. It will be added to the end for now.
    df[new_col_name] = df.apply(
        lambda row: trunc_kernel_details(row, source_col, trunc_length=trunc_length),
        axis=1,
    )

    # 2. Reorder the columns to place the new column next to its source.
    cols = df.columns.tolist()
    # Pop the new column from the end of the list
    new_col = cols.pop(cols.index(new_col_name))
    # Find the position of our source column and insert the new one after it
    source_col_idx = cols.index(source_col)
    cols.insert(source_col_idx + 1, new_col)

    # Return a new DataFrame with the desired column order
    return df[cols]


def generate_perf_report_pytorch(
    profile_json_path: str,
    output_xlsx_path: Optional[str] = None,
    output_csvs_dir: Optional[str] = None,
    # include unlinked kernels in gpu timeline
    include_unlinked_kernels: bool = False,
    # threshold in microseconds for micro idle time
    micro_idle_thresh_us: int = None,
    # collective analysis
    collective_analysis: bool = True,
    # kernel summary sheet
    kernel_summary: bool = False,
    # short kernel study options
    short_kernel_study: bool = False,
    short_kernel_threshold_us: int = 10,
    short_kernel_histogram_bins: int = 100,
    topk_short_kernels: Optional[int] = None,  # include all below thresh by default
    topk_ops: Optional[int] = None,
    topk_roofline_ops: Optional[int] = None,
    extension_file: Optional[str] = None,
    # for gemmologist
    python_path: Optional[str] = None,
    gpu_arch_json_path: Optional[str] = None,
    # inference phase analysis
    inference_phase_analysis: bool = False,
    decode_threshold: int = 10,
    phase_detection_method: str = "union",
    # plotting options
    generate_plots: bool = True,
    disable_plots: bool = False,
    plot_format: str = "png", 
    plot_dpi: int = 300,
) -> Dict[str, pd.DataFrame]:
    if gpu_arch_json_path:
        with open(gpu_arch_json_path, "r") as f:
            gpu_arch_json = json.load(f)
    else:
        gpu_arch_json = None

    perf_analyzer = TreePerfAnalyzer.from_file(
        profile_filepath=profile_json_path,
        arch=gpu_arch_json,
        python_path=python_path,
        include_unlinked_kernels=include_unlinked_kernels,
    )

    if extension_file:
        apply_extension(perf_analyzer, extension_file)

    # Detect GPU-only trace early and inform user
    if perf_analyzer.gpu_only:
        print(
            "Detected GPU-only trace. Skipping CPU-dependent analysis and generating only GPU timeline and kernel summary."
        )

    agg_metrics = ["mean", "median", "std", "min", "max"]

    # Generate base DataFrames
    df_gpu_timeline = perf_analyzer.get_df_gpu_timeline(
        micro_idle_thresh_us=micro_idle_thresh_us
    )

    # TODO: move this to the TreePerfAnalyzer class
    total_time_row = df_gpu_timeline[df_gpu_timeline["type"] == "total_time"]
    total_time_ms = total_time_row["time ms"].values[0]
    perf_analyzer.total_time_ms = total_time_ms

    # Initialize empty DataFrames for GPU-only traces to avoid NameError
    df_kernel_launchers_summary = pd.DataFrame()
    df_kernel_launchers_summary_by_category = pd.DataFrame()
    df_kernel_launchers_unique_args = pd.DataFrame()
    perf_metrics_dfs = {}
    df_hist = pd.DataFrame()
    df_short_kernels = pd.DataFrame()

    # Only process CPU-dependent analysis for non-GPU-only traces
    if not perf_analyzer.gpu_only:
        df_kernel_launchers = perf_analyzer.get_df_kernel_launchers(
            include_kernel_details=True
        )
        df_kernel_launchers_summary = perf_analyzer.get_df_kernel_launchers_summary(
            df_kernel_launchers
        )
        df_kernel_launchers_summary_by_category = (
            perf_analyzer.get_df_kernel_launchers_summary_by_category(
                df_kernel_launchers
            )
        )
        df_kernel_launchers_unique_args = (
            perf_analyzer.get_df_kernel_launchers_unique_args(
                df_kernel_launchers, agg_metrics=agg_metrics, include_pct=True
            )
        )
        df_kernel_launchers_unique_args = add_truncated_kernel_details(
            df_kernel_launchers_unique_args,
            source_col="kernel_details_summary",
            new_col_name="trunc_kernel_details",
        )
        # Dictionary to hold the op-specific DataFrames
        perf_metrics_dfs = {}

        for op_cat, op_names in perf_analyzer.dict_cat2names.items():
            # Filter events belonging to the current category
            op_events = [
                event
                for event in perf_analyzer.tree.events
                if event["name"] in op_names
            ]

            if op_cat in ["GEMM", "UnaryElementwise", "BinaryElementwise"]:
                # For GEMM: create a single table that covers both fwd and bwd.
                df_ops = perf_analyzer.build_df_perf_metrics(
                    op_events, bwd=False, include_kernel_details=True, include_args=True
                )
                df_ops = perf_analyzer.summarize_df_perf_metrics(df_ops, agg_metrics)
                df_ops = add_truncated_kernel_details(
                    df_ops,
                    source_col="kernel_details__summarize_kernel_stats",
                    new_col_name="trunc_kernel_details",
                )
                if not df_ops.empty:
                    perf_metrics_dfs[op_cat] = df_ops
            else:
                # For FLASH_ATTN and CONV: create separate tables for forward and backward passes.
                df_ops_fwd = perf_analyzer.build_df_perf_metrics(
                    op_events, bwd=False, include_kernel_details=True, include_args=True
                )
                df_ops_fwd = perf_analyzer.summarize_df_perf_metrics(
                    df_ops_fwd, agg_metrics
                )
                df_ops_fwd = add_truncated_kernel_details(
                    df_ops_fwd,
                    source_col="kernel_details__summarize_kernel_stats",
                    new_col_name="trunc_kernel_details",
                )
                # For now, flash_attention_varlen_backward is processed with bwd=True, so we need
                # to have a workaround to extract it from the fwd df and append it to the bwd df.
                filtered_df_bwd_ops = None
                if not df_ops_fwd.empty:
                    filtered_df_bwd_ops = df_ops_fwd[
                        df_ops_fwd["name"] == "flash_attn::_flash_attn_varlen_backward"
                    ]
                    df_ops_fwd = df_ops_fwd[
                        df_ops_fwd["name"] != "flash_attn::_flash_attn_varlen_backward"
                    ]
                df_ops_bwd = perf_analyzer.build_df_perf_metrics(
                    op_events, bwd=True, include_kernel_details=True, include_args=True
                )
                df_ops_bwd = perf_analyzer.summarize_df_perf_metrics(
                    df_ops_bwd, agg_metrics
                )
                df_ops_bwd = add_truncated_kernel_details(
                    df_ops_bwd,
                    source_col="kernel_details__summarize_kernel_stats",
                    new_col_name="trunc_kernel_details",
                )
                if filtered_df_bwd_ops is not None:
                    df_ops_bwd = pd.concat([df_ops_bwd, filtered_df_bwd_ops])
                if not df_ops_fwd.empty:
                    perf_metrics_dfs[f"{op_cat}_fwd"] = df_ops_fwd
                if not df_ops_bwd.empty:
                    perf_metrics_dfs[f"{op_cat}_bwd"] = df_ops_bwd

    # Generate inference phase kernel launcher summaries (if enabled)
    if inference_phase_analysis:
        print("Generating inference phase kernel launcher summaries...")
        try:
            df_launcher_phase_summaries = perf_analyzer.get_df_kernel_launchers_summary_by_inference_phase(
                df_kernel_launchers, m_threshold=decode_threshold, detection_method=phase_detection_method
            )
            
            for phase in ['Prefill', 'Decode']:
                if not df_launcher_phase_summaries[phase].empty:
                    print(f"✓ Generated {phase} launcher summary with {len(df_launcher_phase_summaries[phase])} operations")
                else:
                    print(f"⚠ No {phase} phase operations found in launchers")
        except Exception as e:
            print(f"⚠ Error generating inference phase launcher summaries: {e}")
            df_launcher_phase_summaries = {'Prefill': pd.DataFrame(), 'Decode': pd.DataFrame()}
    else:
        # Create empty summaries if inference phase analysis is disabled
        df_launcher_phase_summaries = {'Prefill': pd.DataFrame(), 'Decode': pd.DataFrame()}


    # Short kernel study (works for both GPU-only and regular traces)
    if short_kernel_study:
        df_hist, df_short_kernels = get_dfs_short_kernels(
            perf_analyzer,
            short_kernel_threshold_us=short_kernel_threshold_us,
            histogram_bins=short_kernel_histogram_bins,
            topk=topk_short_kernels,
        )

    # Build dict_name2df - only include sheets that have data
    dict_name2df = {"gpu_timeline": df_gpu_timeline}

    # Add CPU-dependent sheets only if not GPU-only
    if not perf_analyzer.gpu_only:
        if not df_kernel_launchers_summary_by_category.empty:
            dict_name2df["ops_summary_by_category"] = (
                df_kernel_launchers_summary_by_category
            )
        if not df_kernel_launchers_summary.empty:
            dict_name2df["ops_summary"] = df_kernel_launchers_summary
        if not df_kernel_launchers_unique_args.empty:
            dict_name2df["ops_unique_args"] = df_kernel_launchers_unique_args

        if inference_phase_analysis:
            if not df_launcher_phase_summaries['Prefill'].empty:
                dict_name2df["ops_summary_prefill"] = df_launcher_phase_summaries['Prefill']
            if not df_launcher_phase_summaries['Decode'].empty:
                dict_name2df["ops_summary_decode"] = df_launcher_phase_summaries['Decode']
        # update this dict with the perf_metrics_dfs
        dict_name2df.update(perf_metrics_dfs)



    # Kernel summary: aggregate per-kernel durations and counts
    if kernel_summary:
        print("\n" + "="*70)
        print("GENERATING KERNEL SUMMARY")
        print("="*70)
        try:
            print("Extracting kernel details from trace...")
            df_kernels = perf_analyzer.get_df_kernels(
                launcher_detail=True,
                kernel_categorization=True,
                inference_phase_detection=inference_phase_analysis
            )
            print(f"✓ Extracted {len(df_kernels)} kernel events")
        except Exception as e:
            print(f"✗ Error extracting kernel details: {e}")
            import traceback
            traceback.print_exc()
            df_kernels = pd.DataFrame()
        if not df_kernels.empty and "Kernel duration (µs)" in df_kernels.columns:
            print("Processing kernel data for summary...")
            
            # Fallback: If Parent cpu_op is missing, fill it from Launcher (for display purposes)
            if (
                "Parent cpu_op" in df_kernels.columns
                and "Launcher" in df_kernels.columns
            ):
                mask_missing_parent = df_kernels["Parent cpu_op"].isna()
                if mask_missing_parent.any():
                    df_kernels.loc[mask_missing_parent, "Parent cpu_op"] = (
                        df_kernels.loc[mask_missing_parent, "Launcher"]
                    )

            # Fallback categorization for graph/runtime launched kernels with no cpu_op
            # Note: Basic 'Parent op category' is added by get_kernel_details() in tree_perf.py
            # This adds categorization for kernels that don't have a parent cpu_op
            if "Parent op category" not in df_kernels.columns:
                df_kernels["Parent op category"] = np.nan

            if "Launcher" in df_kernels.columns:
                mask_missing_cat = df_kernels["Parent op category"].isna()
                if mask_missing_cat.any():

                    def _launcher_category(name):
                        s = str(name).lower()
                        if "cudagraph" in s or "graphlaunch" in s:
                            return "graph"
                        return "runtime" if s and s != "nan" else np.nan

                    df_kernels.loc[mask_missing_cat, "Parent op category"] = (
                        df_kernels.loc[mask_missing_cat, "Launcher"].apply(
                            _launcher_category
                        )
                    )

            # Group by category/cpu_op along with kernel identifiers when available
            group_cols = []
            for col in [
                "Kernel category",  # New: detailed kernel-level categorization
                "Parent op category",
                "Parent cpu_op",
                "Kernel name",
                "Kernel stream",
            ]:
                if col in df_kernels.columns:
                    group_cols.append(col)
            if not group_cols:
                group_cols = (
                    ["Kernel name"] if "Kernel name" in df_kernels.columns else []
                )
            
            print(f"Grouping kernels by: {', '.join(group_cols)}")

            agg_dict = {"Kernel duration (µs)": ["sum", "count", "mean", "min", "max"]}
            df_kernel_summary = df_kernels.groupby(group_cols, dropna=False).agg(
                agg_dict
            )
            df_kernel_summary.columns = [
                "_".join(col).strip() for col in df_kernel_summary.columns.values
            ]
            df_kernel_summary.reset_index(inplace=True)

            # Percent columns:
            # 1) Percent of kernels time: sums to ~100% across rows
            total_kernels_us = df_kernels["Kernel duration (µs)"].sum()
            if total_kernels_us > 0:
                df_kernel_summary["Percent of kernels time (%)"] = (
                    df_kernel_summary["Kernel duration (µs)_sum"] / total_kernels_us
                ) * 100
            else:
                df_kernel_summary["Percent of kernels time (%)"] = np.nan
            # 2) Percent of total time (GPU timeline baseline; includes idle/non-kernel)
            total_us = (
                perf_analyzer.total_time_ms * 1e3
                if hasattr(perf_analyzer, "total_time_ms")
                else None
            )
            if total_us:
                df_kernel_summary["Percent of total time (%)"] = (
                    df_kernel_summary["Kernel duration (µs)_sum"] / total_us
                ) * 100
            else:
                df_kernel_summary["Percent of total time (%)"] = np.nan

            df_kernel_summary.sort_values(
                by="Kernel duration (µs)_sum", ascending=False, inplace=True
            )
            df_kernel_summary.reset_index(drop=True, inplace=True)
            dict_name2df["kernel_summary"] = df_kernel_summary
            print(f"✓ Generated kernel summary with {len(df_kernel_summary)} unique kernel groups")
            
            # Generate inference phase summaries if inference phase analysis is enabled and column exists
            if inference_phase_analysis and "Inference phase" in df_kernels.columns:
                print("Generating inference phase kernel summaries...")
                try:
                    # Generate per-kernel summaries by inference phase
                    phase_kernel_summaries = perf_analyzer.get_df_kernel_summary_by_inference_phase(
                        df_kernels, m_threshold=decode_threshold
                    )
                    
                    for phase in ['Prefill', 'Decode']:
                        if not phase_kernel_summaries[phase].empty:
                            dict_name2df[f"kernel_summary_{phase.lower()}"] = phase_kernel_summaries[phase]
                            print(f"✓ Generated {phase} kernel summary with {len(phase_kernel_summaries[phase])} groups")
                        else:
                            print(f"⚠ No {phase} phase kernels found")
                            
                except Exception as e:
                    print(f"⚠ Error generating inference phase summaries: {e}")
            elif inference_phase_analysis:
                print("⚠ Inference phase column not found in kernel data (check if kernel_summary is enabled)")
            elif "Inference phase" in df_kernels.columns:
                print("⚠ Inference phase data available but analysis disabled (use --inference_phase_analysis to enable)")
        else:
            if df_kernels.empty:
                print("⚠ No kernel data available for summary")
            else:
                print("⚠ Kernel duration column not found in kernel data")

    if short_kernel_study:
        dict_name2df["short_kernel_histogram"] = df_hist
        dict_name2df["short_kernels_summary"] = df_short_kernels

    # Skip collective analysis for GPU-only traces (no CPU ops means no collectives)
    if collective_analysis and not perf_analyzer.gpu_only:
        nccl_analyser = NcclAnalyser([profile_json_path], None)
        df_nccl_summary = nccl_analyser.build_df_summary_long()
        if not df_nccl_summary.empty:
            dict_name2df["coll_analysis"] = df_nccl_summary

    # Generate inference phase plots if enabled
    if (inference_phase_analysis and generate_plots and not disable_plots and 
        (not df_launcher_phase_summaries['Prefill'].empty or not df_launcher_phase_summaries['Decode'].empty)):
        try:
            # Import the plotting module
            from TraceLens.Reporting.inference_phase_plots import generate_inference_plots_from_report
            
            # Determine output directory for plots
            if output_csvs_dir:
                plot_output_dir = os.path.join(output_csvs_dir, "inference_plots")
            elif output_xlsx_path:
                base_path = output_xlsx_path.rsplit(".", 1)[0]
                plot_output_dir = base_path + "_inference_plots"
            else:
                base_path = profile_json_path.rsplit(".json", 1)[0]
                plot_output_dir = base_path + "_inference_plots"
            
            # Extract trace name from path
            trace_name = os.path.basename(profile_json_path).split('.')[0]
            
            print(f"\n📊 Generating inference phase visualization plots...")
            plot_files = generate_inference_plots_from_report(
                report_data=dict_name2df,
                output_dir=plot_output_dir,
                trace_name=trace_name,
                show_plots=False,
                save_plots=True,
                plot_format=plot_format,
                dpi=plot_dpi
            )
            
            if plot_files:
                print(f"✓ Generated {len(plot_files)} inference phase plots in: {plot_output_dir}/")
                print("📈 Plots include: phase overview, time distributions, operation frequencies,")
                print("   performance bottlenecks, and detailed breakdowns")
                print(f"📄 View summary report: {os.path.join(plot_output_dir, f'{trace_name}_inference_analysis.html')}")
            
        except ImportError as e:
            print(f"⚠ Could not import plotting module: {e}")
            print("💡 Make sure matplotlib and seaborn are installed: pip install matplotlib seaborn")
        except Exception as e:
            print(f"⚠ Error generating inference phase plots: {e}")
            import traceback
            traceback.print_exc()
    
    # Get additional DataFrames from extension if available
    if extension_file:
        extension_path = os.path.abspath(extension_file)
        extension_name = os.path.splitext(os.path.basename(extension_path))[0]
        spec = importlib.util.spec_from_file_location(extension_name, extension_path)
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)

        if hasattr(extension, "get_additional_dataframes_extension"):
            print(f"Getting additional DataFrames from extension: {extension_path}")
            get_additional_dfs = getattr(
                extension, "get_additional_dataframes_extension"
            )
            additional_dfs = get_additional_dfs(perf_analyzer.tree)
            if additional_dfs:
                dict_name2df.update(additional_dfs)
                print(f"Added {len(additional_dfs)} additional sheets from extension")

    # Write all DataFrames to separate sheets in an Excel workbook
    if output_csvs_dir:
        # Ensure the output directory exists
        os.makedirs(output_csvs_dir, exist_ok=True)
        for sheet_name, df in dict_name2df.items():
            csv_path = os.path.join(output_csvs_dir, f"{sheet_name}.csv")
            df.to_csv(csv_path, index=False)
            print(f"DataFrame '{sheet_name}' written to {csv_path}")
    else:
        if output_xlsx_path is None:
            # split input path at 'json' and take the first part and append '.xlsx'
            base_path = profile_json_path.rsplit(".json", 1)[0]
            output_xlsx_path = base_path + "_perf_report.xlsx"
        try:
            import openpyxl
        except (ImportError, ModuleNotFoundError) as e:
            print(f"Error importing openpyxl: {e}")
            request_install("openpyxl")

        with pd.ExcelWriter(output_xlsx_path, engine="openpyxl") as writer:
            for sheet_name, df in dict_name2df.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"DataFrames successfully written to {output_xlsx_path}")

    return dict_name2df


def main():

    parser = argparse.ArgumentParser(
        description="Process a JSON trace profile and generate performance report tables."
    )
    parser.add_argument(
        "--profile_json_path",
        type=str,
        required=True,
        help="Path to the profile.json or .json.gz file",
    )
    parser.add_argument(
        "--output_xlsx_path",
        type=str,
        default=None,
        help="Path to the output Excel file",
    )
    parser.add_argument(
        "--output_csvs_dir",
        type=str,
        default=None,
        help="Directory to save output CSV files",
    )

    # Optional arguments
    parser.add_argument(
        "--include_unlinked_kernels",
        action="store_true",
        help="Include unlinked kernels in the GPU timeline analysis.",
    )
    parser.add_argument(
        "--micro_idle_thresh_us",
        type=int,
        default=None,
        help="Threshold in microseconds to classify idle interval as micro idle in GPU timeline analysis. "
        "Default is None and all idle times are included in one category.",
    )
    parser.add_argument(
        "--disable_coll_analysis",
        action="store_false",
        dest="collective_analysis",
        default=True,
        help="Disable collective analysis section in the report. Enabled by default.",
    )
    parser.add_argument(
        "--enable_kernel_summary",
        action="store_true",
        dest="kernel_summary",
        default=False,
        help="Enable detailed kernel summary sheet in the report. "
             "This generates a comprehensive breakdown of all GPU kernels grouped by "
             "category, parent operation, kernel name, and stream with statistics including "
             "total duration, count, mean, min, max, and percentage of total time. "
             "Note: This can be slow for large traces. Disabled by default.",
    )
    parser.add_argument(
        "--short_kernel_study",
        action="store_true",
        help="Include short kernel study in the report.",
    )
    parser.add_argument(
        "--short_kernel_threshold_us",
        type=int,
        default=10,
        help='Threshold in microseconds to classify a kernel as "short". Defaults to 10 us.',
    )
    parser.add_argument(
        "--short_kernel_histogram_bins",
        type=int,
        default=100,
        help="Number of bins for the short-kernel histogram.",
    )
    parser.add_argument(
        "--topk_short_kernels",
        type=int,
        default=None,
        help="Rows to keep in the short-kernel table.",
    )

    parser.add_argument(
        "--topk_ops",
        type=int,
        default=None,
        help="Rows to keep in the unique-args launcher table.",
    )
    parser.add_argument(
        "--topk_roofline_ops",
        type=int,
        default=None,
        help="Rows to keep in the roofline table.",
    )

    parser.add_argument(
        "--extension_file",
        type=str,
        default=None,
        help="Path to the extension file containing custom extensions for TraceTree and PerfModel.",
    )

    parser.add_argument(
        "--python_path",
        type=str,
        default=None,
        help="Path to the python executable for gemmologist",
    )
    parser.add_argument(
        "--gpu_arch_json_path",
        type=str,
        default=None,
        help="Path to the GPU architecture JSON file",
    )
    
    parser.add_argument(
        "--inference_phase_analysis",
        action="store_true",
        default=False,
        help="Enable inference phase analysis to categorize operations and kernels into Prefill and Decode phases. "
             "This generates separate summary sheets for each phase based on GEMM M parameter patterns and kernel names. "
             "Useful for analyzing LLM inference workloads with vLLM, SGLang, TGI, and other inference engines.",
    )
    
    parser.add_argument(
        "--decode_threshold",
        type=int,
        default=10,
        help="Threshold for GEMM M parameter to distinguish Prefill vs Decode phases. "
             "Operations with M > threshold are classified as Prefill, others as Decode. "
             "Default: 10 (works well for most LLM inference patterns).",
    )
    
    parser.add_argument(
        "--phase_detection_method",
        type=str,
        default="union",
        choices=["union", "gemm_params", "kernel_names", "attention_patterns", "framework_apis", "operation_frequency", "hybrid"],
        help="Method to use for inference phase detection. Options: "
             "'union' (comprehensive union of ALL criteria - RECOMMENDED FOR MAXIMUM COVERAGE), "
             "'hybrid' (combine multiple methods), "
             "'kernel_names' (read from kernel names), "
             "'operation_frequency' (statistical analysis), "
             "'framework_apis' (use vLLM/SGLang/TGI patterns), "
             "'attention_patterns' (analyze attention operations), "
             "'gemm_params' (original M parameter method). "
             "Default: 'union'.",
    )
    
    parser.add_argument(
        "--generate_plots",
        action="store_true",
        default=True,
        help="Generate visualization plots for inference phase analysis. "
             "Creates comprehensive plots showing phase distribution, time analysis, "
             "operation frequencies, and performance bottlenecks. Enabled by default "
             "when --inference_phase_analysis is used.",
    )
    
    parser.add_argument(
        "--disable_plots",
        action="store_true", 
        default=False,
        help="Disable automatic plot generation for inference phase analysis. "
             "Use this flag to skip plot creation if you only need the data tables.",
    )
    
    parser.add_argument(
        "--plot_format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Format for saved inference phase plots. Options: 'png' (default), 'pdf', 'svg'.",
    )
    
    parser.add_argument(
        "--plot_dpi",
        type=int,
        default=300,
        help="DPI (resolution) for saved inference phase plots. Default: 300 (high quality).",
    )

    args = parser.parse_args()
    generate_perf_report_pytorch(
        profile_json_path=args.profile_json_path,
        output_xlsx_path=args.output_xlsx_path,
        output_csvs_dir=args.output_csvs_dir,
        include_unlinked_kernels=args.include_unlinked_kernels,
        micro_idle_thresh_us=args.micro_idle_thresh_us,
        collective_analysis=args.collective_analysis,
        kernel_summary=args.kernel_summary,
        short_kernel_study=args.short_kernel_study,
        short_kernel_threshold_us=args.short_kernel_threshold_us,
        short_kernel_histogram_bins=args.short_kernel_histogram_bins,
        topk_short_kernels=args.topk_short_kernels,
        topk_ops=args.topk_ops,
        topk_roofline_ops=args.topk_roofline_ops,
        extension_file=args.extension_file,
        python_path=args.python_path,
        gpu_arch_json_path=args.gpu_arch_json_path,
        inference_phase_analysis=args.inference_phase_analysis,
        decode_threshold=args.decode_threshold,
        phase_detection_method=args.phase_detection_method,
        generate_plots=args.generate_plots,
        disable_plots=args.disable_plots,
        plot_format=args.plot_format,
        plot_dpi=args.plot_dpi,
    )


if __name__ == "__main__":
    main()
