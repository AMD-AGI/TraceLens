###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import copy
import gzip
import json
import logging
import os, re, sys
import pprint

# TODO: warning should show the stack as well
import warnings
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from ..PerfModel.jax_op_mapping import jax_op_to_perf_model_class_map
from ..PerfModel.torch_op_mapping import (
    categorize_torch_op,
    dict_cat2names,
    op_to_perf_model_class_map,
)
from ..Trace2Tree.trace_to_tree import JaxTraceToTree, TraceToTree
from ..util import DataLoader, JaxProfileProcessor, TraceEventUtils
from .gpu_event_analyser import GPUEventAnalyser, JaxGPUEventAnalyser
from .jax_analyses import JaxAnalyses
from ..Trace2Tree.extensions import apply_pseudo_op_extensions


def normalize_dtype_to_precision(dtype_str):
    """
    Normalize a dtype string to a standard precision identifier.

    Args:
        dtype_str: A dtype string like "c10::half", "c10::bfloat16", "float", etc.

    Returns:
        Normalized precision string like "fp16", "bf16", "fp32", "fp64", "fp8"
        or None if no mapping is found.
    """
    if dtype_str is None:
        return None

    dtype_lower = str(dtype_str).lower()

    # Mapping based on actual PyTorch/c10 dtype strings
    dtype_mapping = {
        # Float16 (half precision)
        "c10::half": "fp16",
        "half": "fp16",
        # BFloat16
        "c10::bfloat16": "bf16",
        "bfloat16": "bf16",
        # Float32
        "float": "fp32",
        "c10::float": "fp32",
        # Float64
        "double": "fp64",
        "c10::double": "fp64",
        # FP8 variants
        "c10::float8_e4m3fnuz": "fp8",
        "c10::float8_e4m3fn": "fp8",
        "c10::float8_e5m2": "fp8",
        "fp8": "fp8",
        "unsigned char": "fp8",
        # Int8
        "signed char": "int8",
        "int8": "int8",
    }

    return dtype_mapping.get(dtype_lower, None)


def get_compute_spec(perf_model):
    """
    Get the compute spec (maf_type + precision) for a perf model.

    Args:
        perf_model: A perf model instance with get_maf_type() and get_compute_precision() methods.

    Returns:
        str: Compute spec like "matrix_fp16", "vector_bf16", or None if not available.
    """
    maf_type = (
        perf_model.get_maf_type() if hasattr(perf_model, "get_maf_type") else None
    )
    precision = (
        perf_model.get_compute_precision()
        if hasattr(perf_model, "get_compute_precision")
        else None
    )
    if maf_type is None or precision is None:
        return None
    return f"{maf_type}_{precision}"


def get_max_achievable_tflops(perf_model, arch):
    """
    Get the max achievable TFLOPS for a perf model based on arch specs.

    Args:
        perf_model: A perf model instance with get_maf_type() and get_compute_precision() methods.
        arch: GPU architecture specs dict with max_achievable_tflops.

    Returns:
        float: Max achievable TFLOPS, or None if not available.
    """
    if arch is None:
        return None
    maf_specs = arch.get("max_achievable_tflops")
    if maf_specs is None:
        return None

    compute_spec = get_compute_spec(perf_model)
    if compute_spec is None:
        return None

    return maf_specs.get(compute_spec)


class TreePerfAnalyzer:
    @staticmethod
    def from_file(
        profile_filepath,
        jax: bool = False,
        enable_pseudo_ops: bool = False,
        tree_postprocess_extension=None,
        *args,
        **kwargs,
    ) -> "TreePerfAnalyzer":
        # Creates a TreePerfAnalyzer from the trace in the provided filepath.
        # *args, **kwargs are passed to the TreePerfAnalyzer constructor.
        data = DataLoader.load_data(profile_filepath)
        data = data["traceEvents"]

        categorizer = (
            TraceToTree.default_categorizer
            if not jax
            else TraceEventUtils.prepare_event_categorizer(data)
        )
        data = data if not jax else TraceEventUtils.non_metadata_events(data)
        tree = TraceToTree(data, event_to_category=categorizer)

        return TreePerfAnalyzer(
            tree,
            jax=jax,
            event_to_category=categorizer,
            enable_pseudo_ops=enable_pseudo_ops,
            tree_postprocess_extension=tree_postprocess_extension,
            *args,
            **kwargs,
        )

    def __init__(
        self,
        tree: TraceToTree,
        add_python_func=False,
        arch=None,
        jax=False,
        python_path=None,
        event_to_category: Callable[[dict], str] = TraceEventUtils.default_categorizer,
        include_unlinked_kernels=False,
        enable_pseudo_ops=False,
        tree_postprocess_extension=None,
    ):
        self.jax = jax
        self.GPUEventAnalyser = GPUEventAnalyser if not jax else JaxGPUEventAnalyser
        self.tree = tree
        self.add_python_func = add_python_func
        self.arch = arch
        self.python_path = python_path
        self.event_to_category = event_to_category
        self.include_unlinked_kernels = include_unlinked_kernels
        self.with_python_stack = any(
            event.get("cat") == "python_func" for event in self.tree.events
        )
        self.gpu_only = self.check_gpu_only()
        self.tree.build_tree(add_python_func=add_python_func)

        # Apply pseudo-op extensions
        if enable_pseudo_ops:
            try:
                apply_pseudo_op_extensions(self.tree)
            except Exception as e:
                logger.warning(f"Failed to apply pseudo-op extensions: {e}")

        # Backward compatibility for custom tree postprocessing
        if tree_postprocess_extension is not None:
            tree_postprocess_extension(self.tree)

        self.op_to_perf_model_class_map = op_to_perf_model_class_map
        self.op_categorizer = categorize_torch_op
        self.dict_cat2names = dict_cat2names

    def check_gpu_only(self):
        for event in self.tree.events:
            if event.get("cat") in {"python_func", "cpu_op"}:
                return False
        return True

    def agg_kernels_in_subtree(self, event, filter_func=None, verbose=False):
        if filter_func is None:
            filter_func = lambda x: True
        if self.event_to_category(event) in {"kernel", "gpu_memcpy", "gpu_memset"}:
            if not filter_func(event):
                return 0, []
            if verbose:
                print(
                    f"Found kernel event, duration: {event['dur']}, name: {event['name']}"
                )
            return event["dur"], [event["UID"]]
        total_dur = 0
        list_kernels = []
        for child_UID in event.get("children", []):
            child = self.tree.get_UID2event(child_UID)
            child_total_dur, child_list_kernels = self.agg_kernels_in_subtree(
                child, filter_func, verbose
            )
            total_dur += child_total_dur
            list_kernels.extend(child_list_kernels)
        return total_dur, list_kernels

    def loop_and_aggregate_kernels(self, events, filter_func=None, verbose=False):
        total_kernel_time = 0
        list_kernels = []
        for event in events:
            this_total_kernel_time, this_list_kernels = self.agg_kernels_in_subtree(
                event, filter_func, verbose=False
            )
            total_kernel_time += this_total_kernel_time
            list_kernels.extend(this_list_kernels)
        return total_kernel_time, list_kernels

    @staticmethod
    def non_data_mov_filter(event):
        DATA_MOVEMENT_PATTERNS = ["at::native::direct_copy_kernel_cuda", "transpose_"]
        return not any(pattern in event["name"] for pattern in DATA_MOVEMENT_PATTERNS)

    def compute_perf_metrics(
        self, event, bwd=False, non_data_mov=False, perf_model_class=None
    ):

        # Handle kernel aggregation
        if bwd:
            # Always use subtree aggregation for backward metrics
            cpu_op_uids = self.tree.get_subtree_bwd_events(event["UID"])
        else:
            cpu_op_uids = [event["UID"]]
        cpu_op_list = [self.tree.get_UID2event(uid) for uid in cpu_op_uids]
        _, list_kernelUIDS = self.loop_and_aggregate_kernels(cpu_op_list)
        list_kernels = [self.tree.events_by_uid[uid] for uid in list_kernelUIDS]
        busy_kernel_time = 0
        if len(list_kernels) > 0:
            busy_kernel_time = self.GPUEventAnalyser(list_kernels).compute_metrics()[
                "busy_time"
            ]
        _, list_non_data_mov_kernelUIDs = self.loop_and_aggregate_kernels(
            cpu_op_list, filter_func=self.non_data_mov_filter
        )
        list_non_data_mov_kernels = [
            self.tree.events_by_uid[uid] for uid in list_non_data_mov_kernelUIDs
        ]
        busy_non_data_mov_time = 0
        if len(list_non_data_mov_kernels) > 0:
            busy_non_data_mov_time = self.GPUEventAnalyser(
                list_non_data_mov_kernels
            ).compute_metrics()["busy_time"]
        event["kernel_details"] = [
            {
                "name": kernel["name"],
                "dur": kernel["dur"],
                "stream": kernel.get("args", {}).get("stream", None),
            }
            for kernel in list_kernels
        ]

        # Select the appropriate dictionary for FLOPS and memory functions
        if perf_model_class is None:
            perf_model_class = self.op_to_perf_model_class_map.get(event["name"])
        perf_model = perf_model_class(
            event, arch=self.arch, python_path=self.python_path
        )

        gflops = (perf_model.flops() if not bwd else perf_model.flops_bwd()) / 1e9

        tflops_per_s = (
            (gflops / 1e3) / (busy_kernel_time / 1e6)
            if busy_kernel_time > 0
            else float("nan")
        )

        non_data_mov_tflops_per_s = (
            (gflops / 1e3) / (busy_non_data_mov_time / 1e6)
            if busy_non_data_mov_time > 0
            else float("nan")
        )
        bytes_moved = perf_model.bytes() if not bwd else perf_model.bytes_bwd()

        dict_metrics = {
            "GFLOPS": gflops,
            "Kernel Time (µs)": busy_kernel_time,
            "TFLOPS/s": tflops_per_s,
        }
        if non_data_mov:
            dict_metrics["Non-Data-Mov Kernel Time (µs)"] = busy_non_data_mov_time
            dict_metrics["Non-Data-Mov TFLOPS/s"] = non_data_mov_tflops_per_s
        if bytes_moved is not None:
            dict_metrics["Data Moved (MB)"] = bytes_moved / (1024 * 1024)
            dict_metrics["FLOPS/Byte"] = (
                (gflops * 1e9) / bytes_moved if bytes_moved > 0 else float("nan")
            )
            dict_metrics["TB/s"] = (
                (bytes_moved / 1e12) / (busy_kernel_time / 1e6)
                if busy_kernel_time > 0
                else float("nan")
            )
        else:
            dict_metrics["Data Moved (MB)"] = float("nan")
            dict_metrics["FLOPS/Byte"] = float("nan")
            dict_metrics["TB/s"] = float("nan")

        # Add compute spec column (e.g., "matrix_fp16", "vector_bf16")
        compute_spec = get_compute_spec(perf_model)
        dict_metrics["Compute Spec"] = compute_spec if compute_spec else ""

        # Compute roofline time and pct_roofline (only if arch is provided)
        if self.arch is not None:
            peak_tflops = get_max_achievable_tflops(perf_model, self.arch)
            mem_bw_gbps = self.arch.get("mem_bw_gbps")

            if (
                peak_tflops is not None
                and mem_bw_gbps is not None
                and bytes_moved is not None
                and gflops > 0
            ):
                # Compute time: flops / (peak_tflops * 1e12) gives seconds, convert to µs
                compute_time_us = (gflops * 1e9 / (peak_tflops * 1e12)) * 1e6
                # Memory time: bytes / (bandwidth_gbps * 1e9) gives seconds, convert to µs
                memory_time_us = (bytes_moved / (mem_bw_gbps * 1e9)) * 1e6
                roofline_time_us = max(compute_time_us, memory_time_us)
                dict_metrics["Roofline Time (µs)"] = roofline_time_us
                dict_metrics["Pct Roofline"] = (
                    (roofline_time_us / busy_kernel_time) * 100
                    if busy_kernel_time > 0
                    else float("nan")
                )

        if hasattr(perf_model, "get_simulation_time") and not bwd:
            # This is for the case where we have a simulated time
            # for the forward pass, but not for the backward pass
            simulated_time = perf_model.get_simulation_time()
            if simulated_time:
                dict_metrics["Simulated Time (µs)"] = simulated_time
                dict_metrics["Simulated TFLOPS/s"] = (
                    (gflops / 1e3) / (simulated_time / 1e6)
                    if simulated_time > 0
                    else float("nan")
                )

        if hasattr(perf_model, "get_simulation_time_bwd") and bwd:
            # This is for the case where we have a simulated time
            # for the forward pass, but not for the backward pass
            simulated_time = perf_model.get_simulation_time_bwd()
            if simulated_time:
                dict_metrics["Simulated Time (µs)"] = simulated_time
                dict_metrics["Simulated TFLOPS/s"] = (
                    (gflops / 1e3) / (simulated_time / 1e6)
                    if simulated_time > 0
                    else float("nan")
                )

        for key, value in perf_model.param_details.items():
            dict_metrics[f"param: {key}"] = value

        return dict_metrics

    def compute_fwd_perf_metrics(self, event, non_data_mov=False):
        return self.compute_perf_metrics(event, bwd=False, non_data_mov=non_data_mov)

    def compute_bwd_perf_metrics(self, event, non_data_mov=False):
        return self.compute_perf_metrics(event, bwd=True, non_data_mov=non_data_mov)

    def build_df_perf_metrics(
        self,
        events,
        bwd=False,
        non_data_mov=False,
        include_kernel_details=False,
        include_args=False,
        dict_name_to_perf_model=None,
    ):
        if len(events) == 0:
            warnings.warn(
                "Input list of events is empty. Returning an empty DataFrame."
            )
            return pd.DataFrame()
        rows = []
        list_warn_non_zero_flops_and_zero_time = []
        list_warn_perf_metrics_failed = []
        list_no_bwd_events = []
        for event in events:
            metrics_event = {
                "cat": self.event_to_category(event),
                "name": event["name"],
                "UID": event["UID"],
                "pid": event["pid"],
                "tid": event["tid"],
                "process_name": self.tree.metadata.get(event["pid"], {})
                .get(0, {})
                .get("process_name", "Unknown"),
                "process_label": self.tree.metadata.get(event["pid"], {})
                .get(0, {})
                .get("process_labels", "Unknown"),
                "thread_name": self.tree.metadata.get(event["pid"], {})
                .get(event["tid"], {})
                .get("thread_name", "Unknown"),
                "external_id": event["args"].get("External id"),
                "overlapping_kernel_names": event.get("overlapping_kernel_names"),
                "overlapping_kernels_details": event.get("overlapping_kernels_details"),
            }
            if include_args:
                args_cols = [
                    "Input Dims",
                    "Input type",
                    "Input Strides",
                    "Concrete Inputs",
                ]
                metrics_event.update((arg, event["args"].get(arg)) for arg in args_cols)
            if dict_name_to_perf_model and event["name"] in dict_name_to_perf_model:
                perf_model_class = dict_name_to_perf_model[event["name"]]
            else:
                perf_model_class = None
            try:
                dict_perf_metrics = self.compute_perf_metrics(
                    event,
                    bwd=bwd,
                    non_data_mov=non_data_mov,
                    perf_model_class=perf_model_class,
                )
            except Exception as e:
                list_warn_perf_metrics_failed.append(event)
                continue
            # handle warnings
            if bwd and not event.get("bwd_events"):
                list_no_bwd_events.append(event)
                continue
            if (
                dict_perf_metrics["GFLOPS"] > 0
                and dict_perf_metrics["Kernel Time (µs)"] == 0
            ):
                list_warn_non_zero_flops_and_zero_time.append(event)

            if dict_perf_metrics is not None:
                metrics_event.update(dict_perf_metrics)
            if include_kernel_details:
                if "kernel_details" in event:
                    metrics_event["kernel_details"] = event["kernel_details"]
            rows.append(metrics_event)

        self._show_warnings(
            list_warn_non_zero_flops_and_zero_time,
            list_no_bwd_events,
            list_warn_perf_metrics_failed,
            len(events),
        )
        df_perf_metrics = pd.DataFrame(rows)
        return df_perf_metrics

    @staticmethod
    def _show_warnings(
        list_warn_non_zero_flops_and_zero_time,
        list_no_bwd_events,
        list_warn_perf_metrics_failed,
        total_events,
    ):
        # we need to say a/b  events had this issue and one example is following
        # where b is total events
        if len(list_warn_non_zero_flops_and_zero_time) > 0:
            warnings.warn(
                f"Found {len(list_warn_non_zero_flops_and_zero_time)}/{total_events} events with non-zero GFLOPS and zero Kernel Time (µs)."
            )
            warnings.warn(
                f"Example event: {pprint.pformat(list_warn_non_zero_flops_and_zero_time[0])}"
            )
        if len(list_no_bwd_events) > 0:
            warnings.warn(
                f"Found {len(list_no_bwd_events)}/{total_events} events without backward events."
            )
            warnings.warn(f"Example event: {pprint.pformat(list_no_bwd_events[0])}")
        if len(list_warn_perf_metrics_failed) > 0:
            warnings.warn(
                f"Found {len(list_warn_perf_metrics_failed)}/{total_events} events with failed performance metric computation."
            )
            warnings.warn(
                f"Example event: {pprint.pformat(list_warn_perf_metrics_failed[0])}"
            )

    def build_df_fwd_perf_metrics(self, events):
        return self.build_df_perf_metrics(events, bwd=False)

    def build_df_bwd_perf_metrics(self, events):
        return self.build_df_perf_metrics(events, bwd=True)

    @staticmethod
    def summarize_df_perf_metrics(
        df_perf_metrics, agg_metrics=["mean", "std"], include_overlapping_kernels=False
    ):
        if df_perf_metrics.empty:
            warnings.warn(
                "Input DataFrame is empty. Returning an empty summary DataFrame."
            )
            return (
                pd.DataFrame()
            )  # Return an empty DataFrame instead of raising an error

        dict_agg = {}
        # first element for GFLOPS and FLOPS/Byte
        dict_agg["GFLOPS"] = "first"
        dict_agg["Data Moved (MB)"] = "first"
        dict_agg["FLOPS/Byte"] = "first"
        dict_agg["TB/s"] = agg_metrics
        dict_agg["TFLOPS/s"] = agg_metrics
        if "process_name" in df_perf_metrics.columns:
            dict_agg["process_name"] = "first"
        if "process_label" in df_perf_metrics.columns:
            dict_agg["process_label"] = "first"
        if "thread_name" in df_perf_metrics.columns:
            dict_agg["thread_name"] = "first"
        # Compute Spec - static for same args
        if "Compute Spec" in df_perf_metrics.columns:
            dict_agg["Compute Spec"] = "first"
        # Roofline metrics - first since they should be same for the group
        if "Roofline Time (µs)" in df_perf_metrics.columns:
            dict_agg["Roofline Time (µs)"] = "first"
        if "Pct Roofline" in df_perf_metrics.columns:
            dict_agg["Pct Roofline"] = agg_metrics
        if "Simulated Time (µs)" in df_perf_metrics.columns:
            # first since it should be same for the group
            dict_agg["Simulated Time (µs)"] = "first"
            dict_agg["Simulated TFLOPS/s"] = "first"
        if "Non-Data-Mov TFLOPS/s" in df_perf_metrics.columns:
            dict_agg["Non-Data-Mov TFLOPS/s"] = agg_metrics
        if "Non-Data-Mov Kernel Time (µs)" in df_perf_metrics.columns:
            dict_agg["Non-Data-Mov Kernel Time (µs)"] = ["sum"]
        # this is a quick fix, we need to veriify it matches in the group
        if "kernel_details" in df_perf_metrics.columns:
            dict_agg["kernel_details"] = partial(
                TreePerfAnalyzer._summarize_kernel_stats, agg_metrics=agg_metrics
            )
        if include_overlapping_kernels and "overlapping_kernels_details" in df_perf_metrics.columns:
            dict_agg["overlapping_kernels_details"] = partial(
                TreePerfAnalyzer._summarize_kernel_stats, agg_metrics=agg_metrics
            )
        args_cols = ["Input Dims", "Input type", "Input Strides", "Concrete Inputs"]
        for arg in args_cols:
            if arg in df_perf_metrics.columns:
                dict_agg[arg] = "first"
        dict_agg["Kernel Time (µs)"] = agg_metrics + ["sum"]
        # dict_agg['Simulated Kernel Time (us)'] = agg_metrics + ['sum']
        dict_agg["name"] = "count"  # Use the 'name' column as a proxy for counting rows
        dict_agg["UID"] = "first"

        # Identify parameter columns for grouping
        param_cols = [
            col for col in df_perf_metrics.columns if col.startswith("param: ")
        ]
        groupby_cols = ["name"] + param_cols
        if include_overlapping_kernels and "overlapping_kernel_names" in df_perf_metrics.columns:
            groupby_cols.append("overlapping_kernel_names")
        # Convert parameter columns to strings to avoid type comparison issues
        df_perf_metrics = df_perf_metrics.copy()
        for col in param_cols:
            df_perf_metrics[col] = df_perf_metrics[col].astype(str)
        # overlapping_kernel_names is list-like; use str for groupby
        if "overlapping_kernel_names" in df_perf_metrics.columns and "overlapping_kernel_names" in groupby_cols:
            df_perf_metrics["overlapping_kernel_names"] = df_perf_metrics["overlapping_kernel_names"].apply(str)
        # TODO warn user if nans in the performance metrics
        # Perform the aggregation
        df_perf_metrics_summary = df_perf_metrics.groupby(
            groupby_cols, dropna=False
        ).agg(dict_agg)
        df_perf_metrics_summary.columns = [
            "_".join(col).strip() for col in df_perf_metrics_summary.columns.values
        ]
        df_perf_metrics_summary.reset_index(inplace=True)

        # Rename columns for cleaner output
        rename_map = {}

        if "Compute Spec_first" in df_perf_metrics_summary.columns:
            rename_map["Compute Spec_first"] = "Compute Spec"
        if rename_map:
            df_perf_metrics_summary.rename(columns=rename_map, inplace=True)

        # Reorder columns: name, process_name, process_label, thread_name, overlapping_kernel_names (if present), param cols, everything else
        priority_cols = ["name"]
        if "process_name" in df_perf_metrics_summary.columns:
            priority_cols.append("process_name")
        if "process_label" in df_perf_metrics_summary.columns:
            priority_cols.append("process_label")
        if "thread_name" in df_perf_metrics_summary.columns:
            priority_cols.append("thread_name")
        if "overlapping_kernel_names" in df_perf_metrics_summary.columns:
            priority_cols.append("overlapping_kernel_names")
        
        other_cols = [
            col
            for col in df_perf_metrics_summary.columns
            if col not in priority_cols and col not in param_cols
        ]

        df_perf_metrics_summary = df_perf_metrics_summary[
            priority_cols + param_cols + other_cols
        ]

        df_perf_metrics_summary.sort_values(
            by=["Kernel Time (µs)_sum", "UID_first"],
            ascending=[False, True],
            inplace=True,
        )
        # df_perf_metrics_summary.sort_values(by='Simulated Kernel Time (us)_sum', ascending=False, inplace=True)
        df_perf_metrics_summary.reset_index(drop=True, inplace=True)

        # Show blank instead of string "None" or "[]" for empty overlapping_kernel_names
        if "overlapping_kernel_names" in df_perf_metrics_summary.columns:
            col = df_perf_metrics_summary["overlapping_kernel_names"]
            df_perf_metrics_summary["overlapping_kernel_names"] = col.replace(
                ["None", "[]"], pd.NA
            )

        return df_perf_metrics_summary

    def get_kernel_launchers(self, include_nccl=False):
        # This method identifies kernel launchers, which are the events directly responsible for launching GPU kernels.
        #
        # In the ideal case, ops are routed through torch dispatcher to create a clear hierarchy
        # where a "leaf" CPU operation is the caller for runtime events that launch kernels. These CPU ops are
        # valuable for analysis as they contain rich argument information (e.g., input dimensions, strides, dtypes).
        #
        # However, some edge cases exist where the calling CPU context is hidden, and a runtime event appears
        # unlinked to a parent CPU op. In these cases, the runtime event itself is used as the launcher.
        #
        # Implementation note: This method works backwards from kernels to find the launcher.
        # It walks up from each kernel's runtime parent, skipping python_function nodes, to find
        # the first cpu_op ancestor. If no cpu_op is found, the runtime event is used as the launcher.
        # This approach gives consistent results regardless of whether add_python_func=True or False.

        # Step 1: Find all kernel events
        kernel_events = [
            evt
            for evt in self.tree.events
            if self.event_to_category(evt) in {"kernel", "gpu_memcpy", "gpu_memset"}
        ]

        # Step 2: Map each kernel to its launcher (cpu_op if found, else runtime event)
        launcher_to_kernels = defaultdict(list)

        for kernel in kernel_events:
            # Skip nccl if not included
            is_nccl = "nccl" in kernel.get("name", "").lower()
            if is_nccl and not include_nccl:
                continue

            # Walk up to find runtime event (immediate parent should be runtime)
            runtime_evt = self.tree.get_parent_event(kernel)
            if runtime_evt is None:
                continue

            # Walk up from runtime to find first cpu_op (skip python_functions)
            current = self.tree.get_parent_event(runtime_evt)
            leaf_cpu_op = None

            while current is not None:
                cat = self.event_to_category(current)
                if cat == "cpu_op":
                    # Special case: 'execute' is a pass-through cpu_op
                    # (e.g. conv_bn_fused -> execute -> cuda launch -> kernel)
                    # Skip it and use its parent as the launcher instead.
                    if current.get("name") == "execute":
                        current = self.tree.get_parent_event(current)
                        continue
                    leaf_cpu_op = current
                    break
                elif cat == "python_function":
                    # Skip python functions, keep going up
                    current = self.tree.get_parent_event(current)
                else:
                    # Some other category, stop
                    break

            # Use cpu_op if found, otherwise use runtime event as launcher
            if leaf_cpu_op is not None:
                launcher_to_kernels[leaf_cpu_op["UID"]].append(kernel)
            else:
                launcher_to_kernels[runtime_evt["UID"]].append(kernel)

        # Step 3: Build kernel_launchers list (sorted by start time for consistent ordering)
        kernel_launchers = []

        # Sort launcher UIDs by the start time of their events
        sorted_launcher_uids = sorted(
            launcher_to_kernels.keys(),
            key=lambda uid: self.tree.get_UID2event(uid).get("ts", 0),
        )

        for launcher_uid in sorted_launcher_uids:
            kernels = launcher_to_kernels[launcher_uid]
            event = self.tree.get_UID2event(launcher_uid)

            event["total_direct_kernel_time"] = self.GPUEventAnalyser(
                kernels
            ).compute_metrics()["busy_time"]
            event["direct_kernel_count"] = len(kernels)
            event["kernel_details"] = [
                {
                    "name": kernel["name"],
                    "dur": kernel["dur"],
                    "stream": kernel.get("args", {}).get("stream", None),
                }
                for kernel in kernels
            ]
            event["op category"] = self.op_categorizer(event)
            event['overlapping_kernel_names'] = [self.tree.get_UID2event(uid).get('name', None) for kernel in kernels for uid in kernel['overlapping_uids']]
            event['overlapping_kernels_details'] = [
                {
                    "name": self.tree.get_UID2event(uid).get('name', None),
                    "dur": self.tree.get_UID2event(uid).get('dur', None),
                    "stream": self.tree.get_UID2event(uid).get("args", {}).get("stream", None),
                }
                for kernel in kernels for uid in kernel['overlapping_uids']
            ]
            if event['overlapping_kernels_details'] == []:
                event['overlapping_kernels_details'] = None
            if event['overlapping_kernel_names'] == []:
                event['overlapping_kernel_names'] = None
            kernel_launchers.append(event)

        return kernel_launchers

    def get_df_kernel_launchers(
        self, id_cols=False, include_kernel_details=False, include_call_stack=False
    ):

        def list_to_tuple(obj):
            if isinstance(obj, list):
                return tuple(list_to_tuple(item) for item in obj)
            return obj

        kernel_launchers = self.get_kernel_launchers()
        rows = []
        for event in kernel_launchers:
            metrics_event = {
                "name": event["name"],
                "op category": event["op category"],
                "UID": event["UID"],
                "total_direct_kernel_time": event["total_direct_kernel_time"],
                "direct_kernel_count": event["direct_kernel_count"],
                "overlapping_kernel_names": event.get("overlapping_kernel_names"),
                "overlapping_kernels_details": event["overlapping_kernels_details"],
            }
            for arg in ["Input Dims", "Input type", "Input Strides", "Concrete Inputs"]:
                if arg in event["args"]:
                    metrics_event[arg] = list_to_tuple(event["args"][arg])
                else:
                    metrics_event[arg] = None

            if id_cols:
                metrics_event["pid"] = event["pid"]
                metrics_event["tid"] = event["tid"]
                metrics_event["external_id"] = event["args"].get("External id")
            if include_kernel_details:
                if "kernel_details" in event:
                    metrics_event["kernel_details"] = event["kernel_details"]
                if include_call_stack:
                    call_stack = self.tree.traverse_parents_and_get_callstack(
                        event, filter=("nn.Module",)
                    )
                    metrics_event["call_stack"] = call_stack
                    metrics_event["parent_module"] = re.sub(
                        r"_\d+", "", (call_stack.split("=>") + ["NA", "NA"])[1]
                    ).strip("")
            thread_metadata = self.tree.metadata.get(event["pid"], {}).get(
                event["tid"], {}
            )
            process_metadata = self.tree.metadata.get(event["pid"], {}).get(0, {})
            metrics_event["process_name"] = process_metadata.get(
                "process_name", "Unknown"
            )
            metrics_event["process_label"] = process_metadata.get(
                "process_labels", "Unknown"
            )
            metrics_event["thread_name"] = thread_metadata.get("thread_name", "Unknown")
            rows.append(metrics_event)
        df = pd.DataFrame(rows)
        return df

    @staticmethod
    def get_df_kernel_launchers_summary(df_kernel_launchers):
        df_temp = df_kernel_launchers.copy()
        df_agg = df_temp.groupby(["name"]).agg(
            {"total_direct_kernel_time": ["sum", "count"], "op category": set},
        )
        df_agg.columns = ["_".join(col).strip() for col in df_agg.columns.values]
        df_agg.reset_index(inplace=True)
        df_agg.rename(
            columns={
                "total_direct_kernel_time_count": "Count",
                "op category_set": "Categories",
            },
            inplace=True,
        )
        df_agg.sort_values(
            by="total_direct_kernel_time_sum", ascending=False, inplace=True
        )
        df_agg["total_direct_kernel_time_ms"] = (
            df_agg["total_direct_kernel_time_sum"] / 1000
        )
        total_duration_ms = df_agg["total_direct_kernel_time_ms"].sum()
        df_agg["Percentage (%)"] = (
            df_agg["total_direct_kernel_time_ms"] / total_duration_ms
        ) * 100
        df_agg["Cumulative Percentage (%)"] = df_agg["Percentage (%)"].cumsum()
        df_agg.reset_index(drop=True, inplace=True)

        return df_agg

    @staticmethod
    def get_df_kernel_launchers_summary_module(df_kernel_launchers):
        df_temp = df_kernel_launchers.copy()
        groupby_cols = ["name"]
        if "parent_module" in df_temp.columns:
            groupby_cols.append("parent_module")
        agg_dict = {"total_direct_kernel_time": ["sum", "count"], "op category": set}
        if "call_stack" in df_temp.columns:
            agg_dict["call_stack"] = "first"
        df_agg = df_temp.groupby(groupby_cols).agg(agg_dict)

        df_agg.columns = ["_".join(col).strip() for col in df_agg.columns.values]
        df_agg.reset_index(inplace=True)
        df_agg.rename(
            columns={
                "total_direct_kernel_time_count": "Count",
                "op category_set": "Categories",
            },
            inplace=True,
        )
        df_agg.sort_values(
            by="total_direct_kernel_time_sum", ascending=False, inplace=True
        )
        df_agg["total_direct_kernel_time_ms"] = (
            df_agg["total_direct_kernel_time_sum"] / 1000
        )
        total_duration_ms = df_agg["total_direct_kernel_time_ms"].sum()
        df_agg["Percentage (%)"] = (
            df_agg["total_direct_kernel_time_ms"] / total_duration_ms
        ) * 100
        df_agg["Cumulative Percentage (%)"] = df_agg["Percentage (%)"].cumsum()
        df_agg.reset_index(drop=True, inplace=True)

        return df_agg

    # separate out name wise perf breakdown and shape wise perf breakdown for a given name
    @staticmethod
    def get_df_kernel_launchers_summary_by_shape(df_kernel_launchers, name):
        warnings.warn(
            "get_df_kernel_launchers_summary_by_shape is deprecated. Use get_df_kernel_launchers_unique_args instead."
        )
        df_temp = df_kernel_launchers.copy()
        df_temp = df_temp[df_temp["name"] == name]
        dict_agg = {
            "total_direct_kernel_time": ["sum", "count", "mean", "std"],
            "direct_kernel_count": ["max", "min"],
        }
        # df_agg = df_temp.groupby(['Input Dims']).agg(dict_agg)
        # check if the input dims and others are present in the df
        df_agg = df_temp.groupby(
            ["Input Dims", "Input type", "Input Strides"], dropna=False
        ).agg(dict_agg)
        df_agg.columns = ["_".join(col).strip() for col in df_agg.columns.values]
        df_agg.reset_index(inplace=True)
        df_agg.rename(
            columns={
                "total_direct_kernel_time_sum": "Total Kernel Time (µs)",
                "total_direct_kernel_time_count": "Count",
                "total_direct_kernel_time_mean": "Mean Kernel Time (µs)",
                "total_direct_kernel_time_std": "Std Kernel Time (µs)",
                "direct_kernel_count_max": "Max Direct Kernel Count",
                "direct_kernel_count_min": "Min Direct Kernel Count",
            },
            inplace=True,
        )
        df_agg.sort_values(by="Total Kernel Time (µs)", ascending=False, inplace=True)
        df_agg["Total Kernel Time (ms)"] = df_agg["Total Kernel Time (µs)"] / 1000
        total_duration_ms = df_agg["Total Kernel Time (ms)"].sum()
        df_agg["Percentage (%)"] = (
            df_agg["Total Kernel Time (ms)"] / total_duration_ms
        ) * 100
        df_agg["Cumulative Percentage (%)"] = df_agg["Percentage (%)"].cumsum()
        df_agg.reset_index(drop=True, inplace=True)
        return df_agg

    @staticmethod
    def _summarize_kernel_stats(series_of_kernel_lists, agg_metrics=["mean"]):
        """
        Revised implementation for ordered kernel summarization.
        """
        METRIC_MAP = {
            "mean": ("mean_duration_us", np.mean),
            "median": ("median_duration_us", np.median),
            "max": ("max_duration_us", np.max),
            "min": ("min_duration_us", np.min),
            "std": ("std_dev_duration_us", np.std),
        }

        # --- CHANGE: More robust way to get the template ---
        # Find the first valid list in the series to use as a template.
        try:
            template = next(
                item
                for item in series_of_kernel_lists
                if isinstance(item, list) and item
            )
        except StopIteration:
            return None  # The series was empty or contained no valid lists.

        # --- CHANGE: Collect durations BY INDEX, not by name ---
        all_durations = [[] for _ in template]

        for kernel_list in series_of_kernel_lists:
            if isinstance(kernel_list, list):
                # Basic validation to prevent errors and warn about inconsistencies
                if len(kernel_list) != len(template):
                    warnings.warn(
                        f"Inconsistent kernel list length found. Skipping a row.",
                        UserWarning,
                    )
                    continue

                for i, kernel in enumerate(kernel_list):
                    try:
                        # Append the duration to the list corresponding to its position
                        all_durations[i].append(kernel["dur"])
                    except (KeyError, IndexError):
                        warnings.warn(
                            f"Malformed kernel event or index issue at index {i}. Skipping kernel: {kernel}",
                            UserWarning,
                        )
                        continue

        # --- CHANGE: Create a deep copy to avoid modifying original data ---
        summary_list = copy.deepcopy(template)

        # Now, compute statistics and populate the summary list
        for i, kernel_summary in enumerate(summary_list):
            durations_for_this_index = all_durations[i]
            dur_arr = np.array(durations_for_this_index)

            # --- CHANGE: Use consistent naming and clear up original 'dur' key ---
            del kernel_summary["dur"]

            kernel_summary["count"] = len(dur_arr)
            kernel_summary["total_duration_us"] = np.sum(
                dur_arr
            )  # Use consistent key name

            if not durations_for_this_index:
                # If no durations were collected (e.g., all rows skipped), skip metric calculation
                continue

            for metric in agg_metrics:
                if metric in METRIC_MAP:
                    metric_name, agg_func = METRIC_MAP[metric]
                    # --- CHANGE: Use the consistent metric name directly ---
                    kernel_summary[metric_name] = agg_func(dur_arr)

        return summary_list

    @staticmethod
    def get_df_kernel_launchers_unique_args(
        df_kernel_launchers: pd.DataFrame,
        event_name=None,
        agg_metrics=["mean"],
        include_pct=False,
        include_overlapping_kernels=False,
    ) -> pd.DataFrame:
        """
        Generate a DataFrame with unique arguments for each operation in the input DataFrame.

        Args:
            df_kernel_launchers (pd.DataFrame): DataFrame containing kernel launchers.
            event_name (str): Optional name of the event to filter the DataFrame.
            agg_metrics (list): List of aggregation metrics to apply. ex: ['mean', 'std', 'median']
            include_pct (bool): If True, include percentage of total time for each row as well as cumulative percentage.
            include_overlapping_kernels (bool): If True, group by 'overlapping_kernel_names' and
            aggregate overlapping_kernels_details_summary via _summarize_kernel_stats.

        Returns:
            pd.DataFrame: DataFrame with unique arguments for each operation.
        """
        grouping_cols_original = [
            "name",
            "op category",
            "process_name",
            "process_label",
            "thread_name",
            "Input Dims",
            "Input type",
            "Input Strides",
            "Concrete Inputs",
        ]
        if include_overlapping_kernels:
            idx = grouping_cols_original.index("name") + 1
            grouping_cols_original.insert(idx, "overlapping_kernel_names")

        # 0. Filter the DataFrame based on the event name if provided
        if event_name is not None:
            df_filtered = df_kernel_launchers[
                df_kernel_launchers["name"] == event_name
            ].copy()
        else:
            df_filtered = df_kernel_launchers.copy()

        # 1. Create string representations of the grouping columns - so we can group by them
        str_col_names, actual_grouping_cols = [], []
        for col in grouping_cols_original:
            if col not in df_filtered.columns:
                continue
            actual_grouping_cols.append(col)
            str_col_name = f"{col}_str_repr_for_grouping"
            df_filtered[str_col_name] = df_filtered[col].apply(str)
            str_col_names.append(str_col_name)
        if not str_col_names:
            raise ValueError("No valid columns found to group by.")

        # 2. Aggregate the DataFrame by the string representations of the grouping columns
        agg_dict = {}
        if "total_direct_kernel_time" in df_filtered.columns:
            agg_dict["total_direct_kernel_time"] = agg_metrics + (
                ["sum"] if "sum" not in agg_metrics else []
            )
        columns_to_keep_first = []
        if "UID" in df_filtered.columns:
            agg_dict["UID"] = ["first", "count"]
            columns_to_keep_first.append("UID")
        if "kernel_details" in df_filtered.columns:
            agg_dict["kernel_details"] = partial(
                TreePerfAnalyzer._summarize_kernel_stats, agg_metrics=agg_metrics
            )
            columns_to_keep_first.append("kernel_details")
        if include_overlapping_kernels and "overlapping_kernels_details" in df_filtered.columns:
            agg_dict["overlapping_kernels_details"] = partial(
                TreePerfAnalyzer._summarize_kernel_stats, agg_metrics=agg_metrics
            )
            columns_to_keep_first.append("overlapping_kernels_details")
        if "parent_module" in df_filtered.columns:
            agg_dict["parent_module"] = "first"
            columns_to_keep_first.append("parent_module")
        for col in actual_grouping_cols:
            if col not in agg_dict:
                agg_dict[col] = "first"
                columns_to_keep_first.append(col)
        df_unique_args = df_filtered.groupby(
            str_col_names, dropna=False, sort=False
        ).agg(agg_dict)
        df_unique_args.columns = [
            "_".join(col).strip() for col in df_unique_args.columns.values
        ]
        df_unique_args.reset_index(inplace=True)

        # 3. Rename columns for clarity
        rename_map = {"UID_count": "operation_count"}
        for col in columns_to_keep_first:
            col_first = f"{col}_first"
            if col_first in df_unique_args.columns:
                rename_map[col_first] = col
        # uid needs to be mapped to ex_UID
        if "UID_first" in df_unique_args.columns:
            rename_map["UID_first"] = "ex_UID"
        for col in df_unique_args.columns:
            if col.startswith("kernel_details_"):
                rename_map[col] = "kernel_details_summary"
            # Rename aggregated overlapping_kernels_details or overlapping_kernels_details_summary
            if (
                (col.startswith("overlapping_kernels_details_") or col.startswith("overlapping_kernels_details_summary_"))
                and not col.endswith("_str_repr_for_grouping")
            ):
                rename_map[col] = "overlapping_kernels_details_summary"
        df_unique_args.rename(columns=rename_map, inplace=True)

        # 4. Reorder columns: start with grouping + key metrics, then rest
        primary_cols = [
            col for col in grouping_cols_original if col in df_unique_args.columns
        ]
        metric_cols = [
            col
            for col in [
                "UID",
                "operation_count",
                "kernel_names",
                "total_direct_kernel_time_mean",
            ]
            if col in df_unique_args.columns
        ]
        other_cols = [
            col
            for col in df_unique_args.columns
            if col not in primary_cols + metric_cols
            and not col.endswith("_str_repr_for_grouping")
        ]
        df_unique_args = df_unique_args[primary_cols + metric_cols + other_cols]
        # 5. Sort the DataFrame by the sum of total_direct_kernel_time and then by ex_uid for stability
        if "total_direct_kernel_time_sum" in df_unique_args.columns:
            df_unique_args = df_unique_args.sort_values(
                by=["total_direct_kernel_time_sum", "ex_UID"], ascending=[False, True]
            ).reset_index(drop=True)

        # 6. Calculate percentage of total time and cumulative percentage if requested
        if include_pct and "total_direct_kernel_time_sum" in df_unique_args.columns:
            total_duration_ms = df_unique_args["total_direct_kernel_time_sum"].sum()
            df_unique_args["Percentage (%)"] = (
                df_unique_args["total_direct_kernel_time_sum"] / total_duration_ms
            ) * 100
            df_unique_args["Cumulative Percentage (%)"] = df_unique_args[
                "Percentage (%)"
            ].cumsum()
        return df_unique_args

    # =========================================================================
    # Unified Perf Metrics Table Methods
    # =========================================================================

    def _has_perf_model(self, event):
        """Check if an event has a perf model available."""
        return event.get("name") in self.op_to_perf_model_class_map

    def _is_leaf_cpu_op(self, event):
        """
        Check if a cpu_op directly launches GPU kernels (is a kernel launcher).

        A leaf cpu_op follows the pattern: cpu_op -> runtime -> kernel
        This matches the definition used in get_kernel_launchers().
        """
        if self.event_to_category(event) != "cpu_op":
            return False

        # Check if any child's grandchild is a kernel (cpu_op -> runtime -> kernel pattern)
        for child_uid in event.get("children", []):
            child = self.tree.get_UID2event(child_uid)
            for grandchild_uid in child.get("children", []):
                grandchild = self.tree.get_UID2event(grandchild_uid)
                if self.event_to_category(grandchild) in {
                    "kernel",
                    "gpu_memcpy",
                    "gpu_memset",
                }:
                    return True
        return False

    def _launches_gpu_kernels(self, event):
        """
        Check if an event launches any GPU kernels.
        Returns True if the event has gpu_events linked to it.
        """
        return len(event.get("gpu_events", [])) > 0

    def _get_linked_fwd_event(self, event):
        """
        Get the linked forward event for a backward op, if it exists.

        Traverses up to 5 ancestor levels looking for an autograd wrapper with
        fwd_event link. This handles different tree structures where the link
        can be at parent, grandparent, or other levels.

        Returns (fwd_event, is_sole_bwd) tuple, or (None, False) if no link.
        is_sole_bwd is True if this is the only backward op for that forward.
        """
        # Traverse up to 5 levels looking for fwd_event
        current = event
        autograd_wrapper = None
        fwd_uid = None

        for _ in range(5):
            parent = self.tree.get_parent_event(current)
            if not parent:
                break
            fwd_uid = parent.get("fwd_event")
            if fwd_uid:
                autograd_wrapper = parent
                break
            current = parent

        if not fwd_uid or not autograd_wrapper:
            return None, False

        fwd_event = self.tree.get_UID2event(fwd_uid)
        if not fwd_event or not self._has_perf_model(fwd_event):
            return None, False

        # Check if this is the only backward op for this forward (1:1 mapping)
        # Count cpu_op descendants of autograd_wrapper that launch GPU kernels
        # and match the main backward op name (exclude helper ops like aten::copy_)
        wrapper_name = autograd_wrapper.get("name", "")
        main_bwd_name = None
        if ": " in wrapper_name:
            main_bwd_name = wrapper_name.split(": ", 1)[1]

        bwd_ops_in_wrapper = []

        def find_main_bwd_ops(evt):
            for child_uid in evt.get("children", []):
                child = self.tree.get_UID2event(child_uid)
                if not child:
                    continue
                child_name = child.get("name", "")
                if (
                    self.event_to_category(child) == "cpu_op"
                    and self._launches_gpu_kernels(child)
                    and (main_bwd_name is None or child_name == main_bwd_name)
                ):
                    bwd_ops_in_wrapper.append(child_uid)
                if self.event_to_category(child) == "cpu_op":
                    find_main_bwd_ops(child)

        find_main_bwd_ops(autograd_wrapper)

        is_sole_bwd = len(bwd_ops_in_wrapper) == 1
        return fwd_event, is_sole_bwd

    def _is_sole_bwd_with_fwd_perf_model(self, event):
        """
        Check if event is a backward op with 1:1 linking to a forward with perf model
        that has backward metrics defined.

        Returns True if:
        - Event is linked to a forward event via autograd wrapper
        - The forward event has a perf model with backward metrics defined
        - This is the ONLY backward op for that forward (1:1 mapping)
        """
        fwd_event, is_sole_bwd = self._get_linked_fwd_event(event)
        if not fwd_event or not is_sole_bwd:
            return False

        # Check if backward metrics are actually defined for this forward op
        try:
            self.compute_perf_metrics(fwd_event, bwd=True)
            return True
        except NotImplementedError:
            return False
        except Exception:
            return False

    def _is_nccl_event(self, event):
        """Check if an event launches NCCL kernels."""
        gpu_event_uids = event.get("gpu_events", [])
        if not gpu_event_uids:
            return False
        # Check if any GPU kernel is an NCCL kernel
        for gpu_uid in gpu_event_uids:
            gpu_event = self.tree.get_UID2event(gpu_uid)
            if gpu_event and "nccl" in gpu_event.get("name", "").lower():
                return True
        return False

    def collect_unified_perf_events(self, include_nccl=False):
        """
        Traverse the trace tree and collect events for unified perf analysis.

        Traverses from cpu_root_nodes and collects events where:
        1. Event has GPU kernels in subtree (first check - skip CPU-only subtrees)
        2. Event has a perf model -> collect and stop traversing subtree
        3. Event is a 1:1 backward op with linked forward that has perf model -> collect and stop
        4. Event is a leaf cpu_op (direct kernel launcher) -> collect

        Args:
            include_nccl (bool): If False, skip events that launch NCCL kernels.
                Default is False to exclude collective communication ops.

        Returns:
            list: List of collected event dictionaries.
        """
        # Note: 1:1 bwd_events linking is done in build_tree() via link_all_fwd_bwd_events()
        # Use get_subtree_bwd_events() for on-demand subtree aggregation

        collected = []
        visited = set()

        def traverse(event_uid):
            if event_uid in visited:
                return
            visited.add(event_uid)

            event = self.tree.get_UID2event(event_uid)

            # Skip non-cpu_op events
            if self.event_to_category(event) != "cpu_op":
                return

            # First check: Does this subtree have any GPU kernels?
            # gpu_events contains all GPU events from the entire subtree
            if not self._launches_gpu_kernels(event):
                return  # No GPU work in this subtree - skip entirely

            # From here, we know there's GPU work in this subtree

            # Exit condition 1: Has perf model - collect and stop
            if self._has_perf_model(event):
                collected.append(event)
                return

            # Exit condition 2: 1:1 backward op with linked forward that has perf model
            # We can compute backward metrics via forward's perf model
            if self._is_sole_bwd_with_fwd_perf_model(event):
                collected.append(event)
                return

            # Exit condition 3: Leaf cpu_op (direct kernel launcher) with GPU kernels
            if self._is_leaf_cpu_op(event):
                # Before collecting, check if any cpu_op children have perf models
                # (e.g., injected pseudo ops from extensions)
                cpu_op_children_with_perf_model = []
                for child_uid in event.get("children", []):
                    child = self.tree.get_UID2event(child_uid)
                    if (
                        self.event_to_category(child) == "cpu_op"
                        and self._has_perf_model(child)
                        and self._launches_gpu_kernels(child)
                    ):
                        cpu_op_children_with_perf_model.append(child_uid)

                if cpu_op_children_with_perf_model:
                    # Traverse children with perf models instead of collecting this leaf
                    for child_uid in cpu_op_children_with_perf_model:
                        traverse(child_uid)
                else:
                    # No children with perf models - collect this leaf
                    if not include_nccl and self._is_nccl_event(event):
                        return
                    collected.append(event)
                return

            # Non-leaf with GPU kernels in subtree but no perf model
            # Traverse children to find more granular ops
            for child_uid in event.get("children", []):
                traverse(child_uid)

        # Start from cpu_root_nodes
        for root_uid in self.tree.cpu_root_nodes:
            traverse(root_uid)

        return collected

    def build_df_unified_perf_table(
        self,
        events=None,
        include_args=True,
        include_perf_metrics=True,
        include_kernel_details=True,
        include_nccl=False,
    ):
        """
        Build a DataFrame with op details and performance metrics for unified perf analysis.

        Collects events that either have a perf model OR are leaf CPU ops that
        launch GPU kernels. CPU-only ops are excluded.

        Args:
            events (list): List of events to include. If None, collects events
                using collect_unified_perf_events().
            include_args (bool): Include input arguments (dims, types, strides).
            include_perf_metrics (bool): Compute and include perf metrics for
                events with perf models.
            include_kernel_details (bool): Include kernel details (name, duration, stream).
            include_nccl (bool): If False, skip events that launch NCCL kernels.
                Default is False to exclude collective communication ops.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - name, op category, UID, pid, tid, External id
                - Input Dims, Input type, Input Strides, Concrete Inputs (if include_args)
                - duration_us, has_perf_model
                - GFLOPS, Kernel Time (µs), TFLOPS/s, Data Moved (MB), FLOPS/Byte, TB/s
                  (if include_perf_metrics and event has perf model)
                - kernel_details (if include_kernel_details)
        """

        def list_to_tuple(obj):
            """Recursively convert lists to tuples for consistent display."""
            if isinstance(obj, list):
                return tuple(list_to_tuple(item) for item in obj)
            return obj

        if events is None:
            events = self.collect_unified_perf_events(include_nccl=include_nccl)

        if len(events) == 0:
            warnings.warn(
                "No events collected for unified perf table. Returning empty DataFrame."
            )
            return pd.DataFrame()

        rows = []
        perf_metrics_failed = []

        for event in events:
            args = event.get("args", {})
            has_own_perf_model = self._has_perf_model(event)
            is_sole_bwd = self._is_sole_bwd_with_fwd_perf_model(event)

            row = {
                "name": event.get("name"),
                "op category": self.op_categorizer(event),
                "UID": event.get("UID"),
                "pid": event.get("pid"),
                "tid": event.get("tid"),
                "process_name": self.tree.metadata.get(event.get("pid"), {})
                .get(0, {})
                .get("process_name", "Unknown"),
                "process_label": self.tree.metadata.get(event.get("pid"), {})
                .get(0, {})
                .get("process_labels", "Unknown"),
                "thread_name": self.tree.metadata.get(event.get("pid"), {})
                .get(event.get("tid"), {})
                .get("thread_name", "Unknown"),
                "External id": args.get("External id"),
                "duration_us": event.get("dur"),
                "has_perf_model": has_own_perf_model or is_sole_bwd,
                "overlapping_kernel_names": event.get("overlapping_kernel_names"),
                "overlapping_kernels_details": event.get("overlapping_kernels_details"),
            }
            if include_args:
                row["Input Dims"] = list_to_tuple(args.get("Input Dims"))
                row["Input type"] = list_to_tuple(args.get("Input type"))
                row["Input Strides"] = list_to_tuple(args.get("Input Strides"))
                row["Concrete Inputs"] = list_to_tuple(args.get("Concrete Inputs"))

            # Add kernel details from gpu_events
            if include_kernel_details:
                gpu_event_uids = event.get("gpu_events", [])
                kernel_details = []
                for gpu_uid in gpu_event_uids:
                    gpu_event = self.tree.get_UID2event(gpu_uid)
                    if gpu_event and self.event_to_category(gpu_event) in {
                        "kernel",
                        "gpu_memcpy",
                        "gpu_memset",
                    }:
                        kernel_details.append(
                            {
                                "name": gpu_event.get("name"),
                                "dur": gpu_event.get("dur"),
                                "stream": gpu_event.get("args", {}).get("stream"),
                            }
                        )
                row["kernel_details"] = kernel_details if kernel_details else None

            # Add perf metrics if available
            perf_cols = [
                "GFLOPS",
                "Kernel Time (µs)",
                "TFLOPS/s",
                "Data Moved (MB)",
                "FLOPS/Byte",
                "TB/s",
                "Compute Spec",
                "Roofline Time (µs)",
                "Pct Roofline",
            ]

            if include_perf_metrics and has_own_perf_model:
                # Has own perf model - compute forward metrics
                try:
                    metrics = self.compute_perf_metrics(event, bwd=False)
                    for col in perf_cols:
                        if col in metrics:
                            row[col] = metrics[col]
                    # Extract perf model params (e.g., M, N, K for GEMM)
                    perf_params = {
                        k.replace("param: ", ""): v
                        for k, v in metrics.items()
                        if k.startswith("param: ")
                    }
                    row["perf_params"] = perf_params if perf_params else None
                except Exception:
                    perf_metrics_failed.append(event)
                    row["perf_params"] = None
            elif include_perf_metrics and is_sole_bwd:
                # 1:1 backward op - use forward's backward metrics
                fwd_event, _ = self._get_linked_fwd_event(event)
                try:
                    metrics = self.compute_perf_metrics(fwd_event, bwd=True)
                    for col in perf_cols:
                        if col in metrics:
                            row[col] = metrics[col]
                    # Extract perf model params
                    perf_params = {
                        k.replace("param: ", ""): v
                        for k, v in metrics.items()
                        if k.startswith("param: ")
                    }
                    row["perf_params"] = perf_params if perf_params else None
                except Exception:
                    perf_metrics_failed.append(event)
                    row["perf_params"] = None
            else:
                # No perf model - compute kernel time using GPUEventAnalyser busy_time
                row["perf_params"] = None

                gpu_event_uids = event.get("gpu_events", [])
                if gpu_event_uids:
                    gpu_events = [
                        self.tree.get_UID2event(uid)
                        for uid in gpu_event_uids
                        if self.tree.get_UID2event(uid)
                    ]
                    if gpu_events:
                        busy_time = GPUEventAnalyser(gpu_events).compute_metrics()[
                            "busy_time"
                        ]
                        row["Kernel Time (µs)"] = busy_time

            rows.append(row)

        if perf_metrics_failed:
            warnings.warn(
                f"Failed to compute perf metrics for {len(perf_metrics_failed)}/{len(events)} events."
            )

        df = pd.DataFrame(rows)

        # Reorder columns
        col_order = [
            "name",
            "op category",
            "UID",
            "pid",
            "tid",
            "process_name",
            "process_label",
            "thread_name",
            "External id",
            "overlapping_kernel_names",
            "overlapping_kernels_details",
        ]
        if include_args:
            col_order.extend(
                ["Input Dims", "Input type", "Input Strides", "Concrete Inputs"]
            )
        col_order.extend(["duration_us", "has_perf_model"])
        if include_perf_metrics:
            col_order.extend(perf_cols)
            col_order.append("perf_params")
        if include_kernel_details:
            col_order.append("kernel_details")

        col_order = [c for c in col_order if c in df.columns]
        return df[col_order]

    @staticmethod
    def summarize_df_unified_perf_table(
        df_unified_perf: pd.DataFrame,
        agg_metrics=["mean", "std"],
        include_pct=True,
        include_overlapping_kernels=False,
    ):
        """
        Summarize unified perf table by unique (name, Input Dims, Input type, etc.).

        Aggregation behavior matches summarize_df_perf_metrics:
        - Static metrics (GFLOPS, Data Moved, FLOPS/Byte): 'first' only
        - Time-varying metrics (Kernel Time, TFLOPS/s, TB/s): mean/std

        Args:
            df_unified_perf (pd.DataFrame): DataFrame from build_df_unified_perf_table().
            agg_metrics (list): Aggregation metrics for time-varying columns.
            include_pct (bool): Include percentage and cumulative percentage columns.
            include_overlapping_kernels (bool): If True, group by 'overlapping_kernel_names' and
                aggregate overlapping_kernels_details_summary via _summarize_kernel_stats.

        Returns:
            pd.DataFrame: Summarized DataFrame grouped by unique args.
        """
        if df_unified_perf.empty:
            warnings.warn(
                "Input DataFrame is empty. Returning an empty summary DataFrame."
            )
            return pd.DataFrame()

        df_temp = df_unified_perf.copy()
        grouping_cols = [
            "name",
            "op category",
            "process_name",
            "process_label",
            "thread_name",
            "Input Dims",
            "Input type",
            "Input Strides",
            "Concrete Inputs",
        ]
        if include_overlapping_kernels and "overlapping_kernel_names" in df_temp.columns:
            grouping_cols.insert(1, "overlapping_kernel_names")  # after name

        # Convert columns to string for grouping
        str_col_names = []
        actual_grouping_cols = []
        for col in grouping_cols:
            if col not in df_temp.columns:
                continue
            actual_grouping_cols.append(col)
            str_col_name = f"{col}_str_repr_for_grouping"
            df_temp[str_col_name] = df_temp[col].apply(str)
            str_col_names.append(str_col_name)

        if not str_col_names:
            raise ValueError("No valid grouping columns found.")

        # Define aggregations
        agg_dict = {}

        # UID: first and count
        if "UID" in df_temp.columns:
            agg_dict["UID"] = ["first", "count"]

        # Duration: sum, mean, std
        if "duration_us" in df_temp.columns:
            agg_dict["duration_us"] = ["sum"] + agg_metrics

        # Static metrics - 'first' only (same for all instances with same args)
        static_cols = ["GFLOPS", "Data Moved (MB)", "FLOPS/Byte", "Compute Spec"]
        for col in static_cols:
            if col in df_temp.columns:
                agg_dict[col] = "first"

        # Time-varying metrics - mean/std (varies per instance)
        time_varying_cols = ["TB/s", "TFLOPS/s"]
        for col in time_varying_cols:
            if col in df_temp.columns:
                agg_dict[col] = agg_metrics

        # Roofline metrics
        if "Roofline Time (µs)" in df_temp.columns:
            agg_dict["Roofline Time (µs)"] = "first"  # Static for same args
        if "Pct Roofline" in df_temp.columns:
            agg_dict["Pct Roofline"] = agg_metrics  # Varies per instance

        # Kernel Time gets mean/std + sum
        if "Kernel Time (µs)" in df_temp.columns:
            agg_dict["Kernel Time (µs)"] = agg_metrics + ["sum"]

        # Kernel details - summarize using _summarize_kernel_stats
        if "kernel_details" in df_temp.columns:
            agg_dict["kernel_details"] = partial(
                TreePerfAnalyzer._summarize_kernel_stats, agg_metrics=agg_metrics
            )
        if include_overlapping_kernels and "overlapping_kernels_details" in df_temp.columns:
            agg_dict["overlapping_kernels_details"] = partial(
                TreePerfAnalyzer._summarize_kernel_stats, agg_metrics=agg_metrics
            )

        # Perf params - static per unique args (e.g., M, N, K for GEMM)
        if "perf_params" in df_temp.columns:
            agg_dict["perf_params"] = "first"

        # Keep original grouping columns (skip any already assigned, e.g. _summarize_kernel_stats)
        for col in actual_grouping_cols:
            if col not in agg_dict:
                agg_dict[col] = "first"

        if "has_perf_model" in df_temp.columns:
            agg_dict["has_perf_model"] = "first"

        # Group and aggregate
        df_summary = df_temp.groupby(str_col_names, dropna=False, sort=False).agg(
            agg_dict
        )

        # Flatten column names
        df_summary.columns = [
            "_".join(col).strip() if isinstance(col, tuple) and col[1] else col[0]
            for col in df_summary.columns
        ]
        df_summary = df_summary.reset_index(drop=True)

        # Rename columns for clarity
        rename_map = {
            "UID_first": "ex_UID",
            "UID_count": "operation_count",
            "duration_us_sum": "total_duration_us",
            "duration_us_mean": "mean_duration_us",
            "duration_us_std": "std_duration_us",
            "has_perf_model_first": "has_perf_model",
            "process_name_first": "process_name",
            "process_label_first": "process_label",
            "thread_name_first": "thread_name",
        }
        for col in actual_grouping_cols:
            rename_map[f"{col}_first"] = col
        for col in static_cols:
            if f"{col}_first" in df_summary.columns:
                rename_map[f"{col}_first"] = col
        # Rename perf_params aggregation column
        if "perf_params_first" in df_summary.columns:
            rename_map["perf_params_first"] = "perf_params"
        # Rename kernel_details and overlapping_kernels_details_summary aggregation columns
        for col in df_summary.columns:
            if col.startswith("kernel_details_"):
                rename_map[col] = "kernel_details_summary"
            if col.startswith("overlapping_kernels_details_"):
                rename_map[col] = "overlapping_kernels_details_summary"

        df_summary = df_summary.rename(columns=rename_map)

        # Sort by total kernel time (GPU), then by ex_UID for stability
        # This matches the ops_unique_args sorting behavior
        sort_cols = []
        if "Kernel Time (µs)_sum" in df_summary.columns:
            sort_cols.append("Kernel Time (µs)_sum")
        elif "total_duration_us" in df_summary.columns:
            sort_cols.append("total_duration_us")
        if "ex_UID" in df_summary.columns:
            sort_cols.append("ex_UID")
        if sort_cols:
            df_summary = df_summary.sort_values(
                by=sort_cols, ascending=[False] + [True] * (len(sort_cols) - 1)
            )

        # Add percentage columns based on kernel time (GPU time)
        if include_pct and "Kernel Time (µs)_sum" in df_summary.columns:
            total = df_summary["Kernel Time (µs)_sum"].sum()
            df_summary["Percentage (%)"] = (
                df_summary["Kernel Time (µs)_sum"] / total
            ) * 100
            df_summary["Cumulative Percentage (%)"] = df_summary[
                "Percentage (%)"
            ].cumsum()

        df_summary = df_summary.reset_index(drop=True)

        # Reorder columns to match tree perf style
        primary_cols = [col for col in grouping_cols if col in df_summary.columns]

        metric_cols = [
            col
            for col in [
                "ex_UID",
                "operation_count",
                "total_duration_us",
                "mean_duration_us",
                "std_duration_us",
            ]
            if col in df_summary.columns
        ]

        # Static perf metrics (no _mean/_std)
        static_metric_cols = [col for col in static_cols if col in df_summary.columns]

        # Time-varying perf metrics (with _mean/_std)
        time_varying_metric_cols = []
        for col in time_varying_cols + ["Kernel Time (µs)"]:
            for suffix in ["", "_mean", "_std", "_sum"]:
                col_name = f"{col}{suffix}" if suffix else col
                if col_name in df_summary.columns:
                    time_varying_metric_cols.append(col_name)

        pct_cols = [
            col
            for col in ["Percentage (%)", "Cumulative Percentage (%)"]
            if col in df_summary.columns
        ]

        other_cols = [
            col
            for col in df_summary.columns
            if col
            not in primary_cols
            + metric_cols
            + static_metric_cols
            + time_varying_metric_cols
            + pct_cols
        ]

        col_order = (
            primary_cols
            + metric_cols
            + static_metric_cols
            + time_varying_metric_cols
            + other_cols
            + pct_cols
        )

        return df_summary[col_order]

    @staticmethod
    def get_df_kernel_launchers_summary_by_category(
        df_kernel_launchers: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate a DataFrame with breakdown of kernel launchers by category.
        Args:
            df_kernel_launchers (pd.DataFrame): DataFrame containing kernel launchers.
        Returns:
            pd.DataFrame: DataFrame with breakdown of kernel launchers by category.
        """
        df_temp = df_kernel_launchers.copy()
        df_agg = df_temp.groupby("op category").agg(
            {"total_direct_kernel_time": ["sum", "count"]}
        )
        df_agg.columns = ["_".join(col).strip() for col in df_agg.columns.values]
        df_agg.reset_index(inplace=True)
        df_agg.rename(columns={"total_direct_kernel_time_count": "Count"}, inplace=True)
        df_agg.sort_values(
            by="total_direct_kernel_time_sum", ascending=False, inplace=True
        )
        df_agg["total_direct_kernel_time_ms"] = (
            df_agg["total_direct_kernel_time_sum"] / 1000
        )
        # remove the us col as we will use ms col
        df_agg.drop(columns=["total_direct_kernel_time_sum"], inplace=True)
        total_duration_ms = df_agg["total_direct_kernel_time_ms"].sum()
        df_agg["Percentage (%)"] = (
            df_agg["total_direct_kernel_time_ms"] / total_duration_ms
        ) * 100
        df_agg["Cumulative Percentage (%)"] = df_agg["Percentage (%)"].cumsum()
        df_agg.reset_index(drop=True, inplace=True)

        return df_agg

    @staticmethod
    def get_df_kernel_launchers_summary_by_category_module(
        df_kernel_launchers: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate a DataFrame with breakdown of kernel launchers by category.
        Args:
            df_kernel_launchers (pd.DataFrame): DataFrame containing kernel launchers.
        Returns:
            pd.DataFrame: DataFrame with breakdown of kernel launchers by category.
        """
        df_temp = df_kernel_launchers.copy()
        groupby_cols = ["op category"]
        if "parent_module" in df_temp.columns:
            groupby_cols.append("parent_module")
        agg_dict = {"total_direct_kernel_time": ["sum", "count"]}
        if "call_stack" in df_temp.columns:
            agg_dict["call_stack"] = "first"

        df_agg = df_temp.groupby(groupby_cols).agg(agg_dict)
        df_agg.columns = ["_".join(col).strip() for col in df_agg.columns.values]
        df_agg.reset_index(inplace=True)
        df_agg.rename(columns={"total_direct_kernel_time_count": "Count"}, inplace=True)
        df_agg.sort_values(
            by="total_direct_kernel_time_sum", ascending=False, inplace=True
        )
        df_agg["total_direct_kernel_time_ms"] = (
            df_agg["total_direct_kernel_time_sum"] / 1000
        )
        # remove the us col as we will use ms col
        df_agg.drop(columns=["total_direct_kernel_time_sum"], inplace=True)
        total_duration_ms = df_agg["total_direct_kernel_time_ms"].sum()
        df_agg["Percentage (%)"] = (
            df_agg["total_direct_kernel_time_ms"] / total_duration_ms
        ) * 100
        df_agg["Cumulative Percentage (%)"] = df_agg["Percentage (%)"].cumsum()
        df_agg.reset_index(drop=True, inplace=True)

        return df_agg

    def get_df_gpu_timeline(self, micro_idle_thresh_us=None):
        kernel_events = [
            event
            for event in self.tree.events
            if self.event_to_category(event) in {"kernel", "gpu_memcpy", "gpu_memset"}
        ]
        if not self.include_unlinked_kernels:
            kernel_events = [event for event in kernel_events if event.get("tree")]
        gpu_event_analyser = self.GPUEventAnalyser(kernel_events)
        df = gpu_event_analyser.get_breakdown_df(
            micro_idle_thresh_us=micro_idle_thresh_us
        )
        return df

    def get_kernel_details(
        self,
        kernel_event,
        launcher_detail=False,
        cpu_op_detail=True,
        nn_module_detail=False,
    ):
        """
        Extract detailed information for a given kernel event.

        This method traces a kernel event's parent relationships to retrieve
        its launcher and CPU operation details, then returns a dictionary of
        relevant information. If any of the necessary links are missing or invalid,
        the function returns None.

        Args:
            kernel_event (dict): The kernel event dictionary.
            launcher_detail (bool): If True, include details of the kernel's launcher.
            cpu_op_detail (bool): If True, include details of the parent CPU operation.
            nn_module_detail (bool): If True, include details of the parent nn.Module event. Only valid if
                `add_python_func` is True. Else, it will be ignored.

        Returns:
            dict or None: A dictionary containing the kernel details, or None if linking fails.
        """

        def list_to_tuple(obj):
            # Recursively convert lists to tuples.
            return (
                tuple(list_to_tuple(item) for item in obj)
                if isinstance(obj, list)
                else obj
            )

        # Verify that the event is a kernel event.
        if self.event_to_category(kernel_event) != "kernel":
            return None

        kernel_details = {
            "UID": kernel_event["UID"],
            "Kernel name": kernel_event["name"],
            "Kernel t start": kernel_event["ts"],
            "Kernel duration (µs)": kernel_event["dur"],
            "Kernel stream": kernel_event["args"].get("stream"),
        }

        # 1. get launcher event
        launcher = self.tree.get_parent_event(kernel_event)

        # add launcher details
        if launcher and launcher_detail:
            kernel_details["Launcher UID"] = launcher["UID"]
            kernel_details["Launcher"] = launcher["name"]
            kernel_details["Grid"] = list_to_tuple(launcher["args"].get("grid"))
            kernel_details["Block"] = list_to_tuple(launcher["args"].get("block"))

        # 2. get lowest cpu_op event - events of cat 'cpu_op' contain args info
        cpu_op = None
        evt = launcher
        while evt:
            if self.event_to_category(evt) == "cpu_op":
                cpu_op = evt
                break
            evt = self.tree.get_parent_event(evt)

        # add cpu_op details
        if cpu_op and cpu_op_detail:
            kernel_details["Parent cpu_op UID"] = cpu_op["UID"]
            kernel_details["Parent cpu_op"] = cpu_op["name"]
            kernel_details["Input dims"] = list_to_tuple(
                cpu_op["args"].get("Input Dims")
            )
            kernel_details["Input types"] = list_to_tuple(
                cpu_op["args"].get("Input type")
            )
            kernel_details["Input strides"] = list_to_tuple(
                cpu_op["args"].get("Input Strides")
            )
            kernel_details["Concrete Inputs"] = list_to_tuple(
                cpu_op["args"].get("Concrete Inputs")
            )
            kernel_details["kernel_file"] = cpu_op["args"].get("kernel_file")
            if cpu_op.get("gpu_busy_time") is None:
                # If the cpu_op event does not have GPU busy time, compute it.
                gpu_events = [
                    self.tree.get_UID2event(uid) for uid in cpu_op.get("gpu_events", [])
                ]
                cpu_op["gpu_busy_time"] = GPUEventAnalyser(
                    gpu_events
                ).compute_metrics()["busy_time"]
            if cpu_op.get("kernel_count") is None:
                # If the cpu_op event does not have direct kernel count, compute it.
                cpu_op["kernel_count"] = len(cpu_op.get("gpu_events", []))
            kernel_details["Parent cpu_op busy time (µs)"] = cpu_op["gpu_busy_time"]
            kernel_details["Parent cpu_op kernel count"] = cpu_op.get("kernel_count", 0)
            if cpu_op["kernel_count"] == 1:
                pct = 100
            else:
                pct = kernel_event["dur"] / cpu_op["gpu_busy_time"] * 100
            kernel_details["Percent of Parent cpu_op busy time (%)"] = pct
            # Add parent op category
            kernel_details["Parent op category"] = self.op_categorizer(cpu_op)

        # 3. get nn.Module event
        nn_module_event = None
        if nn_module_detail and self.add_python_func:
            # Attempt to find the nn.Module parent event.
            evt = kernel_event
            while evt:
                if self.event_to_category(evt) == "python_function" and evt[
                    "name"
                ].startswith("nn.Module:"):
                    nn_module_event = evt
                    break
                evt = self.tree.get_parent_event(evt)

        # add nn.Module details
        if nn_module_event and nn_module_detail:
            kernel_details["Parent nn.Module UID"] = nn_module_event["UID"]
            kernel_details["Parent nn.Module"] = nn_module_event["name"]
            if nn_module_event.get("gpu_busy_time") is None:
                # If the nn.Module event does not have GPU busy time, compute it.
                gpu_events = [
                    self.tree.get_UID2event(uid)
                    for uid in nn_module_event.get("gpu_events", [])
                ]
                nn_module_event["gpu_busy_time"] = GPUEventAnalyser(
                    gpu_events
                ).compute_metrics()["busy_time"]
            if nn_module_event.get("kernel_count") is None:
                # If the nn.Module event does not have kernel count, compute it.
                nn_module_event["kernel_count"] = len(
                    nn_module_event.get("gpu_events", [])
                )
            kernel_details["Parent nn.Module kernel count"] = nn_module_event.get(
                "kernel_count", 0
            )
            kernel_details["Parent nn.Module GPU busy time (µs)"] = nn_module_event.get(
                "gpu_busy_time"
            )
            if nn_module_event["kernel_count"] == 1:
                pct = 100
            else:
                pct = kernel_event["dur"] / nn_module_event["gpu_busy_time"] * 100
            kernel_details["Percent of Parent nn.Module busy time (%)"] = pct
        return kernel_details

    def get_df_kernels(
        self, launcher_detail=False, cpu_op_detail=True, nn_module_detail=False
    ):
        """
        Build a DataFrame with kernel details augmented with
        additional information such as launcher, CPU operation,
        and nn.Module details.
        Args:
            launcher_detail (bool): If True, include details of the kernel's launcher.
            cpu_op_detail (bool): If True, include details of the parent CPU operation.
            nn_module_detail (bool): If True, include details of the parent nn.Module event.

        Returns:
            pd.DataFrame: A DataFrame containing detailed kernel information and aggregated metrics.
        """
        if self.with_python_stack:
            raise ValueError(
                "This method does not support traces with Python stack events at the moment."
            )
        kernel_details_list = []

        # Extract details for all kernel events.
        for event in self.tree.events:
            if self.event_to_category(event) != "kernel":
                continue
            details = self.get_kernel_details(
                event,
                launcher_detail=launcher_detail,
                cpu_op_detail=cpu_op_detail,
                nn_module_detail=nn_module_detail,
            )
            kernel_details_list.append(details)

        df_kernel_view = pd.DataFrame(kernel_details_list)
        for col in df_kernel_view.columns:
            if "UID" in col or "count" in col:
                df_kernel_view[col] = df_kernel_view[col].astype("Int64")
        df_kernel_view.reset_index(drop=True, inplace=True)
        return df_kernel_view

    def build_nn_module_latency_tree(self, root_nn_module: Dict[str, Any]):
        """
        Compute the GPU time metrics for a subtree of nn.Module events rooted at the provided event.
        We populate the nn.Module events with the following metrics:
        - 'GPU Time': the total GPU busy time of the subtree rooted at the nn.Module event.
        - 'nn Parent GPU Time': the total GPU busy time of the parent nn.Module event.
        - 'Non-nn.Module GPU Time': the GPU busy time not attributed to nn.Module children if any.

        """
        if not self.add_python_func:
            raise ValueError(
                "This method requires the trace to include Python function events."
            )
        if not self.tree._is_nn_module_event(root_nn_module):
            raise ValueError("The provided root event is not an nn.Module event.")
        self._build_nn_modules_subtree_recursive(root_nn_module)

    def _build_nn_modules_subtree_recursive(
        self, node: Dict[str, Any], parent_gpu_time=None
    ):
        gpu_events_subtree_UIDs = node.get("gpu_events", [])
        gpu_events_subtree = [
            self.tree.get_UID2event(uid) for uid in gpu_events_subtree_UIDs
        ]
        gpu_time = GPUEventAnalyser(gpu_events_subtree).compute_metrics()["busy_time"]
        node["GPU Time"] = gpu_time
        node["nn Parent GPU Time"] = parent_gpu_time

        # nn_module_children = node.get('nn_module_children', [])
        nn_module_children = self.tree.get_nn_module_children(node)
        if not nn_module_children:
            return

        for i, child_UID in enumerate(nn_module_children):
            child = self.tree.get_UID2event(child_UID)
            self._build_nn_modules_subtree_recursive(child, parent_gpu_time=gpu_time)

        # Account for GPU time not attributed to nn.Module children.
        union_gpu_events_childrenUIDs = set()
        for child_UID in nn_module_children:
            union_gpu_events_childrenUIDs.update(
                self.tree.get_UID2event(child_UID).get("gpu_events", [])
            )
        remaining_gpu_events_UIDs = (
            set(gpu_events_subtree_UIDs) - union_gpu_events_childrenUIDs
        )
        if remaining_gpu_events_UIDs:
            gpu_events_remaining = [
                self.tree.get_UID2event(uid) for uid in remaining_gpu_events_UIDs
            ]
            gpu_time_remaining = GPUEventAnalyser(
                gpu_events_remaining
            ).compute_metrics()["busy_time"]
            node["Non-nn.Module GPU Time"] = gpu_time_remaining
        return


class JaxTreePerfAnalyzer(TreePerfAnalyzer):
    """
    JaxPerfAnalyser is a specialized performance analyser for JAX traces.
    It extends the TreePerfAnalyzer to provide JAX-specific performance analysis features.
    This class is designed to work with JAX traces and provides methods to analyze
    GPU events, categorize events, and compute performance metrics.

    Jax GPU event analyser supports multiple GPUs. Legacy of TreePerf/jax_analyses.py
    """

    @staticmethod
    def from_file(profile_filepath, *args, **kwargs) -> "JaxTreePerfAnalyzer":
        data = DataLoader.load_data(profile_filepath)
        data_pb = data["traceEvents"]
        categorizer = TraceEventUtils.prepare_event_categorizer(data_pb)
        events = TraceEventUtils.non_metadata_events(data_pb)
        linking_key = "correlation_id"
        tree = JaxTraceToTree(
            events, linking_key=linking_key, event_to_category=categorizer
        )
        return JaxTreePerfAnalyzer(
            tree,
            event_to_category=categorizer,
            pb_file_name=profile_filepath,
            *args,
            **kwargs,
        )

    def __init__(
        self,
        tree: JaxTraceToTree,
        event_to_category: Callable[[dict], str] = TraceEventUtils.default_categorizer,
        pb_file_name=None,
        arch=None,
        python_path=None,
        kernel_metadata_keyword_filters: list[str] = None,
    ):
        # super.__init__(*args, **kwargs)
        self.tree = tree
        self.arch = arch
        self.python_path = python_path
        self.event_to_category = event_to_category
        self.pb_file_name = pb_file_name
        self.arch = arch
        self.tree.build_tree(pb_file_name=pb_file_name)
        self.gpu_event_filter = JaxAnalyses.default_gpu_event_filter
        self.gpu_event_analyser = JaxGPUEventAnalyser(self.tree.events)
        self.jax_op_to_perf_model_class_map = jax_op_to_perf_model_class_map
        self.kernel_metadata_keyword_filters = kernel_metadata_keyword_filters

    #####################################
    ## Parsers for JaxTree Event Metadata
    #####################################

    @staticmethod
    def get_event_metadata(
        event,
        args_cols=["Input Dims", "Input type", "Input Strides", "Concrete Inputs"],
    ):
        """
        Parse jax event metadata to get perf model class name, input dims, input types for kernels.

        Input: JaxTree.event.
        Output: dictionary for event metadata: dims, type, etc.

        Example GEMM:
        'metadata': {'output': '(bf16[67320,3072]{1,0},s8[4194304]{0})',
                    'operands': ['bf16[67320,12288]{1,0}', 'bf16[12288,3072]{0,1}'],
                    'computation': 'gemm',
                    ...
                    "lhs_contracting_dimensions":["1"],
                    "rhs_contracting_dimensions":["0"],

        Returns
        dict = {
            'Input Dims' : ((67320,12288), (12288,3072))
            'Input type' : ('bf16', 'bf16')
            'Input indices' : ((1,0), (0,1))
            'M' : 67320
            'N' : 3072
            'K' : 12288
            ...}
        """

        # initialize dict
        dict_metadata = {}
        for _key in args_cols:
            dict_metadata[_key] = ()
        perf_model_name = JaxTreePerfAnalyzer.get_event_perf_model_name(event)
        if "gemm" in perf_model_name:
            _dict_gemm_meta = JaxTreePerfAnalyzer.parse_gemm_metadata(event)
            _dict_jax_gemm = JaxTreePerfAnalyzer.parse_JaxGemm_metadata(event)
            _dict = _dict_gemm_meta | _dict_jax_gemm
        elif "te_fused_attn" in perf_model_name:
            _dict = JaxTreePerfAnalyzer.parse_te_fused_attn_metadata(event)
        elif "conv" in perf_model_name:
            _dict = JaxTreePerfAnalyzer.parse_conv_metadata(event)
        else:
            # print('Use default parser for event', perf_model_name, event['gpu_kernel_op_cat'])
            _dict = JaxTreePerfAnalyzer.parse_metadata(event)
        if _dict:
            dict_metadata.update(_dict)
        return dict_metadata

    @staticmethod
    def get_event_perf_model_name(event):
        """
        Get perf model class name based on operands shape and 'custom_call_target' in metadata.

        Similar to event['computation']. e.g. 'gemm' for events when 'cublas' or 'matmul' in 'custom_call_target'.
        """
        gemm_keys = ["matmul", "cublas"]  # Used in JaxTrace2Tree
        te_fused_attn_keys = [
            "te_fused_attn_forward_ffi",
        ]
        te_fused_attn_bwd_keys = [
            "te_fused_attn_backward_ffi",
        ]
        conv_keys = ["cudnn$convForward", "cudnn$convBiasActivationForward"]
        conv_bwd_keys = [
            "cudnn$convBackward",
        ]
        _operands = event.get("metadata", {}).get("operands", None)
        _event_custom_call = event.get("metadata", {}).get("custom_call_target", None)
        if _operands and _event_custom_call:
            if (
                any(k in _event_custom_call for k in gemm_keys)
                and event["gpu_kernel_op_cat"].lower() == "gemm"
            ):
                return "jax_gemm"
            elif any(k in _event_custom_call for k in te_fused_attn_keys):
                return "jax_te_fused_attn"
            elif any(k in _event_custom_call for k in te_fused_attn_bwd_keys):
                return "jax_te_fused_attn_bwd"
            elif (
                any(k in _event_custom_call for k in conv_keys)
                and event["gpu_kernel_op_cat"].lower() == "conv"
            ):
                return "jax_conv"
            elif (
                any(k in _event_custom_call for k in conv_bwd_keys)
                and event["gpu_kernel_op_cat"].lower() == "conv"
            ):
                return "jax_conv_bwd"
            else:
                return "rest"  # TODO: PerfModel: 'jax_' + event['gpu_kernel_op_cat']
        else:
            return "rest"

    @staticmethod
    def parse_operands(event, metadata_key="operands"):
        """
        Example:
        # event[12540] Cijk_Alik_Bljk_BBS_BH_Bias_HAS_SAV_
        # 'metadata': {'output': '(bf16[67320,3072]{1,0},s8[4194304]{0})',
        # 'operands': ['bf16[67320,12288]{1,0}', 'bf16[3072,12288]{1,0}'],
        # ... }
        """
        operand_list = ()
        operand_type = ()
        operand_idx = ()
        operands = event["metadata"].get(metadata_key, None)
        if metadata_key == "output":
            operands = [
                operands,
            ]
        assert isinstance(
            operands, list
        )  # filter out incomplete metadata field in JaxTree
        operands = list(
            filter(None, operands)
        )  # filter out empty strings in list e.g. ['']
        try:
            if len(operands) > 0:
                for _operand in operands:
                    # Debug example: ['bf16[8,768]{1,0}', 'bf16[8,384]{1,0}', 'fusion,pred[1]{0}', 's32[8]{0}']
                    # JAX data types: ['f32', 'f64', 'f16', 'bf16', 'f8', 'fp8']
                    _pattern = r"([A-Za-z]+[0-9]+)\[([0-9,]+)\]\{([0-9,]+)\}"  # (type)[(dim)]{(_idx)}
                    _op = re.findall(_pattern, _operand)
                    if len(_op) > 0:
                        _type, _dim, _idx = _op[0]
                        _operand_dim = tuple(
                            int(_dim) for _dim in _dim.split(",") if _dim
                        )
                        _operand_idx = tuple(int(_id) for _id in _idx.split(",") if _id)
                        operand_type += (_type,)
                        operand_list += (_operand_dim,)
                        operand_idx += (_operand_idx,)
        except Exception as e:
            logger.debug(f"\nException occurred when parsing Event: \n\n {event} \n\
                            Event metadata: {event['metadata']}, operands: {operands}")
            raise ValueError(
                f"{e} Exception occurred when parsing Event operands: \n\n {operands}"
            )
        return operand_list, operand_type, operand_idx

    @staticmethod
    def parse_metadata(event):
        dict_metadata = {}
        if event.get("metadata", {}).get("operands", None):
            operand_list, operand_type, _ = JaxTreePerfAnalyzer.parse_operands(event)
            dict_metadata["Input Dims"] = operand_list
            dict_metadata["Input type"] = operand_type
        return dict_metadata

    @staticmethod
    def parse_conv_metadata(event, bwd=False):
        """
        Source: /home/guangphu/perf-profiling/tutorials/jax_conv_profiling.py

        Example:
        # Parameters for the 3D convolution
        # batch_size = 1 * jax.local_device_count()
        time_dim = 32       # 5120 # 32
        height = 60         # 32 # 60
        width = 104         # 30 # 104
        in_channels = 16    # 52 # 16
        out_channels = 5120 # 104 # 5120
        # dtype = jax.numpy.bfloat16

        # Kernel parameters
        kernel_t = 1
        kernel_h = 2
        kernel_w = 2
        stride = (1, 2, 2)

        conv_events[0]
        # 'output': '(bf16[1,5120,34,31,53]{4,3,2,1,0},u8[7150336]{0})',
        # 'operands': ['bf16[1,16,32,60,104]{4,3,2,1,0}', 'bf16[5120,16,1,2,2]{4,3,2,1,0}']

        """
        dict_metadata = {}
        perf_model_name = JaxTreePerfAnalyzer.get_event_perf_model_name(event)
        bwd = "bwd" in perf_model_name
        if event.get("metadata", {}).get("operands", None):
            operand_list, operand_types, _ = JaxTreePerfAnalyzer.parse_operands(event)
            dict_metadata["Input Dims"] = operand_list
            dict_metadata["Input type"] = operand_types
            output_list, _, _ = JaxTreePerfAnalyzer.parse_operands(
                event, metadata_key="output"
            )
            dict_metadata["Output Dims"] = output_list
            dict_metadata["Filter Shape"] = operand_list[1][2:]
            if bwd:
                dict_metadata["Filter Shape"] = output_list[0][2:]
        return dict_metadata

    @staticmethod
    def parse_te_fused_attn_metadata(event):
        """
        Ref:
         - https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/c/fused_attn.html#nvte_fused_attn_fwd
         - https://github.com/ROCm/TransformerEngine/blob/a1e66aae34e023070c04f9e46fe75bf947f207e1/transformer_engine/common/include/transformer_engine/fused_attn.h#L326
         - https://github.com/ROCm/TransformerEngine/blob/a1e66aae34e023070c04f9e46fe75bf947f207e1/transformer_engine/common/fused_attn_rocm/fused_attn.cpp#L775


        nvte_fused_attn_fwd, _bwd
        qkv layout | bias | mask | dropout |  sequence length  | head_dim

        Example:
        Hunyuan video
            - Attention  Heads 24
            - Head dim 128
            - fwd operands: ['bf16[1,67576,24,128]{3,2,1,0}', 'bf16[1,67576,24,128]{3,2,1,0}', 'bf16[1,67576,24,128]{3,2,1,0}', 'bf16[0]{0}', 's32[2]{0}', 's32[2]{0}', 'bf16[0]{0}', 'bf16[0]{0}']
            - bwd operands: ['bf16[1,67576,24,128]{3,2,1,0}', 'bf16[1,67576,24,128]{3,2,1,0}', 'bf16[1,67576,24,128]{3,2,1,0}', 'bf16[0]{0}', 'f32[1,24,67576,1]{3,2,1,0}', 'bf16[1,67576,24,128]{3,2,1,0}', 'bf16[1,67576,24,128]{3,2,1,0}', 's32[2]{0}', 's32[2]{0}', 'bf16[0]{0}', 'bf16[0]{0}']
        """

        dict_metadata = {}
        if event.get("metadata", {}).get("operands", None):
            operand_list, operand_type, _ = JaxTreePerfAnalyzer.parse_operands(event)
            dict_metadata["Input Dims"] = operand_list[:3]
            dict_metadata["Input type"] = operand_type[:3]
            dict_metadata["Concrete Inputs"] = operand_list[3:4]  # bias
        return dict_metadata

    @staticmethod
    def parse_gemm_metadata(event):
        """
        Ideally it would output the same as parse_JaxGemm_metadata(event).
        """
        backend_config = event.get("metadata", {}).get("backend_config", None)
        if backend_config is None:
            beta = 0
            raise ValueError("Backend config information missing!", event["metadata"])
        else:
            dict_backend_config = json.loads(
                backend_config.split("=")[1]
            )  # Note: missing '}' in some jax metadata
            beta = dict_backend_config.get("gemm_backend_config", {}).get("beta", 0)
        operand_list, operand_type, operand_idx = JaxTreePerfAnalyzer.parse_operands(
            event
        )
        output_list, _, output_idx = JaxTreePerfAnalyzer.parse_operands(
            event, metadata_key="output"
        )
        dict_metadata = {}
        dict_metadata["Input Dims"] = operand_list
        dict_metadata["Input type"] = operand_type
        dict_metadata["Input indices"] = operand_idx
        dict_metadata["Output Dims"] = output_list
        dict_metadata["Output indices"] = output_idx
        dict_metadata["Beta"] = beta
        return dict_metadata

    @staticmethod
    def parse_JaxGemm_metadata(event):
        """
        JaxAnalyses uses JaxProfileProcessor.process_gemm_ops to parse event metadata.
        The output was previously used for JaxAnalyses JaxGemm model class.
        It is reused here for consistency.
        Replicating the function is also possile with parse_gemm_metadata via manipulating the operands, output, and contracting dims.

        Return:
        gemm_dict = { "Batch": int(batch),
                    "M": int(m),
                    "N": int(n),
                    "K": int(k),
                    "Beta": int(beta),
                    "Type": op["type"],
                    "Computation": "gemm",
                    }

        Usage: gemm_dict = JaxTreePerfAnalyzer.parse_JaxGemm_metadata(event) # JaxAnalyses for JaxGemm
        """
        _dict_hlo_op = {"op_name": event["metadata"]}
        gemm_dict = JaxProfileProcessor.process_gemm_ops(_dict_hlo_op).get(
            "op_name", None
        )
        return gemm_dict

    ##############
    ## GPU metrics
    ##############
    def get_df_gpu_timeline(self, gpu_pid=None):
        return self.gpu_event_analyser.get_breakdown_df(
            gpu_pid=gpu_pid, event_filter=self.gpu_event_filter
        )

    def get_df_gpu_events_averages(self, gpu_pid=None):
        return self.gpu_event_analyser.get_average_df(
            gpu_pid=gpu_pid, event_filter=self.gpu_event_filter
        )

    #################
    ## Kernel metrics
    #################
    def get_kernel_launchers(self, gpu_pid=None, gpu_kernel_op_cats=None):
        kernel_launchers = []
        # filter out event op cats
        kernel_events = [
            event for event in self.tree.events if event["cat"] == "kernel"
        ]
        # filter out gpu kernel op cats
        if gpu_kernel_op_cats:
            kernel_events = [
                event
                for event in kernel_events
                if event["gpu_kernel_op_cat"] in gpu_kernel_op_cats
            ]
        if len(kernel_events) == 0:
            logger.warning(
                "Input list of events is empty. Returning an empty DataFrame."
            )
            return pd.DataFrame()
        for event in kernel_events:
            event["op category"] = event["gpu_kernel_op_cat"]
            event["total_direct_kernel_time"] = event["dur"]
            event["direct_kernel_count"] = int(1)
            # Note: 'dur' in 'kernel_details' is required from tree perf.
            event["kernel_details"] = [
                {
                    "name": event["name"],
                    "dur": event["dur"],
                    "custom_call_target": event.get("metadata", {}).get(
                        "custom_call_target", "NA"
                    ),
                    "operands": event.get("metadata", {}).get("operands", "NA"),
                    "outputs": event.get("metadata", {}).get("outputs", "NA"),
                    "metadata": event.get("metadata", {}).get("metadata", "NA"),
                }
            ]
            event["perf_model_name"] = JaxTreePerfAnalyzer.get_event_perf_model_name(
                event
            )
            dict_jax_metadata = JaxTreePerfAnalyzer.get_event_metadata(event)
            for _key, _val in dict_jax_metadata.items():
                event["args"][_key] = _val
            kernel_launchers.append(event)

        if gpu_pid:
            return [event for event in kernel_launchers if event["pid"] == int(gpu_pid)]
        else:
            return kernel_launchers

    def get_df_xla_perf(self, df_xla_events: pd.DataFrame) -> pd.DataFrame:

        dtype_to_bytes = {
            "f32": 4,
            "bf16": 2,
            "s32": 4,
            "fp16": 2,
            "u32": 4,
            "f16": 2,
            "u64": 8,
        }

        def parse_dtype_shape_layout(operand):
            # Match dtype, shape, and layout
            match = re.match(r"(\w+)\[([0-9,]*)\](?:\{([0-9,]*)\})?", operand)
            if match:
                dtype = match.group(1)
                shape_str = match.group(2)
                layout_str = match.group(3)
                shape = [int(x) for x in shape_str.split(",") if x]
                layout = [int(x) for x in layout_str.split(",")] if layout_str else None
                return dtype, shape, layout
            return None, None, None

        total_input_bytes_list = []
        for index, row in df_xla_events.iterrows():

            kernel_details = row.get("kernel_details")[0]
            operands = kernel_details.get("operands")

            total_input_bytes = 0
            for operand in operands:
                dtype, shape, layout = parse_dtype_shape_layout(operand)
                if shape and dtype:
                    total_input_bytes = (
                        total_input_bytes + np.prod(shape) * dtype_to_bytes[dtype]
                    )

            total_input_bytes_list.append(total_input_bytes)

        df_xla_events["total_input_bytes"] = total_input_bytes_list

        return df_xla_events

    def get_GPU_kernel_launch_latency(self, event: dict) -> float:

        GPU_kernel_launch_latency = event.get("ts") - self.tree.events_by_uid[
            event.get("parent")
        ].get("ts")

        return GPU_kernel_launch_latency

    def get_df_kernel_launchers(
        self,
        id_cols=True,
        gpu_pid=None,
        gpu_kernel_op_cats=None,
        include_kernel_details=False,
        include_args=True,
        args_cols=["Input Dims", "Input type", "Input Strides", "Concrete Inputs"],
    ):
        kernel_launchers = self.get_kernel_launchers(
            gpu_pid=gpu_pid, gpu_kernel_op_cats=gpu_kernel_op_cats
        )
        rows = []
        for event in kernel_launchers:
            metrics_event = {
                "name": event["name"],
                "UID": event["UID"],
                "op category": event["gpu_kernel_op_cat"],
                "total_direct_kernel_time": event["total_direct_kernel_time"],
                "direct_kernel_count": event["direct_kernel_count"],
            }
            if id_cols:
                metrics_event["pid"] = event["pid"]
                metrics_event["tid"] = event["tid"]
            if include_args:
                metrics_event.update((arg, event["args"].get(arg)) for arg in args_cols)
            if include_kernel_details:
                metrics_event["kernel_details"] = event["kernel_details"]

            metrics_event["GPU_kernel_launch_latency"] = (
                self.get_GPU_kernel_launch_latency(event)
            )

            metadata = event.get("metadata")

            if self.kernel_metadata_keyword_filters is not None:
                if metadata:
                    metadata = metadata.get("metadata", "")
                    if any(
                        kernel_metadata_keyword_filter in metadata
                        for kernel_metadata_keyword_filter in self.kernel_metadata_keyword_filters
                    ):
                        rows.append(metrics_event)
            else:
                rows.append(metrics_event)

        df = pd.DataFrame(rows)
        return df

    @staticmethod
    def get_df_kernel_launchers_summary(df_kernel_launchers):
        if df_kernel_launchers.empty:
            logger.warning("Input Dataframe is empty.")
            return pd.DataFrame()
        df = TreePerfAnalyzer.get_df_kernel_launchers_summary(df_kernel_launchers)
        num_gpus = df_kernel_launchers["pid"].nunique()
        df["time ms per gpu"] = df["total_direct_kernel_time_ms"] / num_gpus
        return df

    @staticmethod
    def get_df_kernel_launchers_summary_by_category(df_kernel_launchers):
        if df_kernel_launchers.empty:
            logger.warning("Input Dataframe is empty.")
            return pd.DataFrame()
        num_gpus = df_kernel_launchers["pid"].nunique()
        df = TreePerfAnalyzer.get_df_kernel_launchers_summary_by_category(
            df_kernel_launchers
        )
        df["time ms per gpu"] = df["total_direct_kernel_time_ms"] / num_gpus
        return df

    #############
    ## OP metrics
    #############
    def compute_perf_metrics(self, event, bwd=False):
        list_warn_non_zero_flops_and_zero_time = []
        list_warn_perf_metrics_failed = []
        list_no_bwd_events = []
        # Select the appropriate dictionary for FLOPS and memory functions
        perf_model_name = JaxTreePerfAnalyzer.get_event_perf_model_name(event)
        perf_model_class = self.jax_op_to_perf_model_class_map.get(
            perf_model_name, None
        )
        if perf_model_class is None:
            logger.warning(f"\nPerf model is not implemented. \n\nEvent: {event}")
            return dict()
        perf_model = perf_model_class(
            event, arch=self.arch, python_path=self.python_path
        )

        gflops = (perf_model.flops() if not bwd else perf_model.flops_bwd()) / 1e9
        busy_kernel_time = event[TraceEventUtils.TraceKeys.Duration]

        tflops_per_s = (
            (gflops / 1e3) / (busy_kernel_time / 1e6)
            if busy_kernel_time > 0
            else float("nan")
        )

        bytes_moved = perf_model.bytes() if not bwd else perf_model.bytes_bwd()

        dict_metrics = {
            "GFLOPS": gflops,
            "Kernel Time (µs)": busy_kernel_time,
            "TFLOPS/s": tflops_per_s,
        }
        if bytes_moved is not None:
            dict_metrics["Data Moved (MB)"] = bytes_moved / (1024 * 1024)
            dict_metrics["FLOPS/Byte"] = (
                (gflops * 1e9) / bytes_moved if bytes_moved > 0 else float("nan")
            )
            dict_metrics["TB/s"] = (
                (bytes_moved / 1e12) / (busy_kernel_time / 1e6)
                if busy_kernel_time > 0
                else float("nan")
            )
        else:
            dict_metrics["Data Moved (MB)"] = float("nan")
            dict_metrics["FLOPS/Byte"] = float("nan")
            dict_metrics["TB/s"] = float("nan")

        # JaxGemm
        if hasattr(perf_model, "simulation_time"):
            dict_metrics["Simulation Time (µs)"] = perf_model.simulation_time
            dict_metrics["Simulation TFLOPS/s"] = (
                (gflops / 1e3) / (perf_model.simulation_time / 1e6)
                if perf_model.simulation_time > 0
                else float("nan")
            )

        if hasattr(perf_model, "get_simulation_time") and not bwd:
            # This is for the case where we have a simulated time
            # for the forward pass, but not for the backward pass
            simulated_time = perf_model.get_simulation_time()
            if simulated_time:
                dict_metrics["Simulated Time (µs)"] = simulated_time
                dict_metrics["Simulated TFLOPS/s"] = (
                    (gflops / 1e3) / (simulated_time / 1e6)
                    if simulated_time > 0
                    else float("nan")
                )

        if hasattr(perf_model, "get_simulation_time_bwd") and bwd:
            # This is for the case where we have a simulated time
            # for the forward pass, but not for the backward pass
            simulated_time = perf_model.get_simulation_time_bwd()
            if simulated_time:
                dict_metrics["Simulated Time (µs)"] = simulated_time
                dict_metrics["Simulated TFLOPS/s"] = (
                    (gflops / 1e3) / (simulated_time / 1e6)
                    if simulated_time > 0
                    else float("nan")
                )

        for key, value in perf_model.param_details.items():
            dict_metrics[f"param: {key}"] = value

        return dict_metrics

    def build_df_perf_metrics(
        self,
        events,
        include_kernel_details=False,
        include_args=False,
        args_cols=["Input Dims", "Input type"],
    ):
        rows = []
        list_warn_non_zero_flops_and_zero_time = []
        list_warn_perf_metrics_failed = []
        list_no_bwd_events = []
        for event in events:
            # update event metadata, required for perf model: perf_model_name, kernel names, args['Input Dims']
            event["kernel_details"] = [
                {
                    "name": event["name"],
                    "dur": event["dur"],
                }
            ]
            perf_model_name = JaxTreePerfAnalyzer.get_event_perf_model_name(event)
            dict_jax_metadata = JaxTreePerfAnalyzer.get_event_metadata(event)
            for _key, _val in dict_jax_metadata.items():
                event["args"][_key] = _val
            dict_perf_metrics = None
            if not perf_model_name == "rest":
                try:
                    bwd = perf_model_name.endswith("_bwd")
                    dict_perf_metrics = self.compute_perf_metrics(event, bwd=bwd)
                except Exception as e:
                    list_warn_perf_metrics_failed.append(event)
                    logger.debug(
                        f"\nException occurred when computing perf metrics for Event: \n\n {event}"
                    )
                    raise ValueError(
                        f"\n{e} Exception occurred when computing perf metrics for Event: \n\n {event}"
                    )
            if dict_perf_metrics is not None:
                metrics_event = {
                    "name": event["name"],
                    "UID": event["UID"],
                    "pid": event["pid"],
                    "process_name": self.tree.metadata.get(event["pid"], {})
                    .get(0, {})
                    .get("process_name", "Unknown"),
                    "process_label": self.tree.metadata.get(event["pid"], {})
                    .get(0, {})
                    .get("process_labels", "Unknown"),
                    "thread_name": self.tree.metadata.get(event["pid"], {})
                    .get(event["tid"], {})
                    .get("thread_name", "Unknown"),
                    "dur": event["dur"],
                    "cat": event["cat"],
                    "op category": event["gpu_kernel_op_cat"],
                    "perf model": perf_model_name,
                }
                metrics_event.update(dict_perf_metrics)
                if (
                    dict_perf_metrics["GFLOPS"] > 0
                    and dict_perf_metrics["Kernel Time (µs)"] == 0
                ):
                    list_warn_non_zero_flops_and_zero_time.append(event)
                if include_args:
                    metrics_event.update(
                        (arg, event["args"].get(arg, None)) for arg in args_cols
                    )
                if include_kernel_details:
                    metrics_event["kernel_details"] = event["kernel_details"]
                rows.append(metrics_event)

        self._show_warnings(
            list_warn_non_zero_flops_and_zero_time,
            list_no_bwd_events,
            list_warn_perf_metrics_failed,
            len(events),
        )
        df_perf_metrics = pd.DataFrame(rows)

        return df_perf_metrics
