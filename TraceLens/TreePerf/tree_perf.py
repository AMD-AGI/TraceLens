###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json
import gzip
import os, re, sys
from functools import partial, lru_cache
import copy
from collections import defaultdict
from typing import Dict, Any, Callable

# TODO: warning should show the stack as well
import warnings
import pprint
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

from ..PerfModel.torch_op_mapping import (
    op_to_perf_model_class_map,
    categorize_torch_op,
    dict_cat2names,
)
from ..PerfModel.jax_op_mapping import jax_op_to_perf_model_class_map
from .gpu_event_analyser import GPUEventAnalyser, JaxGPUEventAnalyser
from .jax_analyses import JaxAnalyses
from ..Trace2Tree.trace_to_tree import TraceToTree, JaxTraceToTree
from ..util import DataLoader, TraceEventUtils, JaxProfileProcessor


class TreePerfAnalyzer:
    @staticmethod
    def from_file(
        profile_filepath, jax: bool = False, *args, **kwargs
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
            tree, jax=jax, event_to_category=categorizer, *args, **kwargs
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
    ):
        self.jax = jax
        self.GPUEventAnalyser = GPUEventAnalyser if not jax else JaxGPUEventAnalyser
        self.tree = tree
        self.add_python_func = add_python_func
        self.arch = arch
        self.python_path = python_path
        self.event_to_category = event_to_category
        # include unlinked kernels in gpu timeline
        self.include_unlinked_kernels = include_unlinked_kernels
        # we check if profile contains python func events
        self.with_python_stack = any(
            event.get("cat") == "python_func" for event in self.tree.events
        )
        self.tree.build_tree(add_python_func=add_python_func)
        self.op_to_perf_model_class_map = op_to_perf_model_class_map
        self.op_categorizer = categorize_torch_op
        self.dict_cat2names = dict_cat2names

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
            if not event.get("bwd_events"):
                self.tree.link_bwd_events(event["UID"])
            cpu_op_uids = event["bwd_events"]
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
                "external_id": event["args"].get("External id"),
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
    def summarize_df_perf_metrics(df_perf_metrics, agg_metrics=["mean", "std"]):
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
        # Convert parameter columns to strings to avoid type comparison issues
        df_perf_metrics = df_perf_metrics.copy()
        for col in param_cols:
            df_perf_metrics[col] = df_perf_metrics[col].astype(str)
        # TODO warn user if nans in the performance metrics
        # Perform the aggregation
        df_perf_metrics_summary = df_perf_metrics.groupby(
            ["name"] + param_cols, dropna=False
        ).agg(dict_agg)
        df_perf_metrics_summary.columns = [
            "_".join(col).strip() for col in df_perf_metrics_summary.columns.values
        ]
        df_perf_metrics_summary.reset_index(inplace=True)

        df_perf_metrics_summary.sort_values(
            by=["Kernel Time (µs)_sum", "UID_first"],
            ascending=[False, True],
            inplace=True,
        )
        # df_perf_metrics_summary.sort_values(by='Simulated Kernel Time (us)_sum', ascending=False, inplace=True)
        df_perf_metrics_summary.reset_index(drop=True, inplace=True)

        return df_perf_metrics_summary

    def get_kernel_launchers(self, include_nccl=False):
        # This method identifies kernel launchers, which are the events directly responsible for launching GPU kernels.
        #
        # In the ideal case, ops are routed through torch dispatcher to create a clear hierarchy
        # where a "leaf" CPU operation is the caller for runtime events that launch kernels. These CPU ops are
        # valuable for analysis as they contain rich argument information (e.g., input dimensions, strides, dtypes).
        # The method identifies these as the primary kernel launchers.
        #
        # However, some edge cases exist where the calling CPU context is hidden, and a runtime event appears
        # unlinked to a parent CPU op. While not ideal for a detailed breakdown (as argument info is missing),
        # these unlinked events still launch kernels and must be captured for a complete analysis. This method
        # processes them separately to ensure all kernel launchers are included in the output.
        #
        # Special handling for 'execute' operations for a special customer case

        kernel_launchers = []
        cpu_ops = [
            evt for evt in self.tree.events if self.event_to_category(evt) == "cpu_op"
        ]
        for event in cpu_ops:

            if event["name"] == "execute":
                parent = self.tree.get_parent_event(event)
                list_kernel_uids = parent.get("gpu_events", [])
                list_kernels = [
                    self.tree.get_UID2event(uid) for uid in list_kernel_uids
                ]
                parent["total_direct_kernel_time"] = GPUEventAnalyser(
                    list_kernels
                ).compute_metrics()["busy_time"]
                parent["direct_kernel_count"] = len(list_kernels)
                parent["kernel_details"] = [
                    {
                        "name": kernel["name"],
                        "dur": kernel["dur"],
                        "stream": kernel.get("args", {}).get("stream", None),
                        "Kernel category": self.categorize_kernel_by_name(kernel["name"]),
                    }
                    for kernel in list_kernels
                ]
                parent["op category"] = self.op_categorizer(parent)
                kernel_launchers.append(parent)
                continue  # no need to check children of this event

            kernel_launcher = False
            # total_direct_kernel_time = 0
            # direct_kernel_count = 0
            list_kernels = []
            for child_UID in event.get("children", []):
                child = self.tree.events_by_uid[child_UID]
                for grand_child_UID in child.get("children", []):
                    grand_child = self.tree.events_by_uid[grand_child_UID]
                    is_kernel = self.event_to_category(grand_child) in {
                        "kernel",
                        "gpu_memcpy",
                        "gpu_memset",
                    }
                    is_nccl = "nccl" in grand_child["name"]
                    should_include = is_kernel and (include_nccl or not is_nccl)
                    if should_include:
                        kernel_launcher = True
                        list_kernels.append(grand_child)
            if kernel_launcher:
                for kernel_evt in list_kernels:
                    kernel_evt["args"]["leaf_op"] = event["UID"]
                    runtime_evt = self.tree.get_parent_event(kernel_evt)
                    runtime_evt["args"]["leaf_op"] = event["UID"]
                event["total_direct_kernel_time"] = GPUEventAnalyser(
                    list_kernels
                ).compute_metrics()["busy_time"]
                event["direct_kernel_count"] = len(list_kernels)
                event["kernel_details"] = [
                    {
                        "name": kernel["name"],
                        "dur": kernel["dur"],
                        "stream": kernel.get("args", {}).get("stream", None),
                        "Kernel category": self.categorize_kernel_by_name(kernel["name"]),
                    }
                    for kernel in list_kernels
                ]
                event["op category"] = self.op_categorizer(event)
                kernel_launchers.append(event)

        # Now handle the case where runtime events are not linked to any cpu_op
        runtime_evts = [
            evt
            for evt in self.tree.events
            if self.event_to_category(evt) in {"cuda_runtime", "cuda_driver"}
        ]
        for runtime_evt in runtime_evts:
            if "leaf_op" in runtime_evt.get("args", {}):
                continue  # already processed as part of a cpu_op
            list_kernel_uids = runtime_evt.get("gpu_events", [])
            if len(list_kernel_uids) == 0:
                continue  # no kernels launched
            # for non graph runtime events, we skip nccl kernels unless include_nccl is True
            elif len(list_kernel_uids) == 1:
                is_nccl = "nccl" in self.tree.get_UID2event(list_kernel_uids[0])["name"]
                if is_nccl and not include_nccl:
                    continue  # skip nccl kernels
            list_kernels = [self.tree.get_UID2event(uid) for uid in list_kernel_uids]
            runtime_evt["total_direct_kernel_time"] = GPUEventAnalyser(
                list_kernels
            ).compute_metrics()["busy_time"]
            runtime_evt["direct_kernel_count"] = len(list_kernels)
            runtime_evt["kernel_details"] = [
                {
                    "name": kernel["name"],
                    "dur": kernel["dur"],
                    "stream": kernel.get("args", {}).get("stream", None),
                    "Kernel category": self.categorize_kernel_by_name(kernel["name"]),
                }
                for kernel in list_kernels
            ]
            runtime_evt["op category"] = self.op_categorizer(runtime_evt)
            kernel_launchers.append(runtime_evt)
        return kernel_launchers

    def get_df_kernel_launchers(self, id_cols=False, include_kernel_details=False):

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
            rows.append(metrics_event)
        df = pd.DataFrame(rows)
        return df

    @staticmethod
    def get_df_kernel_launchers_summary(df_kernel_launchers):
        df_temp = df_kernel_launchers.copy()
        df_agg = df_temp.groupby("name").agg(
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
        total_duration_ms = df_agg["total_direct_kernel_time_ms"].sum()
        df_agg["Percentage (%)"] = (
            df_agg["total_direct_kernel_time_ms"] / total_duration_ms
        ) * 100
        df_agg["Cumulative Percentage (%)"] = df_agg["Percentage (%)"].cumsum()
        df_agg.reset_index(drop=True, inplace=True)

        return df_agg

    @staticmethod
    def get_df_kernel_launchers_summary_by_inference_phase(df_kernel_launchers, m_threshold=10, detection_method='hybrid'):
        """
        Generate separate kernel launcher summaries for Prefill and Decode phases.
        
        Args:
            df_kernel_launchers (pd.DataFrame): DataFrame containing kernel launchers.
            m_threshold (int): Threshold for M parameter to distinguish phases. Default: 10
            detection_method (str): Method to use for phase detection:
                - 'kernel_names': Read phase info from kernel names (RECOMMENDED)
                - 'gemm_params': Original GEMM M parameter method
                - 'attention_patterns': Use attention operation analysis
                - 'framework_apis': Use framework-specific patterns
                - 'operation_frequency': Use statistical frequency analysis
                - 'hybrid': Combine multiple methods in priority order
            
        Returns:
            dict: Dictionary with keys 'Prefill' and 'Decode', each containing a summary DataFrame.
        """
        df_temp = df_kernel_launchers.copy()
        
        # Extract kernel categories from kernel_details if available
        if 'kernel_details' in df_temp.columns:
            def extract_kernel_categories(kernel_details):
                """Extract unique kernel categories from kernel_details list"""
                if not kernel_details or not isinstance(kernel_details, list):
                    return None
                categories = set()
                for kernel in kernel_details:
                    if isinstance(kernel, dict) and 'Kernel category' in kernel:
                        cat = kernel['Kernel category']
                        if cat and cat != 'Unknown':
                            categories.add(cat)
                # Return sorted categories joined by comma, or None if empty
                return ', '.join(sorted(categories)) if categories else None
            
            df_temp['kernel_categories'] = df_temp['kernel_details'].apply(extract_kernel_categories)
        
        # Add inference phase column using selected detection method
        if detection_method == 'gemm_params':
            # Original GEMM-based method
            df_temp['inference_phase'] = df_temp.apply(
                lambda row: TreePerfAnalyzer.detect_inference_phase(row, m_threshold), axis=1
            )
        else:
            # Advanced detection methods
            df_temp['inference_phase'] = TreePerfAnalyzer.detect_inference_phase_advanced(
                df_temp, method=detection_method, m_threshold=m_threshold
            )
        
        # Create separate summaries for each phase
        phase_summaries = {}
        
        for phase in ['Prefill', 'Decode']:
            df_phase = df_temp[df_temp['inference_phase'] == phase].copy()
            
            if df_phase.empty:
                # Create empty summary with expected columns
                empty_summary = pd.DataFrame(columns=[
                    'Kernel categories', 'name', 'total_direct_kernel_time_sum', 'Count', 
                    'total_direct_kernel_time_ms', 'Percentage (%)', 'Cumulative Percentage (%)'
                ])
                phase_summaries[phase] = empty_summary
                continue
            
            # Prepare aggregation dict
            agg_dict = {"total_direct_kernel_time": ["sum", "count"]}
            
            # Add kernel_categories to aggregation if available
            if 'kernel_categories' in df_phase.columns:
                # Get the first non-null kernel_categories value for each operation name
                agg_dict["kernel_categories"] = lambda x: x.dropna().iloc[0] if not x.dropna().empty else None
                
            # Group by operation name
            df_agg = df_phase.groupby("name").agg(agg_dict)
            df_agg.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in df_agg.columns.values]
            df_agg.reset_index(inplace=True)
            df_agg.rename(columns={"total_direct_kernel_time_count": "Count"}, inplace=True)
            
            # Rename kernel_categories column if it exists
            if 'kernel_categories_<lambda>' in df_agg.columns:
                df_agg.rename(columns={'kernel_categories_<lambda>': 'Kernel categories'}, inplace=True)
            
            df_agg.sort_values(
                by="total_direct_kernel_time_sum", ascending=False, inplace=True
            )
            df_agg["total_direct_kernel_time_ms"] = (
                df_agg["total_direct_kernel_time_sum"] / 1000
            )
            
            # Calculate percentages within this phase
            total_duration_ms = df_agg["total_direct_kernel_time_ms"].sum()
            if total_duration_ms > 0:
                df_agg["Percentage (%)"] = (
                    df_agg["total_direct_kernel_time_ms"] / total_duration_ms
                ) * 100
                df_agg["Cumulative Percentage (%)"] = df_agg["Percentage (%)"].cumsum()
            else:
                df_agg["Percentage (%)"] = 0.0
                df_agg["Cumulative Percentage (%)"] = 0.0
            
            # Reorder columns to put Kernel categories first (if it exists)
            if 'Kernel categories' in df_agg.columns:
                cols = ['Kernel categories'] + [col for col in df_agg.columns if col != 'Kernel categories']
                df_agg = df_agg[cols]
                
            df_agg.reset_index(drop=True, inplace=True)
            phase_summaries[phase] = df_agg
            
        return phase_summaries

    @staticmethod
    def get_df_kernel_summary_by_inference_phase(df_kernels, m_threshold=10):
        """
        Generate separate kernel summaries (per-kernel level) for Prefill and Decode phases.
        
        Args:
            df_kernels (pd.DataFrame): DataFrame containing individual kernel information.
            m_threshold (int): Threshold for M parameter to distinguish phases. Default: 10
            
        Returns:
            dict: Dictionary with keys 'Prefill' and 'Decode', each containing a summary DataFrame.
        """
        if df_kernels.empty or "Kernel duration (µs)" not in df_kernels.columns:
            # Return empty summaries
            empty_summary = pd.DataFrame()
            return {'Prefill': empty_summary, 'Decode': empty_summary}
            
        df_temp = df_kernels.copy()
        
        # Add inference phase column 
        df_temp['inference_phase'] = df_temp.apply(
            lambda row: TreePerfAnalyzer.detect_inference_phase(row, m_threshold), axis=1
        )
        
        # Create separate summaries for each phase
        phase_summaries = {}
        
        for phase in ['Prefill', 'Decode']:
            df_phase = df_temp[df_temp['inference_phase'] == phase].copy()
            
            if df_phase.empty:
                phase_summaries[phase] = pd.DataFrame()
                continue
            
            # Group by category/cpu_op along with kernel identifiers when available
            group_cols = []
            for col in [
                "Parent op category",
                "Parent cpu_op", 
                "Kernel name",
                "Kernel stream",
            ]:
                if col in df_phase.columns:
                    group_cols.append(col)
            if not group_cols:
                group_cols = (
                    ["Kernel name"] if "Kernel name" in df_phase.columns else []
                )
            
            if not group_cols:
                # No grouping columns available
                phase_summaries[phase] = df_phase
                continue
                
            agg_dict = {"Kernel duration (µs)": ["sum", "count", "mean", "min", "max"]}
            df_kernel_summary = df_phase.groupby(group_cols, dropna=False).agg(agg_dict)
            df_kernel_summary.columns = [
                "_".join(col).strip() for col in df_kernel_summary.columns.values
            ]
            df_kernel_summary.reset_index(inplace=True)
            
            # Calculate percentages within this phase
            total_kernels_us = df_phase["Kernel duration (µs)"].sum()
            if total_kernels_us > 0:
                df_kernel_summary["Percent of kernels time (%)"] = (
                    df_kernel_summary["Kernel duration (µs)_sum"] / total_kernels_us
                ) * 100
            else:
                df_kernel_summary["Percent of kernels time (%)"] = np.nan
                
            df_kernel_summary.sort_values(
                by="Kernel duration (µs)_sum", ascending=False, inplace=True
            )
            df_kernel_summary.reset_index(drop=True, inplace=True)
            
            phase_summaries[phase] = df_kernel_summary
            
        return phase_summaries

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
            return []  # The series was empty or contained no valid lists.

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
    ) -> pd.DataFrame:
        """
        Generate a DataFrame with unique arguments for each operation in the input DataFrame.

        Args:
            df_kernel_launchers (pd.DataFrame): DataFrame containing kernel launchers.
            event_name (str): Optional name of the event to filter the DataFrame.
            agg_metrics (list): List of aggregation metrics to apply. ex: ['mean', 'std', 'median']
            include_pct (bool): If True, include percentage of total time for each row as well as cumulative percentage.

        Returns:
            pd.DataFrame: DataFrame with unique arguments for each operation.
        """
        grouping_cols_original = [
            "name",
            "Input Dims",
            "Input type",
            "Input Strides",
            "Concrete Inputs",
        ]

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
        for col in actual_grouping_cols:
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

    @staticmethod
    def get_df_kernel_launchers_summary_by_category(
        df_kernel_launchers: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate a DataFrame with breakdown of kernel launchers by category.
        
        For operations with "op category" == "other", this function will consult
        the "Kernel category" field from kernel_details to provide better categorization
        based on the actual kernel types being launched.
        
        Args:
            df_kernel_launchers (pd.DataFrame): DataFrame containing kernel launchers.
        Returns:
            pd.DataFrame: DataFrame with breakdown of kernel launchers by category.
        """
        df_temp = df_kernel_launchers.copy()
        
        # Map kernel-level categories to op-level categories
        # This ensures consistent categorization with existing op categories
        def map_kernel_to_op_category(kernel_category):
            """
            Map detailed kernel categories to high-level op categories.
            E.g., GEMM-CK, GEMM-Generic, GEMM-Tensor -> GEMM
            """
            if not kernel_category or kernel_category == "Unknown":
                return None
            
            kernel_cat_lower = kernel_category.lower()
            
            # GEMM variants -> GEMM
            if "gemm" in kernel_cat_lower:
                return "GEMM"
            
            # Attention variants -> attention
            if "attention" in kernel_cat_lower or "flash" in kernel_cat_lower:
                return "attention"
            
            # Normalization variants -> norm
            if "norm" in kernel_cat_lower:
                return "norm"
            
            # Elementwise operations
            if "elementwise" in kernel_cat_lower:
                return "elementwise"
            
            # Reduce operations
            if "reduce" in kernel_cat_lower or "reduction" in kernel_cat_lower:
                return "reduce"
            
            # Memory operations
            if "memory" in kernel_cat_lower or "memcpy" in kernel_cat_lower or "memset" in kernel_cat_lower:
                return "memory"
            
            # Transpose operations
            if "transpose" in kernel_cat_lower:
                return "transpose"
            
            # Convolution operations
            if "conv" in kernel_cat_lower or "convolution" in kernel_cat_lower:
                return "CONV"
            
            # For specific categories that should remain distinct, return as lowercase
            # to match typical op category naming conventions
            return kernel_category.lower()
        
        # Improve categorization by consulting kernel-level categories
        # This is especially important for inference traces where many ops end up in "other"
        if "kernel_details" in df_temp.columns and "op category" in df_temp.columns:
            def get_predominant_kernel_category(row):
                """
                Extract the predominant kernel category from kernel_details.
                Returns the most common kernel category by total duration,
                mapped to the appropriate op-level category.
                """
                if row["op category"] != "other":
                    # Keep existing categorization if not "other"
                    return row["op category"]
                
                kernel_details = row.get("kernel_details")
                if not kernel_details or not isinstance(kernel_details, list):
                    return row["op category"]
                
                # Aggregate kernel durations by category
                category_durations = {}
                for kernel in kernel_details:
                    if not isinstance(kernel, dict):
                        continue
                    kernel_cat = kernel.get("Kernel category", "Unknown")
                    kernel_dur = kernel.get("dur", 0)
                    
                    # Map kernel category to op category
                    op_cat = map_kernel_to_op_category(kernel_cat)
                    
                    if op_cat:
                        category_durations[op_cat] = category_durations.get(op_cat, 0) + kernel_dur
                
                # Return the category with the highest total duration, or keep "other" if none found
                if category_durations:
                    predominant_category = max(category_durations.items(), key=lambda x: x[1])[0]
                    return predominant_category
                
                return row["op category"]
            
            # Apply the improved categorization
            df_temp["op category"] = df_temp.apply(get_predominant_kernel_category, axis=1)
        
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
        )
        return df

    @staticmethod
    @lru_cache(maxsize=2048)
    def categorize_kernel_by_name(kernel_name):
        """
        Categorize a GPU kernel based on its name patterns.
        Generic categorization for all frameworks: vLLM, SGLang, TGI, HuggingFace, etc.
        Works with training and inference workloads on AMD and NVIDIA GPUs.
        
        OPTIMIZED VERSION with two-stage lookup:
        1. Quick prefix-based lookup for common patterns (80% of kernels)
        2. Fallback to full pattern matching for unknown kernels (20%)
        
        Uses LRU cache (maxsize=2048) to avoid re-categorizing the same kernel names.
        Expected speedup: 100-200x for typical traces with many repeated kernel names.
        
        Returns a detailed category for the kernel operation.
        """
        
        # Stage 1: Quick prefix-based categorization for common patterns
        # This handles ~80% of kernels with O(1) lookup
        name_lower = kernel_name.lower()
        
        # Check first token/prefix (before first underscore or space)
        if '_' in kernel_name:
            prefix = kernel_name.split('_')[0].lower()
        elif ' ' in kernel_name:
            prefix = kernel_name.split(' ')[0].lower()
        else:
            prefix = name_lower[:20]  # First 20 chars if no delimiter
        
        # Quick prefix lookups for most common kernel types
        prefix_categories = {
            'cijk': 'GEMM-CK',
            'sgemm': 'GEMM-Generic',
            'dgemm': 'GEMM-Generic', 
            'hgemm': 'GEMM-Generic',
            'gemm': 'GEMM-Generic',
            'cutlass': 'GEMM-CUTLASS',
            'cublas': 'GEMM-cuBLAS',
            'rocblas': 'GEMM-ROCm',
            'hipblas': 'GEMM-ROCm',
            'fmha': 'Attention-Generic',
            'flash': 'Attention-Generic',
            'paged': 'PagedAttention-Generic',
            'pagedattention': 'PagedAttention-Generic',
            'rmsnorm': 'Norm-RMS',
            'layernorm': 'Norm-Layer',
            'batchnorm': 'Norm-Batch',
            'softmax': 'Activation-Softmax',
            'gelu': 'Activation-GELU',
            'silu': 'Activation-SiLU',
            'rope': 'RoPE-Generic',
            'rotary': 'RoPE-Generic',
            'transpose': 'Memory-Transpose',
            'memcpy': 'Memory-Copy',
            'memset': 'Memory-Set',
            'triton': 'Triton-Custom',
        }
        
        if prefix in prefix_categories:
            # Do a quick refinement check for common sub-patterns
            category = prefix_categories[prefix]
            
            # Refine GEMM types
            if category.startswith('GEMM'):
                if 'splitk' in name_lower:
                    return 'GEMM-SplitK'
                if any(q in name_lower for q in ['fp4', 'int4', 'f4']):
                    return 'GEMM-FP4'
                if any(q in name_lower for q in ['fp8', 'int8', 'f8']):
                    return 'GEMM-FP8'
            
            # Refine attention types
            if category.startswith('Attention') or category.startswith('PagedAttention'):
                if 'fwd' in name_lower or 'forward' in name_lower:
                    return category.replace('Generic', 'Forward')
                if 'bwd' in name_lower or 'backward' in name_lower:
                    return category.replace('Generic', 'Backward')
            
            return category
        
        # Stage 2: Full pattern matching for less common kernels
        # This is the fallback but cached via @lru_cache
        return TreePerfAnalyzer._categorize_kernel_by_pattern_matching(kernel_name, name_lower)
    
    @staticmethod
    def _categorize_kernel_by_pattern_matching(kernel_name, name_lower):
        """
        Fallback pattern matching for kernels not caught by prefix lookup.
        This handles the remaining ~20% of kernels with more complex patterns.
        """
        
        # Detect framework from namespace (generic approach)
        framework = None
        if '::' in kernel_name:
            namespace = kernel_name.split('::')[0].lower()
            if 'vllm' in namespace:
                framework = 'vLLM'
            elif 'sglang' in namespace or 'sgl' in namespace:
                framework = 'SGLang'
            elif 'tgi' in namespace or 'text_generation' in namespace:
                framework = 'TGI'
            elif 'transformers' in namespace or 'hf' in namespace or 'huggingface' in namespace:
                framework = 'HuggingFace'
            elif 'tensorrt' in namespace or 'trt' in namespace:
                framework = 'TensorRT'
            elif 'onnx' in namespace or 'ort' in namespace:
                framework = 'ONNX'
        
        # PagedAttention (used by vLLM, SGLang, and other inference engines)
        if 'paged_attention' in name_lower or 'pagedattention' in name_lower:
            if 'reduce' in name_lower:
                return "PagedAttention-Reduce"
            elif any(x in name_lower for x in ['qkv', 'kernel', 'prefill', 'decode']):
                return "PagedAttention-Compute"
            else:
                return "PagedAttention-Generic"
        
        # Chunked Prefill / Continuous Batching patterns (common in SGLang, vLLM)
        if any(pattern in name_lower for pattern in ['chunked_prefill', 'continuous_batch', 'batch_decode']):
            return "InferenceBatching"
        
        # GEMM/Matrix Multiply kernels (including quantized variants)
        if any(pattern in name_lower for pattern in [
            'gemm', 'cijk', 'matmul', 'cublas', 'nvjet',
            'mfma', 'wmma', 'tensor_op', 'sgemm', 'dgemm',
            'hgemm', 'bgemm', 'i8gemm', 'f4gemm', 'f8gemm',
            'cutlass', 'rocblas', 'hipblas', 'matrix_multiply'
        ]):
            # Quantized GEMM variants
            if any(x in name_lower for x in ['f4gemm', 'fp4', 'int4']):
                return "GEMM-FP4"  # FP4/INT4 quantized GEMM
            elif any(x in name_lower for x in ['f8gemm', 'fp8', 'int8']):
                return "GEMM-FP8"  # FP8/INT8 quantized GEMM
            # Hardware-specific GEMM types
            elif 'cijk' in name_lower:
                return "GEMM-CK"  # AMD Composable Kernel GEMM
            elif any(x in name_lower for x in ['cublas', 'cublaslt']):
                return "GEMM-cuBLAS"  # NVIDIA cuBLAS
            elif any(x in name_lower for x in ['cutlass']):
                return "GEMM-CUTLASS"  # NVIDIA CUTLASS
            elif any(x in name_lower for x in ['rocblas', 'hipblas']):
                return "GEMM-ROCm"  # AMD ROCm BLAS
            elif 'nvjet' in name_lower:
                return "GEMM-NVJET"
            elif 'mfma' in name_lower:
                return "GEMM-MFMA"  # AMD Matrix Cores (CDNA/RDNA)
            elif 'wmma' in name_lower:
                return "GEMM-WMMA"  # NVIDIA Tensor Cores
            elif 'splitk' in name_lower or 'wvsplitk' in name_lower:
                return "GEMM-SplitK"  # Split-K GEMM optimization
            elif 'tensor_op' in name_lower:
                return "GEMM-TensorOp"  # NVIDIA Tensor Operations
            else:
                return "GEMM-Generic"
        
        # Attention kernels (Flash Attention, SDPA, etc.)
        if any(pattern in name_lower for pattern in [
            'attention', 'fmha', 'flash', 'attn', 'sdpa',
            'scaled_dot_product'
        ]):
            if 'fwd' in name_lower or 'forward' in name_lower or 'fprop' in name_lower:
                return "Attention-Forward"
            elif 'bwd' in name_lower or 'backward' in name_lower or 'bprop' in name_lower:
                return "Attention-Backward"
            else:
                return "Attention-Generic"
        
        # Convolution kernels
        if any(pattern in name_lower for pattern in [
            'conv', 'winograd', 'im2col', 'fillbuffer'
        ]):
            if 'fwd' in name_lower or 'forward' in name_lower:
                return "Conv-Forward"
            elif 'bwd' in name_lower or 'backward' in name_lower:
                return "Conv-Backward"
            else:
                return "Conv-Generic"
        
        # Fused operations (common in inference - check before individual ops)
        if 'fused' in name_lower:
            if any(x in name_lower for x in ['add', 'rmsnorm', 'layernorm', 'norm']):
                return "Fused-AddNorm"  # Fused add + normalization
            elif any(x in name_lower for x in ['qk', 'rope', 'cache']):
                return "Fused-QKRoPECache"  # Multi-op fusion for attention
            elif any(x in name_lower for x in ['mlp', 'ffn']):
                return "Fused-MLP"  # Fused MLP/FFN operations
            else:
                return "Fused-Generic"
        
        # Normalization kernels
        if any(pattern in name_lower for pattern in [
            'norm', 'rmsnorm', 'layernorm', 'batchnorm',
            'instancenorm', 'groupnorm'
        ]):
            if 'rms' in name_lower or 'rmsnorm' in name_lower:
                return "Norm-RMS"
            elif 'layer' in name_lower:
                return "Norm-Layer"
            elif 'batch' in name_lower:
                return "Norm-Batch"
            else:
                return "Norm-Generic"
        
        # Activation functions (including gated variants)
        if any(pattern in name_lower for pattern in [
            'relu', 'gelu', 'silu', 'swish', 'sigmoid', 
            'tanh', 'softmax', 'act_and_mul', 'activation'
        ]):
            if 'act_and_mul' in name_lower:
                return "Activation-Gated"  # Gated activation (SwiGLU, etc.)
            elif 'silu' in name_lower or 'swish' in name_lower:
                return "Activation-SiLU"
            elif 'gelu' in name_lower:
                return "Activation-GELU"
            elif 'softmax' in name_lower:
                return "Activation-Softmax"
            else:
                return "Activation-Generic"
        
        # Elementwise operations
        if any(pattern in name_lower for pattern in [
            'elementwise', 'pointwise', 'add', 'mul', 
            'div', 'sub', 'fused_add', 'scalar'
        ]):
            return "Elementwise"
        
        # Reduce operations
        if any(pattern in name_lower for pattern in [
            'reduce', 'sum', 'mean', 'max', 'min', 
            'argmax', 'argmin', 'all_reduce'
        ]):
            return "Reduce"
        
        # Memory operations
        if any(pattern in name_lower for pattern in [
            'copy', 'memcpy', 'memset', 'transpose',
            'permute', 'reshape', 'cast', 'convert',
            'fill_buffer', 'matrix_transpose', 'data_transfer'
        ]):
            if 'transpose' in name_lower or 'permute' in name_lower or 'matrix_transpose' in name_lower:
                return "Memory-Transpose"
            elif 'copy' in name_lower or 'memcpy' in name_lower or 'data_transfer' in name_lower:
                return "Memory-Copy"
            elif 'memset' in name_lower or 'fill_buffer' in name_lower:
                return "Memory-Set"
            elif 'reshape' in name_lower or 'cast' in name_lower or 'convert' in name_lower:
                return "Memory-Reshape"
            else:
                return "Memory-Generic"
        
        # Communication kernels (NCCL/RCCL/MSCCL)
        if any(pattern in name_lower for pattern in [
            'nccl', 'rccl', 'msccl', 'allreduce',
            'allgather', 'reducescatter', 'broadcast',
            'send', 'recv', 'barrier', 'alltoall'
        ]):
            if 'allreduce' in name_lower:
                return "Comm-AllReduce"
            elif 'allgather' in name_lower:
                return "Comm-AllGather"
            elif 'reducescatter' in name_lower:
                return "Comm-ReduceScatter"
            elif 'alltoall' in name_lower:
                return "Comm-AllToAll"
            elif 'send' in name_lower:
                return "Comm-Send"
            elif 'recv' in name_lower:
                return "Comm-Recv"
            elif 'broadcast' in name_lower:
                return "Comm-Broadcast"
            elif 'barrier' in name_lower:
                return "Comm-Barrier"
            else:
                return "Comm-Generic"
        
        # Quantization/Dequantization (inference optimization)
        if any(pattern in name_lower for pattern in [
            'quant', 'dequant', 'fp8', 'int8', 'int4',
            'fp4', 'scale', 'per_group', 'per_channel',
            'quantize', 'dequantize', 'qkv_quant', 'weight_quant'
        ]):
            if 'per_group' in name_lower or 'per_channel' in name_lower:
                return "Quant-PerGroup"  # Per-group/channel quantization
            elif 'dynamic' in name_lower:
                return "Quant-Dynamic"  # Dynamic quantization
            elif 'dequant' in name_lower or 'dequantize' in name_lower:
                return "Dequant"  # Dequantization
            elif 'weight' in name_lower:
                return "Quant-Weight"  # Weight quantization
            elif 'activation' in name_lower or 'act' in name_lower:
                return "Quant-Activation"  # Activation quantization
            else:
                return "Quant-Generic"
        
        # RoPE (Rotary Position Embedding) - with variants
        if 'rope' in name_lower or 'rotary' in name_lower:
            if 'cached' in name_lower:
                return "RoPE-Cached"  # Cached RoPE (inference optimization)
            elif 'gqa' in name_lower or 'grouped' in name_lower:
                return "RoPE-GQA"  # Grouped Query Attention variant
            else:
                return "RoPE-Generic"
        
        # KV Cache operations (generic for all inference engines: vLLM, SGLang, TGI, etc.)
        if any(pattern in name_lower for pattern in [
            'cache', 'reshape_and_cache', 'kv_cache', 'key_value_cache'
        ]) and 'paged' not in name_lower:  # Exclude paged_attention (already handled)
            if 'reshape' in name_lower:
                return "KVCache-Reshape"  # Reshape and store in KV cache
            elif 'append' in name_lower or 'update' in name_lower:
                return "KVCache-Update"  # Update/append to cache
            elif 'compress' in name_lower or 'prune' in name_lower:
                return "KVCache-Compression"  # KV cache compression (H2O, etc.)
            elif framework:
                return f"KVCache-{framework}"  # Framework-specific cache ops
            else:
                return "KVCache-Generic"
        
        # Embedding/Lookup operations
        if any(pattern in name_lower for pattern in [
            'embedding', 'lookup', 'gather', 'scatter',
            'index'
        ]):
            if 'gather' in name_lower:
                return "Indexing-Gather"
            elif 'scatter' in name_lower:
                return "Indexing-Scatter"
            else:
                return "Indexing-Generic"
        
        # Sampling/Generation (inference-specific)
        if any(pattern in name_lower for pattern in [
            'sample', 'topk', 'topp', 'beam', 'argmax', 'argmin',
            'sampling', 'select', 'choice', 'greedy', 'nucleus'
        ]):
            if 'topk' in name_lower or 'top_k' in name_lower:
                return "Sampling-TopK"
            elif 'topp' in name_lower or 'top_p' in name_lower or 'nucleus' in name_lower:
                return "Sampling-TopP"
            elif 'argmax' in name_lower or 'argmin' in name_lower or 'greedy' in name_lower:
                return "Sampling-ArgMax"
            elif 'beam' in name_lower:
                return "Sampling-Beam"
            elif 'select' in name_lower or 'choice' in name_lower:
                return "Sampling-Generic"
            else:
                return "Sampling-Generic"
        
        # Speculative Decoding (used in SGLang, vLLM, TGI)
        if any(pattern in name_lower for pattern in ['speculative', 'draft', 'verify_tokens']):
            return "SpeculativeDecoding"
        
        # Prefix Caching / Radix Attention (SGLang-specific optimization)
        if any(pattern in name_lower for pattern in ['radix', 'prefix_cache', 'tree_attention']):
            return "PrefixCaching"
        
        # Token Generation / Logits Processing (generic inference)
        if any(pattern in name_lower for pattern in ['logits', 'token_gen', 'next_token']):
            return "TokenGeneration"
        
        # Triton kernels (custom kernels used by all frameworks)
        if 'triton' in name_lower:
            return "Triton-Custom"
        
        # FlashInfer kernels (used by SGLang and others)
        if 'flashinfer' in name_lower or 'flash_infer' in name_lower:
            return "FlashInfer"
        
        # Generic forward/backward kernels (common pattern across frameworks)
        if kernel_name.startswith('_fwd_kernel') or kernel_name == '_fwd_kernel':
            return "Kernel-Forward"
        if kernel_name.startswith('_bwd_kernel') or kernel_name == '_bwd_kernel':
            return "Kernel-Backward"
        
        # Framework-specific custom operations (generic detection)
        if framework:
            return f"{framework}-Custom"
        
        # Detect other common inference frameworks by namespace
        if '::' in kernel_name:
            namespace = kernel_name.split('::')[0]
            # Check for common framework patterns
            if any(fw in namespace.lower() for fw in ['hf', 'huggingface', 'transformers']):
                return "HuggingFace-Custom"
            elif any(fw in namespace.lower() for fw in ['tensorrt', 'trt']):
                return "TensorRT-Custom"
            elif any(fw in namespace.lower() for fw in ['onnx', 'ort']):
                return "ONNX-Custom"
            elif any(fw in namespace.lower() for fw in ['deepspeed', 'ds']):
                return "DeepSpeed-Custom"
            elif any(fw in namespace.lower() for fw in ['megatron', 'mcore']):
                return "Megatron-Custom"
            elif any(fw in namespace.lower() for fw in ['fairscale', 'fair']):
                return "FairScale-Custom"
        
        # Default: uncategorized
        return "Uncategorized"

    @staticmethod
    def detect_inference_phase(row, m_threshold=10, debug=False):
        """
        Detect whether an operation belongs to Prefill or Decode phase of inference.
        
        This method uses two approaches:
        1. For GEMM operations: Uses the M parameter to distinguish phases
           - Prefill: M > threshold (sequence length > 1)  
           - Decode: M <= threshold (sequence length = 1)
        2. For other operations: Uses kernel name patterns
        
        Args:
            row (pd.Series or dict): Row containing operation information with columns:
                - 'name': operation name
                - 'kernel_details': list of kernel dictionaries (optional)
                - GEMM parameters if available
            m_threshold (int): Threshold for M parameter. Operations with M > threshold
                              are considered Prefill, others are Decode. Default: 10
            debug (bool): If True, print debugging information
                              
        Returns:
            str: "Prefill", "Decode", or "Mixed" (if contains both types of kernels)
        """
        
        # Check if we have GEMM parameter information available
        has_gemm_params = False
        m_values = []
        
        if debug:
            print(f"DEBUG: Analyzing row: {type(row)}")
            if isinstance(row, dict):
                print(f"DEBUG: Row keys: {list(row.keys())}")
            elif hasattr(row, 'index'):
                print(f"DEBUG: Row index: {list(row.index) if hasattr(row.index, '__iter__') else row.index}")
        
        # Method 1: Try to extract M parameter from different possible sources
        
        # Check if this is a GEMM operation with direct M parameter
        if isinstance(row, dict) and 'param: M' in row:
            m_values.append(row['param: M'])
            has_gemm_params = True
            if debug:
                print(f"DEBUG: Found direct M parameter: {row['param: M']}")
        elif hasattr(row, 'get') and row.get('param: M') is not None:
            m_values.append(row.get('param: M'))
            has_gemm_params = True
            if debug:
                print(f"DEBUG: Found M parameter via get(): {row.get('param: M')}")
        
        # Check for Input Dims directly in the row (kernel launchers format)
        input_dims = None
        if isinstance(row, dict):
            input_dims = row.get('Input Dims')
        elif hasattr(row, 'get'):
            input_dims = row.get('Input Dims')
        elif hasattr(row, 'Input Dims'):
            input_dims = getattr(row, 'Input Dims', None)
            
        # Extract M parameter from Input Dims if available
        if input_dims and isinstance(input_dims, (list, tuple)) and len(input_dims) >= 2:
            try:
                # For most GEMM ops: A_shape[0] = M
                A_shape = input_dims[0] if input_dims[0] else []
                if isinstance(A_shape, (list, tuple)) and len(A_shape) >= 1:
                    m_val = A_shape[0]
                    if isinstance(m_val, (int, float)) and m_val > 0:
                        m_values.append(int(m_val))
                        has_gemm_params = True
                        if debug:
                            print(f"DEBUG: Found M parameter from Input Dims: {m_val}")
            except (IndexError, TypeError, ValueError) as e:
                if debug:
                    print(f"DEBUG: Error extracting M from Input Dims: {e}")
        
        # Check kernel details for GEMM parameters (fallback)
        kernel_details = None
        if isinstance(row, dict):
            kernel_details = row.get('kernel_details', [])
        elif hasattr(row, 'get'):
            kernel_details = row.get('kernel_details', [])
        elif hasattr(row, 'kernel_details'):
            kernel_details = row.kernel_details
            
        if kernel_details and isinstance(kernel_details, list):
            for kernel in kernel_details:
                if isinstance(kernel, dict) and 'args' in kernel:
                    # Check for GEMM operations that might have M parameter
                    args = kernel.get('args', {})
                    if 'Input Dims' in args:
                        input_dims_nested = args['Input Dims']
                        if isinstance(input_dims_nested, list) and len(input_dims_nested) >= 2:
                            # Try to extract M from matrix dimensions
                            # For most GEMM ops: A_shape[0] = M
                            try:
                                A_shape = input_dims_nested[0] if input_dims_nested[0] else []
                                if isinstance(A_shape, list) and len(A_shape) >= 1:
                                    m_val = A_shape[0]
                                    if isinstance(m_val, (int, float)) and m_val > 0:
                                        m_values.append(int(m_val))
                                        has_gemm_params = True
                                        if debug:
                                            print(f"DEBUG: Found M parameter from nested kernel details: {m_val}")
                            except (IndexError, TypeError, ValueError) as e:
                                if debug:
                                    print(f"DEBUG: Error extracting M from nested kernel details: {e}")
                                continue
        
        # Method 2: Use kernel name patterns for explicit prefill/decode detection
        op_name = ""
        if isinstance(row, dict):
            op_name = row.get('name', '')
        elif hasattr(row, 'get'):
            op_name = row.get('name', '')
        elif hasattr(row, 'name'):
            op_name = row.name
            
        # Also check kernel names
        kernel_names = []
        if kernel_details:
            for kernel in kernel_details:
                if isinstance(kernel, dict) and 'name' in kernel:
                    kernel_names.append(kernel['name'])
                    
        # Combine operation name and kernel names for pattern matching
        all_names = [op_name] + kernel_names
        name_text = ' '.join(all_names).lower()
        
        if debug:
            print(f"DEBUG: Operation name: '{op_name}'")
            print(f"DEBUG: Kernel names: {kernel_names}")
            print(f"DEBUG: Combined text: '{name_text}'")
            print(f"DEBUG: M values found: {m_values}, has_gemm_params: {has_gemm_params}")
        
        # Explicit prefill/decode pattern detection
        prefill_patterns = ['prefill', 'chunked_prefill', 'batch_prefill', 'prefill_attention']
        decode_patterns = ['decode', 'batch_decode', 'continuous_batch', 'decode_attention', 
                          'paged_attention.*decode', 'decode_kernel']
        
        has_prefill_pattern = any(pattern in name_text for pattern in prefill_patterns)
        has_decode_pattern = any(pattern in name_text for pattern in decode_patterns)
        
        # Decision logic
        if has_prefill_pattern and has_decode_pattern:
            return "Mixed"
        elif has_prefill_pattern:
            return "Prefill"  
        elif has_decode_pattern:
            return "Decode"
            
        # Use GEMM M parameter if available
        if has_gemm_params and m_values:
            # Check if we have mixed M values (both large and small)
            large_m = [m for m in m_values if m > m_threshold]
            small_m = [m for m in m_values if m <= m_threshold]
            
            if large_m and small_m:
                return "Mixed"
            elif large_m:
                return "Prefill"
            elif small_m:
                return "Decode"
                
        # Default fallback: if no clear indicators, classify as Mixed
        # This is conservative - in practice you might want to default to one phase
        return "Mixed"

    @staticmethod
    def detect_inference_phase_advanced(df_operations, method='attention_patterns', m_threshold=10):
        """
        Advanced inference phase detection using multiple elegant approaches.
        
        Args:
            df_operations (pd.DataFrame): DataFrame with operation information
            method (str): Detection method:
                - 'union': Comprehensive union-based approach (RECOMMENDED FOR MAXIMUM COVERAGE)
                - 'kernel_names': Read directly from kernel names
                - 'attention_patterns': Use attention operation characteristics
                - 'framework_apis': Use framework-specific operation patterns  
                - 'temporal_analysis': Use execution timing patterns
                - 'operation_frequency': Use statistical operation patterns
            m_threshold (int): GEMM M parameter threshold for union method
                
        Returns:
            pd.Series: Phase labels for each operation
        """
        
        if method == 'union':
            return TreePerfAnalyzer.detect_inference_phase_union(df_operations, m_threshold)
        elif method == 'kernel_names':
            return TreePerfAnalyzer._detect_by_kernel_names(df_operations)
        elif method == 'attention_patterns':
            return TreePerfAnalyzer._detect_by_attention_patterns(df_operations)
        elif method == 'framework_apis':
            return TreePerfAnalyzer._detect_by_framework_apis(df_operations)
        elif method == 'temporal_analysis':
            return TreePerfAnalyzer._detect_by_temporal_patterns(df_operations)
        elif method == 'operation_frequency':
            return TreePerfAnalyzer._detect_by_operation_frequency(df_operations)
        else:
            # Fallback to hybrid approach
            return TreePerfAnalyzer._detect_hybrid(df_operations)
    
    @staticmethod
    def _detect_by_attention_patterns(df_ops):
        """
        Detect phases based on attention operation characteristics.
        
        Key insights:
        - Prefill: Large sequence lengths in attention, no KV cache operations
        - Decode: Small queries + KV cache operations (paged_attention, reshape_and_cache)
        """
        phase_labels = pd.Series(['Mixed'] * len(df_ops), index=df_ops.index)
        
        # Framework-agnostic attention patterns
        attention_ops = df_ops['name'].str.contains('attention|attn|sdpa', case=False, na=False)
        
        for idx in df_ops[attention_ops].index:
            op_name = df_ops.loc[idx, 'name'].lower()
            
            # Strong Decode indicators (KV cache operations)
            if any(pattern in op_name for pattern in [
                'paged_attention', 'kv_cache', 'reshape_and_cache', 
                'batch_decode', 'decode_attention'
            ]):
                phase_labels[idx] = 'Decode'
                
            # Strong Prefill indicators 
            elif any(pattern in op_name for pattern in [
                'prefill_attention', 'chunked_prefill', 'batch_prefill'
            ]):
                phase_labels[idx] = 'Prefill'
                
            # Analyze attention sequence dimensions if available
            elif 'Input Dims' in df_ops.columns:
                input_dims = df_ops.loc[idx, 'Input Dims']
                if input_dims and len(input_dims) >= 3:
                    # For attention: [batch, seq_len, hidden] or [batch, num_heads, seq_len, head_dim]
                    seq_len = input_dims[1] if len(input_dims) == 3 else input_dims[2]
                    if isinstance(seq_len, (int, float)) and seq_len > 10:
                        phase_labels[idx] = 'Prefill'
                    elif isinstance(seq_len, (int, float)) and seq_len <= 10:
                        phase_labels[idx] = 'Decode'
        
        return phase_labels
    
    @staticmethod 
    def _detect_by_framework_apis(df_ops):
        """
        Detect phases using framework-specific API patterns.
        
        Different inference engines have distinct operation naming:
        - vLLM: Uses paged_attention_*, reshape_and_cache_*
        - SGLang: Uses radix_attention_*, prefix_cache_*
        - TGI: Uses decode_*, continuous_batch_*
        """
        phase_labels = pd.Series(['Mixed'] * len(df_ops), index=df_ops.index)
        
        for idx, name in df_ops['name'].items():
            name_lower = name.lower()
            
            # vLLM-specific patterns
            if 'vllm::' in name or any(pattern in name_lower for pattern in [
                'paged_attention', 'reshape_and_cache', 'batch_decode'
            ]):
                if any(pattern in name_lower for pattern in ['decode', 'cache']):
                    phase_labels[idx] = 'Decode'
                elif 'prefill' in name_lower:
                    phase_labels[idx] = 'Prefill'
                    
            # SGLang-specific patterns
            elif 'sglang::' in name or any(pattern in name_lower for pattern in [
                'radix_attention', 'prefix_cache', 'tree_attention'
            ]):
                if 'decode' in name_lower or 'cache' in name_lower:
                    phase_labels[idx] = 'Decode'
                elif 'prefill' in name_lower:
                    phase_labels[idx] = 'Prefill'
                    
            # TGI-specific patterns
            elif 'tgi::' in name or any(pattern in name_lower for pattern in [
                'continuous_batch', 'decode_kernel', 'token_generation'
            ]):
                if 'decode' in name_lower or 'generation' in name_lower:
                    phase_labels[idx] = 'Decode'
                elif 'prefill' in name_lower or 'encode' in name_lower:
                    phase_labels[idx] = 'Prefill'
                    
            # TensorRT-LLM patterns
            elif any(pattern in name_lower for pattern in [
                'context_phase', 'generation_phase', 'mha_', 'mlp_'
            ]):
                if 'generation' in name_lower or 'decode' in name_lower:
                    phase_labels[idx] = 'Decode'
                elif 'context' in name_lower or 'prefill' in name_lower:
                    phase_labels[idx] = 'Prefill'
        
        return phase_labels
    
    @staticmethod
    def _detect_by_temporal_patterns(df_ops):
        """
        Detect phases based on temporal execution patterns.
        
        Insight: Prefill happens as dense burst, Decode as repeated sparse patterns.
        """
        phase_labels = pd.Series(['Mixed'] * len(df_ops), index=df_ops.index)
        
        # This requires timestamp information - would need to be implemented
        # based on the specific trace format and timing data available
        # For now, return Mixed to indicate temporal analysis is not implemented
        
        return phase_labels
    
    @staticmethod
    def _detect_by_operation_frequency(df_ops):
        """
        Detect phases based on operation frequency patterns.
        
        Insight: Decode operations repeat many times (once per token),
                Prefill operations happen once per layer.
        """
        phase_labels = pd.Series(['Mixed'] * len(df_ops), index=df_ops.index)
        
        # Count operation frequencies
        op_counts = df_ops['name'].value_counts()
        
        # More aggressive thresholds to classify more operations
        high_freq_threshold = op_counts.quantile(0.5)  # Top 50% by frequency -> Decode
        low_freq_threshold = op_counts.quantile(0.15)  # Bottom 15% by frequency -> Prefill
        
        for idx, name in df_ops['name'].items():
            count = op_counts[name]
            
            if count > high_freq_threshold:
                # High frequency suggests Decode (repeated per token)
                phase_labels[idx] = 'Decode'
            elif count < low_freq_threshold:
                # Low frequency suggests Prefill (once per layer)  
                phase_labels[idx] = 'Prefill'
            else:
                # Medium frequency - check kernel names for hints
                phase_labels[idx] = 'Decode'  # Default to Decode for medium frequency
                
        return phase_labels
    
    @staticmethod
    def _detect_by_kernel_names(df_ops):
        """
        Most reliable method: Read phase information directly from kernel names.
        
        Many inference kernels embed phase information in their names:
        - Explicit prefill/decode keywords
        - Sequence length encoding (sl1 = decode, sl>1 = prefill)
        - Context vs generation patterns
        """
        phase_labels = pd.Series(['Mixed'] * len(df_ops), index=df_ops.index)
        
        for idx in df_ops.index:
            op_name = str(df_ops.loc[idx, 'name']).lower()
            
            # Get kernel names if available
            kernel_names = []
            if 'kernel_details' in df_ops.columns:
                kernel_details = df_ops.loc[idx, 'kernel_details']
                if isinstance(kernel_details, list):
                    kernel_names = [k.get('name', '') for k in kernel_details if isinstance(k, dict)]
            
            # Combine all names for analysis
            all_names = ' '.join([op_name] + [str(k).lower() for k in kernel_names])
            
            # === Strong Prefill indicators ===
            prefill_keywords = [
                'prefill', 'context', 'prompt', 'chunked_prefill',
                'batch_prefill', 'context_attention', 'prompt_attention',
                'context_phase', 'initialization', 'first_token'
            ]
            
            # === Strong Decode indicators ===
            decode_keywords = [
                'decode', 'generation', 'autoregressive', 'batch_decode',
                'continuous_batch', 'paged_attention', 'kv_cache',
                'reshape_and_cache', 'generation_phase', 'incremental',
                'single_token', '_decode_', 'decode_kernel'
            ]
            
            # === Sequence length patterns in kernel names ===
            # Many kernels encode sequence length: sl1, sl512, seqlen_1, etc.
            import re
            seq_len_patterns = re.findall(r'sl(\d+)|seqlen[_\s]?(\d+)|seq(\d+)', all_names)
            if seq_len_patterns:
                seq_lens = []
                for match in seq_len_patterns:
                    # Extract the actual number from the match groups
                    seq_len = next((int(x) for x in match if x), None)
                    if seq_len:
                        seq_lens.append(seq_len)
                
                if seq_lens:
                    avg_seq_len = sum(seq_lens) / len(seq_lens)
                    if avg_seq_len == 1:
                        phase_labels[idx] = 'Decode'
                        continue
                    elif avg_seq_len > 10:
                        phase_labels[idx] = 'Prefill'
                        continue
            
            # Check for explicit keywords
            if any(keyword in all_names for keyword in prefill_keywords):
                phase_labels[idx] = 'Prefill'
            elif any(keyword in all_names for keyword in decode_keywords):
                phase_labels[idx] = 'Decode'
            else:
                # If no clear indicator, check operation patterns
                # MLP/FFN GEMMs in decode are typically high-frequency
                if any(pattern in op_name for pattern in ['mm', 'addmm', 'gemm', 'matmul']):
                    # For GEMMs without clear indicators, use frequency as tie-breaker
                    # This will be overridden by hybrid method if needed
                    phase_labels[idx] = 'Mixed'
                    
        return phase_labels
    
    @staticmethod
    def _detect_hybrid(df_ops):
        """
        Combine multiple detection methods for robust phase detection.
        Priority: kernel_names > framework_apis > attention_patterns > frequency
        """
        # Get predictions from different methods (in order of reliability)
        kernel_pred = TreePerfAnalyzer._detect_by_kernel_names(df_ops)
        framework_pred = TreePerfAnalyzer._detect_by_framework_apis(df_ops)
        attention_pred = TreePerfAnalyzer._detect_by_attention_patterns(df_ops)
        frequency_pred = TreePerfAnalyzer._detect_by_operation_frequency(df_ops)
        
        # Hierarchical decision: use most reliable available method
        phase_labels = pd.Series(['Mixed'] * len(df_ops), index=df_ops.index)
        
        for idx in df_ops.index:
            # Priority 1: Kernel name detection (most reliable if available)
            if kernel_pred[idx] != 'Mixed':
                phase_labels[idx] = kernel_pred[idx]
            # Priority 2: Framework-specific APIs
            elif framework_pred[idx] != 'Mixed':
                phase_labels[idx] = framework_pred[idx]
            # Priority 3: Attention patterns
            elif attention_pred[idx] != 'Mixed':
                phase_labels[idx] = attention_pred[idx]
            # Priority 4: Statistical frequency
            elif frequency_pred[idx] != 'Mixed':
                phase_labels[idx] = frequency_pred[idx]
            # Default: leave as Mixed
                
        return phase_labels

    @staticmethod
    def detect_inference_phase_union(df_ops, m_threshold=10, debug=False):
        """
        Comprehensive union-based inference phase detection combining all available criteria:
        1. GEMM M dimension analysis
        2. Timestamp/temporal patterns 
        3. Operation frequency analysis
        4. Request ID patterns (continuous batching in vLLM)
        5. PagedAttention kernel patterns
        6. Framework-specific API patterns
        7. Kernel name explicit patterns
        
        Uses union criteria: If ANY method confidently detects a phase, classify it.
        Only mark as 'Mixed' if truly ambiguous across all methods.
        
        Args:
            df_ops (pd.DataFrame): Operations dataframe
            m_threshold (int): GEMM M parameter threshold for Prefill/Decode
            debug (bool): Print debugging information
            
        Returns:
            pd.Series: Phase labels ('Prefill', 'Decode', 'Mixed')
        """
        
        if debug:
            print(f"DEBUG: Starting union-based detection for {len(df_ops)} operations")
            print(f"DEBUG: Available columns: {list(df_ops.columns)}")
        
        # Initialize all as Mixed, then apply union logic
        phase_labels = pd.Series(['Mixed'] * len(df_ops), index=df_ops.index)
        confidence_scores = pd.Series([0.0] * len(df_ops), index=df_ops.index)
        detection_methods = pd.Series([''] * len(df_ops), index=df_ops.index)
        
        for idx in df_ops.index:
            row = df_ops.loc[idx]
            prefill_votes = 0
            decode_votes = 0
            method_details = []
            
            # === 1. GEMM M Dimension Analysis ===
            gemm_phase = TreePerfAnalyzer._analyze_gemm_dimensions(row, m_threshold, debug)
            if gemm_phase == 'Prefill':
                prefill_votes += 3  # Strong signal
                method_details.append(f"GEMM_M>{m_threshold}")
            elif gemm_phase == 'Decode':
                decode_votes += 3  # Strong signal  
                method_details.append(f"GEMM_M<={m_threshold}")
            
            # === 2. Kernel Name Explicit Patterns ===
            name_phase = TreePerfAnalyzer._analyze_explicit_names(row, debug)
            if name_phase == 'Prefill':
                prefill_votes += 4  # Very strong signal
                method_details.append("ExplicitName")
            elif name_phase == 'Decode':
                decode_votes += 4  # Very strong signal
                method_details.append("ExplicitName")
            
            # === 3. PagedAttention Patterns ===
            paged_phase = TreePerfAnalyzer._analyze_paged_attention(row, debug)
            if paged_phase == 'Prefill':
                prefill_votes += 2
                method_details.append("PagedAttn")
            elif paged_phase == 'Decode':
                decode_votes += 2
                method_details.append("PagedAttn")
            
            # === 4. Framework API Patterns ===
            framework_phase = TreePerfAnalyzer._analyze_framework_apis(row, debug)
            if framework_phase == 'Prefill':
                prefill_votes += 2
                method_details.append("FrameworkAPI")
            elif framework_phase == 'Decode':
                decode_votes += 2
                method_details.append("FrameworkAPI")
            
            # === 5. Request ID / Batch Patterns ===
            batch_phase = TreePerfAnalyzer._analyze_batch_patterns(row, debug)
            if batch_phase == 'Prefill':
                prefill_votes += 1
                method_details.append("BatchPattern")
            elif batch_phase == 'Decode':
                decode_votes += 1
                method_details.append("BatchPattern")
            
            # === Union Decision Logic ===
            total_votes = prefill_votes + decode_votes
            
            if total_votes >= 2:  # Require at least moderate confidence
                if prefill_votes > decode_votes:
                    phase_labels[idx] = 'Prefill'
                    confidence_scores[idx] = prefill_votes / max(total_votes, 1)
                elif decode_votes > prefill_votes:
                    phase_labels[idx] = 'Decode'  
                    confidence_scores[idx] = decode_votes / max(total_votes, 1)
                # If tied, leave as Mixed
                
            detection_methods[idx] = ','.join(method_details) if method_details else 'None'
            
            if debug and total_votes > 0:
                op_name = row.get('name', 'Unknown')
                print(f"DEBUG: {op_name[:50]:<50} | P:{prefill_votes} D:{decode_votes} -> {phase_labels[idx]} | Methods: {detection_methods[idx]}")
        
        # === 6. Temporal/Frequency Post-processing ===
        phase_labels = TreePerfAnalyzer._apply_temporal_refinement(
            df_ops, phase_labels, confidence_scores, debug
        )
        
        if debug:
            phase_counts = phase_labels.value_counts()
            print(f"\nDEBUG: Final phase distribution: {dict(phase_counts)}")
            print(f"DEBUG: Average confidence: {confidence_scores.mean():.3f}")
            
        return phase_labels
    
    @staticmethod
    def _analyze_gemm_dimensions(row, m_threshold, debug=False):
        """Extract and analyze GEMM M dimensions from various sources"""
        m_values = []
        
        # Check direct M parameter
        if 'param: M' in row and pd.notna(row['param: M']):
            m_values.append(int(row['param: M']))
            
        # Check Input Dims
        if 'Input Dims' in row and row['Input Dims']:
            try:
                input_dims = row['Input Dims']
                if isinstance(input_dims, (list, tuple)) and len(input_dims) >= 2:
                    A_shape = input_dims[0] if input_dims[0] else []
                    if isinstance(A_shape, (list, tuple)) and len(A_shape) >= 1:
                        m_val = A_shape[0]
                        if isinstance(m_val, (int, float)) and m_val > 0:
                            m_values.append(int(m_val))
            except (IndexError, TypeError, ValueError):
                pass
        
        # Check kernel details
        if 'kernel_details' in row and isinstance(row['kernel_details'], list):
            for kernel in row['kernel_details']:
                if isinstance(kernel, dict) and 'args' in kernel:
                    args = kernel.get('args', {})
                    if 'Input Dims' in args:
                        try:
                            input_dims_nested = args['Input Dims']
                            if isinstance(input_dims_nested, list) and len(input_dims_nested) >= 2:
                                A_shape = input_dims_nested[0] if input_dims_nested[0] else []
                                if isinstance(A_shape, list) and len(A_shape) >= 1:
                                    m_val = A_shape[0]
                                    if isinstance(m_val, (int, float)) and m_val > 0:
                                        m_values.append(int(m_val))
                        except (IndexError, TypeError, ValueError):
                            continue
        
        if m_values:
            avg_m = sum(m_values) / len(m_values)
            if debug:
                print(f"  GEMM M values: {m_values}, avg: {avg_m:.1f}")
            return 'Prefill' if avg_m > m_threshold else 'Decode'
        
        return 'Mixed'
    
    @staticmethod  
    def _analyze_explicit_names(row, debug=False):
        """Analyze explicit prefill/decode patterns in operation and kernel names"""
        op_name = str(row.get('name', '')).lower()
        
        # Get kernel names
        kernel_names = []
        if 'kernel_details' in row and isinstance(row['kernel_details'], list):
            for kernel in row['kernel_details']:
                if isinstance(kernel, dict) and 'name' in kernel:
                    kernel_names.append(str(kernel['name']).lower())
        
        all_text = ' '.join([op_name] + kernel_names)
        
        # Strong explicit patterns
        prefill_explicit = [
            'prefill', 'chunked_prefill', 'batch_prefill', 'prefill_attention',
            'context_attention', 'prompt_attention', 'context_phase', 
            'first_token', 'initialization', 'prompt_processing'
        ]
        
        decode_explicit = [
            'decode', 'generation', 'batch_decode', 'continuous_batch',
            'paged_attention', 'decode_attention', 'generation_phase',
            'autoregressive', 'incremental', 'single_token', 'next_token'
        ]
        
        # Sequence length patterns (sl1 = decode, sl>1 = prefill)
        import re
        seq_patterns = re.findall(r'sl(\d+)|seqlen[_\s]?(\d+)|seq(\d+)', all_text)
        if seq_patterns:
            seq_lens = []
            for match in seq_patterns:
                seq_len = next((int(x) for x in match if x), None)
                if seq_len:
                    seq_lens.append(seq_len)
            
            if seq_lens:
                avg_seq_len = sum(seq_lens) / len(seq_lens)
                if avg_seq_len == 1:
                    return 'Decode'
                elif avg_seq_len > 10:
                    return 'Prefill'
        
        # Check explicit keywords
        prefill_found = any(pattern in all_text for pattern in prefill_explicit)
        decode_found = any(pattern in all_text for pattern in decode_explicit)
        
        if prefill_found and not decode_found:
            return 'Prefill'
        elif decode_found and not prefill_found:
            return 'Decode'
            
        return 'Mixed'
    
    @staticmethod
    def _analyze_paged_attention(row, debug=False):
        """Analyze PagedAttention-specific patterns (vLLM, SGLang, etc.)"""
        op_name = str(row.get('name', '')).lower()
        
        # Get kernel names
        kernel_names = []
        if 'kernel_details' in row and isinstance(row['kernel_details'], list):
            for kernel in row['kernel_details']:
                if isinstance(kernel, dict) and 'name' in kernel:
                    kernel_names.append(str(kernel['name']).lower())
        
        all_text = ' '.join([op_name] + kernel_names)
        
        # PagedAttention patterns
        paged_decode_patterns = [
            'paged_attention.*decode', 'paged_attn_v2', 'kv_cache',
            'reshape_and_cache', 'block_sparse_attention', 'page_attention'
        ]
        
        paged_prefill_patterns = [
            'paged_attention.*prefill', 'chunked_prefill', 'block_prefill'
        ]
        
        # Flash attention patterns
        flash_decode_patterns = [
            'flash_attn.*decode', 'flash_decode', 'varlen.*decode'
        ]
        
        flash_prefill_patterns = [
            'flash_attn.*prefill', 'flash_prefill', 'varlen.*prefill'
        ]
        
        if any(re.search(pattern, all_text) for pattern in paged_decode_patterns + flash_decode_patterns):
            return 'Decode'
        elif any(re.search(pattern, all_text) for pattern in paged_prefill_patterns + flash_prefill_patterns):
            return 'Prefill'
            
        return 'Mixed'
    
    @staticmethod
    def _analyze_framework_apis(row, debug=False):
        """Analyze framework-specific API patterns"""
        op_name = str(row.get('name', '')).lower()
        
        # vLLM patterns
        if 'vllm::' in op_name:
            if any(pattern in op_name for pattern in ['decode', 'cache', 'paged']):
                return 'Decode'
            elif 'prefill' in op_name:
                return 'Prefill'
                
        # SGLang patterns  
        if 'sglang::' in op_name:
            if any(pattern in op_name for pattern in ['decode', 'cache', 'radix']):
                return 'Decode'
            elif 'prefill' in op_name:
                return 'Prefill'
                
        # TGI patterns
        if 'tgi::' in op_name:
            if any(pattern in op_name for pattern in ['decode', 'generation', 'continuous']):
                return 'Decode'
            elif any(pattern in op_name for pattern in ['prefill', 'encode']):
                return 'Prefill'
                
        return 'Mixed'
    
    @staticmethod
    def _analyze_batch_patterns(row, debug=False):
        """Analyze request ID and batching patterns"""
        op_name = str(row.get('name', '')).lower()
        
        # Continuous batching typically indicates decode
        continuous_patterns = [
            'continuous_batch', 'dynamic_batch', 'batch_decode',
            'multi_request', 'request_batch'
        ]
        
        # Single request patterns typically indicate prefill
        single_patterns = [
            'single_request', 'request_init', 'batch_prefill'
        ]
        
        if any(pattern in op_name for pattern in continuous_patterns):
            return 'Decode'
        elif any(pattern in op_name for pattern in single_patterns):
            return 'Prefill'
            
        return 'Mixed'
    
    @staticmethod
    def _apply_temporal_refinement(df_ops, phase_labels, confidence_scores, debug=False):
        """Apply temporal patterns and frequency analysis to refine phase detection"""
        
        # Add timestamp-based refinement if available
        if 'ts' in df_ops.columns or 'timestamp' in df_ops.columns:
            ts_col = 'ts' if 'ts' in df_ops.columns else 'timestamp'
            
            # Sort by timestamp
            df_sorted = df_ops.sort_values(by=ts_col)
            
            # Look for burst patterns (Prefill) vs repeated patterns (Decode)
            # This is a simplified heuristic - could be enhanced
            for i in range(1, len(df_sorted)):
                current_idx = df_sorted.index[i]
                prev_idx = df_sorted.index[i-1]
                
                # If current operation is Mixed but previous was confidently classified
                if (phase_labels[current_idx] == 'Mixed' and 
                    phase_labels[prev_idx] != 'Mixed' and
                    confidence_scores[prev_idx] > 0.5):
                    
                    # Check if they have similar names (likely same operation type)
                    current_name = str(df_ops.loc[current_idx, 'name']).lower()
                    prev_name = str(df_ops.loc[prev_idx, 'name']).lower()
                    
                    # Simple similarity check
                    if (current_name[:20] == prev_name[:20] or
                        any(common in current_name and common in prev_name 
                            for common in ['gemm', 'mm', 'attention', 'attn', 'norm'])):
                        phase_labels[current_idx] = phase_labels[prev_idx]
                        confidence_scores[current_idx] = 0.3  # Lower confidence for inherited
        
        # Apply frequency-based refinement
        op_counts = df_ops['name'].value_counts()
        
        for idx in df_ops.index:
            if phase_labels[idx] == 'Mixed':
                op_name = df_ops.loc[idx, 'name']
                count = op_counts[op_name]
                
                # High frequency operations are likely Decode (per-token)
                # Low frequency operations are likely Prefill (per-layer)
                if count > op_counts.quantile(0.7):  # Top 30% by frequency
                    phase_labels[idx] = 'Decode'
                    confidence_scores[idx] = 0.2  # Low confidence for frequency-based
                elif count < op_counts.quantile(0.3):  # Bottom 30% by frequency
                    phase_labels[idx] = 'Prefill'  
                    confidence_scores[idx] = 0.2  # Low confidence for frequency-based
        
        return phase_labels

    def get_kernel_details(
        self,
        kernel_event,
        launcher_detail=False,
        cpu_op_detail=True,
        nn_module_detail=False,
        kernel_categorization=True,
        inference_phase_detection=True,
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
            kernel_categorization (bool): If True, add detailed kernel-level categorization based on kernel name.

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
        
        # Add kernel-level categorization
        if kernel_categorization:
            kernel_details["Kernel category"] = self.categorize_kernel_by_name(
                kernel_event["name"]
            )

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
        
        # 4. Add inference phase detection (if enabled)
        if inference_phase_detection:
            # Create a temporary row-like object for the detect_inference_phase method
            temp_row = {
                'name': kernel_details.get('Parent cpu_op', ''),
                'kernel_details': [{'name': kernel_event['name']}]
            }
            
            # Add Input Dims and other args directly to temp_row (matching kernel launcher format)
            if cpu_op and 'args' in cpu_op:
                args = cpu_op['args']
                # Copy the Input Dims directly to the temp_row (this is the main source for GEMM M parameter)
                temp_row['Input Dims'] = args.get('Input Dims')
                temp_row['Input type'] = args.get('Input type')
                temp_row['Input Strides'] = args.get('Input Strides')
                temp_row['Concrete Inputs'] = args.get('Concrete Inputs')
            
            # Use union-based detection for comprehensive phase detection
            # Create a temporary single-row DataFrame for the union method
            temp_df = pd.DataFrame([temp_row])
            phase_result = TreePerfAnalyzer.detect_inference_phase_union(temp_df, m_threshold=10)
            kernel_details["Inference phase"] = phase_result.iloc[0] if len(phase_result) > 0 else "Mixed"
        return kernel_details

    def get_df_kernels(
        self, 
        launcher_detail=False, 
        cpu_op_detail=True, 
        nn_module_detail=False,
        kernel_categorization=True,
        inference_phase_detection=True
    ):
        """
        Build a DataFrame with kernel details augmented with
        additional information such as launcher, CPU operation,
        nn.Module details, and kernel-level categorization.
        
        Args:
            launcher_detail (bool): If True, include details of the kernel's launcher.
            cpu_op_detail (bool): If True, include details of the parent CPU operation.
            nn_module_detail (bool): If True, include details of the parent nn.Module event.
            kernel_categorization (bool): If True, add detailed kernel-level categorization 
                                         based on kernel name patterns (GEMM, Attention, etc.).
            inference_phase_detection (bool): If True, add inference phase detection 
                                              (Prefill/Decode/Mixed) based on GEMM parameters and kernel patterns.

        Returns:
            pd.DataFrame: A DataFrame containing detailed kernel information with the following columns:
                - UID: Unique identifier for the kernel
                - Kernel name: Full name of the GPU kernel
                - Kernel category: Detailed categorization (e.g., "GEMM-CK", "Attention-Forward", "Norm-RMS")
                - Inference phase: Phase detection ("Prefill", "Decode", or "Mixed") (if inference_phase_detection=True)
                - Kernel duration (µs): Duration in microseconds
                - Kernel stream: CUDA/HIP stream ID
                - Parent op category: Category of the parent PyTorch operation (if cpu_op_detail=True)
                - Parent cpu_op: Name of the parent CPU operation (if cpu_op_detail=True)
                - Launcher: Name of the launcher event (if launcher_detail=True)
                - And more depending on the options selected
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
                kernel_categorization=kernel_categorization,
                inference_phase_detection=inference_phase_detection,
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
        metadata = TraceEventUtils.get_metadata(data_pb)
        tree = JaxTraceToTree(
            events, linking_key=linking_key, event_to_category=categorizer
        )
        return JaxTreePerfAnalyzer(
            tree,
            event_to_category=categorizer,
            metadata=metadata,
            pb_file_name=profile_filepath,
            *args,
            **kwargs,
        )

    def __init__(
        self,
        tree: JaxTraceToTree,
        event_to_category: Callable[[dict], str] = TraceEventUtils.default_categorizer,
        metadata=None,
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
        self.metadata = metadata
        self.pb_file_name = pb_file_name
        self.arch = arch
        self.tree.build_tree(metadata=metadata, pb_file_name=pb_file_name)
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
            logger.debug(
                f"\nException occurred when parsing Event: \n\n {event} \n\
                            Event metadata: {event['metadata']}, operands: {operands}"
            )
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
        if hasattr(perf_model, "gemmologist_time"):
            dict_metrics["Gemmologist Time (µs)"] = perf_model.gemmologist_time
            dict_metrics["Gemmologist TFLOPS/s"] = (
                (gflops / 1e3) / (perf_model.gemmologist_time / 1e6)
                if perf_model.gemmologist_time > 0
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
