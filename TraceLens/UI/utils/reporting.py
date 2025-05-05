import gzip
import json
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pandas as pd

from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

@dataclass
class TraceLensColumns:
    TYPE = "Input type"
    DIMS = "Input Dims"
    STRIDES = "Input Strides"
    PERCENTAGE = "Percentage (%)"

    SUMMARY_TOTAL_KERNEL_TIME = "Total Kernel Time (µs)"
    SUMMARY_COUNT = "Count"


@dataclass
class FAReportColumns:
    NAME = "Name"
    PASS = "Pass"
    DURATION = "avg duration (µs)"


@dataclass
class GEMMReportColumns:
    TYPES = "types"
    DIMS = "dims"
    STRIDES = "strides"
    DURATION = "avg duration (µs)"
    TFLOPS = "avg performance (TFLOPS)"

    DEFAULT_MAPPING = {
        TraceLensColumns.TYPE: ("Input", TYPES),
        TraceLensColumns.DIMS: ("Input", DIMS),
        TraceLensColumns.STRIDES: ("Input", STRIDES),
    }

    @staticmethod
    def map_columns(df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
        df.columns = pd.MultiIndex.from_tuples([
            GEMMReportColumns.DEFAULT_MAPPING[col] if col in GEMMReportColumns.DEFAULT_MAPPING else (experiment_name, col)
            for col in df.columns
        ])
        return df


@dataclass
class ConvReportColumns:
    TYPES = "types"
    DIMS = "dims"
    STRIDES = "strides"
    DURATION = "avg duration (µs)"
    TFLOPS = "avg performance (TFLOPS)"

    DEFAULT_MAPPING = {
        TraceLensColumns.TYPE: ("Input", TYPES),
        TraceLensColumns.DIMS: ("Input", DIMS),
        TraceLensColumns.STRIDES: ("Input", STRIDES),
    }

    @staticmethod
    def map_columns(df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
        df.columns = pd.MultiIndex.from_tuples([
            ConvReportColumns.DEFAULT_MAPPING[col] if col in ConvReportColumns.DEFAULT_MAPPING else (experiment_name, col)
            for col in df.columns
        ])
        return df


def read_trace(file: BytesIO) -> TreePerfAnalyzer:
    if file.name.endswith(".json.gz"):
        with gzip.GzipFile(fileobj=file) as f:
            data = json.load(f)
    elif file.name.endswith(".json"):
        data = json.load(file)

    tree = TraceToTree(data['traceEvents'])
    return TreePerfAnalyzer(tree)


def get_fa_config(analyzer: TreePerfAnalyzer, index: str) -> pd.DataFrame:
    kernel_launchers = analyzer.get_kernel_launchers()
    fa_kernel = None
    for event in kernel_launchers:
        if "flash" in event["name"].lower() and "backward" not in event["name"].lower():
            fa_kernel = event
            break

    if not fa_kernel:
        return None

    CASUAL_ARG_IDX = 9
    ARGS_KEY = "args"
    q, k, v = fa_kernel[ARGS_KEY][TraceLensColumns.DIMS][:3]
    config_df = pd.DataFrame(
        {
            ("Sequence length", "", ""): [q[0]],
            ("Attention type", "", ""): (
                "Multi-Head" if q[1] == k[1] else
                "Multi-Query" if k[1] == 1 else
                "Groupped-Query"
            ),
            ("Causal masking", "", ""): [fa_kernel[ARGS_KEY]["Concrete Inputs"][CASUAL_ARG_IDX]],
            ("Heads", "Q", "#"): [q[1]],
            ("Heads", "Q", "Dims"): [q[0]],
            ("Heads", "Q", "Type"): [fa_kernel[ARGS_KEY][TraceLensColumns.TYPE][0]],
            ("Heads", "K", "#"): [k[1]],
            ("Heads", "K", "Dims"): [k[0]],
            ("Heads", "K", "Type"): [fa_kernel[ARGS_KEY][TraceLensColumns.TYPE][1]],
            ("Heads", "V", "#"): [v[1]],
            ("Heads", "V", "Dims"): [v[0]],
            ("Heads", "V", "Type"): [fa_kernel[ARGS_KEY][TraceLensColumns.TYPE][2]],
        },
        index=[index],
    )
    return pd.DataFrame(config_df)


def get_fa_perf_df(summary: pd.DataFrame) -> pd.DataFrame:
    # on MI300X forward operations doesn't have "forward" in the name,
    # so we have to search "backward" for identification
    lower_name_str = summary["name"].str.lower().str
    summary[FAReportColumns.PASS] = np.where(lower_name_str.contains("backward"), "backward", "forward")
    summary[FAReportColumns.DURATION] = summary["total_direct_kernel_time_sum"] / summary[TraceLensColumns.SUMMARY_COUNT]
    return (
        summary[lower_name_str.contains("flash")]
        .rename(columns={"name": FAReportColumns.NAME})
        [[FAReportColumns.PASS, FAReportColumns.NAME, FAReportColumns.DURATION]]
    )


def _calc_tflops(row: pd.Series) -> float:
    dims = row[TraceLensColumns.DIMS]
    m = dims[0][0]
    n = dims[0][1]
    k = dims[1][1]
    return 2 * m * n * k / row[GEMMReportColumns.DURATION] / 1e6


def get_gemm_perf_df(summary: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    if summary.empty:
        # Return an empty DataFrame with the expected structure
        return pd.DataFrame(columns=[
            TraceLensColumns.TYPE,
            TraceLensColumns.DIMS,
            TraceLensColumns.STRIDES,
            TraceLensColumns.PERCENTAGE,
            GEMMReportColumns.DURATION,
            GEMMReportColumns.TFLOPS
        ])

    summary[GEMMReportColumns.DURATION] = summary[TraceLensColumns.SUMMARY_TOTAL_KERNEL_TIME] / summary[TraceLensColumns.SUMMARY_COUNT]
    summary[GEMMReportColumns.TFLOPS] = summary.apply(_calc_tflops, axis=1)
    return GEMMReportColumns.map_columns(
        summary[[
            TraceLensColumns.TYPE,
            TraceLensColumns.DIMS,
            TraceLensColumns.STRIDES,
            TraceLensColumns.PERCENTAGE,
            GEMMReportColumns.DURATION,
            GEMMReportColumns.TFLOPS
        ]],
        experiment_name,
    )


def get_conv_perf_df(summary: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    # Filter for convolution kernels (e.g., "aten::conv" or similar)
    conv_kernels = summary[summary["name"].str.contains("aten::conv", case=False, na=False)]

    if conv_kernels.empty:
        return pd.DataFrame()

    # Calculate duration and TFLOPS
    conv_kernels[ConvReportColumns.DURATION] = conv_kernels[TraceLensColumns.SUMMARY_TOTAL_KERNEL_TIME] / conv_kernels[TraceLensColumns.SUMMARY_COUNT]

    # Example TFLOPS calculation for convolution (modify based on actual kernel details)
    def _calc_conv_tflops(row: pd.Series) -> float:
        # Assuming dims are in the format: [input_channels, output_channels, kernel_h, kernel_w, input_h, input_w]
        input_channels, output_channels, kernel_h, kernel_w, input_h, input_w = row[TraceLensColumns.DIMS][0]
        return (2 * input_channels * output_channels * kernel_h * kernel_w * input_h * input_w) / row[ConvReportColumns.DURATION] / 1e6

    conv_kernels[ConvReportColumns.TFLOPS] = conv_kernels.apply(_calc_conv_tflops, axis=1)

    return ConvReportColumns.map_columns(
        conv_kernels[[
            TraceLensColumns.TYPE,
            TraceLensColumns.DIMS,
            TraceLensColumns.STRIDES,
            TraceLensColumns.PERCENTAGE,
            ConvReportColumns.DURATION,
            ConvReportColumns.TFLOPS
        ]],
        experiment_name,
    )
