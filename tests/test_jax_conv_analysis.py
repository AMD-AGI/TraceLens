###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import math
from collections import Counter
import logging
import numpy as np

np.random.seed(42)

from TraceLens.TreePerf import JaxTreePerfAnalyzer, TreePerfAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Working directory: %s", os.getcwd())
jax_conv_minimal = "./tests/traces/mi300/jax_conv_minimal/chi-mi300x-013.ord.vultr.cpe.ice.amd.com.xplane.pb"
assert os.path.exists(jax_conv_minimal)
perf_analyzer = JaxTreePerfAnalyzer.from_file(profile_filepath=jax_conv_minimal)


def profile_jax_conv(path=None):
    """
    Parameters for the 3D convolution
    batch_size = 1
    time_dim = 32
    height = 60
    width = 104
    in_channels = 16
    out_channels = 5120
    dtype = jax.numpy.bfloat16

    # Kernel parameters
    kernel_t = 1
    kernel_h = 2
    kernel_w = 2
    stride = (1, 2, 2)
    """

    if path is None:
        path = "/tmp/jax_trace.xplane.pb"
    # Profile trace
    # Jesus' example: ./jax-minimal/jax_conv.py
    return path


##################
# Event Statistics
##################


def test_num_tree_events():
    expected_result = 5903

    result = len(perf_analyzer.tree.events)
    assert result == expected_result


def test_tree_event_cats():
    expected_result = {
        "Unknown": 4658,
        "cpu_op": 1147,
        "memcpy": 53,
        "kernel": 25,
        "python function": 20,
    }

    result = Counter([event["cat"] for event in perf_analyzer.tree.events])
    assert result == expected_result


def test_kernel_event_cats():
    expected_result = {"Uncategorized Events/XLA": 15, "Conv": 10}

    result = Counter(
        [
            event["gpu_kernel_op_cat"]
            for event in perf_analyzer.tree.events
            if event["cat"] == "kernel"
        ]
    )
    assert result == expected_result


################
# GPU Statistics
################


def test_gpu_pids():
    expected_result = set([1, 8])

    result = set(perf_analyzer.gpu_event_analyser.gpu_pids)
    assert result == expected_result


def test_gpu_timeline():
    # gpu 1
    busy_time = perf_analyzer.get_df_gpu_timeline(gpu_pid=1).set_index("type")[
        "time ms"
    ]["busy_time"]
    assert math.isclose(0.889028, busy_time, rel_tol=1e-5)

    # gpu 8
    busy_time = perf_analyzer.get_df_gpu_timeline(gpu_pid=8).set_index("type")[
        "time ms"
    ]["busy_time"]
    assert math.isclose(3.586493, busy_time, rel_tol=1e-5)

    # average
    busy_time = perf_analyzer.get_df_gpu_events_averages().set_index("type")["time ms"][
        "busy_time"
    ]
    assert math.isclose(2.237760, busy_time, rel_tol=1e-5)


###############
# Kernel Events
###############


def test_kernel_launchers():
    # kernel launchers
    kernel_launchers = perf_analyzer.get_kernel_launchers()
    assert len(kernel_launchers) == 25


def test_df_kernel_launchers():
    """
    Alternatively provide trace and desired output xlsx files (with tabs) for testing.
    """
    # dataframe
    df_kernel_launchers = perf_analyzer.get_df_kernel_launchers()
    df_kernel_launchers_summary = perf_analyzer.get_df_kernel_launchers_summary(
        df_kernel_launchers
    )
    df_kernel_launchers_summary_by_category = (
        perf_analyzer.get_df_kernel_launchers_summary_by_category(df_kernel_launchers)
    )

    assert df_kernel_launchers.shape == (25, 12)
    assert df_kernel_launchers_summary.shape == (5, 8)
    assert df_kernel_launchers_summary_by_category.shape == (2, 6)


#####################
# Performance Metrics
#####################

kernel_events = [
    event for event in perf_analyzer.tree.events if event["cat"] == "kernel"
]
assert len(kernel_events) == 25

conv_events = [
    event for event in kernel_events if event["gpu_kernel_op_cat"].lower() == "conv"
]
assert len(conv_events) == 10

result = Counter(
    [perf_analyzer.get_event_perf_model_name(event) for event in conv_events]
)
assert result == {"jax_conv": 10}

rand_idx = np.random.randint(0, len(conv_events))
event = conv_events[rand_idx]


def test_conv_event_bytes_and_flops():
    """
    The total bytes moved during a single forward pass of a convolution can be estimated using the following formula:
    Bytes Moved = (Input Size) + (Kernel Size) + (Output Size) = (16*32*60*104 + 4 + 5120*34*31*53)*2 = 578416648

    The Floating Point Operations (FLOPs) of a standard convolutional layer can be calculated using the following formula:
    FLOPs: bytes per element * Number of Kernel * Kernel Shape * Output Shape
    FLOPs = 2*2*2*5120*34*31*53 = 2.288107520*1E09

    Where:
    C_out: Number of output channels (or filters).
    C_in: Number of input channels.
    K_h: Height of the convolutional kernel (filter).
    K_w: Width of the convolutional kernel (filter).
    H_out: Height of the output feature map.
    W_out: Width of the output feature map.
    """

    perf_model_name = perf_analyzer.get_event_perf_model_name(event)
    perf_model_class = perf_analyzer.jax_op_to_perf_model_class_map.get(
        perf_model_name, None
    )
    perf_model = perf_model_class(event)
    assert perf_model.bytes() == 578416648
    assert perf_model.flops() == 2288107520


def test_conv_event_metrics():

    dict_perf_metrics = perf_analyzer.compute_perf_metrics(conv_events[rand_idx])
    assert dict_perf_metrics["param: input_shape"] == (1, 16, 32, 60, 104)
    assert dict_perf_metrics["param: filter_shape"] == (1, 2, 2)
    assert dict_perf_metrics["param: output_shape"] == (1, 5120, 34, 31, 53)
    assert dict_perf_metrics["param: bias"] == False
    assert math.isclose(2.288108, dict_perf_metrics["GFLOPS"], rel_tol=1e-5)
    assert math.isclose(
        578416648 / (1024 * 1024), dict_perf_metrics["Data Moved (MB)"], rel_tol=1e-5
    )


def test_conv_perf_metrics():

    df = perf_analyzer.build_df_perf_metrics(conv_events)
    assert df.shape == (10, 20)

    df_conv = df[df["perf model"].str.contains("jax_conv")]
    df_metrics = perf_analyzer.summarize_df_perf_metrics(
        df_conv, agg_metrics=["mean", "std"]
    )
    assert df_metrics.shape == (2, 20)
