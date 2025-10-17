###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure basic logging to stdout with DEBUG level
logging.basicConfig(
    stream=sys.stdout,  # Output to console
    level=logging.DEBUG,  # Set the minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from TraceLens import TreePerfAnalyzer, JaxTreePerfAnalyzer


def generate_roofline_plot(
    df_ops,
    peak_tflops=700,
    peak_bandwidth=5.3,
    output_filename="gemm_roofline",
    output_dir="./",
    output_format="png",
):
    """
    Generates a roofline plot based on the provided performance metrics DataFrame.

    Args:
        df_ops (pd.DataFrame): DataFrame containing performance metrics with columns "FLOPS/Byte" and "TFLOPS/s".
        peak_tflops (float, optional): The peak computational performance of the hardware in TFLOPS/s. Default is 700.
        peak_bandwidth (float, optional): The peak memory bandwidth of the hardware in TB/s. Default is 5.3.
        output_filename (str, optional): Filename (without extension) for the output plot. Default is "gemm_roofline".
        output_dir (str, optional): Directory to save the output plot. Default is "./".
        output_format (str, optional): File format for the output plot (e.g., "png", "pdf", "svg"). Default is "png".

    Outputs:
        Saves the roofline plot as a file in the specified output directory and format.

    References:
        - TraceLens/examples/roofline_plots_example.ipynb
        - JaxTrace_Analysis/gemm_roofline.py
    """
    required_columns = {"FLOPS/Byte", "TFLOPS/s"}
    missing = required_columns - set(df_ops.columns)
    if missing:
        raise ValueError(f"Input dataframe must contain columns: {', '.join(required_columns)}. Missing: {', '.join(missing)}")
    logger.info(
        f"Using peak_bandwidth: {peak_bandwidth} TB/s, peak_tflops: {peak_tflops} TFLOPS/s"
    )

    x_realized_intensity = list(df_ops["FLOPS/Byte"])
    y_realized_performance = list(df_ops["TFLOPS/s"])

    # Filter to strictly positive values for log-scale plotting
    x_realized_intensity_pos = [x for x in x_realized_intensity if x > 0]
    if not x_realized_intensity_pos:
        raise ValueError("All realized intensity values are zero or negative; cannot plot on log scale.")

    # Compute intensity and bounds
    log_max_intensity = np.log10(
        max(x_realized_intensity_pos) * 2
    )  # times 2 for better visualization
    log_min_intensity = np.log10(
        min(x_realized_intensity_pos) / 2
    )  # divided by 2 for better visualization
    x_intensity = np.logspace(log_min_intensity, log_max_intensity, 100)  # FLOPs/Byte
    y_memory_bound = x_intensity * peak_bandwidth  # FLOPS/Byte * TB/s = TFLOPS/s
    y_compute_bound = np.full_like(x_intensity, peak_tflops)  # TFLOPS/s

    # Roofline
    plt.figure(figsize=(8, 5))
    plt.loglog(
        x_intensity,
        y_memory_bound,
        color="orange",
        linestyle="--",
        label=f"Memory Bound ({peak_bandwidth} TB/s)",
    )
    plt.loglog(
        x_intensity,
        y_compute_bound,
        color="red",
        linestyle=":",
        label=f"Compute Bound ({peak_tflops} TFLOPS/s)",
    )
    plt.scatter(
        x_realized_intensity,
        y_realized_performance,
        marker="x",
        s=6,
        color="blue",
        label="Performance data",
    )
    plt.xlabel("Operational Intensity (FLOPs/Byte)")
    plt.ylabel("Performance (TFLOPS/s)")
    plt.title("Roofline Model")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, output_filename + "." + output_format),
        format=output_format,
    )
    plt.close()
    logger.info(f"Outputs saved to {output_dir}/{output_filename}.{output_format}")


def main():
    """Generate roofline analysis from trace file or from performance metrics DataFrame.

    Note: PyTorch trace event filtering is by 'op_names' or by 'op_cat'.
          JAX trace event filtering is by 'gpu_kernel_op_cat'. 'op_names' are implemented but not tested.

    op_cat example:
        for pytorch ['GEMM', 'CONV', 'SDPA', 'UnaryElementwise', 'BinaryElementwise'];
        for JAX ['GEMM', 'CONV', 'TE'];
        More details see torch_op_mapping.py and jax_op_mapping.py in TraceLens/TreePerf/

    Usage example:
    python3 TraceLens/Reporting/generate_roofline_analysis.py --profile_path $trace --op_cats GEMM
    python3 TraceLens/Reporting/generate_roofline_analysis.py --profile_path $trace --op_cats GEMM CONV
    python3 TraceLens/Reporting/generate_roofline_analysis.py --profile_path $trace_pytorch --op_names 'aten::copy_'
    python3 TraceLens/Reporting/generate_roofline_analysis.py --profile_path $trace_pytorch --op_names 'aten::copy_' 'aten::matmul_'
    """

    parser = argparse.ArgumentParser(
        description="Generate roofline analysis from trace file or from performance metrics DataFrame."
    )
    parser.add_argument(
        "--profile_path",
        type=str,
        required=True,
        help="Path to the trace file: pytorch trace.json or jax xplane.pb",
    )
    parser.add_argument(
        "--op_cats",
        type=str,
        nargs="+",
        default=["GEMM"],
        help="Filter event by op category. Example: --op_cat GEMM CONV",
    )
    parser.add_argument(
        "--op_names",
        type=str,
        nargs="+",
        default=[],
        help="Filter operations by names for roofline analysis (Torch and JAX). Example: --op_names aten::matmul aten::copy_",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Directory to save the output roofline plot.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="roofline",
        help="Prefix for the output plot filename.",
    )
    parser.add_argument(
        "--peak_tflops",
        type=float,
        default=1307.4,
        help="Peak computational performance in TFLOPS/s.",
    )
    parser.add_argument(
        "--peak_bandwidth",
        type=float,
        default=5.3,
        help="Peak memory bandwidth in TB/s.",
    )
    args = parser.parse_args()

    # Config peak performance (TODO) mi300x_config.yaml, mi355x_config.yaml, etc.
    peak_bandwidth = args.peak_bandwidth  # TB/s
    peak_tflops = args.peak_tflops  # TFLOPS/s
    op_cats = args.op_cats
    op_names = args.op_names
    profile_path = args.profile_path
    output_dir = args.output_path
    output_filename = args.output_filename
    os.makedirs(output_dir, exist_ok=True)

    # Load input trace file & filter events
    if profile_path.endswith("xplane.pb"):
        perf_analyzer = JaxTreePerfAnalyzer.from_file(profile_path)
        if not op_names:
            logger.info(f"Filtering Jax events by op category: {op_cats}")
            op_events = [
                event
                for event in perf_analyzer.tree.events
                for op_cat in op_cats
                if event.get("gpu_kernel_op_cat", "None").lower() == op_cat.lower()
            ]
        else:
            logger.info(f"Filtering Jax events by op names: {op_names}")
            op_events = [
                event
                for event in perf_analyzer.tree.events
                if event["name"] in op_names
            ]

    elif profile_path.endswith("trace.json"):
        perf_analyzer = TreePerfAnalyzer.from_file(profile_path)
        if not op_names:
            logger.info(f"Filtering Torch events by op category: {op_cats}")
            op_names = [
                item
                for op_cat in op_cats
                for item in perf_analyzer.dict_cat2names.get(op_cat.upper(), [])
            ]
        else:
            logger.info(f"Filtering Torch events by op names: {op_names}")
        op_events = [
            event for event in perf_analyzer.tree.events if event["name"] in op_names
        ]
    else:
        logger.error(f"Trace file {profile_path} format not recognized.")
        sys.exit(1)

    if not op_events:
        logger.error("No events selected in the trace.")
        sys.exit(1)
    df_ops = perf_analyzer.build_df_perf_metrics(op_events)

    # Generate roofline plot
    generate_roofline_plot(
        df_ops,
        peak_tflops=peak_tflops,
        peak_bandwidth=peak_bandwidth,
        output_dir=output_dir,
        output_filename=output_filename,
    )


if __name__ == "__main__":
    main()
