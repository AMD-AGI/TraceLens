###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure basic logging to stdout with WARNING level
logging.basicConfig(
    stream=sys.stdout,  # Output to console
    level=logging.WARNING,  # Set the minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from TraceLens import TreePerfAnalyzer, JaxTreePerfAnalyzer


def generate_roofline_plot(
    df_ops, peak_tflops=700, peak_bandwidth=5.3, output_filename="gemm_roofline", output_dir="./", output_format="png"
):
    """
    This function generates a roofline plot based on the provided performance metrics DataFrame.

    Args:
    - df_ops (pd.DataFrame): DataFrame containing performance metrics with columns "FLOPS/Byte" and "TFLOPS/s".
    - peak_tflops (float): The peak computational performance of the hardware in TFLOPS/s.
    - peak_bandwidth (float): The peak memory bandwidth of the hardware in TB/s.
    - prefix (str): Prefix for the output plot filename.
    - outdir (str): Directory to save the output plot.

    Outputs:
    - Saves the roofline plot as a PNG file in the specified output directory.

    References:
    - TraceLens/examples/roofline_plots_example.ipynb
    - JaxTrace_Analysis/gemm_roofline.py
    """
    assert "FLOPS/Byte" in df_ops.columns and "TFLOPS/s" in df_ops.columns, "Input dataframe must contain 'FLOPS/Byte' and 'TFLOPS/s' columns."
    print(f"Using peak_bandwidth: {peak_bandwidth} TB/s, peak_tflops: {peak_tflops} TFLOPS/s")
    
    x_realized_intensity = list(df_ops["FLOPS/Byte"])
    y_realized_performance = list(df_ops["TFLOPS/s"])

    # Compute intensity and bounds
    log_max_intensity = np.log10(
        max(x_realized_intensity) * 2
    )  # 2 for better visualization
    log_min_intensity = np.log10(
        min(x_realized_intensity) / 2
    )  # 2 for better visualization
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
        color="blue",
        label="Performance data",
    )
    plt.xlabel("Operational Intensity (FLOPs/Byte)")
    plt.ylabel("Performance (TFLOPS/s)")
    plt.title("Roofline Model")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_filename + '.' + output_format), format=output_format)
    plt.show()
    print(f"Outputs saved to {output_dir}/{output_filename}.{output_format}")


def main():
    """Generate roofline analysis from trace file or from performance metrics DataFrame.
    
    Note: Pytorch trace event filtering by op names or by op category is supported.
          Jax trace event filtering is only supported by op category.
          
    op_cat example:
        for pytorch ['GEMM', 'CONV', 'SDPA', 'UnaryElementwise', 'BinaryElementwise'];
        for jax ['GEMM', 'CONV', 'TE'];
        More details see torch_op_mapping.py and jax_op_mapping.py in TraceLens/TreePerf/ 
    
    Usage example:
    python3 TraceLens/Reporting/generate_roofline_analysis.py --profile_path $trace_jax --op_cat GEMM 
    python3 TraceLens/Reporting/generate_roofline_analysis.py --profile_path $trace_pytorch --op_names 'aten::copy_' 
    python3 TraceLens/Reporting/generate_roofline_analysis.py --profile_path $trace_pytorch --op_cat GEMM  
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
        "--op_cat",
        type=str,
        default="GEMM",
        help="Filter event by op category.",
    )
    parser.add_argument(
        "--op_names",
        type=str,
        default=None,
        help="Filter Torch operation by names for roofline analysis. Example: 'aten::matmul', 'aten::copy_', ",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='./',
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

    # TODO peak performance configs from mi300x_config.yaml, mi355x_config.yaml, etc.
    peak_bandwidth = args.peak_bandwidth  # TB/s
    peak_tflops = args.peak_tflops  # TFLOPS/s
    op_cat = args.op_cat
    op_names = args.op_names
    profile_path = args.profile_path
    output_dir = args.output_path
    output_filename = args.output_filename
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input trace file
    if profile_path.endswith("xplane.pb"):
        perf_analyzer = JaxTreePerfAnalyzer.from_file(profile_path)
        op_events = [
            event
            for event in perf_analyzer.tree.events
            if event.get("gpu_kernel_op_cat", "None").lower() == op_cat.lower()
        ]
    elif profile_path.endswith("trace.json"):
        perf_analyzer = TreePerfAnalyzer.from_file(profile_path)
        if not op_names:
            op_names = perf_analyzer.dict_cat2names[op_cat.upper()]
        else:
            op_names = [op_names] # TODO support list input
        op_events = [
            event for event in perf_analyzer.tree.events if event["name"] in op_names
        ]
    else:
        logger.warning(f"Trace file {profile_path} format not recognized.")
        sys.exit(0)

    assert len(op_events) > 0, f"No events found for category '{op_cat}' in the trace."
    print(f"Number of events found for category '{op_cat}': {len(op_events)}")
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
