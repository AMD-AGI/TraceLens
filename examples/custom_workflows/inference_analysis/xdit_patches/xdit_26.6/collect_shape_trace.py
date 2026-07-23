###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Collect a shape trace for diffusion models using xDiT's built-in runner.

This script wraps xDiT's runner to collect a shape trace with per-kernel
metadata (Input Dims, Concrete Inputs, Input type).  It does this by
profiling the graph CAPTURE phase — the first forward pass after
torch.compile(mode="max-autotune"), where kernels dispatch individually.

Use the EXACT SAME model configuration (fp8, attention backend, resolution,
etc.) as the customer's timing trace to ensure kernel names match.

Usage:
    # Collect shape trace with same flags as customer's trace
    python collect_shape_trace.py \
        --model /path/to/FLUX.1-dev \
        --use_fp8_gemms \
        --height 1024 --width 1024 \
        --num_inference_steps 28 \
        --guidance_scale 4.0 \
        --output_directory /path/to/output

    # Then merge into customer's timing trace:
    python TraceLens/Reporting/generate_perf_report_pytorch.py \
        --profile_json_path customer_rank_0.trace.json.gz \
        --diffusion_shape_trace output/shape_trace_rank_0.json.gz \
        --output_csvs_dir report/

How it works:
    1. Initializes xDiT runner with user's config (same as production)
    2. Profiles the FIRST forward pass (graph capture phase):
       - torch.compile triggers Inductor compilation + autotune
       - HIP graph capture records kernel sequence
       - During capture, each kernel dispatches individually
       - Profiler records per-kernel cpu_op events with shapes
    3. Saves the capture-phase trace as the shape trace

    Since graph capture runs the same compiled kernels as graph replay,
    kernel names are identical → 100% match rate when merging.
"""

import os
import sys
import logging

import torch
from torch.profiler import profile, record_function, ProfilerActivity


def main():
    # Use xDiT's own argument parsing so all model flags work
    from xfuser.config import FlexibleArgumentParser
    from xfuser import xFuserArgs
    from xfuser.runner import xFuserModelRunner, setup_logging
    from xfuser.core.utils.runner_utils import log

    setup_logging()

    parser = FlexibleArgumentParser(description="Collect shape trace for TraceLens")
    xfuser_args = xFuserArgs.add_runner_args(parser).parse_args()
    args = vars(xfuser_args)

    runner = xFuserModelRunner(args)
    runner.print_args(args)

    input_args = runner.preprocess_args(args)
    runner.initialize(input_args)

    # Profile the FIRST forward pass — this is the graph capture phase.
    # During capture, each kernel dispatches individually via
    # hipLaunchKernel, so the profiler records per-kernel cpu_op events
    # with Input Dims, Concrete Inputs, Input type.
    log("Profiling graph capture phase for shape trace...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with record_function("shape_capture"):
            runner.model._run_pipe(input_args)

    # Save shape trace
    os.makedirs(xfuser_args.output_directory, exist_ok=True)
    from xfuser.core.distributed import get_world_group
    rank = get_world_group().rank
    shape_trace_path = os.path.join(
        xfuser_args.output_directory,
        f"shape_trace_rank_{rank}.json.gz",
    )
    prof.export_chrome_trace(shape_trace_path)
    log(f"Shape trace saved to: {shape_trace_path}")

    log("To merge into a timing trace:")
    log(f"  python TraceLens/Reporting/generate_perf_report_pytorch.py \\")
    log(f"    --profile_json_path <timing_trace>.json.gz \\")
    log(f"    --diffusion_shape_trace {shape_trace_path} \\")
    log(f"    --output_csvs_dir report/")

    runner.cleanup()


if __name__ == "__main__":
    main()
