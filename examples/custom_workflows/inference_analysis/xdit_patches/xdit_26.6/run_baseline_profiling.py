###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Minimal baseline profiling for FLUX.1 on unpatched rocm/pytorch-xdit.

No kernel_shape_profiler — just torch.profiler with record_shapes=True.
AITER FMHA shapes will be missing; GEMM and Triton kernel metadata will
be present via standard profiler instrumentation.  Inductor cudagraph
capture is disabled so each kernel dispatches individually and the
profiler records per-kernel Concrete Inputs / Input Dims.

Usage:
    python /workspace/run_baseline_profiling.py \
        --model /model \
        --trace-dir /traces \
        [--height 512] [--width 512] [--steps 20]
"""

import argparse
import os

import torch
from torch.profiler import ProfilerActivity


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to FLUX.1-dev model")
    p.add_argument("--trace-dir", required=True, help="Directory to save traces")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--prompt", default="a photo of a cat")
    p.add_argument("--no-compile", action="store_true",
                   help="Skip torch.compile (eager mode)")
    p.add_argument("--warmup-steps", type=int, default=1,
                   help="Warmup runs before profiling (triggers compilation)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.trace_dir, exist_ok=True)

    from diffusers import FluxPipeline

    print("Loading pipeline...")
    pipe = FluxPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    if not args.no_compile:
        # Use max-autotune-no-cudagraphs so each kernel dispatches
        # individually — the profiler then captures per-kernel metadata
        # (Concrete Inputs, Input Dims) needed for roofline analysis.
        # "max-autotune" would force cudagraph capture, making all kernels
        # replay via a single hipGraphLaunch with no per-kernel metadata.
        print("Compiling transformer (max-autotune-no-cudagraphs)...")
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
        )

    gen_kwargs = dict(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        output_type="latent",
    )

    # Warmup: triggers compilation, not profiled
    print(f"Warmup ({args.warmup_steps} step(s))...")
    for _ in range(args.warmup_steps):
        with torch.no_grad():
            pipe(**gen_kwargs)

    # Profiled run
    worker_name = f"flux_{args.height}x{args.width}_baseline"
    print("Profiling...")

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            args.trace_dir,
            worker_name=worker_name,
            use_gzip=True,
        ),
    ):
        with torch.no_grad():
            pipe(**gen_kwargs)

    print(f"Trace saved to: {args.trace_dir}/")


if __name__ == "__main__":
    main()
