###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
TraceLens profiling wrapper for xDiT + FLUX.1 (or other DiT models).

Two-pass profiling analogous to vLLM/SGLang graph-capture patches:

  Pass 1 (shape):  torch.compile with max-autotune-no-cudagraphs.
      Each compiled kernel dispatches individually, so the profiler
      captures per-kernel Concrete Inputs, Input Dims, Input type.

  Pass 2 (timing): torch.compile with max-autotune (with cudagraphs).
      Kernels replay via hipGraphLaunch — representative timing.

  Merge:  Handled at the tree level by
      TraceLens.Trace2Tree.trace_capture_merge_diffusion, which
      re-parents GPU kernel nodes under synthetic cpu_op events
      carrying shape metadata from the shape trace.

Usage (single GPU):
    python run_with_profiling.py \\
        --model /path/to/FLUX.1-dev \\
        --trace-dir /path/to/traces \\
        [--height 1024] [--width 1024] [--steps 20]

Usage (multi-GPU via torchrun):
    torchrun --nproc-per-node=8 run_with_profiling.py \\
        --model /path/to/FLUX.1-dev \\
        --trace-dir /path/to/traces \\
        --ulysses-degree 8

Output:
    <trace-dir>/flux_<H>x<W>_rank<R>_timing.json.gz   (graph-replay trace)
    <trace-dir>/flux_<H>x<W>_rank<R>_shapes.json.gz   (shape trace)

    When generating a perf report, pass both traces:
        from TraceLens.Trace2Tree.trace_capture_merge_diffusion import (
            merge_diffusion_shape_trace,
        )
        # Build tree from timing trace, then merge shapes:
        augmented_tree = merge_diffusion_shape_trace(shape_path, timing_tree)
"""

import argparse
import glob
import os
import sys

import torch
import torch.distributed as dist


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to model directory")
    p.add_argument("--trace-dir", required=True, help="Directory to save traces")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--prompt", default="a photo of a cat")
    p.add_argument("--ulysses-degree", type=int, default=1,
                   help="Ulysses sequence parallel degree (1 = no SP)")
    p.add_argument("--no-compile", action="store_true",
                   help="Skip torch.compile (eager mode)")
    p.add_argument("--warmup-steps", type=int, default=1,
                   help="Number of untraced warmup runs before profiling")
    p.add_argument("--shapes-only", action="store_true",
                   help="Single-pass: shapes only (no cudagraphs, no timing pass)")
    return p.parse_args()


def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, dist.get_world_size()


def build_pipeline(args, rank, world_size, compile_mode):
    """Load the xDiT FLUX pipeline."""
    use_xfuser = world_size > 1
    if use_xfuser:
        try:
            from xfuser import xFuserFluxPipeline, xFuserArgs
        except ImportError:
            use_xfuser = False

    if not use_xfuser:
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        if not args.no_compile:
            pipe.transformer = torch.compile(
                pipe.transformer,
                mode=compile_mode,
            )
        return pipe, None

    engine_args = xFuserArgs(
        model=args.model,
        ulysses_degree=args.ulysses_degree,
        use_torch_compile=not args.no_compile,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
    )
    engine_config, input_config = engine_args.create_config()
    pipe = xFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=args.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )
    pipe.prepare_run(input_config)
    return pipe, input_config


def run_pipeline(pipe, input_config, args):
    """Run one generation step."""
    kwargs = dict(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        output_type="latent",
    )
    pipe(**kwargs)


def profile_one_pass(pipe, input_config, args, trace_dir, worker_name):
    """Profile a single forward pass, save trace, return the trace path."""
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            trace_dir,
            worker_name=worker_name,
            use_gzip=True,
        ),
    ) as prof:
        with torch.profiler.record_function(
            f"dit_forward_{args.height}x{args.width}_steps{args.steps}"
        ):
            with torch.no_grad():
                run_pipeline(pipe, input_config, args)

    traces = sorted(glob.glob(os.path.join(trace_dir, f"{worker_name}.*.json.gz")))
    return traces[-1] if traces else None


def main():
    args = parse_args()
    rank, world_size = setup_distributed()

    os.makedirs(args.trace_dir, exist_ok=True)

    # ── Enable kernel shape profiler BEFORE torch.compile ──
    _ksp_available = False
    try:
        from kernel_shape_profiler import enable as ksp_enable
        from kernel_shape_profiler import disable as ksp_disable
        ksp_enable()
        _ksp_available = True
        print(f"[rank {rank}] kernel_shape_profiler enabled")
    except ImportError:
        print(f"[rank {rank}] kernel_shape_profiler not found — AITER shapes will be missing")

    if args.no_compile or args.shapes_only:
        # ── Single-pass mode ──
        compile_mode = "max-autotune-no-cudagraphs"
        pipe, input_config = build_pipeline(args, rank, world_size, compile_mode)

        print(f"[rank {rank}] Warmup ({args.warmup_steps} step(s))...")
        for _ in range(args.warmup_steps):
            with torch.no_grad():
                run_pipeline(pipe, input_config, args)
        if world_size > 1:
            dist.barrier()

        worker_name = f"flux_{args.height}x{args.width}_rank{rank}_shapes"
        print(f"[rank {rank}] Profiling (shapes only)...")
        trace_path = profile_one_pass(pipe, input_config, args, args.trace_dir, worker_name)

        if _ksp_available:
            ksp_disable()

        if rank == 0 and trace_path:
            print(f"\nShape trace saved to: {trace_path}")
    else:
        # ── Two-pass mode ──

        # Pass 1: shapes (no cudagraphs)
        print(f"[rank {rank}] Pass 1: compiling (max-autotune-no-cudagraphs) for shapes...")
        pipe, input_config = build_pipeline(
            args, rank, world_size,
            compile_mode="max-autotune-no-cudagraphs",
        )

        print(f"[rank {rank}] Warmup ({args.warmup_steps} step(s))...")
        for _ in range(args.warmup_steps):
            with torch.no_grad():
                run_pipeline(pipe, input_config, args)
        if world_size > 1:
            dist.barrier()

        shape_worker = f"flux_{args.height}x{args.width}_rank{rank}_shapes"
        print(f"[rank {rank}] Profiling shapes...")
        shape_trace = profile_one_pass(pipe, input_config, args, args.trace_dir, shape_worker)
        print(f"[rank {rank}] Shape trace: {shape_trace}")

        # Pass 2: timing (with cudagraphs)
        torch._dynamo.reset()
        print(f"[rank {rank}] Pass 2: recompiling (max-autotune) for timing...")
        pipe, input_config = build_pipeline(
            args, rank, world_size,
            compile_mode="max-autotune",
        )

        print(f"[rank {rank}] Warmup ({args.warmup_steps} step(s)) — triggers graph capture...")
        for _ in range(args.warmup_steps):
            with torch.no_grad():
                run_pipeline(pipe, input_config, args)
        if world_size > 1:
            dist.barrier()

        timing_worker = f"flux_{args.height}x{args.width}_rank{rank}_timing"
        print(f"[rank {rank}] Profiling timing (graph replay)...")
        timing_trace = profile_one_pass(pipe, input_config, args, args.trace_dir, timing_worker)
        print(f"[rank {rank}] Timing trace: {timing_trace}")

        if _ksp_available:
            ksp_disable()

        if rank == 0:
            print(f"\nTraces saved to: {args.trace_dir}/")
            print(f"  Shape trace:  {shape_trace}")
            print(f"  Timing trace: {timing_trace}")
            print(f"\nTo generate a perf report with merged shapes:")
            print(f"  from TraceLens.Trace2Tree.trace_capture_merge_diffusion import merge_diffusion_shape_trace")
            print(f"  # Pass shape_trace_path and the built timing tree to merge_diffusion_shape_trace()")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
