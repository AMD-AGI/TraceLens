###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
TraceLens profiling wrapper for xDiT + FLUX.1 (or other DiT models).

Single-compilation, two-pass profiling analogous to vLLM/SGLang patches:

  torch.compile(mode="max-autotune") captures HIP/CUDA graphs internally.
  The FIRST forward pass after compilation is the graph capture phase —
  kernels dispatch individually via hipLaunchKernel while the graph is
  being recorded.  Subsequent forward passes replay the graph via
  hipGraphLaunch.

  Pass 1 (shape):  Profile the graph CAPTURE pass (first forward after
      compile).  Individual kernel dispatches → per-kernel Concrete Inputs,
      Input Dims, Input type.

  Pass 2 (timing): Profile a graph REPLAY pass (subsequent forward).
      hipGraphLaunch → representative timing.

  Merge:  Handled at the tree level by
      TraceLens.Trace2Tree.trace_capture_merge_experimental.

  Same compilation for both passes → identical kernel names → 100% match.

Usage (single GPU):
    python run_with_profiling.py \\
        --model /path/to/FLUX.1-dev \\
        --trace-dir /path/to/traces \\
        [--height 1024] [--width 1024] [--steps 20]

Output:
    <trace-dir>/flux_<H>x<W>_rank<R>_shapes.json.gz   (capture-phase trace)
    <trace-dir>/flux_<H>x<W>_rank<R>_timing.json.gz   (replay-phase trace)
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
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--ulysses-degree", type=int, default=1,
                   help="Ulysses sequence parallel degree (1 = no SP)")
    p.add_argument("--no-compile", action="store_true",
                   help="Skip torch.compile (eager mode)")
    p.add_argument("--shapes-only", action="store_true",
                   help="Single-pass: capture phase only (shapes, no timing pass)")
    p.add_argument("--warmup-steps", type=int, default=5,
                   help="Number of untraced warmup calls before profiling")
    p.add_argument("--precision", choices=["bf16", "fp8"], default="bf16",
                   help="Model precision (bf16 or fp8)")
    p.add_argument("--attention-backend", default=None,
                   help="Attention backend (e.g. aiter). Sets XDIT_ATTENTION_BACKEND env var.")
    return p.parse_args()


def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, dist.get_world_size()


def build_pipeline(args, rank, world_size):
    """Load the xDiT FLUX pipeline with torch.compile(max-autotune)."""
    # Set attention backend env var if specified
    if args.attention_backend:
        os.environ["XDIT_ATTENTION_BACKEND"] = args.attention_backend

    model_dtype = torch.bfloat16

    # xfuser requires torchrun (RANK/WORLD_SIZE env vars).
    # For single-GPU without torchrun, fall back to plain diffusers.
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
            torch_dtype=model_dtype,
        ).to("cuda")
        if not args.no_compile:
            pipe.transformer = torch.compile(
                pipe.transformer,
                mode="max-autotune",
                fullgraph=True,
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
        torch_dtype=model_dtype,
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
        guidance_scale=args.guidance_scale,
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
            f"execute_diffusion_{args.height}x{args.width}_{args.precision}"
        ):
            with torch.no_grad():
                run_pipeline(pipe, input_config, args)

    traces = sorted(glob.glob(os.path.join(trace_dir, f"{worker_name}.*.json.gz")))
    return traces[-1] if traces else None


def main():
    args = parse_args()
    rank, world_size = setup_distributed()

    os.makedirs(args.trace_dir, exist_ok=True)

    # Note: kernel_shape_profiler is NOT used here.  AITER attention
    # kernels already get Input Dims via their aten::_flash_attention_forward
    # parent in the timing trace.  Avoiding the profiler wrappers lets us
    # keep fullgraph=True → one monolithic compiled graph per denoise step,
    # matching production execution structure.

    # ── Compile once with max-autotune ──
    print(f"[rank {rank}] Loading pipeline and compiling (max-autotune)...")
    pipe, input_config = build_pipeline(args, rank, world_size)

    # ── Pass 1: Profile the graph CAPTURE phase (first forward) ──
    # The first forward pass after torch.compile(mode="max-autotune")
    # triggers Inductor compilation + autotune + HIP graph capture.
    # During capture, each kernel dispatches individually via
    # hipLaunchKernel — the profiler records per-kernel cpu_op events
    # with Input Dims, Concrete Inputs, Input type.
    shape_worker = f"flux_{args.height}x{args.width}_rank{rank}_shapes"
    print(f"[rank {rank}] Pass 1: profiling graph capture phase (shapes)...")
    shape_trace = profile_one_pass(pipe, input_config, args, args.trace_dir, shape_worker)
    print(f"[rank {rank}] Shape trace: {shape_trace}")

    if world_size > 1:
        dist.barrier()

    if args.shapes_only:
        if rank == 0:
            print(f"\nShape trace saved to: {shape_trace}")
    else:
        # ── Warmup: run additional forward passes to stabilize graph replay ──
        if args.warmup_steps > 0:
            print(f"[rank {rank}] Warmup ({args.warmup_steps} step(s))...")
            for _ in range(args.warmup_steps):
                with torch.no_grad():
                    run_pipeline(pipe, input_config, args)
            if world_size > 1:
                dist.barrier()

        # ── Pass 2: Profile graph REPLAY (subsequent forward) ──
        # The graph is now captured and warmed up. This forward pass
        # replays via hipGraphLaunch — representative production timing.
        timing_worker = f"flux_{args.height}x{args.width}_rank{rank}_timing"
        print(f"[rank {rank}] Pass 2: profiling graph replay (timing)...")
        timing_trace = profile_one_pass(pipe, input_config, args, args.trace_dir, timing_worker)
        print(f"[rank {rank}] Timing trace: {timing_trace}")

        if rank == 0:
            print(f"\nTraces saved to: {args.trace_dir}/")
            print(f"  Shape trace:  {shape_trace}")
            print(f"  Timing trace: {timing_trace}")
            print(f"\nGenerate perf report with merged shapes:")
            print(f"  python TraceLens/Reporting/generate_perf_report_pytorch.py \\")
            print(f"    --profile_json_path {timing_trace} \\")
            print(f"    --capture_trace {shape_trace} \\")
            print(f"    --output_csvs_dir report/")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
