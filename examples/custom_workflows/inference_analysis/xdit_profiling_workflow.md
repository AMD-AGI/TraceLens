# xDiT Diffusion Profiling Instructions

## Step 1: Build the Patched Docker Image

Base image: `rocm/pytorch-xdit:v26.6`

```bash
bash examples/custom_workflows/inference_analysis/build_docker_xdit.sh \
    v26.6 /path/to/TraceLens -t tracelens-xdit:v26.6
```

To use a custom base image:

```bash
bash examples/custom_workflows/inference_analysis/build_docker_xdit.sh \
    v26.6 /path/to/TraceLens --base-image my-custom/xdit:latest -t tracelens-xdit:custom
```

## Step 2: Run Profiling

```bash
docker run --rm \
    --device /dev/kfd --device /dev/dri \
    --group-add video --ipc=host \
    -e HSA_NO_SCRATCH_RECLAIM=1 \
    -e ROCR_VISIBLE_DEVICES=0 \
    -e HIP_VISIBLE_DEVICES=0 \
    -e MODEL=/path/to/FLUX.1-dev \
    -e XDIT_MODEL_NAME=FLUX.1-dev \
    -e RESULT_DIR=/workspace/output \
    -e RUNNER_TYPE=mi300x \
    -e TP=1 \
    -e PROFILE=1 \
    -e XDIT_SUPPORTS_PROFILER=1 \
    -e XDIT_USE_TORCH_COMPILE=1 \
    -e XDIT_ATTENTION_BACKEND=aiter \
    -e XDIT_HEIGHT=1024 -e XDIT_WIDTH=1024 \
    -e XDIT_NUM_STEPS=28 \
    -e XDIT_NUM_ITERATIONS=25 \
    -e XDIT_WARMUP_CALLS=5 \
    -e XDIT_GUIDANCE_SCALE=4.0 \
    -e XDIT_ULYSSES_DEGREE=1 \
    -e XDIT_PROMPT="a photo of a cat" \
    -e EXTRA_XDIT_ARGS="--profile_wait 1 --profile_capture_phase" \
    -v /path/to/hf_cache:/root/.cache/huggingface \
    -v /path/to/output:/workspace/output \
    -v /path/to/Magpie/scripts/benchmark:/workspace/bench_scripts:ro \
    tracelens-xdit:v26.6 \
    bash /workspace/bench_scripts/xdit_mi300x.sh
```

### Key flags

| Flag | Purpose |
|------|---------|
| `PROFILE=1` | Enable xDiT profiler |
| `XDIT_SUPPORTS_PROFILER=1` | Gate for profiling flags in xdit_bench_common.sh |
| `--profile_wait 1` | Fix for ROCTracer empty trace bug with `wait=0` ([ROCm #6102](https://github.com/ROCm/ROCm/issues/6102)) |
| `--profile_capture_phase` | Enable capture trace for TraceLens shape merging (rank 0 only) |

### Output

```
<output_dir>/xdit_run.<id>/
    capture_traces/capture_rank_0.json.gz   # Capture trace (shapes)
    profile_trace_rank_0.json.gz            # Timing trace (graph replay)
```

## Step 3: Generate Performance Report

```bash
python TraceLens/Reporting/generate_perf_report_pytorch.py \
    --profile_json_path <output>/xdit_run.<id>/profile_trace_rank_0.json.gz \
    --capture_trace <output>/xdit_run.<id>/capture_traces \
    --output_csvs_dir <report_dir> \
    --output_xlsx_path <report_dir>/perf_report.xlsx \
    --include_call_stack \
    --enable_pseudo_ops \
    --gpu_arch_json_path TraceLens/Agent/Analysis/utils/arch/MI300X.json
```

### Expected output (FLUX.1-dev 1024x1024)

| Category | % of kernel time |
|----------|-----------------|
| GEMM | ~63% |
| SDPA_fwd | ~18% |
| Triton | ~10% |
| CONV_fwd | ~5% |

## How It Works

The patch splits torch.compile's compilation from graph capture:

1. **Step 1 (no profiler)**: `_compile_model()` runs with 1 denoise step
   (compilation + autotuning). Graph capture is deferred.
2. **Step 2 (with profiler)**: One more `_run_timed_pipe()` triggers graph
   capture (`hipStreamBeginCapture`). The profiler records per-kernel `cpu_op`
   events with `Input Dims` and `Concrete Inputs`.
3. **Replay profiling**: xDiT's `profile()` method records graph replay
   timing via `hipGraphLaunch`.
4. **Merge**: TraceLens grafts the capture subtree (shapes) into the replay
   tree (timing), enabling roofline analysis.

See [docs/xdit_e2e_workflow.md](../../../docs/xdit_e2e_workflow.md) for full
technical details.
