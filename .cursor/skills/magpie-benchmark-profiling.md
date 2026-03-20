<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: magpie-benchmark-profiling
description: Run Magpie LLM inference benchmarks (vLLM/SGLang) and collect PyTorch profiler traces on remote GPU nodes. Use when the user asks to benchmark, profile, or collect traces for LLM inference workloads using the Magpie framework.
---

# Magpie Benchmark + Profiling

Run LLM inference benchmarks via the Magpie framework and collect PyTorch profiler traces for downstream analysis.

## Prerequisites

- SSH access to a GPU node with Docker and AMD/NVIDIA GPUs
- A conda environment with Magpie installed (`pip install -e .` from the Magpie repo root)
- TraceLens installed on the host if TraceLens analysis is desired (`pip install -e .` from `TraceLens-internal/`)
- HuggingFace model weights cached or an `HF_TOKEN` for gated models

## Workflow

### Step 0: Gather Execution Environment Details

Before doing anything else, check whether the user has provided all three of these:

1. **Target node** — the hostname or IP to SSH into (e.g., `gpu-node-3`)
2. **Conda environment name** — the env where Magpie is installed
3. **Conda install path** — defaults to `~/miniconda3`; ask only if conda activation fails

If any are missing, ask the user before proceeding. Example:

> "To run the benchmark I need a few details about your execution environment:
> 1. Which node should I SSH into?
> 2. What is the name of the conda environment with Magpie installed?
> 3. Is conda installed at `~/miniconda3`, or a different path?"

Do not guess node names or environment names — these are site-specific and getting them wrong wastes time on SSH failures or missing-package errors.

### Step 1: Read the Benchmark Config

The user provides a YAML config (typically under `Magpie/examples/`). Read it to understand the run parameters.

Key fields:

```yaml
benchmark:
  framework: vllm | sglang
  model: <HuggingFace model name>
  precision: fp8 | fp16 | bf16 | fp4

  envs:
    TP: 8              # tensor parallelism (GPU count)
    CONC: 32           # request concurrency
    ISL: 1024          # input sequence length
    OSL: 1024          # output sequence length

  profiler:
    torch_profiler:
      enabled: true    # MUST be true to collect traces
    system_profiler:
      enabled: false

  benchmark_script: "dsr1_fp8_mi300x.sh"  # optional override
  timeout_seconds: 3600
  hf_cache_path: "/path/to/hf/cache"
```

### Step 1b: Check Docker Image for Profiling Patches

After reading the config, ask the user whether their Docker image already includes TraceLens profiling patches. These patches are required for capturing kernel-level detail (shapes, roofline data, call stacks) in both eager and graph-mode traces.

Ask:

> "Does the Docker image in your config (or the default image Magpie will select) already have TraceLens profiling patches applied? If you're unsure, I can build a patched image for you."

#### If the user's image is already patched

Proceed to Step 2. No changes needed.

#### If the user needs a patched image

Present the supported inference server and version options based on the `framework` field in the YAML config:

**For vLLM (`framework: vllm`):**

> "Which vLLM version would you like to build a patched image for?
>
> | Option | vLLM Version | Base Image |
> |--------|-------------|------------|
> | 1 | v0.14.0 | `rocm/vllm-dev:preview_releases_rocm_v0.14.0_20260120` |
> | 2 | v0.15.0 | `rocm/vllm-dev:preview_releases_rocm_v0.15.0_20260130` |
> | 3 | v0.16.0 | `rocm/vllm-dev:preview_rocm70_releases_rocm_v0.16.0_20260223` |"
> | 4 | v0.17.0 | `vllm/vllm-openai-rocm:v0.17.0` |"

Once the user selects a version, build the patched image on the remote node:

```bash
ssh <node> "cd <TraceLens_repo> && \
  bash examples/custom_workflows/inference_analysis/build_docker_vllm.sh \
    <version_tag> \
    <TraceLens_repo> \
    -t tracelens-vllm"
```

Where `<version_tag>` is `v14`, `v15`, `v16`, or `v17` based on the user's selection.

**For SGLang (`framework: sglang`):**

> "Which GPU type are you targeting?
>
> | Option | GPU | Base Image |
> |--------|-----|------------|
> | 1 | MI300X | `lmsysorg/sglang:v0.5.9-rocm700-mi30x` |
> | 2 | MI355X | `lmsysorg/sglang:v0.5.9-rocm700-mi35x` |"

Once the user selects, build the patched image on the remote node:

```bash
ssh <node> "cd <TraceLens_repo> && \
  bash examples/custom_workflows/inference_analysis/build_docker_sglang_v059.sh \
    <TraceLens_repo> \
    <gpu_type>"
```

Where `<gpu_type>` is `mi300` or `mi355` based on the user's selection. The script starts a container, applies sglang roofline patches, and installs TraceLens inside it.

**After building:** Update the `docker_image` field in the benchmark YAML config to use the newly built image:

```yaml
benchmark:
  framework: vllm  # or sglang
  docker_image: tracelens-vllm  # or the sglang container name
  ...
```

This overrides the default auto-selected image from `benchmark_images.yaml`.

#### If the user declines

Proceed with the user's chosen image. Warn that traces may lack kernel-level detail for graph-replayed operations, and roofline analysis may not be available.

### Step 2: Ensure Profiling Is Enabled

If the user wants a profiler trace, verify `profiler.torch_profiler.enabled` is `true` in the YAML. If it is `false`, edit it to `true` before running.

When enabled, the benchmarker automatically sets these env vars inside the container:
- `PROFILE=1`
- `VLLM_TORCH_PROFILER_DIR=/workspace/torch_trace`
- `SGLANG_TORCH_PROFILER_DIR=/workspace/torch_trace`

No manual env var configuration is needed.

### Step 2b: Ask About Profiler Tuning (vLLM only)

Before running the benchmark, ask the user whether they want to profile a targeted steady-state window or the entire benchmark run.

#### Profiling mode

Ask:

> "Would you like to profile a **targeted steady-state window** (recommended — smaller traces, captures steady-state decode) or the **entire benchmark** run?"

##### Option A: Targeted steady-state window (delay + max iterations)

Compute recommended `delay_iterations` and `max_iterations` from the YAML config values. Read `OSL`, `CONC`, and `RANDOM_RANGE_RATIO` from the `envs` section (default `RANDOM_RANGE_RATIO` to `1.0` if absent).

**Formulas:**

```
max_iters = min(OSL, OSL * 16 / CONC)

if RANDOM_RANGE_RATIO < 1:
    delay_iters = OSL * RANDOM_RANGE_RATIO * 6
else:                          # RANDOM_RANGE_RATIO == 1
    delay_iters = OSL * 5 - max_iters / 2
```

Compute the values, round to integers, then present them to the user for confirmation:

> "Based on your config (OSL=X, CONC=Y, RANDOM_RANGE_RATIO=Z), I'd suggest:
> - **delay_iterations** = A  (skip A engine iterations before profiling)
> - **max_iterations** = B  (profile B iterations then stop)
>
> This targets steady-state decode after warmup. Does this look good, or would you like different values?"

Once the user confirms (or provides overrides), apply **three** temporary edits on the remote node:

**1. Add profiler iteration args to the benchmark script.**

The script lives at `<magpie_repo>/InferenceMAX/benchmarks/<benchmark_script>` (the script name comes from the YAML `benchmark_script` field, defaulting to `vllm_mi300x.sh`). Find the `PROFILER_ARGS` block (the `if [[ "${PROFILE:-}" == "1" ]]` section) and append these lines:

```bash
ssh <node> "sed -i '/profiler-config.torch_profiler_use_gzip/a\\
  PROFILER_ARGS+=(--profiler-config.delay_iterations <DELAY>)\\
  PROFILER_ARGS+=(--profiler-config.max_iterations <MAX>)\\
  PROFILER_ARGS+=(--profiler-config.ignore_frontend True)' \
  <magpie_repo>/InferenceMAX/benchmarks/<benchmark_script>"
```

`ignore_frontend` must be `True` when using delay/max iterations, otherwise the AsyncLLM front-end profiler captures the entire range and adds significant overhead.

**2. Ensure `PROFILER_ARGS` is passed to `vllm serve`.**

The benchmark script may not include `"${PROFILER_ARGS[@]}"` in the `vllm serve` command. Check whether the `vllm serve` line already references `PROFILER_ARGS`. If it does not, inject it before the output redirection (`> $SERVER_LOG`):

```bash
ssh <node> "sed -i 's|\$EXTRA_VLLM_ARGS > \$SERVER_LOG|\$EXTRA_VLLM_ARGS \"\${PROFILER_ARGS[@]}\" > \$SERVER_LOG|' \
  <magpie_repo>/InferenceMAX/benchmarks/<benchmark_script>"
```

Without this, the `PROFILER_ARGS` array is built but never actually passed to the vLLM server.

**3. MANDATORY: Increase `num_prompts` in `benchmark_lib.sh` so the benchmark runs long enough for the profiling window.**

**This patch is always required when using delay + max iterations.** The `run_benchmark_serving` function in `benchmark_lib.sh` **unconditionally overrides** `num_prompts` when `PROFILE=1` — this happens *after* parsing the `--num-prompts` argument, so it stomps whatever value the calling script (e.g. `vllm_mi300x.sh`) passes. Do NOT skip this patch even if the calling script already sets a larger `--num-prompts`. Without this fix, the benchmark finishes before the delay window is reached, and no steady-state trace is captured.

```bash
ssh <node> "sed -i 's/num_prompts=\"\$((max_concurrency \* 1))\"/num_prompts=\"\$((max_concurrency * 10))\"/' \
  <magpie_repo>/InferenceMAX/benchmarks/benchmark_lib.sh"
```

Replace `10` with a different multiplier if the user requests it.

##### Option B: Profile the entire benchmark

Do **not** edit either script. The defaults are appropriate:
- No `delay_iterations` / `max_iterations` — the profiler captures everything from start to finish.
- `num_prompts` stays at `CONC` — keeps the trace to a manageable size since every iteration is profiled.

Warn the user that full-benchmark traces will be very large (potentially several GB per rank for large models).

#### Cleanup

These edits are temporary. Magpie re-clones InferenceMAX when the directory is absent, so deleting `<magpie_repo>/InferenceMAX` before a future run restores defaults. No permanent code changes are made.

### Step 3: Run the Benchmark

SSH into the target node, activate the conda environment, and run:

```bash
ssh <node> "cd <magpie_repo_root> && \
  source ~/miniconda3/etc/profile.d/conda.sh && \
  conda activate <env_name> && \
  python -m Magpie benchmark --benchmark-config <path/to/config.yaml> 2>&1"
```

Run this as a background command (`block_until_ms: 0`) since benchmarks take 5-30+ minutes depending on model size.

### Step 4: Monitor Progress

The main process logs are sparse during container execution. To see detailed progress:

1. **Check container status:**
   ```bash
   ssh <node> "docker ps --filter 'name=magpie' --format '{{.Names}}\t{{.Status}}'"
   ```

2. **Check container logs for model loading and benchmark progress:**
   ```bash
   ssh <node> "docker logs <container_name> 2>&1 | tail -40"
   ```

3. **Key milestones in container logs:**
   - `"The server is fired up and ready to roll!"` — model loaded, server ready
   - `"Warming up with N requests..."` — warmup phase (pre-profiling)
   - `"Starting profiler..."` — profiling active, benchmark running
   - Progress bars for warmup and benchmark requests

4. **Container disappears when done** (it runs with `--rm`). Once the container is gone, check the main process terminal for the final summary.

### Step 5: Verify Trace Quality

After completion, verify trace artifacts in the results workspace:

```bash
# List all trace files and sizes
ls -lh <workspace>/torch_trace/

# Verify trace contains GPU kernels
python3 -c "
import json, gzip
with gzip.open('<workspace>/torch_trace/<trace_file>', 'rt') as f:
    data = json.load(f)
events = data.get('traceEvents', [])
cats = {}
for e in events:
    cat = e.get('cat', 'no_cat')
    cats[cat] = cats.get(cat, 0) + 1
print(f'Total events: {len(events)}')
for cat, cnt in sorted(cats.items(), key=lambda x: -x[1])[:10]:
    print(f'  {cat}: {cnt}')
"
```

**Quality checklist:**

| Check | Expected | If it fails |
|-------|----------|-------------|
| Trace files exist in `torch_trace/` | At least one `.trace.json.gz` per rank | Profiler not enabled; check config |
| Each file > 100KB | SGLang: EXTEND ~4-5MB, DECODE ~150KB. vLLM: ~100-150MB per rank for large models with graph replay | Profiling window too short or no GPU ops |
| `kernel` category present | Hundreds to thousands of kernel events | `ProfilerActivity.CUDA` not captured |
| Multiple event categories | `cpu_op`, `kernel`, `cuda_runtime`, `python_function` | Partial trace; re-run |

**SGLang trace naming convention:**
- `<id>-TP-{rank}-DECODE.trace.json.gz` — decode phase per rank
- `<id>-TP-{rank}-EXTEND.trace.json.gz` — prefill/extend phase per rank
- `merged-<id>.trace.json.gz` — combined view across all ranks
- Expect `2 * TP` per-rank files + 1 merged file

**vLLM trace naming convention:**
- `*-rank-{N}.*.pt.trace.json.gz` — one GPU trace per rank
- `*async_llm.*.pt.trace.json.gz` — CPU-side AsyncLLM trace
- `profiler_out_{N}.txt` — profiler summary text per rank
- Expect `TP` rank files + 1 async_llm file + `TP` profiler_out files

### Step 6: Split Traces for Analysis

All vLLM and sglang traces — regardless of whether they were collected in eager mode or graph-replay mode — must be split before running TraceLens analysis. The raw per-rank traces are large (vLLM: ~100-150MB, 5-9M events; sglang: smaller but still benefit from splitting) and `TraceLens_generate_perf_report_pytorch` cannot handle them efficiently without preprocessing (it will hang or run for 10+ minutes with no output).

Run trace preprocessing on the rank-0 trace file:

```bash
ssh <node> "source ~/miniconda3/etc/profile.d/conda.sh && conda activate <env> && \
  cd <TraceLens_repo> && \
  python examples/custom_workflows/split_vllm_trace_annotation.py \
    <workspace>/torch_trace/<rank-0-trace>.pt.trace.json.gz \
    -o <workspace>/torch_trace/trace_split \
    --find-steady-state --num-steps 32"
```

Despite the script name referencing "vllm", it handles both vLLM and sglang traces.

This produces ~3 files in `trace_split/`:
- `*_annotation_iteration_0_*.json.gz` — full iteration (prefill + decode), ~16MB. **Use this for analysis.**
- `prefilldecode_*_.json.gz` — prefill/decode phase only
- `decode_*_.json.gz` — decode-only phase
- `execution_details.json` — iteration metadata

Then construct and print the following command for the user to run manually (do **not** execute it). Fill in the absolute paths based on the workspace and split output:

```
python <TraceLens_repo>/TraceLens/Reporting/generate_perf_report_pytorch_inference.py \
    --capture_folder <workspace>/torch_trace/capture_traces \
    --profile_json_path <workspace>/torch_trace/trace_split/<split_trace>.json.gz \
    --output_csvs_dir <workspace>/torch_trace/analysis_output \
    --group_by_parent_module --enable_pseudo_ops
```

Use absolute paths for all three arguments. The `--profile_json_path` should point to the `*_annotation_iteration_0_*.json.gz` file from the split output.

## Output Structure

```
results/benchmark_{framework}_{timestamp}/
├── benchmark_report.json      # parsed benchmark metrics
├── summary.txt                # human-readable summary
├── config.yaml                # snapshot of run configuration
├── benchmark_stdout.log       # container stdout
├── benchmark_stderr.log       # container stderr
├── server.log                 # inference server log
├── inferencemax_result.json   # raw InferenceMAX output
└── torch_trace/               # PyTorch profiler traces
    ├── *-TP-0-DECODE.trace.json.gz
    ├── *-TP-0-EXTEND.trace.json.gz
    ├── ...
    └── merged-*.trace.json.gz
```

## CLI Reference

```
python -m Magpie benchmark --benchmark-config <config.yaml>
python -m Magpie benchmark --benchmark-config <config.yaml> --run-mode local
python -m Magpie benchmark gap-analysis --trace-dir <results_dir>
```

CLI args can override YAML fields: `--torch-profiler`, `--tp N`, `--model <name>`, `--timeout N`, `--docker-image <img>`, `--output-dir <dir>`.

## Pitfalls

### 1. `--log-level` is not a valid CLI argument

The benchmark subcommand does not accept `--log-level`. Passing it causes an argparse error because it gets parsed as a positional argument for the `benchmark_action` sub-subparser (which only accepts `gap-analysis`). Omit it entirely.

### 2. No output during Docker container startup

The main Magpie process emits a "Running benchmark in container with image: ..." log line and then goes silent for several minutes while Docker starts the container, loads the model, and runs warmup. This is normal. Monitor progress by checking `docker logs` on the remote node rather than waiting for the main process.

### 3. Large models need long timeouts

DeepSeek-R1 and similarly large models can take 3-5 minutes just to load weights. Use `timeout_seconds: 3600` (1 hour) or more. The default is usually sufficient but do not reduce it for large models.

### 4. Traces are bind-mounted, not copied

The workspace directory is mounted into the container at `/workspace`. Trace files written inside the container appear directly on the host. There is no `docker cp` step. If the workspace path is wrong, traces will be missing.

### 5. Container is auto-removed

The container runs with `--rm`, so it disappears after completion. If you need to inspect container state after a failure, look at `benchmark_stdout.log` and `benchmark_stderr.log` in the workspace, or `server.log` for server-side issues.

### 6. SSH command quoting

When running the benchmark over SSH via the Shell tool, quote the entire remote command as a single string. Conda activation requires sourcing the conda init script explicitly:

```bash
ssh <node> "cd <repo> && source ~/miniconda3/etc/profile.d/conda.sh && conda activate <env> && python -m Magpie benchmark ... 2>&1"
```

### 7. Config field: `torch_profiler.enabled` defaults to `true`

In the Python config dataclass, `TorchProfilerConfig.enabled` defaults to `True`. But many example YAML files explicitly set it to `false`. Always check the actual YAML the user points to.

### 8. `benchmark_lib.sh` unconditionally overrides `num_prompts` when profiling

When `PROFILE=1`, the `run_benchmark_serving` function in `benchmark_lib.sh` **unconditionally** sets `num_prompts="$((max_concurrency * 1))"` *after* parsing all `--num-prompts` arguments. This means even if the calling benchmark script (e.g. `vllm_mi300x.sh`) explicitly passes `--num-prompts $(( $CONC * 10 ))`, that value gets overwritten. When using `delay_iterations` + `max_iterations` for steady-state profiling, you **must** patch `benchmark_lib.sh` to increase this multiplier, otherwise the benchmark ends before the profiling window starts and no trace is captured.

### 9. The `hf_cache_path` field avoids re-downloading models

If the node has a shared HuggingFace cache, set `hf_cache_path` in the config to avoid multi-hour model downloads. This path is mounted into the container.


## Optional: Post-Benchmark Analysis

After collecting traces, you can run additional analysis:

**Gap analysis** (kernel bottleneck report):
```bash
python -m Magpie benchmark gap-analysis --trace-dir <results_dir> \
    --start-pct 50 --end-pct 80 --top-k 20
```

**TraceLens analysis** (enable in YAML):
```yaml
  profiler:
    tracelens:
      enabled: true
      export_format: csv
      perf_report_enabled: true
      multi_rank_report_enabled: true
```

TraceLens requires the `TraceLens` package installed on the host (not inside the container).
