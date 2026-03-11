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

> "Would you like to profile a **targeted window** (recommended — smaller traces, captures steady-state decode) or the **entire benchmark** run?"

##### Option A: Targeted window (delay + max iterations)

Compute recommended `delay_iterations` and `max_iterations` from the YAML config values. Read `OSL`, `CONC`, and `RANDOM_RANGE_RATIO` from the `envs` section (default `RANDOM_RANGE_RATIO` to `1.0` if absent).

**Formulas:**

```
max_iters = max(OSL, OSL * 5 / CONC)

if RANDOM_RANGE_RATIO < 1:
    delay_iters = OSL * RANDOM_RANGE_RATIO * 5
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

**3. Increase `num_prompts` so the benchmark runs long enough for the profiling window.**

By default, when profiling is enabled, `benchmark_lib.sh` caps `num_prompts` to `max_concurrency` (i.e. `CONC`). With delay + max iterations the benchmark needs many more prompts to reach and complete the profiling window. Override to `CONC * 10`:

```bash
ssh <node> "sed -i 's/num_prompts=\"\$max_concurrency\"/num_prompts=\"\$((max_concurrency * 10))\"/' \
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

### 8. The `hf_cache_path` field avoids re-downloading models

If the node has a shared HuggingFace cache, set `hf_cache_path` in the config to avoid multi-hour model downloads. This path is mounted into the container.

### 9. vLLM graph-mode traces cannot be processed directly

vLLM v0.15+ captures traces with GPU graph replay enabled by default. These raw per-rank traces are very large (~100-150MB, 5-9M events) and contain graph replay events that the standard `TraceLens_generate_perf_report_pytorch` tool cannot handle efficiently (it will hang or run for 10+ minutes with no output). You must split the trace first, then run the perf report on the smaller split output. See the "vLLM Graph-Mode Trace Processing" section below.

To detect graph mode: check the raw trace for `hipGraphLaunch` or `cudaGraphLaunch` events in `cuda_runtime` category, or look for `StreamBeginCapture`/`StreamEndCapture` events. If these are absent, the trace is eager-mode and can be processed directly with `TraceLens_generate_perf_report_pytorch` without splitting.

## vLLM Graph-Mode Trace Processing

When working with vLLM traces captured in graph-replay mode, split them before running TraceLens analysis. If the trace is eager-mode (no graph replay events), skip this section and use `TraceLens_generate_perf_report_pytorch` directly on the raw trace.

### Prerequisites: Patched vLLM for graph capture profiling

For full graph-mode analysis (shapes, roofline, call stacks for graph-replayed kernels), vLLM must be patched to profile the graph capture phase. Without the patch, the trace only contains opaque graph replay events with no kernel-level detail inside the graphs.

**Option A: Build a patched Docker image (recommended)**

TraceLens provides a build script that applies the patch automatically:

```bash
cd <TraceLens_repo>
bash examples/custom_workflows/inference_analysis/build_docker_vllm.sh \
    <version_tag> \
    /path/to/TraceLens-internal \
    -t tracelens-vllm
```

Supported version tags and base images:

| Tag | Base Image | vLLM Version |
|-----|-----------|--------------|
| `v14` | `rocm/vllm-dev:preview_releases_rocm_v0.14.0_20260120` | v0.14.0 |
| `v15` | `rocm/vllm-dev:preview_releases_rocm_v0.15.0_20260130` | v0.15.0 |
| `v16` | `rocm/vllm-dev:preview_rocm70_releases_rocm_v0.16.0_20260223` | v0.16.0 |

Then tell Magpie to use this image by adding `docker_image` to the benchmark YAML config:

```yaml
benchmark:
  framework: vllm
  docker_image: tracelens-vllm
  ...
```

This overrides the default auto-selected image from `benchmark_images.yaml`.

**Option B: Apply the patch manually inside the Magpie Docker container**

If you cannot build a custom image, you can patch the running container that Magpie creates. This requires exec-ing into the container before or during the benchmark run, since the patch must be applied to the vLLM installation inside the container (not on the host).

```bash
# Exec into the running Magpie benchmark container
docker exec -it <container_name> bash

# Find where vLLM is installed inside the container
python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))"

# Apply the matching patch (TraceLens repo must be mounted or copied in)
cd /path/to/vllm/../
git apply /path/to/TraceLens-internal/examples/custom_workflows/inference_analysis/vllm_v0.15.0.patch
```

Available patches: `vllm_v0.13.0.patch`, `vllm_v0.14.0.patch`, `vllm_v0.15.0.patch`, `vllm_v0.16.0.patch`.

Note: Option A (pre-built image) is strongly preferred since it avoids the need to patch a running container and ensures the patch is applied before profiling begins.

### Step 1: Split the trace by annotation iteration

```bash
ssh <node> "source ~/miniconda3/etc/profile.d/conda.sh && conda activate <env> && \
  cd <TraceLens_repo> && \
  python examples/custom_workflows/split_vllm_trace_annotation.py \
    <workspace>/torch_trace/<rank-0-trace>.pt.trace.json.gz \
    -o <workspace>/graph_split \
    --find-steady-state --num-steps 10"
```

This produces ~3 files in `graph_split/`:
- `*_annotation_iteration_0_*.json.gz` — full iteration (prefill + decode), ~16MB. **Use this for analysis.**
- `prefilldecode_*_.json.gz` — prefill/decode phase only
- `decode_*_.json.gz` — decode-only phase
- `execution_details.json` — iteration metadata

### Step 2: Process the split trace

Run the standard perf report tool on the split iteration trace (not the raw trace).

When feeding into the standalone analysis skill, use the split trace path (not the raw trace).

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
