<!--
Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Kernel shape profiler

Adds `Input Dims` / `Input type` / `Input Strides` to PyTorch profiler traces
for GPU kernels (Triton / aiter / FlashInfer) that would otherwise appear with
no operand metadata, so they can be rooflined.

It works by wrapping **kernel launcher functions** — Python entry points like
`gemm_a8w8_blockscale`, `invoke_fused_moe_kernel` or `rmsnorm` — as
`torch.library` custom ops. Each wrapped launcher then shows up in the trace as
a `cpu_op` named after itself, carrying its tensor operands:

```
sglang_profiler::fp8_utils_gemm_a8w8_blockscale_12   <- cpu_op, named after the launcher
    Input Dims:    [[1025, 7168], [7168, 2112], [1025, 56], [56, 17]]
    Input Strides: [[7168, 1], [1, 7168], [56, 1], [1, 56]]
    Input type:    ['c10::Float8_e4m3fnuz', 'c10::Float8_e4m3fnuz', 'float', 'float']
  _gemm_a8w8_blockscale_kernel_GROUP_K_128_...        <- the real GPU kernel
```

Launchers come from an explicit registry plus auto-discovery, so no
serving-framework source is patched.

## Files

| File | Role |
|------|------|
| `kernel_shape_profiler.py` | Wraps launcher functions as custom ops and rebinds every module-level reference to them. |
| `sitecustomize.py` | Auto-loaded shim that drives `enable()` / `disable()` from the torch-profiler window so nothing is wrapped outside a profiling run, and runs discovery at SGLang scheduler startup. |

## Activation

Put this directory on `PYTHONPATH` and set the flag:

```bash
export PYTHONPATH=/path/to/kernel_shape_tool:$PYTHONPATH
export TRACELENS_SHAPE_DISCOVERY=1
```

CPython auto-imports `sitecustomize` at interpreter startup for every process,
so the server and all TP workers pick it up. When `TRACELENS_SHAPE_DISCOVERY`
is unset or `0`, every hook short-circuits, so it is safe to leave the directory
on `PYTHONPATH` permanently.

### When discovery runs

Auto-discovery imports and scans every module under the target prefixes, which
takes several seconds on a large server. On SGLang it runs once per scheduler
process, at the end of scheduler init (from `Scheduler.get_init_info()`, just
before the scheduler reports ready), so it adds to startup instead of stalling
the first profiler window. Each window's `enable()` then only rebinds
references (well under a second). On other servers, or if the hook can't be installed,
the first `enable()` runs discovery itself; the result is cached for later
windows either way. `kernel_shape_profiler.prepare()` is the same step for
callers that want to trigger it themselves.

Each log line reports the split, for example:

```
kernel_shape_profiler prepared: 3590 entry points (41 explicit + 3549 auto-discovered), 3553 wrappers; 6.4 s (discover 6.3)
kernel_shape_profiler enabled: 5644 references patched across 3590 entry points (41 explicit + 3549 auto-discovered), 0 of them decorator closures; 0.1 s (discover 0.0, wrap 0.0, patch 0.1)
```

### Optional knobs

| Env var | Default | Meaning |
|---------|---------|---------|
| `TRACELENS_SHAPE_DISCOVERY` | `0` | Master switch |
| `TRACELENS_SHAPE_FORCE_RECORD_SHAPES` | `1` | Force `record_shapes=True` on the profiler (`Input Dims` only surface when shapes are recorded). Set `0` to respect the server's own setting. |

### Launchers with unannotated tensors

Discovery and custom-op registration read tensor params from type annotations.
A launcher whose tensors are bare can't be registered from its signature alone;
`_TENSOR_PARAM_HINTS` names their schema types (aiter's `unified_attention`
annotates only its flags). A launcher with some params unannotated falls back
to the source check instead of being excluded by its non-tensor annotations.

> Note on CUDA graphs: kernels that only run inside replayed graphs execute no
> Python, so their shapes are recorded at graph **capture** time. A decode-path
> analysis therefore needs the graph-capture trace, not just the serving-window
> trace.
