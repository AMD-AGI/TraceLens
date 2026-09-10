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
| `sitecustomize.py` | Auto-loaded shim that drives `enable()` / `disable()` from the torch-profiler window so nothing is wrapped outside a profiling run. |

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

### Optional knobs

| Env var | Default | Meaning |
|---------|---------|---------|
| `TRACELENS_SHAPE_DISCOVERY` | `0` | Master switch |
| `TRACELENS_SHAPE_FORCE_RECORD_SHAPES` | `1` | Force `record_shapes=True` on the profiler (`Input Dims` only surface when shapes are recorded). Set `0` to respect the server's own setting. |

> Note on CUDA graphs: kernels that only run inside replayed graphs execute no
> Python, so their shapes are recorded at graph **capture** time. A decode-path
> analysis therefore needs the graph-capture trace, not just the serving-window
> trace.
