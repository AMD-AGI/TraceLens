<!--
Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Tensor shape metadata in PyTorch profiler traces in TraceLens
```{meta}
:description: A conceptual explanation of when tensor shapes appear in PyTorch profiler traces, why some operations lack them, and how to register operations so shapes are recorded.
:keywords: tensor shapes, PyTorch profiler, cpu_op, dispatcher, torch.library, custom op, Triton, FlashInfer, vLLM, SGLang, Input Dims, backward events, TraceLens
```

Tensor shapes are one of the most useful pieces of metadata in a PyTorch profiler trace: they let you match trace events to model architecture, analyze kernel efficiency, and spot unusual stride or dtype patterns. But shapes aren't always present. This topic explains when they're available, why they sometimes go missing, and how to get them back.

## When are shapes available?

Shapes are available when an operation goes through PyTorch's dispatcher as a `cpu_op`.

In profiler traces, look for events with:

- `"cat": "cpu_op"`
- `"args": { "Input Dims": [...], "Input type": [...] }`

### Operations that have shapes

The following operation types record tensor shapes in the trace.

| Operation type | Example | Why it works |
|---|---|---|
| Native ATen ops | `aten::mm`, `aten::linear` | Built into the PyTorch dispatcher |
| Registered custom ops | `torch.ops.mylib.my_op` | Registered through `torch.library` |
| TorchScript ops | JIT-compiled functions | Goes through the dispatcher |
| Custom `autograd.Function` | User-defined forward | Forward call is dispatched |
| Distributed collectives | `record_param_comms` | Instrumented by PyTorch |

### Operations that don't have shapes

The following operation types bypass the dispatcher and do not record shapes.

| Operation type | Example | Why it fails |
|---|---|---|
| Plain Python functions | `def my_kernel(x, y): ...` | Bypasses the dispatcher |
| Triton kernels | `@triton.jit` | Called as a Python function |
| FlashInfer ops | GEMM, MoE, attention | Registration intentionally disabled |
| Backward engine events | `autograd::engine::evaluate_function:*Backward` | Empty inputs passed to the profiler |

### Quick reference: event categories

The following summary shows which event categories carry shape information.

```
cpu_op          → Usually has shapes (exception: backward events)
python_function → No shapes
kernel          → No shapes (GPU-side event)
cuda_runtime    → No shapes (API-level event)
```

## How to get shapes when they're missing

If an operation lacks shapes, the fix is to register it with the dispatcher through `torch.library`. There are two approaches.

### Option 1: Register as a custom op

Wrap the operation with `torch.library.custom_op`:

```python
@torch.library.custom_op("mylib::triton_matmul", mutates_args=())
def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return my_triton_kernel(a, b)

# Now call via torch.ops
result = torch.ops.mylib.triton_matmul(a, b)
```

The event then appears as `cpu_op` with `Input Dims`.

The `mutates_args` argument tells PyTorch which input arguments the function modifies in-place:

- `mutates_args=()` — no inputs are modified (a pure function).
- `mutates_args=("output",)` — the `output` argument is modified in-place.
- Required for correctness with `torch.compile` and autograd.

### Option 2: Lighter-weight registration (vLLM approach)

Use the lower-level `Library` API directly. For reference, see [`vllm/utils/torch_utils.py:742` — `direct_register_custom_op`](https://github.com/vllm-project/vllm/blob/0b225fb7b22f8ae1f5fc8ee618640ae0983c76de/vllm/utils/torch_utils.py#L742-L780).

```python
from torch.library import Library
from torch._library.infer_schema import infer_schema

my_lib = Library("mylib", "FRAGMENT")

def register_op(op_name, op_func, mutates_args=None):
    schema = infer_schema(op_func, mutates_args=mutates_args or [])
    my_lib.define(op_name + schema)
    my_lib.impl(op_name, op_func, dispatch_key="CUDA")

# Register
register_op("triton_matmul", triton_matmul_impl)
```

### Choosing between the two options

The following table compares the two registration approaches to help you select the right one for your use case.

| Aspect | Option 1: `custom_op` | Option 2: `Library.define()` |
|---|---|---|
| Simplicity | Decorator, minimal code | More boilerplate |
| Overhead | Higher (full dispatcher) | Lower (direct to CUDA) |
| `torch.compile` | Full support | Works with `register_fake` |
| Use when | Prototyping, one-off ops | Performance-critical paths |

## Framework-specific status

The following table shows the current shape-recording status for each supported framework.

| Framework | Current state | How to get shapes |
|---|---|---|
| PyTorch ATen | Has shapes | Already works |
| vLLM (standard) | Has shapes | Uses `direct_register_custom_op` |
| vLLM (OAI Triton) | Missing | Needs registration |
| FlashInfer | Disabled | Set `FLASHINFER_ENABLE_PROFILER_METADATA=1` (proposed) |
| SGLang (CUDA) | Has shapes | Uses `torch.ops.sgl_kernel.*` |
| SGLang (Triton) | Missing | Needs registration |

## Key takeaways

The following points summarize when shapes are available and how to recover them when they are missing.

- Shapes are tied to dispatcher registration. If it's a `cpu_op`, it has shapes.
- The fix is straightforward: register operations through `torch.library`.
- Start simple with `@torch.library.custom_op`.
- Optimize if needed by switching to `Library.define()` for lower overhead.
- FlashInfer disabled shape recording on purpose — a performance versus observability trade-off that can be re-enabled.

## Appendix

This section provides supplementary detail on how backward events relate to their forward counterparts via sequence number linking.

### Backward events and sequence number linking

Backward engine events (`autograd::engine::evaluate_function:*Backward`) are `cpu_op` but don't have `Input Dims`. They have `Sequence number` instead, which links to the corresponding forward op.

```python
def get_backward_shapes(trace_events, backward_event):
    seq_num = backward_event["args"].get("Sequence number")
    if seq_num is None:
        return None

    # Find forward op with same sequence number
    for event in trace_events:
        if event.get("cat") == "cpu_op" and \
           event["args"].get("Sequence number") == seq_num and \
           "Backward" not in event.get("name", ""):
            return event["args"].get("Input Dims")
    return None
```

## Related topics

- [Trace2Tree](../conceptual/trace2tree.md)
- [Understanding PyTorch traces](../conceptual/torch-profiling-analysis.md)
- [Generate a performance report from PyTorch traces](../how-to/generate-perf-report-pytorch.md)
- [Performance report columns](../reference/perf-report-columns.md)
- [API reference](../reference/api-reference.md)
