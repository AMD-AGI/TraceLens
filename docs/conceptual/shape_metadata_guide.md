# Tensor Shape Metadata in PyTorch Profiler Traces

A practical guide to understanding when tensor shapes are available in profiler traces and how to get them when they're missing.

---

## Part 1: When Do We Have Shapes?

### The Simple Rule

**Shapes are available when an operation goes through PyTorch's dispatcher as a `cpu_op`.**

In profiler traces, look for events with:
- `"cat": "cpu_op"`
- `"args": { "Input Dims": [...], "Input type": [...] }`

### What HAS Shapes

| Operation Type | Example | Why It Works |
|---------------|---------|--------------|
| Native ATen ops | `aten::mm`, `aten::linear` | Built into PyTorch dispatcher |
| Registered custom ops | `torch.ops.mylib.my_op` | Registered via `torch.library` |
| TorchScript ops | JIT-compiled functions | Goes through dispatcher |
| Custom `autograd.Function` | User-defined forward | Forward call is dispatched |
| Distributed collectives | `record_param_comms` | Instrumented by PyTorch |

### What DOESN'T Have Shapes

| Operation Type | Example | Why It Fails |
|---------------|---------|--------------|
| Plain Python functions | `def my_kernel(x, y): ...` | Bypasses dispatcher |
| Triton kernels | `@triton.jit` | Called as Python function |
| FlashInfer ops | GEMM, MoE, Attention | Registration intentionally disabled |
| Backward engine events | `autograd::engine::evaluate_function:*Backward` | Empty inputs passed to profiler |

### Quick Reference: Event Categories

```
cpu_op          → Usually HAS shapes (exception: backward events)
python_function → NO shapes
kernel          → NO shapes (GPU-side event)
cuda_runtime    → NO shapes (API-level event)
```

---

## Part 2: How to Get Shapes When Missing

### Option 1: Register as Custom Op

Wrap the operation with `torch.library.custom_op`:

```python
@torch.library.custom_op("mylib::triton_matmul", mutates_args=())
def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return my_triton_kernel(a, b)

# Now call via torch.ops
result = torch.ops.mylib.triton_matmul(a, b)
```

**Result:** Event appears as `cpu_op` with `Input Dims`.

**What is `mutates_args`?**

Tells PyTorch which input arguments the function modifies in-place:
- `mutates_args=()` — No inputs are modified (pure function)
- `mutates_args=("output",)` — The `output` argument is modified in-place
- Required for correctness with `torch.compile` and autograd

### Option 2: Lighter-Weight Registration (vLLM approach)

Use the lower-level `Library` API directly:

**Reference:** [vllm/utils/torch_utils.py:742 - `direct_register_custom_op`](https://github.com/vllm-project/vllm/blob/0b225fb7b22f8ae1f5fc8ee618640ae0983c76de/vllm/utils/torch_utils.py#L742-L780)

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

### Option 1 vs Option 2: When to Use Which?

| Aspect | Option 1: `custom_op` | Option 2: `Library.define()` |
|--------|----------------------|------------------------------|
| **Simplicity** | ✅ Decorator, minimal code | ❌ More boilerplate |
| **Overhead** | Higher (full dispatcher) | Lower (direct to CUDA) |
| **torch.compile** | ✅ Full support | ✅ Works with `register_fake` |
| **Use when...** | Prototyping, one-off ops | Performance-critical paths |

---

## Framework-Specific Status

| Framework | Current State | How to Get Shapes |
|-----------|--------------|-------------------|
| **PyTorch ATen** | ✅ Has shapes | Already works |
| **vLLM (standard)** | ✅ Has shapes | Uses `direct_register_custom_op` |
| **vLLM (OAI Triton)** | ❌ Missing | Needs registration |
| **FlashInfer** | ❌ Disabled | Set `FLASHINFER_ENABLE_PROFILER_METADATA=1` (proposed) |
| **SGLang (CUDA)** | ✅ Has shapes | Uses `torch.ops.sgl_kernel.*` |
| **SGLang (Triton)** | ❌ Missing | Needs registration |

---

## Summary

1. **Shapes are tied to dispatcher registration** — If it's a `cpu_op`, it has shapes
2. **The fix is straightforward** — Register operations via `torch.library`
3. **Start simple** — Use `@torch.library.custom_op` first
4. **Optimize if needed** — Switch to `Library.define()` for lower overhead
5. **FlashInfer disabled it on purpose** — Performance vs. observability trade-off; can be re-enabled

---

## Appendix

### A1: Backward Events and Sequence Number Linking

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
