<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# generic_shape_profiler

Annotate **tensor shapes and dtypes of GPU kernels that bypass the PyTorch
dispatcher** (Triton, FlyDSL, TileLang, tvm-ffi / SGLang `load_jit`, aiter
`cpp_itfs` and its off-dispatcher `compile_ops`) into PyTorch profiler traces
**without patching the inference server**.

It is framework-agnostic: the same two files work under SGLang, vLLM, ATOM or
any other torch workload, with no changes to the server source: the tool
activates itself through a `sitecustomize.py` shim that CPython auto-loads in
every process on `PYTHONPATH`.

## What problem it solves

A profiler trace already carries full operand metadata for anything that goes
through the PyTorch dispatcher — aten ops and every `torch.ops.*` custom op
(aiter, FlashInfer, sgl-kernel, DeepGEMM wrappers, …) are recorded with
`Input Dims` / `Input type` for free.

JIT-compiled kernels are the exception. They are launched straight from Python
onto the stream, bypassing the dispatcher entirely, so they appear in the trace
as bare GPU kernels with no operand information, which makes them impossible to
roofline.

This tool closes exactly that gap: it hooks the JIT launch entry points and, for
each launch, calls a no-op dispatcher op named after the kernel whose arguments
are the launch's, bound to the kernel's signature. The profiler records that op
like any other, so the kernel's shapes, strides, dtypes and scalar arguments land
in the trace.

| Launch path | Hooked at | Example kernels |
|-------------|-----------|-----------------|
| Triton | `JITFunction.run`, `Autotuner.run`, `Heuristics.run` | aiter / SGLang / vLLM Triton GEMMs, `_sparse_mla_split_k_kernel` |
| Triton, compiled-kernel relaunch | `CompiledKernel.__getitem__` | ATOM's packed H2D scatter, which keeps the `CompiledKernel` and relaunches it with `compiled[grid](*args)` |
| FlyDSL | `JitFunction.__call__` | FlyDSL kernels launched through `@flyc.jit` |
| aiter FlyDSL dispatch | `aiter.ops.flydsl.kernels.tensor_shim._run_compiled` | aiter's FlyDSL kernels, launched from a cached executable with raw pointers |
| TileLang | `JITKernel.__call__` (named via `JITImpl.compile`) | SGLang's TileLang DSA prefill / decode kernels |
| tvm-ffi (SGLang `load_jit`) | `tvm_ffi.Module.__getattr__` | `clamp_position` |
| aiter `csrc.cpp_itfs` | `run_lib` (ctypes entry points), `run_hsaco` | `paged_attention_ragged` (`paged_attention_ll4mi_*`) |
| aiter `compile_ops` | `aiter.jit.core.compile_ops`, only for `NONE_WRAPPED_OP` ops | custom all-reduce / all-gather: `all_reduce`, `fused_allreduce_rmsnorm`, `all_gather_reg` |

Every other aiter `compile_ops` op is registered as a `torch.ops.aiter.*`
dispatcher op, so those kernels are already shaped natively and are left
alone. A `cpp_itfs` entry point or an aiter FlyDSL executable is called with
raw pointers, so its operands are read from the calling Python wrapper's
arguments instead.

TileLang needs its own hook even though it is built on TVM: it compiles TIR to a
HIP/CUDA library and calls it through its own runtime adapter, never through a
`tvm_ffi.Module` attribute. Every TileLang kernel's entry symbol is `main` (so
`main_kernel` on the GPU); the `JITImpl.compile` hook tags each compiled kernel
with its `@tilelang.jit` factory's name so the annotation is readable.

## Files

| File | Role |
|------|------|
| `generic_shape_profiler.py` | The core. Hooks the launch paths above and registers each launch's operands as dispatcher ops — `kernel_shapes::<kernel>` and `launcher_shapes::<calling function>`. Knows nothing about any serving framework. |
| `sitecustomize.py` | Auto-loaded shim. Installs each hook when its library is imported, and drives `enable()` / `disable()` from the torch-profiler window. Gated by an env var. |

## Activation

Two environment variables — no source changes:

```bash
export PYTHONPATH=/path/to/TraceLens/TraceLens/TraceUtils/generic_shape_profiler:$PYTHONPATH
export TRACELENS_SHAPE_DISCOVERY=1
```

Put the directory itself on `PYTHONPATH` (not `TraceLens/` or its parent):
`sitecustomize` is only auto-loaded as a top-level module.

Then launch the server and profile as usual (e.g. SGLang's `/start_profile`, or
any `torch.profiler.profile` window). Kernel shapes appear in the resulting
trace with no other changes. See [Profiling serving
frameworks](#profiling-serving-frameworks) for the framework-specific flags that
produce graph-capture traces.

When `TRACELENS_SHAPE_DISCOVERY` is unset or `0`, every hook short-circuits, so
it is safe to leave the directory on `PYTHONPATH` permanently (for example baked
into a container image).

## What lands in the trace

For `add_launcher(x, y)`, which allocates `out` and launches
`add_kernel(x, y, out)` with 4096-element fp16 tensors:

```
launcher_shapes::add_launcher          <- cpu_op enclosing the whole launcher call
    Input Dims:    [[4096], [4096]]       (the launcher's own tensor arguments)
    Input type:    ['c10::Half', 'c10::Half']
  add_launcher                         <- the launcher body runs inside the op
    aten::empty_like                   <- everything it does nests here ...
    jit::add_kernel                    <- record_function span around the launch
      kernel_shapes::add_kernel        <- leaf op: the kernel's own operands
          Input Dims:    [[4096], [4096], [4096]]
          Input Strides: [[1], [1], [1]]
      add_kernel                       <- ... including the real GPU kernel
```

The launcher op sits where a hand-registered wrapper op would: it is an
ancestor of every GPU kernel the launcher issues, so a trace tree attributes
those kernels to it. The `jit::` span gives the exact kernel-level association,
and the kernel op name matches the kernel name for name-based lookup.

### Argument slots

Every call is bound to the function's signature the way Python binds it
(positionally, by keyword, or from the default), so each parameter has one fixed
slot, in signature order, whatever the call site does. This is the same rule the
dispatcher applies to native ops, so the slots of `aiter::*` and
`launcher_shapes::*` ops read the same way:

| Parameter value | Slot type | `Input Dims` / `Input type` / `Concrete Inputs` |
|---|---|---|
| tensor | `Tensor?` | its shape / its dtype / `""` |
| `None` or not passed (optional tensor) | `Tensor?` | `[]` / `""` / `""` |
| `int`, `float`, `bool` (incl. Triton `constexpr`s) | `Scalar?` | `[]` / `Scalar` / the value |
| `torch.dtype` | `ScalarType?` | `[]` / `Scalar` / the dtype's enum value (`15` = bf16) |
| list or tuple of tensors | `Tensor[]?` | the shapes / `TensorList` / `""` |
| list or tuple of ints | `int[]?` | `[]` / `ScalarList` / e.g. `[2, 3]` |
| anything else (`self`, configs, strings) | no slot | – |

For `axpy(x, y, bias=None, out=None, alpha=1.0, *, block=1024)`, the calls
`axpy(X, Y)`, `axpy(X, Y, out=O, bias=B, alpha=2.0)` and
`axpy(y=Y, x=X, block=512)` all record `launcher_shapes::axpy` with the slots
`x, y, bias, out, alpha, block`; absent tensors are `[]` and the scalars appear in
`Concrete Inputs`. Triton kernels bind to the `@triton.jit` function's own
parameters, so `kernel_shapes::<kernel>` records every pointer, size and
`constexpr` passed to the launch (autotuned `constexpr`s are not arguments and
stay empty).

The parameter names are written once into the trace's top-level metadata under
`generic_shape_profiler_schemas` when the profiler stops:

```json
"generic_shape_profiler_schemas": {
  "launcher_shapes::axpy": [["x", "Tensor?"], ["y", "Tensor?"], ["bias", "Tensor?"],
                            ["out", "Tensor?"], ["alpha", "Scalar?"], ["block", "Scalar?"]]
}
```

so a perf model can address operands by name: slot *i* of an op is
`schemas[op][i]`. `*args` and `**kwargs` parameters become one trailing
`Tensor[]?` slot each.

A parameter's slot type is fixed by the first value it is seen with, or by its
annotation when that value is `None`. An unannotated parameter first seen as
`None` is guessed to be a tensor, and that guess is provisional: the first call
that passes a real value of another type (say `eps=1e-5`) re-decides the slot,
and the function continues under a new op `<op>__v2` with the corrected
schema. Slot positions are unchanged, and the old op only holds calls on which
that parameter was `None`. Each re-decided op is listed under
`generic_shape_profiler_aliases`, so it can be merged into its replacement:

```json
"generic_shape_profiler_aliases": {
  "launcher_shapes::fused_rms_fp8_group_quant": "launcher_shapes::fused_rms_fp8_group_quant__v2"
}
```

A slot is re-decided at most once; any other type mismatch records as empty
rather than changing the op. Launch paths with no Python signature —
tvm-ffi functions, `run_hsaco`, `compiled[grid](...)` relaunches, TileLang
kernels and `mark_launch` — record their arguments in call order instead
(keywords sorted by name), with the same slot types; their slots are named
`a0, a1, ...`. A second function with the same name, or a positional call of
another shape, gets a separate op with an `__<N>t` suffix (`N` slots).

### Two granularities

Each launch is annotated twice, under two dispatcher namespaces:

| Namespace | Named after | Use it when |
|-----------|-------------|-------------|
| `kernel_shapes::<kernel>` | the Triton / FlyDSL / TileLang kernel, e.g. `_gemm_a8w8_blockscale_kernel` | joining to GPU kernels by name (the op name *equals* the kernel name, so it prefix-matches the mangled GPU name) |
| `launcher_shapes::<caller>` | the Python function that issued the launch, e.g. `gemm_a8w8_blockscale` | you want the semantic op ("this is a blockscaled fp8 GEMM") rather than whichever kernel implements it |

Why the launcher level exists: a Triton kernel is nearly always reached through
a Python wrapper, and with `with_stack` that wrapper appears in the trace only
as a `python_function` frame — which carries no operand metadata. Annotating it
as an op is what makes `aiter/ops/triton/.../gemm_a8w8_blockscale.py: gemm_a8w8_blockscale`
show up as a `cpu_op` with `Input Dims`.

The wrapper is found by walking out of the JIT machinery *by module*: Triton
reaches `.run()` through its own `KernelInterface.__getitem__` lambda, so the
first frame outside `triton` / `flydsl` / `tilelang` / `tvm` / `tvm_ffi`, outside
aiter's launch helpers and outside this module is the launcher. No registry, no
source scanning, and no matching on function names. Methods are named by
`__qualname__` (`CustomAllreduce.fused_ar_rms`, rendered as
`CustomAllreduce_fused_ar_rms`), so two classes' `forward` stay distinct.

That first launch is also how the launcher is *promoted*: its frame yields the
function object, whose code object is swapped for a trampoline. Every later
call goes through `launcher_shapes::<launcher>(<launcher's arguments>)`,
whose implementation runs the original function, so the op encloses the
launcher's whole body. Swapping `__code__` changes the one function object in
place, so all references to it are covered without patching modules. The call
that discovers a launcher carries only the kernel-level op. Closures,
generators and `torch` internals are never swapped and keep a leaf
`launcher_shapes` op next to the launch, carrying the launcher's own arguments
read from its frame (so a parameter the launcher reassigned before the launch
records its new value).

> The two levels describe the same work, so pick one granularity when summing
> FLOPs or bytes — don't add them. Set `TRACELENS_SHAPE_LAUNCHER_OPS=0` to emit
> kernel-level annotations only.

### Optional knobs

| Env var | Default | Meaning |
|---------|---------|---------|
| `TRACELENS_SHAPE_DISCOVERY` | `0` | Master switch. Must be truthy to do anything. |
| `TRACELENS_SHAPE_FORCE_RECORD_SHAPES` | `1` | Force `record_shapes=True` on the profiler (`Input Dims` only surface when shapes are recorded). Set `0` to respect the server's own setting. |
| `TRACELENS_SHAPE_SPANS` | `1` | Emit the enclosing `jit::<kernel>` span. Set `0` for shape ops only (marginally lower overhead, association by name only). |
| `TRACELENS_SHAPE_LAUNCHER_OPS` | `1` | Also annotate the calling Python launcher as `launcher_shapes::<caller>`. Set `0` for kernel-level annotations only. |
| `TRACELENS_SHAPE_LAUNCHER_PROMOTE` | `1` | Run discovered launchers inside their `launcher_shapes` op, so it encloses the whole launcher. Set `0` to keep the launcher op as a leaf next to the launch. |
| `TRACELENS_SHAPE_SCALARS` | `1` | Also record `int` / `float` / `bool` / `dtype` / int-list arguments. Set `0` for tensor slots only (smaller traces; the tensor slots stay in signature order). |
| `TRACELENS_SHAPE_BREAK` | unset | Debug only. Drop into `breakpoint()` at each hooked launch: `1` / `all` for every hook, or a comma-separated list of hook-id prefixes (`triton`, `flydsl`, `tilelang`, `tvm_ffi`, `aiter.cpp_itfs`, `aiter.hsaco`, `aiter.flydsl`, `aiter.flydsl.moe`, `aiter.compile_ops`). Honours `PYTHONBREAKPOINT`. |
| `TRACELENS_SHAPE_BREAK_RANK` | unset | Debug only. Restrict `TRACELENS_SHAPE_BREAK` to one `torch.distributed` rank, so a TP=N server stops in one process instead of N. |

## Profiling serving frameworks

Graph replay executes no Python, so a kernel that only runs inside a replayed
CUDA / HIP graph — in practice most of decode — is **not** annotated in the
serving trace. Its shapes are recorded when the graph is *captured*, so profile
the capture as well. Capture inside a profiler window is covered automatically,
because the hooks are gated by that window and need no special case.

| Framework | Capture traces | Notes |
|-----------|----------------|-------|
| SGLang | `--enable-profile-cuda-graph` and `SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE=1`; traces land in `<SGLANG_TORCH_PROFILER_DIR>/graph_capture_profile/` | SGLang 0.5.23 renamed `--cuda-graph-max-bs` to `--cuda-graph-max-bs-decode`. With MTP, the EAGLE draft and draft-extend runners are captured too. |
| vLLM | `--profiler-config '{"profiler": "torch", "torch_profiler_record_shapes": true, "capture_torch_profiler": true, ...}'`; traces land in `<torch_profiler_dir>/capture_traces/` | `capture_torch_profiler` is only implemented in the V1 model runner: set `VLLM_USE_V2_MODEL_RUNNER=0`. |
| ATOM | `--mark-trace` with `ATOM_PROFILER_RECORD_SHAPES=1`; traces land in `<trace dir>/rank_<N>/capture_traces/` | Needs an ATOM build that has both options. |

Use the serving trace for everything that still launches from Python (prefill,
eager paths, metadata kernels) and the capture traces for decode. On the
DeepSeek-R1 TP=8 SGLang run under [Validated on](#validated-on) that split is
large: 22,220 annotated launches per rank in the capture window versus 1,770 in
the serving window.

**Trace size.** The tool adds a few ops per launch and does not change trace
size materially. Very large vLLM traces come from `with_stack` recording the
engine's `shm_broadcast` busy-wait loop, independent of this tool; set
`"torch_profiler_with_stack": false` in the vLLM profiler config if call stacks
are not needed.

## Annotating a non-JIT launcher

For a launch path that is not hooked (a hand-written HIP extension, a CuTe
kernel, …), one call is enough — no need to teach the tool about it:

```python
import generic_shape_profiler

generic_shape_profiler.mark_launch("my_fused_kernel", [q, k, v, out])
```

It is a no-op unless tracing is enabled. With no signature to bind to, the
list is recorded in the order given, as slots `a0, a1, ...`.

## Design

- **No framework knowledge.** Hooks are installed on the JIT libraries' own
  classes and loaders, so every launch is observed regardless of who calls it.
  There is no registry of kernel entry points, no `sys.modules` scanning, no
  source-code heuristics, and no awareness of SGLang / vLLM / ATOM / any stack.
  Adding support for a new server is a no-op; supporting a new JIT is one
  function.
- **Hook before the first kernel loads.** tvm-ffi modules, aiter's `run_lib`,
  aiter's `_run_compiled` (imported by name), `compile_ops` (applied when
  `aiter.ops.*` import) and TileLang's `JITImpl.compile` (kernels are compiled
  during warmup) all hand out or cache things before any profiler window opens,
  so those hooks are installed the moment their module finishes importing, not
  when the profiler starts.
- **Install once, gate with a flag.** Hooks are installed permanently the first
  time the JIT modules appear and short-circuit on a boolean. `enable()` /
  `disable()` flip that boolean and swap promoted launchers' code in and out,
  so outside a profiler window every launcher runs its original code.
- **Outermost launch only.** `Autotuner.run` calls `JITFunction.run`
  internally; a thread-local depth counter records each launch once, at the
  outermost hook where the caller's real arguments are still visible.
- **Deferred patching.** `torch` is usually not imported when `sitecustomize`
  runs, so patches are registered as *pending* and applied by a `sys.meta_path`
  finder the moment the target module finishes importing. The finder is only
  consulted on a module's first import, so the `import` statements a server
  runs on its hot paths cost nothing and add no frames to a `with_stack` trace.
  All patches are idempotent and wrapped in `try/except`, so they can never
  break the workload.
- **Few Python calls per launch.** With `with_stack`, every Python call made
  inside a profiler window is a trace event — builtins such as `isinstance`,
  `len` and `list.append` included — so the per-launch path (argument binding,
  the outermost-launch check) uses flat loops, indexing, membership tests and
  exact-type lookups. A function's parameter layout is worked out once and
  cached on its code object.
- **Lazy core import.** `generic_shape_profiler` (and thus `torch`) is only
  imported once a hook needs it and `torch` is already loaded, so the shim adds
  no measurable startup cost and never drags torch into a process that avoids
  it.

## Docker usage (non-invasive)

Copy the tool into the image and set the env vars. No server source is modified:

```dockerfile
FROM <sglang / vllm / atom base image>

COPY TraceLens/TraceUtils/generic_shape_profiler /opt/generic_shape_profiler

ENV PYTHONPATH=/opt/generic_shape_profiler:${PYTHONPATH}
# Leave unset to bake the tool in inert and switch it on per run.
ENV TRACELENS_SHAPE_DISCOVERY=1
```

## Validated on

All runs on MI355X unless noted, with and without the tool on the same workload.

| Stack | Image | Models |
|-------|-------|--------|
| SGLang 0.5.23 | `lmsysorg/sglang:v0.5.23-rocm720-mi35x` | Qwen3-8B (TP1), DeepSeek-R1-0528 fp8 (TP8), GLM-5.2-MXFP4 with and without EAGLE MTP (TP8, TileLang DSA backends) |
| SGLang 0.5.23 under Magpie AgentX | `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261113` | GLM-5.2-MXFP4 + EAGLE MTP (TP4/EP4, HiCache), agentic workload |
| vLLM 0.33.0 | `vllm/vllm-openai-rocm:v0.33.0-rocm100` | Qwen3-8B, Qwen3-14B (TP1), DeepSeek-R1-0528 fp8 (TP8) |
| ATOM nightly | `rocm/atom-dev:nightly_202611091503` | DeepSeek-V4-Pro (TP8) |
| SGLang 0.5.18 | `sglang-rocm v0.5.18-rocm724-mi30x` (MI300X) | DeepSeek-R1-0528 fp8 (TP8) |

**Unit level:** hooks install on every hooked path; a real `@triton.jit` launch
is recorded exactly once with correct fp16 shapes and dtypes; an autotuned
launch is not double-counted; the tracer is inert outside a profiler window;
and the GPU kernel runs inside the annotation span.

Underscore-prefixed kernel names (`_gemm_a8w8_blockscale_kernel` and friends —
the common case in aiter) keep their leading underscore all the way through
`torch.library`'s schema parser, so the op name equals the kernel name and
prefix-matches the mangled GPU kernel name.

**Full workload (MI300X):** DeepSeek-R1-0528 fp8, TP=8, ISL/OSL 1024/1024,
concurrency 64 — 64/64 requests, 875 tok/s total, 40 ms TPOT. All 8 ranks
annotated. In the serving window each rank produced 1,770 annotation ops, 1,770
`user_annotation` spans and 1,770 `gpu_user_annotation` spans — a 1:1:1 match,
i.e. every annotated launch had GPU work attributed to its span. Example
recorded operand set for `gemm_a8w8_blockscale_kernel`:

```
Input Dims:    [[1025, 7168], [7168, 2112], [1025, 2112], [1025, 56], [56, 17]]
Input type:    ['c10::Float8_e4m3fnuz', 'c10::Float8_e4m3fnuz', 'c10::BFloat16', 'float', 'float']
Input Strides: [[7168, 1], [1, 7168], [2112, 1], [56, 1], [1, 56]]
```

That is an M=1025, K=7168, N=2112 fp8 blockscale GEMM with a column-major B
(stride `[1, 7168]`), bf16 output and two fp32 scale tensors — enough to compute
both FLOPs and bytes moved without guessing.
