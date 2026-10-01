# kernel_shape_tool

Annotate **tensor shapes and dtypes of Triton / FlyDSL kernels** into PyTorch
profiler traces **without patching the inference server**.

This is a drop-in, non-invasive replacement for the
`sglang_roofline_patches/sglang_0_5_18/` patch set (and the equivalent
`vllm_patches` / `atom_roofline_patches` wiring), with no `git apply` into the
server source tree: the tool activates itself through a `sitecustomize.py` shim
that CPython auto-loads in every process on `PYTHONPATH`.

## What problem it solves

A profiler trace already carries full operand metadata for anything that goes
through the PyTorch dispatcher — aten ops and every `torch.ops.*` custom op
(aiter, FlashInfer, sgl-kernel, DeepGEMM wrappers, …) are recorded with
`Input Dims` / `Input type` for free.

Triton and FlyDSL kernels are the exception. They are launched straight from
Python onto the stream, bypassing the dispatcher entirely, so they appear in the
trace as bare GPU kernels with no operand information, which makes them
impossible to roofline.

This tool closes exactly that gap: it hooks the JIT launch entry points and, for
each launch, calls a no-op dispatcher op named after the kernel with one `Tensor`
parameter per operand. The profiler records that op like any other, so the
kernel's shapes, strides and dtypes land in the trace.

## Files

| File | Role |
|------|------|
| `jit_shape_tracer.py` | The core. Hooks Triton / FlyDSL JIT launches and registers each launch's operands as dispatcher ops — `kernel_shapes::<kernel>` and `launcher_shapes::<calling function>`. Knows nothing about any serving framework. |
| `sitecustomize.py` | Auto-loaded shim. Installs the JIT hooks when Triton / FlyDSL are imported, and drives `enable()` / `disable()` from the torch-profiler window. Gated by an env var. |

## Activation

Two environment variables — no source changes:

```bash
export PYTHONPATH=/path/to/TraceLens/examples/custom_workflows/inference_analysis/kernel_shape_tool:$PYTHONPATH
export TRACELENS_SHAPE_DISCOVERY=1
```

Then launch the server and profile as usual (e.g. SGLang's `/start_profile`, or
any `torch.profiler.profile` window). Kernel shapes appear in the resulting
trace with no other changes.

When `TRACELENS_SHAPE_DISCOVERY` is unset or `0`, every hook short-circuits, so
it is safe to leave the directory on `PYTHONPATH` permanently.

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

### Two granularities

Each launch is annotated twice, under two dispatcher namespaces:

| Namespace | Named after | Use it when |
|-----------|-------------|-------------|
| `kernel_shapes::<kernel>` | the Triton / FlyDSL kernel, e.g. `_gemm_a8w8_blockscale_kernel` | joining to GPU kernels by name (the op name *equals* the kernel name, so it prefix-matches the mangled GPU name) |
| `launcher_shapes::<caller>` | the Python function that issued the launch, e.g. `gemm_a8w8_blockscale` | you want the semantic op ("this is a blockscaled fp8 GEMM") rather than whichever kernel implements it |

Why the launcher level exists: a Triton kernel is nearly always reached through
a Python wrapper, and with `with_stack` that wrapper appears in the trace only
as a `python_function` frame — which carries no operand metadata. Annotating it
as an op is what makes `aiter/ops/triton/.../gemm_a8w8_blockscale.py: gemm_a8w8_blockscale`
show up as a `cpu_op` with `Input Dims`.

The wrapper is found by walking out of the JIT machinery *by module*: Triton
reaches `.run()` through its own `KernelInterface.__getitem__` lambda, so the
first frame outside `triton` / `flydsl` and outside this module is the launcher.
No registry, no source scanning, and no matching on function names.

That first launch is also how the launcher is *promoted*: its frame yields the
function object, whose code object is swapped for a trampoline. Every later
call goes through `launcher_shapes::<launcher>(<launcher's tensor args>)`,
whose implementation runs the original function, so the op encloses the
launcher's whole body. Swapping `__code__` changes the one function object in
place, so all references to it are covered without patching modules. The call
that discovers a launcher carries only the kernel-level op. Closures,
generators and `torch` internals are never swapped and keep a leaf
`launcher_shapes` op next to the launch, carrying the kernel's operands.

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

## Design

- **No framework knowledge.** Hooks are installed on Triton's and FlyDSL's own
  classes (`JITFunction.run`, `Autotuner.run`, `Heuristics.run`,
  `JitFunction.__call__`), so every launch is observed regardless of who calls
  it. There is no registry of kernel entry points, no `sys.modules` scanning, no
  source-code heuristics, and no awareness of SGLang / vLLM / any stack. Adding
  support for a new server is a no-op; supporting a new JIT is one function.
- **Install once, gate with a flag.** Hooks are installed permanently the first
  time the JIT modules appear and short-circuit on a boolean. `enable()` /
  `disable()` flip that boolean and swap promoted launchers' code in and out,
  so outside a profiler window every launcher runs its original code.
- **Outermost launch only.** `Autotuner.run` calls `JITFunction.run`
  internally; a thread-local depth counter records each launch once, at the
  outermost hook where the caller's real arguments are still visible.
- **Deferred patching.** `torch` is usually not imported when `sitecustomize`
  runs, so patches are registered as *pending* and applied by an `__import__`
  hook the moment the target module appears. All patches are idempotent and
  wrapped in `try/except`, so they can never break the workload.
- **Lazy core import.** `jit_shape_tracer` (and thus `torch`) is only imported
  once a hook needs it and `torch` is already loaded, so the shim adds no
  measurable startup cost and never drags torch into a process that avoids it.
- **CUDA graphs need no special case.** Kernels launched during graph *replay*
  run no Python, so their shapes are recorded when the graph is *captured* — and
  capture inside a profiler window is already covered by the flag being on.

### Where to look for decode shapes

This follows from the point above and matters when analysing a serving trace.
Because graph replay executes no Python, a kernel that only ever runs inside a
replayed decode graph is **not** annotated in the serving trace; its shapes are
in the graph-capture trace instead. On the DeepSeek-R1 TP=8 run below that is a
large difference: 22,220 annotated launches per rank in the capture window
versus 1,770 in the serving window (prefill and other non-graph paths). Attribute
decode kernels from `graph_capture_profile/`, and use the serving trace for
everything that still launches from Python.

### Annotating a non-JIT launcher

For a launcher that is neither Triton nor FlyDSL (a hand-written HIP extension,
a CuTe kernel, …), one call is enough — no need to teach the tool about it:

```python
import jit_shape_tracer

jit_shape_tracer.mark_launch("my_fused_kernel", [q, k, v, out])
```

It is a no-op unless tracing is enabled.

## How it maps onto the retired patches

Everything that made the old approach require SGLang source edits was pure
*wiring*. Each piece has a non-invasive equivalent here:

| Old patch | What it did | Replacement in this tool |
|-----------|-------------|--------------------------|
| `server_args.patch`, `io_struct.patch`, `grpc_server.patch` | Added a `shape_discovery` flag and plumbed it through config / gRPC | `TRACELENS_SHAPE_DISCOVERY` env var |
| `profiler_manager.patch`, `profile_utils.patch` | Called `enable()` / `disable()` around `torch_profiler.start()` / `.stop()` | `sitecustomize.py` hooks `torch.profiler.profile.start` / `.stop` |
| `decode_cuda_graph_runner.patch`, `eagle_*_cuda_graph_runner.patch`, `multi_layer_*` | Enabled shape discovery around CUDA-graph capture | Not needed: the JIT hooks are always installed and gated by the profiler window, so a capture inside that window is traced |
| `kernel_shape_profiler.patch` | Wrapped a registry of known SGLang / aiter / FlashInfer launcher functions | `jit_shape_tracer.py` hooks the JIT boundary instead, so all Triton / FlyDSL launches are covered without a registry. `torch.ops.*` launchers (aiter, FlashInfer) already carry shapes natively and need no wrapping. |

## Docker usage (non-invasive)

Instead of the `git apply` loop in `build_docker_sglang.sh`, copy the tool into
the image and set the env vars. No SGLang source is modified:

```dockerfile
FROM <sglang-base-image>

COPY examples/custom_workflows/inference_analysis/kernel_shape_tool \
     /opt/kernel_shape_tool

ENV PYTHONPATH=/opt/kernel_shape_tool:${PYTHONPATH}
ENV TRACELENS_SHAPE_DISCOVERY=1
```

The ROCm 7.2.4 HIP / roctracer overwrite step from `build_docker_sglang.sh`
(needed so kernels inside HIP-graph replay reach the trace) is orthogonal to
shape annotation and should be kept if you target that base.

## Validated on

sglang-rocm `v0.5.18-rocm724-mi30x`, Triton 3.7.0, MI300X.

**Unit level:** hooks install on `JITFunction.run` / `Autotuner.run` /
`Heuristics.run` (and FlyDSL's `JitFunction.__call__` when present); a real
`@triton.jit` launch is recorded exactly once with correct fp16 shapes and
dtypes; an autotuned launch is not double-counted; the tracer is inert outside a
profiler window; and the GPU kernel runs inside the annotation span.

Underscore-prefixed kernel names (`_gemm_a8w8_blockscale_kernel` and friends —
the common case in aiter) keep their leading underscore all the way through
`torch.library`'s schema parser, so the op name equals the kernel name and
prefix-matches the mangled GPU kernel name.

**Full workload:** DeepSeek-R1-0528 fp8, TP=8, ISL/OSL 1024/1024, concurrency
64 — 64/64 requests, 875 tok/s total, 40 ms TPOT. All 8 ranks annotated. In the
serving window each rank produced 1,770 annotation ops, 1,770 `user_annotation`
spans and 1,770 `gpu_user_annotation` spans — a 1:1:1 match, i.e. every
annotated launch had GPU work attributed to its span. Example recorded operand
set for `gemm_a8w8_blockscale_kernel`:

```
Input Dims:    [[1025, 7168], [7168, 2112], [1025, 2112], [1025, 56], [56, 17]]
Input type:    ['c10::Float8_e4m3fnuz', 'c10::Float8_e4m3fnuz', 'c10::BFloat16', 'float', 'float']
Input Strides: [[7168, 1], [1, 7168], [2112, 1], [56, 1], [1, 56]]
```

That is an M=1025, K=7168, N=2112 fp8 blockscale GEMM with a column-major B
(stride `[1, 7168]`), bf16 output and two fp32 scale tensors — enough to compute
both FLOPs and bytes moved without guessing.
