# Kernel shape profiler (launcher-wrapping approach)

Adds `Input Dims` / `Input type` / `Input Strides` to PyTorch profiler traces
for GPU kernels that would otherwise appear with no operand metadata, so they
can be rooflined.

It works by wrapping **kernel launcher functions** — the Python entry points
like `gemm_a8w8_blockscale`, `invoke_fused_moe_kernel` or `rmsnorm` — as
`torch.library` custom ops. Each wrapped launcher then shows up in the trace as
a `cpu_op` named after itself, carrying its tensor operands.

> This is the registry-based approach. The alternative, on the
> `jit-shape-tracer` branch, hooks the Triton/FlyDSL JIT launch boundary
> instead. See [Comparison](#comparison-with-the-jit-hook-approach) for which
> one to use.

## Files

| File | Role |
|------|------|
| `kernel_shape_profiler.py` | The core. Wraps launcher functions as `sglang_profiler::<module>_<fn>_<n>` custom ops via an explicit registry plus filtered auto-discovery, and rebinds every module-level reference to them. |
| `sitecustomize.py` | Auto-loaded shim. Drives `enable()` / `disable()` from the torch-profiler window so nothing is wrapped outside a profiling run. |

## Activation

No serving-framework source is patched. Put this directory on `PYTHONPATH` and
set the flag:

```bash
export PYTHONPATH=/path/to/kernel_shape_tool:$PYTHONPATH
export TRACELENS_SHAPE_DISCOVERY=1
```

CPython auto-imports `sitecustomize` at interpreter startup for every process,
so the server and all TP workers pick it up. When
`TRACELENS_SHAPE_DISCOVERY` is unset or `0`, every hook short-circuits, so it is
safe to leave the directory on `PYTHONPATH` permanently.

### Optional knobs

| Env var | Default | Meaning |
|---------|---------|---------|
| `TRACELENS_SHAPE_DISCOVERY` | `0` | Master switch. Must be truthy to do anything. |
| `TRACELENS_SHAPE_FORCE_RECORD_SHAPES` | `1` | Force `record_shapes=True` on the profiler (`Input Dims` only surface when shapes are recorded). Set `0` to respect the server's own setting. |

## What lands in the trace

For a launch of `gemm_a8w8_blockscale(x, w, x_scale, w_scale)`:

```
sglang_profiler::fp8_utils_gemm_a8w8_blockscale_12   <- cpu_op, named after the launcher
    Input Dims:    [[1025, 7168], [7168, 2112], [1025, 56], [56, 17]]
    Input Strides: [[7168, 1], [1, 7168], [56, 1], [1, 56]]
    Input type:    ['c10::Float8_e4m3fnuz', 'c10::Float8_e4m3fnuz', 'float', 'float']
  _gemm_a8w8_blockscale_kernel_GROUP_K_128_...        <- the real GPU kernel
```

The op is named `<last module component>_<function>_<counter>`. The counter is
monotonic per process and never reset, so op names from earlier
`enable()`/`disable()` cycles stay valid.

## How it works

### 1. Choosing what to wrap

Two sources are merged in `enable()`:

**An explicit registry** (`_KERNEL_ENTRY_POINTS`) of `(module_path, function)`
pairs covering SGLang Triton attention, fused MoE, layernorm, FP8 quantization,
LoRA, aiter ops and FlashInfer MoE.

**Filtered auto-discovery** (`_discover_kernel_entry_points`) which force-imports
everything under `sglang.srt.`, `aiter.ops.` and `flashinfer.`, then keeps only
functions that look like kernel launchers — either they have a `Tensor`
annotation, or (when unannotated) their source contains a launch pattern such as
`[grid`, `torch.ops.` or `sgl_kernel.` (`_is_likely_kernel_launcher`).

Two things are deliberately excluded:
- `@triton.jit` objects (`JITFunction` / `Autotuner`). Replacing them in module
  globals breaks Triton's global resolution for device-side calls between JIT
  kernels. Only plain Python functions are wrapped.
- Test / benchmark / autotune modules, which are not entry points and some of
  which call `torch.set_default_device("cuda")` at import time.

### 2. Building the op

`_build_schema_from_sig` derives a `torch.library` schema from the function
signature, mapping annotated parameters to schema types (`_infer_schema_type`
handles both real annotations and PEP 563 string annotations). Tensor
parameters become schema arguments; everything else is passed through a
thread-local side channel (`_stash_non_tensor_args`), because a schema of
`-> ()` cannot carry them. The return value comes back the same way
(`_stash_return_value`).

If no schema can be built (no annotations, `*args`/`**kwargs`), the function
falls back to `_make_record_function_wrapper`, which emits a `record_function`
event with the shapes embedded in the *event name* instead of as structured
args.

### 3. Applying the wrapper

`_patch_all_references` scans all of `sys.modules` and replaces **every**
module-level attribute pointing at the original function. This is required
because of the `from X import Y` pattern: patching only the defining module
would miss callers that already captured a local binding.

Launchers whose reference was captured as an *instance attribute* before
`enable()` (e.g. `self.fn = dispatch_w8a8_block_fp8_linear()`) cannot be
intercepted this way. For those, the registry targets the **inner** kernel that
the wrapper looks up from module globals on every call — which is why
`gemm_a8w8_blockscale` is registered rather than the outer dispatch wrapper.

## Safety properties

The implementation carries a few hard-won guards worth preserving:

**The `Library` is never torn down.** Dropping it destroys the registered ops.
A wrapper reference can outlive `disable()` — a module imported lazily *during*
a profiling window may have captured the wrapper via `from X import Y` and is
not in `_patches` to be restored. If the backing op were freed, that leaked
wrapper would dispatch into freed memory and segfault. The `Library`, the
monotonic op counter and the wrapper cache are all kept alive for the process
lifetime, and the `if not _enabled` guard in each wrapper routes leaked calls
straight to the original function.

**Global torch state is restored around imports.** `enable()` imports a large
number of modules, some of which mutate process-global torch state at import
time — most notably `torch.set_default_device("cuda")`. That mutation is not
undone by `disable()`. A leaked default device corrupts downstream CPU tensor
creation: a buffer such as `seq_lens_cpu` is suddenly allocated on CUDA, which
surfaces as `Buffer seq_lens_cpu has different device than before` and later as
out-of-bounds GPU memory faults in index kernels.
`_preserve_global_torch_state` and the per-import restore in
`_force_import_submodules` keep that from escaping into the serving path.

**Every wrapper falls back to the original.** Unbindable signatures, all-`None`
tensor arguments, and dispatch failures all call the original function
directly, so wrapping can never change behaviour.

**Wrappers are never wrapped again.** Frameworks often keep a compat
re-export, so the same function is reachable by two module paths. Once the
first is wrapped, `_patch_all_references` rebinds *every* reference to it —
including the second path — so the second registry entry would otherwise
resolve to our own wrapper and wrap it a second time, nesting two annotations
around a single call and double-counting it. Wrappers carry a
`_kernel_shape_wrapper` marker and `enable()` skips them.

## Registry staleness

The explicit registry is coupled to framework internals, and it drifts. Measured
against SGLang 0.5.18 + matching aiter, only **5 of the original 24 entries
still resolved**: SGLang had moved its Triton kernels out of
`sglang.srt.layers.*` into a separate `sglang.kernels.ops.*` package, and aiter
had regrouped its Triton ops into subpackages.

Current-layout paths have been added alongside the legacy ones (**21 of 40**
resolve now; the 19 that do not are the legacy paths, kept deliberately).
`_resolve_target` returns `None` for a path that does not exist and the entry is
skipped silently, so one registry serves several framework versions.

Two entries cannot be fixed by a path change: aiter's batched blockscale GEMM
was *renamed* (now
`batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant`), and
`flashinfer` is not installed in this image. Auto-discovery covers the former.

The practical lesson: **auto-discovery is doing most of the work**, and the
registry is best treated as a hint list for launchers the heuristics miss. Check
what actually resolved before trusting the registry on a new stack.

## Validation

Verified with `validate_kernel_shape_profiler.py`:

- A synthetic launcher is registered, produces a `cpu_op` with
  `Input Dims: [[4, 8], [8, 16]]` and per-tensor dtypes, returns the correct
  value, and passes its non-tensor argument through the side channel.
- Both the definition-site binding and a `from X import Y` binding are rebound,
  and both are restored on `disable()`.
- A wrapper captured while enabled still returns correct results after
  `disable()` (the leaked-reference case the persistent `Library` protects).
- The real aiter `gemm_a8w8_blockscale` is annotated with
  `Input Dims: [[1025, 7168], [2112, 7168], [1025, 56], [17, 56], []]` and
  `['c10::Float8_e4m3fnuz', 'c10::Float8_e4m3fnuz', 'float', 'float', '']`,
  exactly once per call.
- `enable()` took **18–19 s** with auto-discovery scoped to `aiter.ops.` alone,
  which is the cost of the package walk. Budget for more with `sglang.srt.`
  included.

### What the operands mean at launcher level

Note the difference from a JIT-boundary tracer. Here the recorded operands are
the launcher's **call arguments**, so for the same GEMM you see `w` as
`[2112, 7168]` (before the launcher transposes it) and the optional `y` output
as an empty slot because it was not passed. A JIT-level tracer sees the
**kernel's** operands: `[7168, 2112]` for the transposed view and the
materialized `[1025, 2112]` output. Neither is wrong; they describe different
boundaries. For roofline work the kernel-level view is usually the one that
matches the GPU kernel's actual traffic.

## Comparison with the JIT-hook approach

| | This branch (launcher wrapping) | `jit-shape-tracer` branch (JIT hooks) |
|---|---|---|
| Interception point | the Python launcher function | `JITFunction.run` / `Autotuner.run` / FlyDSL `__call__` |
| Backend coverage | **any** backend — Triton, ASM, CK, aiter C++ bindings, RCCL collectives, torch ops | Triton and FlyDSL only |
| Needs to know the framework | **yes** — module paths in the registry are coupled to SGLang / aiter versions | no |
| Finds unlisted kernels | only via auto-discovery heuristics | automatically, all of them |
| `enable()` cost | high — walks and imports whole package trees, rebinds `sys.modules` | negligible — installs a few method wrappers |
| Risk | rebinding module globals and import side effects (see Safety properties) | contained to the JIT classes |
| Naming | op named after the launcher, with a numeric suffix | op name *equals* the kernel name, plus a launcher-level op |

Use this branch when you need shapes for non-Triton work (ASM GEMMs, CK MoE,
aiter C++ kernels, collectives). Use the JIT-hook branch when you want
framework independence, complete Triton coverage without a registry, and a much
smaller blast radius.

## Limitation: CUDA graphs

Kernels that only run inside replayed CUDA graphs execute no Python, so no
wrapper fires during replay. Their shapes are recorded when the graph is
**captured**. If the capture happens inside a profiler window, the capture-time
trace holds those shapes — so a decode-path analysis needs the graph-capture
trace, not just the serving-window trace.
