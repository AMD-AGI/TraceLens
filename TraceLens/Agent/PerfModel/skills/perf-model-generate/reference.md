<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Perf-model authoring: reference

Detail for [SKILL.md](SKILL.md). Run every `python3` command below in the Python
environment (or command prefix) given as an input, not the system `python3`.

## Entry points

Only in standalone use, when the op is not given. A caller that supplies the op
skips this section.

### EP1: CSV-driven, top ops missing a perf model

Use when a `unified_perf_summary.csv` is available and you want to prioritize by
runtime impact. The top-ops ranking reads each op's call stack from the CSV's
`call_stack_full` column.

**Run the top-ops ranking** (`utils/unified_perf_ops.py`):
```bash
python3 -m TraceLens.Agent.PerfModel.utils.unified_perf_ops \
    /path/to/unified_perf_summary.csv \
    --mode top-ops
```

This groups rows by `name`, sums runtime across all shapes, keeps names whose
summed runtime is at least `--threshold` of the global total and that have no
perf model (`row_has_perf_model`: `has_perf_model` true with non-zero GFLOPS or
Data Moved), and prints a confirmable candidate table. Add `--emit-extension` to
write a starter `<csv_stem>_triage_extension.py` beside the CSV, holding
`TRIAGE_OP_NAMES` (the ranked candidate names) and empty `perf_model_extension`
and `op_category_extension` dicts.

Also run the legacy `other-bucket` mode (default) to catch ops in the "other"
category even if they are individually below `--threshold`:
```bash
python3 -m TraceLens.Agent.PerfModel.utils.unified_perf_ops \
    /path/to/unified_perf_summary.csv --also-global-pareto
```

### EP2: Manual op

Use when the cpu_op is named directly: the exact profiler `name` string (e.g.
`aiter::my_op`), and optionally the kernel name from rocprofv3 /
`kernel_details_summary` and the call stack.

Proceed to "Authoring steps" below with that single op. No script required.

### EP3: Framework scan

Use to enumerate the cpu_ops of a whole framework (aiter, vLLM, SGLang, or
ATOM), optionally filtered (e.g. "attention kernels").

```bash
python3 -m TraceLens.Agent.PerfModel.utils.scan_framework_ops \
    --aiter /path/to/aiter \
    --filter attention
```

The script performs a **static source scan** (no GPU/run required):
- aiter: `@compile_ops("aiter::<name>")`, `direct_register_custom_op`,
  `torch.library` → `aiter::<name>`
- vLLM: `direct_register_custom_op(op_name=...)` / `torch.library` → `vllm::<name>`
- SGLang: `torch.library` / `direct_register_custom_op` → `sgl_kernel::*` and
  `sglang_profiler::*`

Add `--check-mapping` to mark ops that already have a built-in perf model
(needs TraceLens importable, e.g. `pip install -e .`). It checks the core map
before `apply_extension`, so an op that an extension registers shows False; that
is not a gap.

For each matched op it outputs: registered name, source file, a regex-based
hint of the call chain to the kernel (confirm it in the source), and any sibling `test_<op>` / `benchmark_<op>` files
(roofline reference).

For deep or ambiguous chains, instruct an explore sub-agent to traverse the
repo (same technique used to map FlyDSL ops).

The scanner covers aiter, vLLM, and SGLang. For ATOM, use an explore sub-agent
on `atom/model_ops/` and the model files: ATOM mostly calls aiter ops, so the
trace names are usually `aiter::*`, but it also has its own wrappers and
FlyDSL kernels. When one kernel is shared, look in all of these frameworks
(and FlyDSL) for other registrations of it. Layouts and data types often
differ, and those names will show up in other traces.

---

## Per-op choices

Each op gets a depth and an output mode. Use the defaults from SKILL.md's Inputs
unless the caller chooses otherwise:

| Choice | What to produce |
|--------|----------------|
| **full** | A perf model class with `get_param_details`, `flops()`, `bytes()`, `get_compute_precision()` + registration |
| **categorize-only** | Category label only (`op_category_extension` or `get_pseudo_op_category_only_mappings()`); no class; `has_perf_model` stays False. Always the choice for a `(Synthetic Op)` name: the name embeds templates and shapes, and its time is only part of the logical op |

Then pick the output mode:

| Mode | Where the result goes |
|------|-----------------------|
| **integrate** | Class added to `TraceLens/PerfModel/extensions/*_perf_model_extensions.py`; name registered in `pseudo_ops_perf_utils.py` |
| **extension-only** | Class + mappings added to an extension file ([Mode B](#mode-b-extension-only-file)) |

---

## Authoring steps (per op, full model)

### Locate

From the call stack (EP1/EP3) or the info given directly (EP2):
- Read the paper or blog for architectural choices that change the math
  (sparsity, compression, routing, cache layout), starting from the model
  config's `paper` link when one is given and finding it yourself otherwise.
  Some of these are not obvious from the trace or the kernel.
- Find the Python binding in the vendor repo.  Record the **relative path**
  (used in the class docstring `Reference implementation:` line).
- Find the underlying kernel (HIP/Triton/CUDA).  Look for:
  - aiter: `aiter/ops/<op>.py` → `@compile_ops(fc_name=...)` or HIP `__global__`
  - vLLM: `vllm/model_executor/layers/...` → `torch.ops.vllm.<name>`
  - SGLang: `sglang/srt/...` → `kernel_shape_profiler` wrapper
  - ATOM: `atom/model_ops/...` → usually an aiter op, sometimes a FlyDSL or
    Triton kernel called directly

### Vendor reference

vLLM, aiter, SGLang, and ATOM frequently ship `test_<op>.py` /
`bench_<op>.py` / `op_tests/` beside the kernel.  These often print
theoretical FLOPs and bytes.

- Check `aiter/op_tests/`, `aiter/aiter/ops/*/test_*.py`
- Check `vllm/benchmarks/`, `vllm/tests/kernels/`
- Check `sglang/benchmark/`, `sglang/test/`
- ATOM: when it calls an aiter op, the aiter test applies; otherwise check
  ATOM's own tests for the kernel

**Use these expressions as the derivation reference** when writing `flops()` /
`bytes()`, and as an **independent cross-check** of your model (separate from
HW-counter validation).  If the wrapper, the kernel, and the vendor formula
disagree, find out why. Do not pick the most convenient number.

### Map Input Dims

Inspect several events, not one shape. Map `event["args"]["Input Dims"][i]`
and `event["args"]["Input type"][i]` to the function's arguments. Also check
`Input Strides`, `Concrete Inputs` (scalars such as `group_size`, boolean
flags), `annotation`, and which parent or child actually owns the GPU time.

Do not assume `Input Dims[0]` is the main input, that it sets the compute
precision, or that the output dtype matches the first input. Cover every dtype
the binding can emit, not only the dtype in this trace. Read sizes from
`Input Dims` or `Concrete Inputs`, never from the op name.

- MoE: token counts may be before or after routing and padding. Weights may be
  read for every expert or only the active ones.
- Quantized ops, keep separate: storage dtype, unpacked dtype, scale dtype and
  shape, accumulator dtype, output dtype, and the matrix-instruction dtype
  used for the roofline peak.

### Base class

| Op kind | Base class | File |
|---------|-----------|------|
| Dense GEMM (any dtype) | `GEMM` | `perf_model.py` |
| RMSNorm / LayerNorm | `RMSNorm` (subclass of `Normalization`) | `perf_model.py` |
| Elementwise unary | `UnaryElementwise` | `perf_model.py` |
| Elementwise binary | `BinaryElementwise` | `perf_model.py` |
| Reduction | `Reduce` | `perf_model.py` |
| Per-group quantization | `GroupQuant` | `perf_model_extensions.py` |
| Inference attention | `InferenceAttention` | `attention_perf_model_extensions.py` |
| MoE | `moe_aiter_*` family | `moe_perf_model_extensions.py` |

**Reuse-first**: before writing a new class, check if an existing one already
matches the op's `Input Dims` / `Input type` layout.  If yes, just add a
registry mapping row. Same math with different shape extraction or dtypes:
subclass and override `get_param_details`. Add a new base only when several
models share a concept that has no owner. `get_pseudo_op_mappings()` also
holds direct op names that are not pseudo ops.

### Implement

Use the mandatory [docstring template](#perf-model-class-docstring-template).  Key rules:

**FLOPs, from the real kernel:**
- Read the kernel source; count **major MFMA / tensor-core operations** of the
  operation for FLOPs.
- Do not estimate from the output buffer alone.
- For GEMM: `flops = 2 * M * N * K`. Fused epilogues, recomputation, sparsity,
  and routing can add or remove work.
- The kernel implementation may pad or tile shapes (block sizes, matrix-
  instruction sizes, split-K). That is a kernel artifact: leave it out of both
  FLOPs and bytes. Count padding only when the padded tensor is part of the op's
  contract, such as a padded output the caller receives.
- State what is included (top-k, activation, normalization, QK recomputation).
- For RMSNorm: `flops ≈ 5 * T * N` (variance + norm + scale).
- For attention prefill: `flops = 4 * T^2 * H * d` (QK + softmax + V).
- For recurrent attention (GDN): per-token flops from the state update rule.

**Bytes, all major loads and stores:**
- Every input tensor that is read: `nelems * bpe_in`.
- Every output tensor that is written: `nelems * bpe_out`.
- For quantized ops: activations, weights, and scales are separate read terms;
  output dtype may differ from input (`output_bpe != input_bpe`).
- For packed weights (fp4/mxfp4): `bpe = 0.5`.
- Do **not** assume `output_bpe == input_bpe`.
- Roofline bytes are algorithmic traffic at one memory level. They will not
  match hardware counters, which also see caches, re-reads, and write policy.

**compute_precision pitfalls:**
- `get_compute_precision()` must return the **dominant MFMA dtype** the kernel
  actually uses (fp8, int8, bf16, fp16 …).
- Packed / compressed weights: the MFMA dtype is determined by the **unpacked**
  logical dtype the MMA uses internally (often fp8 or fp16 even if weights
  arrive as fp4).
- Weight-dtype-dependent precision: if the kernel selects the MFMA path based
  on a runtime flag or dtype check, model that branch. Fused MoE is a common
  case: the weight dtype, not the activation, selects the precision.
- Also set the math type (matrix vs vector). If precision or the compute unit
  is missing, the report can show TFLOP/s and bandwidth but no roofline
  percentage, and the impact score falls back to a heuristic.

#### Attention annotations

`InferenceAttention` subclasses do **not** read FLOPs/bytes from `Input Dims`
alone: for chunked-prefill / paged-decode the per-request KV context lengths
are runtime-only. They come from the trace's per-step `user_annotation` event,
which `InferenceAttention._parse_chunk_stats(event["annotation"])` passes to
`IterationAnnotation(...).chunk_stats()` in
`TraceLens/TraceUtils/utils/annotation_utils.py`. The result is:
- context (prefill) aggregates: `c_sq` (Σ query tokens), `c_sk` (Σ kv
  tokens), `c_sqsq` (Σ sq·sq), `c_sqsk` (Σ sq·sk)
- generation (decode) aggregates: `g_sq`, `g_sk`, `g_sqsq`, `g_sqsk`

The annotation text depends on the framework. Any format works if
`IterationAnnotation` parses it and it carries the sq/sk sums (`has_sqsk`);
otherwise `chunk_stats()` raises and the op is no-perf. Detailed formats it
parses today:
```
# vLLM
execute_64_context_0(sq0sk0sqsq0sqsk0)_generation_64(sq64sk131072sqsq64sqsk131072)
# SGLang
step[MIXED bs=2 c=1 g=1 c_sq=5 c_sk=8 c_sqsq=25 c_sqsk=40 g_sq=1 g_sk=12 g_sqsq=1 g_sqsk=12]
# ATOM (needs ATOM_ENABLE_DETAILED_ANNOTATION=1)
decode[bs=32 tok=128 d=32 spec=3 sqsq=512 sqsk=262144 sk=65536]
```
The native formats (`execute_context_2(14721)_generation_0(0)`,
`step[DECODE bs=64]`, `decode[bs=64 tok=64 d=64]`) have counts only, so they
cannot drive attention perf models. Enable the framework's detailed annotation
when profiling. See `TraceLens/Agent/Profiling/README.md`.

For a framework whose format is not parsed yet, add the parser in TraceUtils
with `IterationAnnotation.register_format(kind, pattern, parser)` (or a new
entry in `FORMATS`). Do not parse annotation text in the perf model.

The stats mean the same thing for every format:
- **Prefill-only step**: all `g_*` are 0.
- **Decode-only step**: all `c_*` are 0. Each request has `sq=1` and
  `sk=ctx_len`, so `g_sq = #requests`, `g_sk = Σ ctx_len`, `g_sqsk = g_sk`,
  `g_sqsq = #requests`. With speculative decoding (MTP) `sq > 1`, so
  `g_sq > #requests`.

You normally do **not** synthesize the annotation yourself. It must be attached
to the event before the perf model runs (see
[Annotation propagation](#annotation-propagation)). Your job is to map
`Input Dims` to Q/K/V shapes and `Input type` to dtypes in `get_param_details`,
then let the base `flops()`/`bytes()` consume the parsed `c_*`/`g_*` stats.

**FP8 KV cache:** paged decode kernels often store K/V as FP8 (1 byte) while Q
and the output stay BF16. KV reads dominate decode bytes, so override `bytes()`
to use the cache dtype (`name2bpe(Input type[k_cache])`) for the K/V read terms
rather than the Q dtype.

- Document the `Input Dims` layout your class expects and that it needs a
  detailed (sq/sk) annotation.

#### Annotation propagation

Required when the model reads `event["annotation"]`, attention or not. The
inference report copies the enclosing step annotation only onto ops whose name
starts with an entry in `_INFERENCE_ANNOTATION_OP_PREFIXES` in
`TraceLens/Reporting/generate_perf_report_pytorch_inference.py`. If your op is
not listed, its events have **no** `annotation` and the model silently degrades
to no-perf.

- **Integrate mode:** add the op name to `_INFERENCE_ANNOTATION_OP_PREFIXES`.
- **Extension-only mode:** core files stay untouched, so re-run annotation from
  the extension with a module-level `tree_postprocess_extension`, which the
  report generator calls on the built tree after the built-in
  `apply_annotation` and before perf models execute:
  ```python
  def tree_postprocess_extension(trace_tree):
      trace_tree.apply_annotation(name_filters=["<your op name>"])
  ```

Either way, `apply_annotation` matches by `name.startswith(...)`. It
finds the enclosing `user_annotation` event (by timestamp containment) and sets
`event["annotation"]` to its name, whatever the format. If a trace has no such
annotation event, or its format has no sq/sk sums, wrap your
`get_param_details` body in `try/except` returning
`InferenceAttention.no_perf_param_details()` so `flops()`/`bytes()` return
`None` gracefully.

#### Annotation use for other inference ops

Inference ops often depend on the true KV length, while `Input Dims` only show
the allocated buffer sizes (the KV cache pool, padded or maximum lengths).
Modeling from those sizes overstates the work. Before making such an op
category-only, check whether the step annotation can supply what is missing:
its sq/sk sums give the query and KV lengths per step, as the existing
attention models use. Where the annotation only bounds the value, for example
for a per-request quantity, an approximation is acceptable if the class
docstring says so. The annotation reaches the event as described in
[Annotation propagation](#annotation-propagation).

The annotation describes the whole step, not necessarily this op's inputs.
Guard its use:
- **Rows match:** use the sums only when the op covers the whole step, that is,
  when Σ sq (`c_sq + g_sq`) equals the op's query or token rows in
  `Input Dims`. Otherwise return no-perf.
- **Cap at the buffer:** never let an annotation length exceed the matching
  buffer dimension in `Input Dims`. Clamp it, and say so in `param_details`.
- **Compressed inputs:** the sums count raw tokens. When the op works on
  compressed or pooled KV (sparse indexers, compressed caches), scale `sk` by
  the compression ratio, or return no-perf if that ratio is not known.

### Register

**Integrate mode**: add to `pseudo_ops_perf_utils.py`:
```python
# in get_pseudo_op_mappings():
"aiter::my_op": perf_model_extensions.MyOp,

# in get_pseudo_op_category_only_mappings() (categorize-only):
"aiter::my_op": "GEMM",
```

**Extension-only mode**: add to the extension file ([Mode B](#mode-b-extension-only-file)):
```python
perf_model_extension = {
    "aiter::my_op": MyOp,   # category comes from the class
}
op_category_extension = {
    "aiter::my_other_op": "GEMM",   # categorize-only
}
```

Key the mapping on the CPU op name that appears in the trace (or a registered
pseudo op), never on a kernel name. A generic key (e.g. `aten::topk`, or a kernel
shared by several models) must not map to a model-specific class: list its
callers across frameworks, and keep the model generic or guard it so other
callers get no-perf.

### Regenerate and check

```bash
python3 -m py_compile /path/to/extension_or_module.py
```

Regenerate the report from the checkout that contains the new perf model, with
the same config path the model-bringup skill uses:
```bash
python3 -m TraceLens.Agent.Analysis.utils.config_utils build --scope standalone \
    --analysis_mode inference --output_dir /path/to/report \
    --job trace_path=/path/to/trace.json.gz,platform=<platform>[,capture_folder=/path/to/captures] \
    [--set extension_file=/path/to/extension.py]   # extension-only mode only
python3 -m TraceLens.Reporting.generate_perf_report_from_config \
    --config /path/to/report/cache/run_config.yaml
```

Re-check `has_perf_model`, `op category`, GFLOPS, Data Moved, and
`param_details`. Confirm values such as top-k, `num_experts` and `hidden_size`
against the model config and launch settings when given; without them, name the
values the trace alone could not confirm. One trace is one example; do not
hardcode a dtype, layout, or variant that the op can vary.

---

## Pseudo ops

Never add a pseudo op on your own judgment: run the stability and
cost check below and report the table; the caller decides. This section says
when one is a candidate and what it needs if the caller approves.

**A pseudo op is a candidate only when the profiler event is the wrong unit.** A stable
CPU op that already owns the runtime and the arguments needs a mapping, not a
pseudo op. Add one when:

- one logical op appears as several stages under a broad parent;
- the useful shapes are on the parent and the runtime is on the children;
- Python ranges and GPU kernels must be one modelable unit;
- modes need separate names, as with `pseudo_v4_paged_decode_swa`,
  `pseudo_v4_paged_decode_csa`, and `pseudo_v4_paged_decode_hca`.

If synthetic ops are among the top kernels, check the tree visually before
modeling the logical op. The `(Synthetic Op)` name itself is categorize-only (see
[Per-op choices](#per-op-choices)).

If the caller approves, a pseudo op needs both an injector in
`TraceLens/Trace2Tree/extensions` (stable detection and a shape donor) and the
same name in `get_pseudo_op_mappings()`. Detect from an op name, call-stack
range, or kernel family, not from one timestamp, UID, or an unstable full kernel
name. The report must run with pseudo ops enabled. Check that each kernel is
attached once, parents are correct, neighbors are not captured, and injection
time stays low as the tree grows.

**Stability and cost check (required before adding a pseudo op).** Do both
checks below.

1. **Brittleness.** The injector matches on certain Python functions and events
   between the parent `cpu_op` and the insertion point. Those must appear on
   every call chain that can reach the op, not just the one in this trace.
   Read the vendor library source (framework, aiter, kernel wrappers) and list
   every path from the parent `cpu_op` to the insertion point. Cover dispatch
   branches, config and env flags, GPU-arch specific paths, eager versus graph
   capture, prefill versus decode, and speculative decoding. Also cover the
   other framework versions TraceLens supports, as listed in the
   `build_docker_*.sh` scripts. For each path, note whether every function
   and event the injector relies on is present with the same name and nesting.
   The verdict is **stable** if they match on all paths, **fragile** otherwise,
   with the differing paths named.
2. **Overhead.** State the injector's complexity in terms of the tree size
   `n`. Examples: one pass over events is O(n); a descendant search per
   candidate is O(n·m); a scan of all events per candidate is O(n²). Name the
   loop that sets it. Then measure it on a real trace: tree build time with
   and without the injector, and the number of events it adds.

Report it like this:

| Pseudo op | Brittleness | Paths checked | Differing paths | Complexity | Measured overhead |
|---|---|---|---|---|---|
| `<name>` | stable / fragile | `<count and list>` | `<paths, or none>` | O(...) | `<seconds added on <trace>, events added>` |

---

## Hand off to validation

Hand the regenerated `perf_report_csvs` directory and the op name to the
validate-perf-model skill
(`TraceLens/Agent/PerfModel/skills/validate-perf-model/SKILL.md`), which takes
them as its Inputs along with the GPU, `--arch` and output directory it
requires. The validator loads only perf models integrated into TraceLens core,
so integrate an extension-only model first.

---

## Perf model class docstring template

Every class registered in `perf_model_extension` or `get_pseudo_op_mappings()`
MUST have a docstring following this layout:

```python
class MyOp(BaseClass):
    """
    Performance model for <exact_profiler_name>.

    Reference implementation:
        <relative/path/inside/vendor/repo.py>

    <One or two sentences: the math, including fused steps and optional
    outputs.>

    Signature:
        def <fn_name>(
            <arg0>: torch.Tensor,  # [0] input, <dtype> [<shape>]
            <arg1>: torch.Tensor,  # [1] output (in place), <dtype> [<shape>]
            <arg2>: int,           # [2] scalar, <meaning and allowed values>
            <arg3>: Optional[torch.Tensor] = None,  # [3] optional input, ...
        ) -> <return>

    Options:
        Shape variants: <decode / prefill, padded / ragged, compressed /
            uncompressed, kernel-path switches>; this model covers <which>.
        Allowed values: <e.g. topk <= 1024, page_size a power of two>.
        Dtypes: <every input and output dtype the binding accepts>.

    Expected Input Dims from trace:
        [<index>] = <shape description>
        e.g. [(M, K), (N, K), (M, 1), (N, 1)]

    Expected Input type from trace:
        [<dtype0>, <dtype1>, ...]

    Concrete Inputs[<k>] = <semantics>   # omit if unused

    Roofline -- FLOPs:
        <closed-form expression>
        Counts: <what is included>. Leaves out: <what is not>.

    Roofline -- bytes moved:
        bytes_read_A   = M * K * bpe_in
        bytes_read_B   = N * K * bpe_in
        bytes_write    = M * N * bpe_out
        Total          = bytes_read_A + bytes_read_B + bytes_write

    Notes:
        <Annotation format, index conventions, bpe assumptions.>

    Vendor roofline reference:
        <path/to/test_my_op.py or benchmark_my_op.py>  # if found
    """
```

Minimal variant when inheriting the roofline unchanged and adding nothing:
```python
class MyOp(ExistingClass):
    """
    Performance model for <exact_profiler_name>.

    Reference implementation: <path>

    flops/bytes inherited from <ExistingClass>.
    """
```

When the class reuses the roofline but adds its own `get_param_details`, the
parsing is new, so document what it parses:
```python
class MyOp(ExistingClass):
    """
    Performance model for <exact_profiler_name>.

    Reference implementation:
        <relative/path/inside/vendor/repo.py>

    Signature:
        <annotated Python signature, one comment per argument, as above>

    Options:
        <variants, allowed values and dtypes, and which ones are covered>

    Expected Input Dims from trace:
        [<index>] = <shape description>

    Expected Input type from trace:
        [<dtype0>, <dtype1>, ...]

    Concrete Inputs[<k>] = <semantics>   # omit if unused

    Annotation: <fields read from it, or "not used">

    flops/bytes inherited from <ExistingClass>.
    """
```

**Rules:**
- First line: `Performance model for <exact_profiler_name>.`
- Write the signature as an annotated Python signature with one comment per
  argument: its index in `Input Dims`; whether it is an input, an in-place
  output, optional, or a scalar; its shape and dtype; and, for scalars and
  flags, what they mean. No tables.
- Options lists every variant the binding accepts and says which ones the model
  covers. A variant the model does not cover must return no-perf. A documented
  approximation, such as an approximate KV length from the annotation, is fine;
  an undocumented guess is not.
- Always state `output_bpe` explicitly when it differs from `input_bpe`.
- `get_param_details` returns every value `flops()` and `bytes()` use, including
  those parsed from the annotation. They become `perf_params` in the report,
  which is what validation runs from.

---

## Output modes

### Mode A: integrate into TraceLens core

Edit these files in `TraceLens/PerfModel/extensions/`:

| What to add | File |
|------------|------|
| New class | `perf_model_extensions.py` / `attention_perf_model_extensions.py` / `rmsnorm_perf_model_extensions.py` / `moe_perf_model_extensions.py` |
| Full model mapping | `pseudo_ops_perf_utils.py` → `get_pseudo_op_mappings()` |
| Category-only mapping | `pseudo_ops_perf_utils.py` → `get_pseudo_op_category_only_mappings()` |

### Mode B: extension-only file

Add the class and mappings to an extension file: the one EP1's `--emit-extension`
writes, or a new `.py` with the same dicts. The dict layout is under
[Register](#register), the annotation hook under
[Annotation propagation](#annotation-propagation).

**Do not** edit `torch_op_mapping.py` for perf-model authoring; it is product
code.

---

## Model-specific configuration

Hardcode or read a model-specific value only when it changes FLOPs or bytes,
it is not in the event, annotation, or kernel name, it is constant for the
whole report, and the chosen value is copied onto the event. Prefer a value
parsed from the trace. Do not read an environment variable inside `flops()` or
`bytes()`; the same event would then depend on hidden process state. Read it
while building the pseudo op and store it in the event arguments.

A silent fallback between variants is risky. Prefer a warning or a no-perf
result when a wrong assumption would change the report, unless the
approximation is documented in the class docstring. Show the value that was
used in `param_details`.
