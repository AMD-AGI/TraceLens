---
name: unified-perf-report-postprocess
description: >-
  Authors TraceLens perf models for cpu_ops missing coverage.  Driven from
  three entry points: (1) CSV >4% triage from a unified_perf_summary.csv,
  (2) manual op + optional kernel/callstack, (3) static scan of a vendor
  framework repo (aiter/vLLM/SGLang/ATOM) with optional filter.  For each op the
  user picks full perf model or categorization-only, and either integrates the
  result into TraceLens/PerfModel/extensions/ or emits a standalone
  extension file.  After authoring, asks the user whether to validate with
  the validate-perf-model skill (rocprofv3 hardware counters).
  A generated class is a draft: confirm it against the kernel, several trace
  events, and any model-specific config the trace does not contain.
---

# Unified perf-model authoring

## Entry points — pick one to begin

### EP1 — CSV-driven: top ops missing a perf model

Use when a `unified_perf_summary.csv` is available and you want to prioritize by
runtime impact.  Newer TraceLens embeds the full call stack inline in this CSV
(`call_stack_full` column), so no companion `unified_perf_callstacks.csv` is
needed; the triage script reads the call stack directly from the summary.

**Run the triage script:**
```bash
python3 <skill-dir>/run_other_bucket_triage.py \
    /path/to/unified_perf_summary.csv \
    --mode top-ops --threshold 0.04
```

This groups rows by `name`, sums runtime across all shapes, keeps names whose
summed runtime is **> 4% of the global total** and `has_perf_model == False`,
and prints a confirmable candidate table.  Add `--emit-extension` to write a
starter `<csv_stem>_triage_extension.py` beside the CSV.

Also run the legacy `other-bucket` mode (default) to catch ops in the "other"
category even if they are individually below the 4% threshold:
```bash
python3 <skill-dir>/run_other_bucket_triage.py \
    /path/to/unified_perf_summary.csv --also-global-pareto
```

**After printing the table**, confirm the candidate list with the user before
proceeding to authoring.


### EP2 — Manual op

Use when the user directly names the cpu_op (and optionally the kernel name and
call stack).

1. Ask the user for:
   - `name`: the exact profiler `name` string (e.g. `aiter::my_op`)
   - Optional: kernel name from rocprofv3 / `kernel_details_summary`
   - Optional: call stack (or the user pastes it)
2. Proceed to "Authoring steps" below with that single op.

No script required.  Use `emit_perf_model.py` (`--manual-op NAME`) to generate
a class stub.

### EP3 — Framework scan

Use when the user asks to enumerate cpu_ops for a whole framework (aiter, vLLM,
SGLang, or ATOM), optionally filtered (e.g. "attention kernels").

```bash
python3 <skill-dir>/scan_framework_ops.py \
    --repo /path/to/aiter \
    --filter attention \
    [--trace /path/to/unified_perf_summary.csv]
```

The script performs a **static source scan** (no GPU/run required):
- aiter: `@compile_ops("aiter::<name>")`, `direct_register_custom_op`,
  `torch.library` → `aiter::<name>`
- vLLM: `direct_register_custom_op(op_name=...)` / `torch.library` → `vllm::<name>`
- SGLang: `torch.library` / `direct_register_custom_op` → `sgl_kernel::*` and
  `sglang_profiler::*`

Add `--check-mapping` to mark ops that already have a built-in perf model
(needs TraceLens importable, e.g. `pip install -e .`).

For each matched op it outputs: registered name, source file, reconstructed
call chain to kernel, and any sibling `test_<op>` / `benchmark_<op>` files
(roofline reference).  When `--trace` is given it annotates which ops actually
appeared in the trace.

For deep or ambiguous chains, instruct an explore subagent to traverse the
repo (same technique used to map FlyDSL ops).

**Confirm the candidate table with the user before proceeding.**

The scanner covers aiter, vLLM, and SGLang. For ATOM, use an explore subagent
on `atom/model_ops/` and the model files: ATOM mostly calls aiter ops, so the
trace names are usually `aiter::*`, but it also has its own wrappers and
FlyDSL kernels. When one kernel is shared, look in all of these frameworks
(and FlyDSL) for other registrations of it. Layouts and data types often
differ, and those names will show up in other traces.

---

## Per-op choices (confirm with user before authoring)

For **each** candidate op the user picks:

| Choice | What to produce |
|--------|----------------|
| **full** | A perf model class with `get_param_details`, `flops()`, `bytes()`, `get_compute_precision()` + registration |
| **categorize-only** | Category label only; no class; `has_perf_model` stays False |

Then pick the output mode:

| Mode | Where the result goes |
|------|-----------------------|
| **integrate** | Class added to `TraceLens/PerfModel/extensions/*_perf_model_extensions.py`; name registered in `pseudo_ops_perf_utils.py` |
| **extension-only** | Class + mappings added to the `<csv_stem>_triage_extension.py` file; pass via `--extension_file` |

Use `emit_perf_model.py` to scaffold the appropriate output.

---

## Authoring steps (per op, full model)

### Step 1 — Locate the implementation

From the call stack (EP1/EP3) or user-provided info (EP2):
- Read the paper or blog for architectural choices that change the math
  (sparsity, compression, routing, cache layout). Some of these are not
  obvious from the trace or the kernel.
- Find the Python binding in the vendor repo.  Record the **relative path**
  (used in the class docstring `Reference implementation:` line).
- Find the underlying kernel (HIP/Triton/CUDA).  Look for:
  - aiter: `aiter/ops/<op>.py` → `@compile_ops(fc_name=...)` or HIP `__global__`
  - vLLM: `vllm/model_executor/layers/...` → `torch.ops.vllm.<name>`
  - SGLang: `sglang/srt/...` → `kernel_shape_profiler` wrapper
  - ATOM: `atom/model_ops/...` → usually an aiter op, sometimes a FlyDSL or
    Triton kernel called directly

### Step 2 — Check for vendor test_*/benchmark_* roofline

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

### Step 3 — Match Input Dims to the signature

Inspect several events, not one shape. Map `event["args"]["Input Dims"][i]`
and `event["args"]["Input type"][i]` to the function's arguments. Also check
`Input Strides`, `Concrete Inputs` (scalars such as `group_size`, boolean
flags), `annotation`, and which parent or child actually owns the GPU time.

Do not assume `Input Dims[0]` is the main input, that it sets the compute
precision, or that the output dtype matches the first input. Cover every dtype
the binding can emit, not only the dtype in this trace.

- MoE: token counts may be before or after routing and padding. Weights may be
  read for every expert or only the active ones.
- Quantized ops, keep separate: storage dtype, unpacked dtype, scale dtype and
  shape, accumulator dtype, output dtype, and the matrix-instruction dtype
  used for the roofline peak.

### Step 4 — Choose the right base class

| Op kind | Base class | File |
|---------|-----------|------|
| Dense GEMM (any dtype) | `GEMM` | `perf_model.py:22` |
| RMSNorm / LayerNorm | `RMSNorm` (subclass of `Normalization`) | `perf_model.py:4856` |
| Elementwise unary | `UnaryElementwise` | `perf_model.py:3225` |
| Elementwise binary | `BinaryElementwise` | `perf_model.py:3294` |
| Reduction | `Reduce` | `perf_model.py:3433` |
| Per-group quantization | `GroupQuant` | `perf_model_extensions.py:378` |
| Inference attention | `InferenceAttention` | `attention_perf_model_extensions.py:15` |
| MoE | `moe_aiter_*` family | `moe_perf_model_extensions.py` |

**Reuse-first**: before writing a new class, check if an existing one already
matches the op's `Input Dims` / `Input type` layout.  If yes, just add a
registry mapping row. Same math with different shape extraction or dtypes:
subclass and override `get_param_details`. Add a new base only when several
models share a concept that has no owner. `get_pseudo_op_mappings()` also
holds direct op names that are not pseudo ops.

### Step 5 — Implement the class

Use the mandatory docstring template (see below).  Key rules:

**FLOPs — derive from the real kernel:**
- Read the kernel source; count **major MFMA / tensor-core operations** for FLOPs.
- Do not estimate from the output buffer alone.
- For GEMM: `flops = 2 * M * N * K`. Fused epilogues, recomputation, sparsity,
  padding, and routing can add or remove work.
- State what is included (top-k, activation, normalization, QK recomputation).
- For RMSNorm: `flops ≈ 5 * T * N` (variance + norm + scale).
- For attention prefill: `flops = 4 * T^2 * H * d` (QK + softmax + V).
- For recurrent attention (GDN): per-token flops from the state update rule.

**Bytes — account for all major loads and stores:**
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

**Attention annotations:**

`InferenceAttention` subclasses do **not** read FLOPs/bytes from `Input Dims`
alone — for chunked-prefill / paged-decode the per-request KV context lengths
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
to the event before the perf model runs (see propagation below). Your job is to
map `Input Dims` to Q/K/V shapes and `Input type` to dtypes in
`get_param_details`, then let the base `flops()`/`bytes()` consume the parsed
`c_*`/`g_*` stats.

**Propagating the annotation to a new op (required for new attention ops):**
`apply_annotation` (called by the inference report) only copies annotations onto
a **hardcoded `name_filters` list** in
`generate_perf_report_pytorch_inference.py`. If your op is not in that list
(e.g. `aiter::pa_decode_gluon`), its events will have **no** `annotation` and the
model silently degrades to no-perf. Re-run annotation for your op from the
extension via a module-level `tree_postprocess_extension`:
```python
def tree_postprocess_extension(trace_tree):
    trace_tree.apply_annotation(name_filters=["aiter::pa_decode_gluon"])
```
This runs after the built-in pass and matches by `name.startswith(...)`. It
finds the enclosing `user_annotation` event (by timestamp containment) and sets
`event["annotation"]` to its name, whatever the format. If a trace has no such
annotation event, or its format has no sq/sk sums, wrap your
`get_param_details` body in `try/except` returning
`InferenceAttention.no_perf_param_details()` so `flops()`/`bytes()` return
`None` gracefully.

**FP8 KV cache:** paged decode kernels often store K/V as FP8 (1 byte) while Q
and the output stay BF16. KV reads dominate decode bytes, so override `bytes()`
to use the cache dtype (`name2bpe(Input type[k_cache])`) for the K/V read terms
rather than the Q dtype.

- Document the `Input Dims` layout your class expects and that it needs a
  detailed (sq/sk) annotation.

### Step 6 — Register the op

**Integrate mode** — add to `pseudo_ops_perf_utils.py`:
```python
# in get_pseudo_op_mappings():
"aiter::my_op": perf_model_extensions.MyOp,

# in get_pseudo_op_category_only_mappings() (categorize-only):
"aiter::my_op": "GEMM",
```

**Extension-only mode** — add to `<csv_stem>_triage_extension.py`:
```python
perf_model_extension = {
    "aiter::my_op": MyOp,   # category comes from the class
}
op_category_extension = {
    "aiter::my_other_op": "GEMM",   # categorize-only
}
```

Use `emit_perf_model.py` to generate the stub; edit from there.

**Add a pseudo op only when the profiler event is the wrong unit.** A stable
CPU op that already owns the runtime and the arguments needs a mapping, not a
pseudo op. Add one when:

- one logical op appears as several stages under a broad parent;
- the useful shapes are on the parent and the runtime is on the children;
- Python ranges and GPU kernels must be one modelable unit;
- modes need separate names, as with `pseudo_v4_paged_decode_swa`,
  `pseudo_v4_paged_decode_csa`, and `pseudo_v4_paged_decode_hca`.

If synthetic ops are among the top kernels, check the tree visually before
modeling the synthetic name. `(Synthetic Op)` rows are usually category-only:
the names embed templates and shapes, and the time is only part of the logical
op. `aiter::indexer_score_topk` is category-only for the same reason: its KV
context length is not in `Input Dims`, so a guessed roofline would mislead.

A pseudo op needs both an injector in `TraceLens/Trace2Tree/extensions`
(stable detection and a shape donor) and the same name in
`get_pseudo_op_mappings()`. Detect from an op name, call-stack range, or kernel
family, not from one timestamp, UID, or an unstable full kernel name. The
report must run with pseudo ops enabled. Check that each kernel is attached
once, parents are correct, neighbors are not captured, and injection time stays
low as the tree grows.

### Step 7 — Validate with py_compile

```bash
python3 -m py_compile /path/to/extension_or_module.py
```

Regenerate the report:
```bash
/path/to/env/python TraceLens/Reporting/generate_perf_report_pytorch_inference.py \
    --profile-json-path /path/to/profile.json \
    --output-csvs-dir /path/to/perf_report_csvs \
    --enable-pseudo-ops \
    [--extension_file /path/to/extension.py]   # extension-only mode only
```

Re-check `has_perf_model`, `op category`, GFLOPS, Data Moved, and
`param_details`. Using the model config and the parallelism strategy, confirm
values such as topK, `num_experts`, and `hidden_size`. One trace is one
example; do not hardcode a dtype, layout, or variant that the op can vary.

---

## Validate with hardware counters (ask the user)

After authoring, **always ask**:

> "Do you want to validate these perf models against hardware counters using
> the `validate-perf-model` skill (rocprofv3 on a real AMD GPU)?"

If yes, hand off to
`TraceLens/Agent/PerfModel/validate-perf-model/validate_perf_model.md`.
Provide:
- The `OP_REGISTRY` key (op name used in the validator)
- The profiler trace `name(s)` (for `trace_names=` in its `register(...)` call
  in `perf_model_harnesses.py`)
- The perf model class name and module path
- A representative shape and dtype for defaults

---

## Perf model class docstring template (required)

Every class registered in `perf_model_extension` or `get_pseudo_op_mappings()`
MUST have a docstring following this layout:

```python
class MyOp(BaseClass):
    """
    Performance model for <exact_profiler_name>.

    Reference implementation:
        <relative/path/inside/vendor/repo.py>

    <One line: what numerical op this is.>

    Signature: <fn_name>(<args>) -> <return>
        <arg0>  — shape [...], dtype <...>
        <arg1>  — shape [...], dtype <...>

    Expected Input Dims from trace:
        [<index>] = <shape description>
        e.g. [(M, K), (N, K), (M, 1), (N, 1)]

    Expected Input type from trace:
        [<dtype0>, <dtype1>, ...]

    Concrete Inputs[<k>] = <semantics>   # omit if unused

    Roofline -- FLOPs:
        <closed-form expression>

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

Minimal variant when inheriting roofline unchanged:
```python
class MyOp(ExistingClass):
    """
    Performance model for <exact_profiler_name>.

    Reference implementation: <path>

    flops/bytes inherited from <ExistingClass>.
    """
```

**Rules:**
- First line: `Performance model for <exact_profiler_name>.`
- No perf model classes for `(Synthetic Op)` names. Categorize them only
  (`op_category_extension` or `get_pseudo_op_category_only_mappings()`).
- Always state `output_bpe` explicitly when it differs from `input_bpe`.

---

## Output modes — detail

### Mode A: integrate into TraceLens core

Edit these files in `TraceLens/PerfModel/extensions/`:

| What to add | File |
|------------|------|
| New class | `perf_model_extensions.py` / `attention_perf_model_extensions.py` / `rmsnorm_perf_model_extensions.py` / `moe_perf_model_extensions.py` |
| Full model mapping | `pseudo_ops_perf_utils.py` → `get_pseudo_op_mappings()` |
| Category-only mapping | `pseudo_ops_perf_utils.py` → `get_pseudo_op_category_only_mappings()` |

Use `emit_perf_model.py --output-mode integrate` to generate the class stub
pre-placed in the right file.

### Mode B: extension-only file

The `--emit-extension` flag of `run_other_bucket_triage.py` produces
`<csv_stem>_triage_extension.py` with:
- `TRIAGE_OP_NAMES`, the ranked candidate names
- an empty `perf_model_extension` dict (name → class)
- an empty `op_category_extension` dict (name → category) for categorize-only ops

`dict_cat2names_extension` is deprecated; the report generator ignores it.

Optional module-level hooks the report generator will call if present:
- `tree_postprocess_extension(trace_tree)` — runs against the built tree before
  perf models execute (after the built-in `apply_annotation`). Use it to attach
  attention annotations to ops outside the built-in filter list, e.g.
  `trace_tree.apply_annotation(name_filters=["aiter::pa_decode_gluon"])`.

Pass it to the report generator via `--extension_file`.

**Do not** edit `torch_op_mapping.py` or `agentic_perf_model_extensions.py`
for triage work — those are product code / legacy stubs.

---

## Model-specific configuration

Hardcode or read a model-specific value only when it changes FLOPs or bytes,
it is not in the event, annotation, or kernel name, it is constant for the
whole report, and the chosen value is copied onto the event. Prefer a value
parsed from the trace. Do not read an environment variable inside `flops()` or
`bytes()`; the same event would then depend on hidden process state. Read it
while building the pseudo op and store it in the event arguments.

A silent fallback between variants is risky. Prefer a warning or a no-perf
result when a wrong assumption would change the report. Show the value that
was used in `param_details`.

### DeepSeek-V4 Pro and Flash

`create_pseudo_ops_v4_paged_decode` reads `TL_MODEL` (variant) and `TL_TP`
(tensor-parallel size), and tries to take mode, local query-head count, and
head dimension from kernel names. Those are stored as `v4_*` args. If
`TL_MODEL` is unset or unrecognized, the perf model falls back to
DeepSeek-V4-Flash, so a Pro run can look valid with the wrong index top-k and
head count. Set the variant and check the pseudo-op args.

DeepSeek-V4-Flash is not FlashAttention (`flash_attention` / `flash_impl`).
FlashAttention does not use `TL_MODEL` or `TL_TP`.

---

## Pitfalls

| Pitfall | Correct approach |
|---------|----------------|
| Inferring FLOPs from output buffer only | Read the kernel source; count MFMA ops |
| Assuming `output_bpe == input_bpe` | Check the output dtype in the binding |
| Using `bpe = 1` for fp4/mxfp4 weights | Use `bpe = 0.5` |
| Wrong `get_compute_precision()` | Check which MFMA path the kernel actually uses |
| New attention op has no FLOPs/bytes | Its events lack `annotation`; add `tree_postprocess_extension` calling `apply_annotation(name_filters=[...])` |
| Using BF16 bpe for an FP8 KV cache | Override `bytes()` to read the K/V cache dtype from `Input type`; KV reads dominate decode |
| Assuming KV context length is in `Input Dims` | It's runtime-only; comes from the `user_annotation` event via `_parse_chunk_stats` |
| Assuming the annotation is the vLLM `execute_*` format | Any format `IterationAnnotation` parses with sq/sk sums works (vLLM, SGLang, ATOM); add new ones with `register_format` |
| Profiling with native (count-only) annotations | Enable detailed annotations; without sq/sk the attention op is no-perf |
| Adding perf model for `(Synthetic Op)` name | Categorize only; names are unstable |
| Editing TraceLens core in extension-only mode | Only edit extension file; pass via `--extension_file` |
| Trusting `--check-mapping` "False" as gap | It checks core map before `apply_extension`; extensions show False there |
| Using system `python3` for TraceLens import | Ask for conda/venv Python first |
| Skipping vendor test_*/benchmark_* check | Always look for roofline reference before writing formulas |
| Wrapper, kernel, and vendor formula disagree | Investigate; do not pick the convenient number |
| Modeling one dtype from one trace | Cover every dtype the binding can emit |
| Missing compute precision or compute unit | Roofline percentage is blank and impact score uses a heuristic |
| Reading `TL_MODEL` / `TL_TP` inside `flops()` | Stamp them onto the event while injecting the pseudo op |
| Unset `TL_MODEL` on a DeepSeek-V4-Pro trace | Fallback is DeepSeek-V4-Flash (wrong top-k and head count) |
| Pseudo op for an op that already has a stable cpu_op | Register that name directly |
| Pseudo-op rule tied to one UID or timestamp | Detect from op name, call stack, or kernel family |

---

## Checklist

- [ ] Entry point chosen (EP1 / EP2 / EP3)
- [ ] Candidate list confirmed with user
- [ ] Per-op depth (full / categorize-only) confirmed
- [ ] Output mode (integrate / extension-only) confirmed
- [ ] Vendor `test_*` / `benchmark_*` roofline checked and documented in class docstring
- [ ] FLOPs derived from actual kernel MFMA ops (not output buffer)
- [ ] Bytes account for all read/write tensors with correct bpe per role
- [ ] `get_compute_precision()` matches dominant MFMA dtype, and the math type (matrix vs vector) is set
- [ ] Supported dtypes are covered, not only the dtype in this trace
- [ ] Model-specific values that are not in the trace are stamped on the event; no env read inside `flops()` / `bytes()`
- [ ] Pseudo op added only when no single profiler event has both runtime and arguments; detection is stable and injection cost was checked
- [ ] Attention: trace has a detailed annotation that `IterationAnnotation` parses with sq/sk sums
- [ ] Attention: annotation propagated to the op (built-in `apply_annotation` list or a `tree_postprocess_extension`), with graceful no-perf fallback when absent
- [ ] Attention: FP8 KV cache modeled with the cache dtype's bpe in `bytes()`
- [ ] Class docstring follows template (all headings present)
- [ ] `py_compile` clean; report regenerated; `has_perf_model` / `op category` correct
- [ ] User asked whether to run `validate-perf-model` HW-counter validation
