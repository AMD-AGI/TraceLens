---
name: unified-perf-report-postprocess
description: >-
  Authors TraceLens perf models for cpu_ops missing coverage.  Driven from
  three entry points: (1) CSV >4% triage from a unified_perf_summary.csv,
  (2) manual op + optional kernel/callstack, (3) static scan of a vendor
  framework repo (aiter/vLLM/SGLang) with optional filter.  For each op the
  user picks full perf model or categorization-only, and either integrates the
  result into TraceLens/PerfModel/extensions/ or emits a standalone
  extension file.  After authoring, asks the user whether to validate with
  the validate-perf-model skill (rocprofv3 hardware counters).
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
or SGLang), optionally filtered (e.g. "attention kernels").

```bash
python3 <skill-dir>/scan_framework_ops.py \
    --repo /path/to/aiter \
    --filter attention \
    [--trace /path/to/unified_perf_summary.csv]
```

The script performs a **static source scan** (no GPU/run required):
- aiter: finds `@compile_ops(..., fc_name=...)` → `aiter::<fc_name>`
- vLLM: finds `direct_register_custom_op` / `torch.library` → `vllm::<name>`
- SGLang: finds `kernel_shape_profiler` wrappers → `sglang_profiler::<...>` plus `sgl_kernel::*`

For each matched op it outputs: registered name, source file, reconstructed
call chain to kernel, and any sibling `test_<op>` / `benchmark_<op>` files
(roofline reference).  When `--trace` is given it annotates which ops actually
appeared in the trace.

For deep or ambiguous chains, instruct an explore subagent to traverse the
repo (same technique used to map FlyDSL ops).

**Confirm the candidate table with the user before proceeding.**

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
- Find the Python binding in the vendor repo.  Record the **relative path**
  (used in the class docstring `Reference implementation:` line).
- Find the underlying kernel (HIP/Triton/CUDA).  Look for:
  - aiter: `aiter/ops/<op>.py` → `@compile_ops(fc_name=...)` or HIP `__global__`
  - vLLM: `vllm/model_executor/layers/...` → `torch.ops.vllm.<name>`
  - SGLang: `sglang/srt/...` → `kernel_shape_profiler` wrapper

### Step 2 — Check for vendor test_*/benchmark_* roofline

vLLM, aiter, and SGLang frequently ship `test_<op>.py` / `bench_<op>.py` /
`op_tests/` beside the kernel.  These often print theoretical FLOPs and bytes.

- Check `aiter/op_tests/`, `aiter/aiter/ops/*/test_*.py`
- Check `vllm/benchmarks/`, `vllm/tests/kernels/`
- Check `sglang/benchmark/`, `sglang/test/`

**Use these expressions as the derivation reference** when writing `flops()` /
`bytes()`, and as an **independent cross-check** of your model (separate from
HW-counter validation).  If vendor numbers disagree with yours, re-read the
kernel.

### Step 3 — Match Input Dims to the signature

Map `event["args"]["Input Dims"][i]` and `event["args"]["Input type"][i]` to
the function's arguments.  Also check `Input Strides` and `Concrete Inputs`
(for scalars like `group_size`, boolean flags).

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
registry mapping row.

### Step 5 — Implement the class

Use the mandatory docstring template (see below).  Key rules:

**FLOPs — derive from the real kernel:**
- Read the kernel source; count **major MFMA / tensor-core operations** for FLOPs.
- Do not estimate from the output buffer alone.
- For GEMM: `flops = 2 * M * N * K`.
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

**compute_precision pitfalls:**
- `get_compute_precision()` must return the **dominant MFMA dtype** the kernel
  actually uses (fp8, int8, bf16, fp16 …).
- Packed / compressed weights: the MFMA dtype is determined by the **unpacked**
  logical dtype the MMA uses internally (often fp8 or fp16 even if weights
  arrive as fp4).
- Weight-dtype-dependent precision: if the kernel selects the MFMA path based
  on a runtime flag or dtype check, model that branch.

**Attention annotations:**

`InferenceAttention` subclasses do **not** read FLOPs/bytes from `Input Dims`
alone — for chunked-prefill / paged-decode the per-request KV context lengths
are runtime-only. They come from the trace's `user_annotation` events, which
`InferenceAttention._parse_chunk_stats(event["annotation"])` turns into:
- context aggregates: `c_sq` (Σ query tokens), `c_sk` (Σ kv tokens),
  `c_sqsq` (Σ sq·sq), `c_sqsk` (Σ sq·sk)
- generation aggregates: `g_sq`, `g_sk`, `g_sqsq`, `g_sqsk`

The annotation event name is produced by `split_inference_trace_annotation.py`
and looks like:
```
execute_<i>_context_<Nctx>(sq<c_sq>sk<c_sk>sqsq<c_sqsq>sqsk<c_sqsk>)_generation_<Ngen>(sq<g_sq>sk<g_sk>sqsq<g_sqsq>sqsk<g_sqsk>)
```
- **Prefill-only step**: `generation` group is `0(sq0sk0sqsq0sqsk0)`.
- **Decode-only step**: `context` group is `0(...)`; each request has `sq=1`,
  `sk=ctx_len`, so `g_sq = #requests`, `g_sk = Σ ctx_len`, `g_sqsk = g_sk`,
  `g_sqsq = #requests`.

You normally do **not** synthesize this string yourself — it must be attached to
the event before the perf model runs (see propagation below). Your job is to map
`Input Dims` to Q/K/V shapes and `Input type` to dtypes in `get_param_details`,
then let the base `flops()`/`bytes()` consume the parsed `c_*`/`g_*` stats.

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
`event["annotation"]`. If a trace has no such annotation event, wrap your
`get_param_details` body in `try/except` returning
`InferenceAttention.no_perf_param_details()` so `flops()`/`bytes()` return
`None` gracefully.

**FP8 KV cache:** paged decode kernels often store K/V as FP8 (1 byte) while Q
and the output stay BF16. KV reads dominate decode bytes, so override `bytes()`
to use the cache dtype (`name2bpe(Input type[k_cache])`) for the K/V read terms
rather than the Q dtype. See `pa_decode_gluon` in the lucid_2506 extension for a
worked example.

- Document which annotation format and `Input Dims` layout your class expects.

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
    "aiter::my_op": MyOp,
}
dict_cat2names_extension = {
    "GEMM": ["aiter::my_op"],
}
```

Use `emit_perf_model.py` to generate the stub; edit from there.

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

Re-check `has_perf_model`, `op category`, GFLOPS, Data Moved.

---

## Validate with hardware counters (ask the user)

After authoring, **always ask**:

> "Do you want to validate these perf models against hardware counters using
> the `validate-perf-model` skill (rocprofv3 on a real AMD GPU)?"

If yes, hand off to
`TraceLens/Agent/PerfModel/validate-perf-model/validate_perf_model.md`.
Provide:
- The `OP_REGISTRY` key (op name used in the validator)
- The profiler trace `name(s)` (for `CSV_NAME_TO_REGISTRY`)
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
- No perf model classes for `(Synthetic Op)` names — use `categorize_extension` only.
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
- `perf_model_extension` dict (name → class)
- `op_category_extension` dict (name → category) for categorize-only ops
- `categorize_extension(row, plugin)` function

Optional module-level hooks the report generator will call if present:
- `tree_postprocess_extension(trace_tree)` — runs against the built tree before
  perf models execute (after the built-in `apply_annotation`). Use it to attach
  attention annotations to ops outside the built-in filter list, e.g.
  `trace_tree.apply_annotation(name_filters=["aiter::pa_decode_gluon"])`.

Pass it to the report generator via `--extension_file`.

**Do not** edit `torch_op_mapping.py` or `agentic_perf_model_extensions.py`
for triage work — those are product code / legacy stubs.

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
| Adding perf model for `(Synthetic Op)` name | Use `categorize_extension` only; names are unstable |
| Editing TraceLens core in extension-only mode | Only edit extension file; pass via `--extension_file` |
| Trusting `--check-mapping` "False" as gap | It checks core map before `apply_extension`; extensions show False there |
| Using system `python3` for TraceLens import | Ask for conda/venv Python first |
| Skipping vendor test_*/benchmark_* check | Always look for roofline reference before writing formulas |

---

## Checklist

- [ ] Entry point chosen (EP1 / EP2 / EP3)
- [ ] Candidate list confirmed with user
- [ ] Per-op depth (full / categorize-only) confirmed
- [ ] Output mode (integrate / extension-only) confirmed
- [ ] Vendor `test_*` / `benchmark_*` roofline checked and documented in class docstring
- [ ] FLOPs derived from actual kernel MFMA ops (not output buffer)
- [ ] Bytes account for all read/write tensors with correct bpe per role
- [ ] `get_compute_precision()` matches dominant MFMA dtype
- [ ] Attention: annotation propagated to the op (built-in `apply_annotation` list or a `tree_postprocess_extension`), with graceful no-perf fallback when absent
- [ ] Attention: FP8 KV cache modeled with the cache dtype's bpe in `bytes()`
- [ ] Class docstring follows template (all headings present)
- [ ] `py_compile` clean; report regenerated; `has_perf_model` / `op category` correct
- [ ] User asked whether to run `validate-perf-model` HW-counter validation
