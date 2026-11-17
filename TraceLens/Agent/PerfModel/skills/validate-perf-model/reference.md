<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Perf-model validation: reference

Detail for [SKILL.md](SKILL.md). Read each section when its step needs it. The
scripts live in `TraceLens/Agent/PerfModel/validate_perf_model/`. The commands below
run from there; from anywhere else, replace `python validate_perf_model.py` with
`python -m TraceLens.Agent.PerfModel.validate_perf_model.validate_perf_model`.

## Prerequisites

- ROCm with `rocprofv3` on PATH (or pass `--rocprofv3-path`)
- `aiter` installed in the Python environment or container the user named
- vLLM for `vllm_*` ops and SGLang (`sgl_kernel`) for SGLang ops; ops from a
  framework that is not installed are reported as failures, the rest still run
- AMD GPU: MI300X or MI325X (`--arch gfx942`, the default), or MI350X or MI355X
  (`--arch gfx950`)
- TraceLens importable (installed, or this repo checkout)
- The perf model integrated into TraceLens core. Models that exist only in an
  extension file (`--extension_file`) cannot be validated

---

## Layout

```
validate_perf_model/
├── validate_perf_model.py   CLI: profiles ops under rocprofv3, compares, reports
├── perf_model_harnesses.py  Op registry: @register(...) event builders + perf models
└── replay/
    ├── _runner.py           Subprocess rocprofv3 wraps; dispatches --op <key> to test_<key> and measures its call
    ├── generic.py           Replay of traced shapes from --cases (OP_CALL_SPEC)
    ├── _dtypes.py           Shared dtype helpers
    ├── gemm.py moe.py attention.py rmsnorm.py other.py rope.py
    │                        test_<key>(**kwargs) GPU replay harnesses, one module per op family
    └── collectives.py       Collective harnesses; not wired into the runner or registry
```

Each op is declared once, in `perf_model_harnesses.py`:

```python
@register(
    "gemm_a8w8_blockscale",                       # OP_REGISTRY key == test_<key> name
    category="gemm",                              # --category grouping
    description="...",
    model=(PERF_MODEL_EXT, "gemm_a8w8_blockscale"),  # (source, class or trace name)
    defaults={"M": 2048, "N": 4096, "K": 8192},   # fill CLI args the user omits
    trace_names=("aiter::gemm_a8w8_blockscale_ck", ...),  # --cases mapping
)
def _gemm_a8w8_blockscale_event(args):
    return {"name": ..., "args": {"Input Dims": ..., "Input type": ...}}
```

The builder turns CLI dimensions into a synthetic TraceLens event; the
registry scores it with the perf model class and returns `(flops, bytes,
precision)`. `model` sources are the module constants `PERF_MODEL_EXT`,
`MOE_EXT`, `RMSNORM_EXT`, `ATTN_EXT`, `PERF_MODEL` and `CORE_OP_MAP`.

---

## How a run works

For each op, `validate_perf_model.py`:

1. Fills unset CLI dimensions from the op's `defaults` and builds the runner
   command `python replay/_runner.py --op <key> --M ... --num-warmup 3`.
2. Runs it once under `rocprofv3 --kernel-trace` to pick the target kernel
   (non-infrastructure kernel with `warmups + 1` dispatches, longest total
   time) and its runtime. `--kernel-filter` overrides this.
3. Evaluates the perf model.
4. Re-runs the test under `rocprofv3 --pmc` once per counter group (below),
   keeping the last dispatch of the target kernel (the measured iteration).
5. Writes `validation_report.txt` and adds a row to the summary CSV with its
   `run_status` and `match_status` (see Outputs in SKILL.md).

In `--cases` mode the op list comes from a cases JSON that
`python3 -m TraceLens.Agent.ModelBringup.utils.bringup_gates cases --report-dir
<perf_report_csvs> --out cases.json` builds from `unified_perf_summary.csv`: rows with
`has_perf_model == True`, deduplicated by (name, Input Dims, Input type). Cases whose
`name` is in no op's `trace_names` are skipped. The prediction
comes from the traced event (`run_perf_model_from_event`) and the kernel is
replayed either through the op's own `test_<key>` (`csv_harness=True`) or
through `replay/generic.py` with the traced shapes.

Each case's `perf_params` (the values the perf model parsed in the report)
drive both sides, ahead of the attention CSV parameters:
- **Perf model input:** the event's annotation is rebuilt from all eight sums
  (`c_sq`, `c_sk`, `c_sqsq`, `c_sqsk` for prefill and `g_sq`, `g_sk`, `g_sqsq`,
  `g_sqsk` for decode) with `chunk_stats_annotation`, so the model sees the same
  values as in the report.
- **Harness:** sizes come from `perf_params` (for attention: `N_Q`, heads, head
  dim, and the same annotation, which the unified-attention tests use to build
  sequence lengths).

Request counts are not stored in `perf_params`, so they are derived assuming
uniform lengths within each phase (requests = sq² / sqsq).

Ops registered with `perf_model_only=True` (kernels that need model-level
context such as a forward context or valid sorted expert ids) skip profiling
and report the prediction only.

---

## Counters

| Counter | Pass | gfx942 (MI300X, MI325X) | gfx950 (MI350X, MI355X) | Used as |
|---|---|---|---|---|
| `SQ_INSTS_VALU_MFMA_MOPS_I8` | ops | yes | yes | int8 MFMA ops = count × 512 |
| `SQ_INSTS_VALU_MFMA_MOPS_F8` | ops | yes | yes | fp8 MFMA ops = count × 512 |
| `SQ_INSTS_VALU_MFMA_MOPS_F6F4` | ops | no | yes | fp6/fp4 MFMA ops = count × 512 |
| `TOTAL_16_OPS` | ops_derived_16 | yes | yes | 16-bit ops (fp16/bf16) |
| `TOTAL_32_OPS` | ops_derived_32 | yes | yes | 32-bit ops |
| `TCC_EA0_RDREQ_sum` | mem_read | yes | yes | read requests (formula below) |
| `TCC_EA0_RDREQ_32B_sum` | mem_read | yes | yes | 32-byte read requests |
| `TCC_EA0_RDREQ_64B_sum` | mem_read | yes | yes | 64-byte read requests |
| `TCC_BUBBLE_sum` | mem_read | yes | yes | collected but not used |
| `WRITE_SIZE` | mem_write | yes | yes | write bytes = value × 1024 |

gfx942 has no FP6/FP4 matrix instructions, so its FP4 kernels run on FP8 or
16-bit MFMA and the `F6F4` counter does not exist there.

`--skip-derived` skips the two derived ops passes. Without `TOTAL_16_OPS` and
`TOTAL_32_OPS`, the precision check cannot pass for fp16, bf16 or fp32 models,
and their FLOPs ratio has no counter to use.

Read bytes are rebuilt from the L2 (TCC) to memory-fabric (EA) read requests,
weighting each request by its size:

```
read_bytes = 32 * RDREQ_32B + 64 * RDREQ_64B + 128 * (RDREQ - RDREQ_32B - RDREQ_64B)
```

Both read and write counters measure traffic between L2 and the memory
fabric. They do not see L2 hits and do not separate HBM from on-package cache
(MALL).

Comparisons in the report:

- **FLOPs ratio**: measured ops from whichever counter above is largest,
  divided by predicted FLOPs.
- **Bytes ratio**: (read + write bytes) / predicted bytes.
- **Known ratio sources**: common causes when a ratio is outside its range.
  These are hypotheses to check in the code (see "Investigating a deviation"),
  never a cause on their own.

  | Op family | Ratio | Common cause |
  |---|---|---|
  | MoE (fused or two-stage) | FLOPs above 1 | each active expert's rows are padded to `block_m`; worst at small M |
  | MoE, GEMM | Bytes above 1 | weights or activations re-read once per M or N tile when they do not stay in L2 |
  | GEMM, attention | FLOPs slightly above 1 | M, N or K padded to the macro tile |
  | Block-scaled FP8 or FP4 | Bytes above 1 | scale tensors, often left out of the model |
  | Causal attention | FLOPs above 1 | masked tiles computed in full |
  | Top-k and selection (radix top-k) | Bytes above 1 | several passes over the input, not one |
  | Multi-block or split-K kernels | Bytes above 1 | workspace or partial-result buffers written and read back |
  | Sparse or compressed attention, indexers | FLOPs or bytes off | scores or KV compressed, while the model counts raw tokens |
  | GEMM with small M | Bytes above 1 | shared weights re-read by each XCD's L2 |
- **Precision check**: the dominant MFMA counter (or the largest ops counter
  if no MFMA ran) must be one expected for the model's compute precision:

  | Precision | Expected counter |
  |---|---|
  | fp8 | `SQ_INSTS_VALU_MFMA_MOPS_F8` |
  | fp4 | `SQ_INSTS_VALU_MFMA_MOPS_F6F4` |
  | int8 | `SQ_INSTS_VALU_MFMA_MOPS_I8` |
  | fp16, bf16 | `TOTAL_16_OPS` |
  | fp32 | `TOTAL_32_OPS` |

  The check compares the reported precision with the measured counter only;
  it does not adjust for the architecture. When the expected counter does not
  exist on the target (fp4 on gfx942), the result is still a mismatch, and the
  report and the CSV `comment` give that as the reason.

- **Throughput** (TF/s, TB/s): uses the kernel runtime from the
  `--kernel-trace` pass.

The CSV `comment` column is a rule-based hint from the ratios alone, prefixed
`hint:`. It is not a verified cause.

---

## What to validate

One trace shape is one example. Cover the variants that change the work:
- the traced shapes, from `--cases`;
- decode and prefill, when the op runs in both;
- both sides of any kernel-path switch (a size threshold, a flag or a layout
  that picks a different kernel);
- compressed and uncompressed inputs, when the op accepts both;
- every dtype the binding accepts, not only the one in the trace.

A generic or shared key (an `aten::*` op, or a kernel several models call)
also gets one invocation from outside the target model, to show the model
holds for other callers or that its guard returns no-perf for them.

---

## Investigating a deviation

Do this for each row whose `match_status` is `deviation-suspect`; other rows
need nothing.
Read everything in the same Python environment or container that ran the
validator, so the code matches the kernel that was measured.

1. **Follow the call chain to the kernel that ran.** One file is rarely
   enough:
   - the harness, `test_<key>` in `replay/<module>.py`, shows which framework
     function was called;
   - the Python wrapper (aiter, vLLM, SGLang) may choose a path by shape, dtype
     or architecture;
   - the dispatch leads to a Triton kernel (`.py`), a HIP or CK kernel
     (`.cu`/`.hpp` under `csrc`), a JIT-built module (`@compile_ops`), or a
     precompiled assembly kernel (`.co`);
   - the CSV `kernel_name` shows which branch actually ran; read that branch.

   Find the installed sources with, for example,
   `python -c "import aiter; print(aiter.__file__)"`. Assembly kernels have no
   source; take their tile sizes from the config table that selects them
   (aiter keeps CSV configs next to the `.co` files).
2. **Name the mechanism,** with the file and line where the code shows it.
   - **FLOPs:** tile or block sizes and the padding they cause, masked tiles
     computed anyway, recomputation, split-K partial sums.
   - **Bytes moved:**
     - padding: loads and stores of padded tiles;
     - redundant reads or writes in the code: the same tensor loaded once per
       tile, intermediates written and read back, workspace or partial-result
       buffers, atomics, scale tensors left out of the model;
     - cache capacity: the working set does not fit in L2 (or the on-package
       cache), so data is re-read from memory. Small shapes that fit show
       little mismatch, and it grows once the working set passes the cache
       size.
3. **Quantify it** for this shape (for example padded M over M, or the bytes
   of one extra pass over a tensor) and check that it reproduces the measured
   ratio within the band. For cache capacity, compare the working set with
   the cache size of the target GPU.
4. **Only if steps 1-3 do not settle it,** rerun the op at a shape where the
   mechanism should change or vanish (a tile multiple, a larger M, or a shape
   whose working set fits in cache) and check that the ratio moves as
   predicted. A ratio that grows with size points to cache capacity; one that
   stays constant points to padding or redundant traffic in the code. This is
   one extra validator call for that op.
5. **Record** the result in the validation CSV, editing only that row and
   only after the investigation is done: set `match_status` from
   `deviation-suspect` to `deviation-explained`, and replace the `comment` hint
   with the cause, its file and line, and the quantitative check. Without both
   the file and line and the check, leave the row `deviation-suspect`. Never
   change ratios, bands or other rows, and never mark `over-predicted` or
   `no prediction` rows as explained.

## Running

```bash
# Single op
python validate_perf_model.py --op gemm_a8w8_blockscale \
    --M 2048 --N 4096 --K 8192 --arch gfx942 --output-dir /tmp/perf_val

# One category, or every registered op, with default dimensions
python validate_perf_model.py --category gemm --output-dir /tmp/perf_val
python validate_perf_model.py --all --output-dir /tmp/perf_val

# Every traced shape in a cases file (see above)
python validate_perf_model.py --cases cases.json --arch gfx950 --output-dir /tmp/perf_val
# ...restricted to some registry keys or trace names
python validate_perf_model.py --cases cases.json \
    --only-ops ck_moe_stage1,aiter::rms_norm --output-dir /tmp/perf_val

# aiter:: / vllm:: ops with TraceLens perf models but no registry entry
python validate_perf_model.py --discover
```

`python validate_perf_model.py --help` lists every op key and category.

Output in `--output-dir`:
- `validation_report_<timestamp>.csv`: one row per op with predictions,
  counter totals, ratios, runtime, perf model class and source location
- per-op directories (`<op>/` for batch runs, `<idx>_<trace name>/` for
  `--cases` runs) with `validation_report.txt` and the rocprofv3 CSVs and logs

Before running, have everything listed under Inputs in SKILL.md: the ops (single op,
category, all, or a cases file), the GPU and `--arch`, the output directory,
and where to run.

---

## Adding an op: where things go

| To... | Edit |
|---|---|
| Validate a new op from CLI dimensions | `@register(...)` builder in `perf_model_harnesses.py`, in its op-family section, **and** `test_<key>(**kwargs)` in the matching `replay/<family>.py` |
| Map more trace names to an existing op | `trace_names=` on its `register(...)` |
| Replay report rows through the generic runner | an `OP_CALL_SPEC` entry keyed by the op key in `replay/generic.py` |
| Replay report rows through the op's own `test_<key>` | `csv_harness=True` on `register(...)` and a branch in `_extract_dims_for_existing_harness` in `validate_perf_model.py` that maps the row to CLI args |
| Validate an op only from `--cases` (no CLI builder) | a row in `_CSV_ONLY_OPS` in `perf_model_harnesses.py` and an `OP_CALL_SPEC` entry |
| Register an op whose kernel cannot run standalone | `perf_model_only=True` |
| Add a test module for a new op family | the module name in `replay/_runner.MODULES` |
| Use a new perf-model module | a constant next to `PERF_MODEL_EXT` |
| Add a CLI dimension or dtype flag | `main()` in `validate_perf_model.py` and `INT_FLAGS` / `STR_FLAGS` in `replay/_runner.py` |
| Support a new GPU arch | `ARCH_COUNTER_CONFIGS` in `validate_perf_model.py` |

Writing a harness for a new op:

1. **Find the call.** Start from the report row's `entry_point` and
   `call_stack_full`: they name the framework function that issued the op.
   Read that code in the environment that ran the trace and copy how it builds
   the arguments: layouts, strides, flags and scalar arguments.
2. **Build realistic values, not only shapes.** Random tensors are fine when
   the work depends only on shapes (dense GEMM, elementwise). Many inference
   ops change their work with input values: sequence lengths, page tables and
   block indices, routing and top-k ids, sorted expert ids, scales and masks.
   Build those the way the framework does, from the case's `perf_params` and
   annotation sums where they apply.
3. **Pick the replay path.** Use an `OP_CALL_SPEC` entry only when random
   tensors of the traced shapes reproduce the work. Otherwise write a
   `test_<key>` and set `csv_harness=True`.
4. **Check that the same kernel ran.** Compare the kernel the validator
   discovered for the harness with the row's `kernel_details_summary`. If they
   differ, the harness took another branch and the run measures the wrong
   kernel; fix the inputs before trusting any ratio.

Harness conventions (see `replay/gemm.py`):

- Signature `test_<key>(<dims>, <dtype kwargs>=<default>, **_)`; `**_` absorbs
  flags meant for other ops.
- Build the inputs and return a zero-argument callable that makes the kernel
  call. `measure()` in `replay/_runner.py` runs it `--num-warmup` times,
  synchronizes, flushes L2 so the measured call starts cold, launches a
  `bitwise_not` marker between the warmup and measured dispatches, then makes
  one measured call and synchronizes again. The validator reads counters from
  the last dispatch.
- Import the kernel library inside the function, so a missing framework only
  breaks its own ops.
- Invoke the op the way the framework does: `aiter.<fn>(...)`,
  `torch.ops.vllm.<fn>(...)` after `import vllm._aiter_ops`, or
  `sgl_kernel.<fn>(...)`.

`OP_CALL_SPEC` entries are `{"call": fn, "output_indices": [...],
"skip_indices": [...]}`. The generic test builds one tensor per traced input
(empty for `output_indices`, none for `skip_indices` or scalars) and calls
`fn(tensors)` or `fn(tensors, concrete_inputs)`, where `tensors` maps input
index to tensor.

---

## Categories

`activation`, `atom_flydsl`, `attention`, `dsv3_mla_decode`,
`dsv3_moe_flydsl`, `ext_mxfp4`, `gemm`, `moe`, `norm`, `quant`,
`vllm_attention`, `vllm_gemm`, `vllm_norm`, `vllm_quant`.

Categories come from the `category=` of registered ops; a new value becomes a
new `--category` choice.

---

## Attention annotation

`unified_attention` and `vllm_unified_attention` take `--annotation`, the
detailed step annotation that carries the per-batch statistics the
`InferenceAttention` perf model needs. Any format `IterationAnnotation` parses
works: vLLM `execute_*`, SGLang `step[...]` or ATOM detailed. The vLLM form is:

```
execute_<iter>_context_<num_prefill_reqs>(sq<Σq>sk<Σkv>sqsq<Σq·q>sqsk<Σq·kv>)_generation_<num_decode_reqs>(sq<Σq>sk<Σkv>sqsq<Σq·q>sqsk<Σq·kv>)
```

Sums are over the prefill (context) or decode (generation) requests.
`vllm_unified_attention` and `unified_attention` also launch the batch the
annotation describes: requests in each phase get uniform lengths of
`sq / n` queries and `sk / n` KV tokens.

The varlen ops (`mha_varlen_fwd`, `fmha_v3_varlen_fwd`) build this from
`--varlen_num_seqs` (default 4), `--varlen_seed` (default 42) and
`--varlen_scenario`:

- `random`: self-attention; `seq_len` tokens split randomly across sequences.
- `mixed_prefill_decode`: one prefill sequence (Q=K=`seq_len`) plus
  `varlen_num_seqs - 1` decode sequences (Q=1, K=`seq_len`).
