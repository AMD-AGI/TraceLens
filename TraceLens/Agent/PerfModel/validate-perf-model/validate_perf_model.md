---
name: validate-perf-model
description: >-
  Validates TraceLens perf model predictions (FLOPs and bytes) against actual
  hardware counter measurements from rocprofv3 --pmc. Three-part workflow:
  Step 1 generates parametric CSV test cases, Step 2 (optional) adds new
  harnesses, Step 3 runs harnesses under rocprofv3 and produces comparison
  CSVs. Covers GEMM, MoE, Attention, RMSNorm, Quant/Activation, FusedRoPE,
  and CustomCollective op families. Supports auto GPU arch detection, cold
  cache flushing, Docker/SSH execution, and bulk missing-harness discovery.
---

# Perf model validation with rocprofv3 hardware counters

## Three-part workflow

```
Step 1: generate_test_cases.py     →   test_cases/<op>.csv
Step 2: tests/<category>.py        →   add harnesses as needed
Step 3: run_validation.py          →   output/<timestamp>_summary.csv
```

Each step is independent.  You can skip Step 1 if test cases already exist,
or skip Step 2 if all needed harnesses are already implemented.

---

## Prerequisites

- ROCm installation with `rocprofv3` on PATH (or `--rocprofv3-path`)
- AITER (`aiter`) installed in the active Python environment
- vLLM installed for `vllm_*` ops (aiter-only ops work without it)
- `sgl_kernel` installed for SGLang ops
- AMD GPU: MI300X (gfx942) or MI355X (gfx950)
- TraceLens repo accessible for perf model class imports

---

## Step 1 — Generate test cases

`generate_test_cases.py` reads `OP_METADATA` from all `tests/*.py` harness
modules and produces `test_cases/<op>.csv` files.

```bash
# Auto sweep (default): M / seq_len sweep for every harness
python generate_test_cases.py --auto --output-dir test_cases/

# Sized to a model config JSON
python generate_test_cases.py --from-config model_configs/llama3_8b.json

# Extract from a TraceLens unified_perf_summary.csv
python generate_test_cases.py --from-report /path/to/unified_perf_summary.csv

# Generate for all harnesses + discover ops missing from traces
python generate_test_cases.py --all-ops \
    --discovery-report /path/to/unified_perf_summary.csv

# Restrict to one op or category
python generate_test_cases.py --op gemm_a8w8_blockscale
python generate_test_cases.py --category GEMM

# List all ops with registered harnesses
python generate_test_cases.py --list-ops
```

The `missing_harnesses.txt` output from `--from-report` / `--all-ops` lists
op names that appear in traces but have no matching harness.

---

## Step 2 — Add new harnesses (when needed)

Copy `tests/_harness_template.py` to `tests/<category>.py`, implement the
`test_<op>` function using the appropriate invocation pattern:

| Pattern | When to use |
|---------|-------------|
| **aiter** (`import aiter; aiter.<fn>(...)`) | aiter package ops |
| **vLLM** (`import vllm._aiter_ops; torch.ops.vllm.<fn>(...)`) | vLLM-registered ops |
| **SGLang** (`import sgl_kernel; sgl_kernel.<fn>(...)`) | SGLang-registered ops |

Then:

1. Add an `OP_METADATA` entry in the harness file (see template for format).
2. Register the harness in `tests/_runner.py`'s `_build_op_table()`.
3. Add a `run_perf_model_<op>(args)` function to `perf_model_harnesses.py` that
   constructs a synthetic event dict and calls the extension class's
   `flops()` / `bytes()` / `get_compute_precision()`.
4. Add an entry to `OP_REGISTRY` in `validate_perf_model.py` pointing to the
   new `run_perf_model_<op>` function, with `defaults`, `required_args`,
   `category`, and `description`.
5. Re-run `generate_test_cases.py --auto --op <new_op>`.

### Harness file → extension category mapping

| `tests/*.py` | Extension base class | `.category` |
|---|---|---|
| `tests/gemm.py` | `GEMM` | `GEMM` |
| `tests/moe.py` | `MoE` (fused/unfused) | `MoE` |
| `tests/attention.py` | `InferenceAttention` | `InferenceAttention` |
| `tests/rmsnorm.py` | `RMSNorm` | `RMSNorm` |
| `tests/other.py` | `GroupQuant`, `UnaryElementwise` | same |
| `tests/rope.py` | `FusedRoPE` | `FusedRoPE` |
| `tests/collectives.py` | `CustomCollective` | `CustomCollective` |
| `tests/extensions_new.py` | `GEMM`, `GroupQuant`, `RMSNorm` | various |

---

## Step 3 — Run validation

`run_validation.py` reads test case CSVs, invokes `validate_perf_model.py`
for each row under `rocprofv3`, flushes the GPU cache between runs, and
writes output CSVs.

```bash
# Run all ops in test_cases/
python run_validation.py --output-dir output/

# Run a single op
python run_validation.py --op gemm_a8w8_blockscale --output-dir output/

# Run inside Docker
python run_validation.py --docker <container_name> --output-dir output/

# Run on a remote host via SSH
python run_validation.py --ssh user@hostname --output-dir output/

# Override GPU arch (default: auto-detected from rocm-smi)
python run_validation.py --arch gfx950 --output-dir output/

# Dry run: print commands without executing
python run_validation.py --dry-run --output-dir output/

# Restrict to a category
python run_validation.py --category InferenceAttention --output-dir output/
```

Output files:
- `output/<timestamp>_<op>/` — per-op rocprofv3 CSVs and logs
- `output/<timestamp>_summary.csv` — merged results across all ops

---

## Direct validate_perf_model.py usage

For single ops or ad-hoc exploration, call `validate_perf_model.py` directly:

```bash
# Single op — arch is auto-detected
python validate_perf_model.py \
    --op gemm_a8w8_blockscale \
    --M 2048 --N 4096 --K 8192 \
    --output-dir /tmp/perf_val

# Explicit arch
python validate_perf_model.py \
    --op gemm_a8w8_blockscale --arch gfx942 \
    --M 2048 --N 4096 --K 8192 \
    --output-dir /tmp/perf_val

# Run all ops in a category
python validate_perf_model.py --category gemm --output-dir /tmp/perf_val

# Run all registered ops
python validate_perf_model.py --all --output-dir /tmp/perf_val

# Validate from a TraceLens report directory
python validate_perf_model.py \
    --from-report-dir /path/to/report_folder \
    --output-dir /tmp/perf_val

# Discover ops missing validation coverage
python validate_perf_model.py --discover
```

---

## GPU architecture and cache flushing

- `--arch` is **optional** in all scripts. When omitted, `detect_gpu_arch()`
  queries `rocm-smi --showproductname` / `rocminfo` to determine the arch
  automatically. Supported: `gfx942` (MI300X) and `gfx950` (MI355X).

- Between every pair of rocprofv3 sub-runs, `flush_gpu_cache()` writes 512 MiB
  of zeros to device memory, then synchronizes and frees the buffer.

  `FETCH_SIZE` and `WRITE_SIZE` are L2 (TCC) counters — they count bytes that
  **missed L2** and were fetched from below. On MI300X/MI355X the level below
  L2 is the **Infinity Cache** (256 MB), not HBM directly. To make these
  counters reflect true HBM traffic, both the L2 (32 MB) and the Infinity
  Cache (256 MB) must be cold. 512 MiB = 2× the Infinity Cache size, which
  guarantees both levels are fully evicted. At ~5 TB/s HBM bandwidth the fill
  takes ~0.1 ms — negligible overhead in the validation loop.

---

## Attention test cases and annotation string

Attention harnesses accept an `annotation` string that encodes the per-sequence
statistics needed by the `InferenceAttention` perf model.  The format is:

```
attn_<prefill_count>_<decode_count>_<total_q_tokens>_<total_kv_tokens>_...
```

For variable-length scenarios, `--varlen-num-seqs` and `--varlen-scenario`
control the number and distribution of sequences:

- `random` — all sequences are self-attention; tokens split randomly.
- `mixed_prefill_decode` — one prefill sequence (Q=K=seq_len) plus
  `varlen_num_seqs - 1` decode sequences (Q=1, K=seq_len each).

The annotation is auto-generated by the harness from these parameters if not
provided explicitly.

---

## Agent workflow — ask the user before running

Before running validation on a GPU machine, ask the user for:

1. **Target machine** — SSH hostname (or "local" / Docker container name).
2. **Output directory** — where to write rocprofv3 CSVs and reports.
3. **Ops / category to validate** — defaults to all registered harnesses.
4. **Test case source** — auto-sweep, model config JSON, or unified_perf_summary.csv.

Then run the three steps in order:

```bash
# Step 1: generate test cases
python generate_test_cases.py --auto --output-dir test_cases/

# Step 3: run validation (no GPU needed for dry-run)
python run_validation.py --output-dir output/ --dry-run
python run_validation.py --output-dir output/
```

---

## Adding OP_METADATA to a new harness

Every `tests/<category>.py` file should expose a top-level `OP_METADATA` dict:

```python
OP_METADATA: dict = {
    "my_op_name": {
        "fn":           test_my_op,          # callable in this module
        "category":     "GEMM",              # extension base class .category
        "description":  "One-line description",
        "dtypes":       ["bf16", "fp16"],    # all supported in_dtype values
        "defaults":     {"M": 2048, "N": 4096, "K": 7168, "in_dtype": "bf16"},
        "required_args": ["M", "N", "K"],
    },
}
```

`generate_test_cases.py` reads this to auto-generate `test_cases/my_op_name.csv`
without any manual configuration.
