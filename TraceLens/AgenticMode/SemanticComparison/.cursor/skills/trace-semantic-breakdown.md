---
name: trace-semantic-breakdown
description: Analyze a single vLLM/PyTorch GPU trace and decompose it into semantic blocks with roofline analysis. Each semantic label maps to a perf model (GEMM, SDPA, Normalization, or Elementwise) so that roofline can be computed for every block. Use when the user wants to break down a trace, label kernels, or run roofline analysis on graph-mode traces.
---

# Trace Semantic Breakdown

Break down a single GPU trace (Chrome trace JSON from PyTorch profiler) into a two-level semantic block structure with roofline analysis. Works standalone on one trace -- no second trace needed.

Supports diverse transformer architectures (dense, MoE, hybrid) across
inference platforms (vLLM, TensorRT-LLM, SGLang, etc.).

**Key constraint**: every `semantic_block` label maps to exactly one of four
perf model categories: **GEMM**, **SDPA**, **Normalization**, or **Elementwise**.
There are no "Other" or unmodeled categories. This ensures roofline analysis
can be computed for every kernel in the trace.

## Prerequisites

- Python 3.8+, pandas, openpyxl (for xlsx output)
- Scripts directory: `TraceLens/AgenticMode/SemanticComparison/trace_breakdown/`
- HuggingFace `config.json` for the model being traced

## Two-Level Semantic Categories

### Low-level: `semantic_block`

Per-kernel functional role (e.g., "QKV Projection", "Attention", "Down Projection").
Assigned by the LLM in Step 4 from the reference vocabulary below.
**Every label must map to a perf model** -- do not invent labels that lack
a roofline formula.

### High-level: `semantic_group`

Functional macro-operation a block belongs to. Derived deterministically from
`semantic_block` via `category_mappings.py`:

- **Self-Attention**: Pre-Attn Norm, QKV/Q/KV Projection, Rotary Embedding, Attention, KV Cache Store, Output Projection, Attention Output Gate, Post-Attn Residual Add
- **MoE / FFN**: Post-Attn Norm, Router Gate, MoE Routing, MoE GateUp+SwiGLU, MoE Quantize, MoE Down Projection, MoE Finalize, Shared Expert GateUp/Down, Post-MoE Residual Add
- **Dense FFN**: FFN Norm, GateUp/Gate/Up/Down Projection, Activation, Post-FFN Residual Add
- **Preamble** (optional): Preamble: Embedding, Preamble: Input Norm
- **Epilogue** (optional): Epilogue: Final Norm, Epilogue: LM Head

### `perf_category`

Maps each `semantic_block` to one of four TraceLens perf model types.
Every block has a roofline formula -- no exceptions.

- **GEMM**: QKV Projection, Q Projection, KV Projection, Output Projection, Router Gate, MoE GateUp+SwiGLU, MoE Down Projection, Shared Expert GateUp, Shared Expert Down, GateUp Projection, Gate Projection, Up Projection, Down Projection, Epilogue: LM Head
- **SDPA**: Attention (includes any splitK reduce as part of the attention op)
- **Normalization**: Pre-Attn Norm, Post-Attn Norm, FFN Norm, Epilogue: Final Norm, Preamble: Input Norm
- **Elementwise**: Post-Attn Residual Add, Post-FFN Residual Add, Post-MoE Residual Add, Activation, Attention Output Gate, Rotary Embedding, KV Cache Store, Preamble: Embedding, MoE Routing, MoE Quantize, MoE Finalize

## Workflow

Copy this checklist and track progress:

```
Task Progress:
- [ ] Step 1: Extract trace data (script)
- [ ] Step 2: Find layer pattern (script)
- [ ] Step 3: Classify kernel types (script)
- [ ] Step 4: Assign semantic labels (LLM)
- [ ] Step 5: Derive shapes from config (script)
- [ ] Step 6: Generate semantic report (script)
- [ ] Step 7: Augment trace with annotations (script)
- [ ] Step 8: Verify outputs (script)
```

### Step 1: Extract trace data [S]

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/extract_trace_data.py <trace.json> -o extracted.json
```

Output: `extracted.json` with ordered kernels, python call stack, graph mode detection.

Verify: script exits 0, reports kernel count and total time.

### Step 2: Find layer pattern [S]

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/find_layer_pattern.py extracted.json -o pattern.json
```

Output: `pattern.json` with best period, layer count, preamble/epilogue sizes, kernel-per-layer list.

Verify: autocorrelation score > 0.5. Check assertion warnings.

### Step 3: Classify kernel types [S]

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/classify_kernels.py extracted.json -o classified.json
```

Output: `classified.json` with per-kernel type (GEMM, RMSNorm, Attention, etc).

Verify: few "Unknown" classifications. Check warnings.

### Step 4: Assign semantic labels [L -- LLM required]

This is the core LLM step. Read `extracted.json`, `pattern.json`, and `classified.json`, then produce `semantic_labels.json`.

**Input to LLM**: The three JSON files from previous steps, plus optionally
the model's `config.json` for architecture context.

**Task**: For each kernel in the trace, assign a `semantic_block` label and a `layer` number (null for preamble/epilogue kernels).

**Rules for labeling**:

1. Read `pattern.json` to know preamble size, layer boundaries, and epilogue.
2. First, **identify the model architecture** from the trace patterns:
   - Is it MoE (expert routing kernels, multiple small GEMMs) or dense FFN (single large GateUp/Down GEMMs)?
   - What norm type? (RMSNorm or LayerNorm -- use generic "Norm" labels either way)
   - Are Q/KV projections fused (QKV Projection) or separate (Q Projection + KV Projection)?
   - Does the trace start/end mid-layer (no preamble/epilogue)?
3. For each kernel, combine its position (from pattern), type (from classified), and name to assign a semantic role.
4. **Only use labels from the vocabulary below.** Every label must map to a
   perf model (GEMM, SDPA, Normalization, or Elementwise). Do not invent
   labels outside this set.

**Complete vocabulary for `semantic_block`**:

   **Preamble** (optional -- only if kernels exist before first repeating layer):
   - `Preamble: Embedding` -- embedding lookup [Elementwise]
   - `Preamble: Input Norm` -- input normalization [Normalization]

   **Attention sub-block** (per layer):
   - `Pre-Attn Norm` -- norm before attention [Normalization]
   - `QKV Projection` -- single fused GEMM for Q, K, V [GEMM]
   - `Q Projection` -- separate Q GEMM [GEMM]
   - `KV Projection` -- separate KV GEMM [GEMM]
   - `Rotary Embedding` -- RoPE application [Elementwise]
   - `Attention` -- flash/paged/SDPA kernel(s), including any splitK reduce [SDPA]
   - `KV Cache Store` -- cache write [Elementwise]
   - `Output Projection` -- GEMM after attention [GEMM]
   - `Attention Output Gate` -- gating on attention output [Elementwise]
   - `Post-Attn Residual Add` -- residual connection [Elementwise]

   **MoE FFN sub-block** (per layer, for MoE models):
   - `Post-Attn Norm` -- norm before MoE [Normalization]
   - `Router Gate` -- GEMM computing expert routing scores [GEMM]
   - `MoE Routing` -- topk, renormalize, index computation [Elementwise]
   - `MoE GateUp+SwiGLU` -- fused GateUp GEMM with SwiGLU [GEMM]
   - `MoE Quantize` -- quantize activations [Elementwise]
   - `MoE Down Projection` -- expert down-projection GEMM [GEMM]
   - `MoE Finalize` -- scatter/reduce expert outputs [Elementwise]
   - `Shared Expert GateUp` -- shared expert gate+up [GEMM]
   - `Shared Expert Down` -- shared expert down [GEMM]
   - `Post-MoE Residual Add` -- residual connection after MoE [Elementwise]

   **Dense FFN sub-block** (per layer, for dense/non-MoE models):
   - `FFN Norm` -- norm before FFN [Normalization]
   - `GateUp Projection` -- fused gate+up GEMM [GEMM]
   - `Gate Projection` -- separate gate GEMM [GEMM]
   - `Up Projection` -- separate up GEMM [GEMM]
   - `Activation` -- SiLU/GELU activation [Elementwise]
   - `Down Projection` -- down projection GEMM [GEMM]
   - `Post-FFN Residual Add` -- residual connection [Elementwise]

   **Epilogue** (optional -- only if kernels exist after last repeating layer):
   - `Epilogue: Final Norm` -- final normalization [Normalization]
   - `Epilogue: LM Head` -- logits GEMM [GEMM]

5. Within each layer, blocks should follow the model's algorithmic order.
   Common patterns:
   - Pre-norm transformer: Norm -> Attention -> Residual -> Norm -> FFN/MoE -> Residual
   - Post-norm transformer: Attention -> Norm -> Residual -> FFN -> Norm -> Residual
   The LLM should infer the actual order from the trace, not assume one.
6. If a kernel cannot be assigned with high confidence, assign it to the
   nearest matching perf category label (prefer Elementwise for unknowns
   so it still gets a roofline formula).

**Output format** -- write `semantic_labels.json`:

```json
{
  "source_file": "<trace path>",
  "total_kernel_time_us": 5351.8,
  "model_info": {
    "architecture": "<identified architecture, e.g. LLaMA-70B dense, DeepSeek-V3 MoE>",
    "num_layers": 36,
    "ffn_type": "<dense | moe>",
    "graph_mode": true
  },
  "labeled_kernels": [
    {
      "index": 0,
      "name": "<full kernel name>",
      "dur": 3.4,
      "kernel_type": "Elementwise Add",
      "semantic_block": "Post-Attn Residual Add",
      "layer": 0
    }
  ]
}
```

**Self-check after labeling**:
- Every kernel has a non-empty `semantic_block` from the vocabulary above
- Every `semantic_block` maps to a known perf_category (GEMM, SDPA, Normalization, or Elementwise)
- Within each layer, the block order is consistent with the identified architecture
- Attention kernel count matches expected per layer
- Total labeled kernel count == total kernels in extracted.json

### Step 5: Derive shapes from config [S]

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/derive_shapes.py semantic_labels.json <config.json> \
    --num_tokens <T> [--context_length <ctx>] -o derived_shapes.json
```

Computes theoretical FLOPS and bytes for each semantic block using the
HuggingFace model `config.json`. Uses the same formulas as TraceLens
perf models (GEMM: `2*M*N*K`, SDPA: QK+PV matmuls, Normalization,
Elementwise). Every block gets a formula -- there should be no blocks
with null FLOPS/bytes.

Handles both MoE and dense FFN models automatically based on which fields
are present in the config.

**Arguments**:
- `--num_tokens T`: number of active tokens. For decode, this is batch_size
  (often 1). For prefill, this is prompt_length * batch_size. Required on
  first run; cached to `run_config.json` for reuse.
- `--context_length ctx`: KV cache length for SDPA roofline (decode only).
  Defaults to `num_tokens` (prefill behavior).

Output: `derived_shapes.json` with per-block FLOPS, bytes, perf_params,
semantic_group, and perf_category.

Verify: script exits 0, all blocks have non-null FLOPS/bytes.

### Step 6: Generate semantic report [S]

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/generate_semantic_report.py \
    semantic_labels.json derived_shapes.json \
    [-o report.xlsx] [--gpu_arch gpu_arch.json] [--output_csvs_dir ./csvs]
```

Produces a multi-sheet report with standalone-compatible column names:

- **category_breakdown**: per perf-category time and GFLOPS breakdown
- **semantic_group_summary**: high-level functional groups with aggregate GFLOPS and data moved
- **unified_perf_summary**: one row per (block, perf_params) with standalone-compatible columns (`name`, `op category`, `Kernel Time (µs)_sum`, `TFLOPS/s_mean`, `TB/s_mean`, `FLOPS/Byte`, `Pct Roofline`, `Input Dims`) plus semantic context
- **ops_summary**: per unique GPU kernel with standalone-compatible columns (`name`, `op category`, `Kernel Time (µs)_sum`)
- **gpu_timeline**: GPU utilization breakdown (100% computation for graph-mode traces)

Optional `--gpu_arch gpu_arch.json` enables Roofline_Time_us and Pct Roofline
columns. The gpu_arch JSON should contain `mem_bw_gbps` and
`max_achievable_tflops` (dict with keys like `matrix_bf16`).

Output: `.xlsx` workbook and/or per-sheet CSV files. When `--output_csvs_dir`
points to `perf_report_csvs/`, the CSVs can be consumed directly by
`orchestrator_prepare.py` for standalone analysis.

### Step 7: Augment trace with annotations [S]

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/augment_trace.py <trace.json> semantic_labels.json \
    -o augmented_trace.json
```

Injects `gpu_user_annotation` events into the original trace JSON. Adds three
annotation tiers on the GPU timeline (same `pid` as kernels, new `tid` values):

- **Semantic Groups** row: functional group spans (Self-Attention, MoE / FFN or Dense FFN, etc.)
- **Semantic Layers** row: layer-level spans (Layer 0..N, plus Preamble/Epilogue if present)
- **Semantic Blocks** row: per-block spans wrapping the kernel(s) of each semantic block

The augmented trace can be opened in Perfetto (ui.perfetto.dev) for visual
inspection.

### Step 8: Verify [S]

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/verify_breakdown.py semantic_labels.json breakdown.csv
```

Must exit 0. Fix any FAIL assertions before proceeding.

## Output Files

| File | Format | Purpose |
|------|--------|---------|
| `extracted.json` | JSON | Raw trace data, kernels, metadata |
| `pattern.json` | JSON | Layer structure, period, boundaries |
| `classified.json` | JSON | Per-kernel type classification |
| `semantic_labels.json` | JSON | Per-kernel semantic labels (LLM output) |
| `derived_shapes.json` | JSON | Per-block FLOPS/bytes from config.json |
| `run_config.json` | JSON | Cached num_tokens / context_length |
| `report.xlsx` | Excel | Multi-sheet semantic performance report |
| `augmented_trace.json` | JSON | Original trace + annotation events for Perfetto |

## Shape Derivation from config.json

For GPU-only (graph mode) traces, kernel shapes are derived from the model
config instead of CPU op arguments. The formulas mirror TraceLens perf models.
Every block has a formula.

**GEMM blocks**:

| semantic_block | Shape formula |
|---|---|
| QKV Projection | M=T, N=n_heads*d+2*n_kv*d, K=hidden |
| Q Projection | M=T, N=n_heads*d, K=hidden |
| KV Projection | M=T, N=2*n_kv*d, K=hidden |
| Output Projection | M=T, N=hidden, K=n_heads*d |
| Router Gate | M=T, N=num_experts, K=hidden |
| MoE GateUp+SwiGLU | M=T*topk, N=2*moe_inter, K=hidden |
| MoE Down Projection | M=T*topk, N=hidden, K=moe_inter |
| Shared Expert GateUp | M=T, N=2*shared_inter, K=hidden |
| Shared Expert Down | M=T, N=hidden, K=shared_inter |
| GateUp Projection | M=T, N=2*intermediate, K=hidden |
| Gate/Up Projection | M=T, N=intermediate, K=hidden |
| Down Projection | M=T, N=hidden, K=intermediate |
| Epilogue: LM Head | M=1 or T, N=vocab, K=hidden |

**SDPA blocks**:

| semantic_block | Shape formula |
|---|---|
| Attention | B=1, H_Q=n_heads, H_KV=n_kv, N_Q=T, N_KV=ctx, d=head_dim |

**Normalization blocks**:

| semantic_block | Shape formula |
|---|---|
| All norm blocks | num_elems=T*hidden, num_channels=hidden |

**Elementwise blocks**:

| semantic_block | Shape formula |
|---|---|
| Residual adds, Activation, Gate | num_elems=T*hidden |
| Rotary Embedding | num_elems=T*n_heads*head_dim |
| KV Cache Store | num_elems=2*T*n_kv*head_dim |
| MoE Routing | num_elems=T*num_experts |
| MoE Quantize | num_elems=T*topk*hidden |
| MoE Finalize | num_elems=T*topk*hidden |

Where T = num_tokens, ctx = context_length.
