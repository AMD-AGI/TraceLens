# Semantic Breakdown Agent

Run the full semantic breakdown pipeline on a single GPU trace, producing
`semantic_labels.json` and `derived_shapes.json`. This agent is invoked by the
semantic comparison orchestrator to process each trace independently (two
instances run in parallel).

**Scripts directory:** `TraceLens/AgenticMode/SemanticComparison/trace_breakdown/`

## Required Execution Context

The orchestrator provides these values when launching this agent:

- `<trace_path>` -- path to the raw trace JSON
- `<config_json>` -- path to HuggingFace model config.json
- `<num_tokens>` -- number of active tokens
- `<context_length>` -- KV cache length (optional, defaults to num_tokens)
- `<output_dir>` -- directory for all outputs
- `<trace_name>` -- short label for this trace (e.g., MI355)

## Workflow

### Step 1: Extract trace data

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/extract_trace_data.py \
    <trace_path> -o <output_dir>/extracted.json
```

Output: `extracted.json` with ordered kernels, python call stack, graph mode detection.

Verify: script exits 0, reports kernel count and total time.

### Step 2: Find layer pattern

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/find_layer_pattern.py \
    <output_dir>/extracted.json -o <output_dir>/pattern.json
```

Output: `pattern.json` with best period, layer count, preamble/epilogue sizes.

Verify: autocorrelation score > 0.5.

### Step 3: Classify kernel types

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/classify_kernels.py \
    <output_dir>/extracted.json -o <output_dir>/classified.json
```

Output: `classified.json` with per-kernel type (GEMM, RMSNorm, Attention, etc).

### Step 4: Assign semantic labels [LLM]

Read `extracted.json`, `pattern.json`, and `classified.json`, then produce
`semantic_labels.json`. This is the core LLM step.

**Task**: For each kernel in the trace, assign a `semantic_block` label and a
`layer` number (null for preamble/epilogue kernels).

**Rules for labeling**:

1. Read `pattern.json` to know preamble size, layer boundaries, and epilogue.
2. First, **identify the model architecture** from the trace patterns:
   - Is it MoE (expert routing kernels, multiple small GEMMs) or dense FFN?
   - What norm type? (RMSNorm or LayerNorm -- use generic "Norm" labels either way)
   - Are Q/KV projections fused (QKV Projection) or separate?
   - Does the trace start/end mid-layer (no preamble/epilogue)?
3. For each kernel, combine its position (from pattern), type (from classified),
   and name to assign a semantic role.
4. **Only use labels from the vocabulary below.** Every label must map to a
   perf model (GEMM, SDPA, Normalization, or Elementwise).

**Complete vocabulary for `semantic_block`**:

   **Preamble** (optional):
   - `Preamble: Embedding` [Elementwise]
   - `Preamble: Input Norm` [Normalization]

   **Attention sub-block** (per layer):
   - `Pre-Attn Norm` [Normalization]
   - `QKV Projection` [GEMM]
   - `Q Projection` [GEMM]
   - `KV Projection` [GEMM]
   - `Rotary Embedding` [Elementwise]
   - `Attention` [SDPA]
   - `KV Cache Store` [Elementwise]
   - `Output Projection` [GEMM]
   - `Attention Output Gate` [Elementwise]
   - `Post-Attn Residual Add` [Elementwise]

   **MoE FFN sub-block** (per layer, for MoE models):
   - `Post-Attn Norm` [Normalization]
   - `Router Gate` [GEMM]
   - `MoE Routing` [Elementwise]
   - `MoE GateUp+SwiGLU` [GEMM]
   - `MoE Quantize` [Elementwise]
   - `MoE Down Projection` [GEMM]
   - `MoE Finalize` [Elementwise]
   - `Shared Expert GateUp` [GEMM]
   - `Shared Expert Down` [GEMM]
   - `Post-MoE Residual Add` [Elementwise]

   **Dense FFN sub-block** (per layer, for dense/non-MoE models):
   - `FFN Norm` [Normalization]
   - `GateUp Projection` [GEMM]
   - `Gate Projection` [GEMM]
   - `Up Projection` [GEMM]
   - `Activation` [Elementwise]
   - `Down Projection` [GEMM]
   - `Post-FFN Residual Add` [Elementwise]

   **Epilogue** (optional):
   - `Epilogue: Final Norm` [Normalization]
   - `Epilogue: LM Head` [GEMM]

5. Within each layer, blocks should follow the model's algorithmic order.
6. If a kernel cannot be assigned with high confidence, assign it to the
   nearest matching perf category label (prefer Elementwise for unknowns).

**Output format** -- write `<output_dir>/semantic_labels.json`:

```json
{
  "source_file": "<trace path>",
  "total_kernel_time_us": 5351.8,
  "model_info": {
    "architecture": "<identified architecture>",
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
- Every `semantic_block` maps to a known perf_category
- Within each layer, the block order is consistent with the identified architecture
- Total labeled kernel count == total kernels in extracted.json

### Step 5: Derive shapes from config

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/derive_shapes.py \
    <output_dir>/semantic_labels.json <config_json> \
    --num_tokens <num_tokens> --context_length <context_length> \
    -o <output_dir>/derived_shapes.json
```

Output: `derived_shapes.json` with per-block FLOPS, bytes, perf_params.

Verify: script exits 0, all blocks have non-null FLOPS/bytes.

### Step 6: Verify outputs

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/verify_breakdown.py \
    <output_dir>/semantic_labels.json <output_dir>/breakdown.csv
```

Must exit 0. If it fails, fix the labeling issues and re-run.

## Required Outputs

The agent **must** produce these files in `<output_dir>/`:

| File | Required | Purpose |
|------|----------|---------|
| `semantic_labels.json` | Yes | Per-kernel semantic labels |
| `derived_shapes.json` | Yes | Per-block FLOPS/bytes from config |
| `extracted.json` | Yes | Raw trace data |
| `pattern.json` | Yes | Layer structure |
| `classified.json` | Yes | Per-kernel type classification |

## Return Value

When complete, return a summary to the orchestrator:
- `status`: SUCCESS or ERROR
- `trace_name`: the short label for this trace
- `output_dir`: path to the output directory
- `kernel_count`: number of kernels labeled
- `total_time_us`: total kernel time
- `architecture`: identified model architecture
- `num_layers`: number of layers found
- `ffn_type`: dense or moe
- Any errors or warnings encountered

## Shape Derivation Reference

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
