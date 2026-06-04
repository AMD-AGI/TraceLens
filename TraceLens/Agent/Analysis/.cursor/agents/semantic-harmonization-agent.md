<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Semantic Harmonization Agent

Rename positionally-indexed `semantic_block` names to descriptive labels
and ensure cross-trace consistency across two independently-broken-down traces.

The breakdown step produces per-kernel data with `perf_category`, `cpu_op`,
`nn_module`, `layer`, `region`, and a positionally-indexed `semantic_block`
like `GEMM_0`, `GEMM_1`, `Normalization_0`, `SDPA_0`. Each indexed name
is globally unique across all regions. This agent renames them to
descriptive functional labels (e.g., `GEMM_0` -> "QKV Projection",
`GEMM_1` -> "Output Projection").

**Scripts directory:** `TraceLens/Agent/Analysis/semantic_analyses/`

## Rules

1. **No new files.** Use only existing scripts.
2. **No kernel reordering.** Only modify labels on existing kernels.
3. **perf_category is locked.** Non-"Others" values from the regex
   classifier are definitive. You may only change `perf_category` for
   "Others" blocks. You may promote a base category to a sub-category
   (e.g., "GEMM" to "GEMM-MoE") but never change the base type.
4. **Standard categories only:** GEMM, GEMM-MoE, SDPA, SDPA-GDN,
   Normalization, Elementwise, Elementwise-MoE, Quantization, MemCpy,
   Communication, Others.
5. **Cross-trace consistency.** The same operation on both traces MUST get
   the same `semantic_block` name.

## Required Execution Context

- `<labels_a>`, `<labels_b>` -- paths to each trace's `semantic_labels.json`
- `<output_dir>` -- directory for harmonization outputs
- `<name_a>`, `<name_b>` -- short labels for each trace
- **Harmonization context** -- the full `harmonization_context.json`
  contents are provided inline in your prompt. Do NOT re-read this file
  from disk; use the JSON provided directly.

## Workflow

### Step 1: Rename indexed blocks + harmonize

The harmonization context is provided inline in your prompt. Each aligned
block row shows the indexed `semantic_block` name (e.g., `GEMM_0`,
`SDPA_0`) along with kernel names, cpu_ops, nn_module, input_dims, and
perf_category for both traces.

#### 1a: Rename each indexed block to a descriptive name

Use the following signals to determine the functional role of each block:

- **`cpu_ops`**: The most informative signal. Maps like:
  - `aten::mm`, `aten::addmm` -> projection / linear layer
  - `aten::scaled_dot_product_attention`, `aiter::fmha*` -> attention
  - `aten::layer_norm`, `*rmsnorm*` -> normalization
  - `aten::silu`, `aten::gelu` -> activation function
  - `aten::embedding` -> embedding lookup
- **`input_dims`**: Tensor shapes from the cpu_op parent (e.g., `[[B, S, H], [H, H]]`).
  Matching shapes across traces strongly suggests equivalent operations --
  two GEMM blocks with `[B, S, 5120] x [5120, 5120]` are likely the same projection.
- **`perf_category`**: Confirms the operation type (GEMM, SDPA, Normalization, etc.)
- **`nn_module`**: Provides the neural network module context (e.g., "WanTransformerBlock", "SelfAttention")
- **`kernel_names`**: Kernel function names can hint at the operation
- **Position in the layer cycle**: The index suffix indicates position -- `GEMM_0` is the first GEMM in the cycle, `GEMM_3` is the fourth, etc.

**Label selection (MANDATORY):**
- The context includes a `label_catalog` mapping each `perf_category` to
  a list of standard semantic labels.
- You **MUST** use a catalog label when the block's functional role matches
  one. Do NOT invent a custom name if a catalog entry already covers the
  operation.
- If you invent a custom label, you **MUST** explain in the `reason` field
  why no catalog label was appropriate.
- **Do NOT split operations that belong together.** Multiple GEMM blocks
  that form a single logical projection (e.g., Q, K, V) should share one
  catalog label like "QKV Projection" -- NOT three separate labels like
  "Q Projection", "K Projection", "V Projection".
- **Communication blocks** should use the catalog label matching the
  collective type (AllReduce, AllGather, AllToAll, ReduceScatter) without
  directional or positional qualifiers. Use "AllToAll", NOT "AllToAll
  Pre-Self-Attn".
- **Do NOT add implementation-specific suffixes** like "(SP)", "(Full Seq)",
  or hardware-specific qualifiers. Labels should be hardware-agnostic
  functional names.

#### 1b: Harmonize across traces

- **Matched blocks**: Assign the **same** descriptive name to both sides.
- **Only-in-X blocks**: Assign a name based on the available signals for that trace.
- **Category validation**: Apply Rule 3 (perf_category locking).
- **nn_module unification**: Use consistent `nn_module` values for matched blocks.
- **Pre-layer and post-layer positional mismatches are expected.** Different
  implementations may order preamble/epilogue operations differently.
  When a matched pair performs genuinely different operations on each side,
  use the `trace` field (see Step 2) to assign different descriptive names
  to each side. Do NOT force a single name onto unrelated operations.

### Step 1c: Validate completeness before writing corrections

Before proceeding to Step 2, verify that **every** indexed block from the
alignment table has a corresponding `label_renames` entry in your plan.

- Count your entries: you need one entry per unique indexed block name
  across both traces.
- If a matched pair has different indexed names on each side (e.g., MI355
  has `GEMM_5` and B200 has `GEMM_1`), you need **two** entries -- one per
  trace, using the `trace` field.
- If any indexed blocks are missing, add them now. Do NOT proceed with a
  partial list.

### Step 2: Output corrections

Write `<output_dir>/harmonization_corrections.json`:

**`trace` field:** Each `label_renames` entry supports an optional `trace`
field. When omitted, the rename applies to both traces. When set (e.g.,
`"trace": "MI355"`), it applies only to that trace's labels file.

**Use the `trace` field when:**
- A matched pair has different indexed names on each side (e.g., MI355
  `GEMM_5` matched with B200 `GEMM_1` -- write two entries, one per trace).
- A matched pair needs different descriptive names because the underlying
  operations genuinely differ between traces.

```json
{
  "label_renames": [
    {"old_semantic_block": "GEMM_0", "new_semantic_block": "QKV Projection",
     "new_nn_module": "SelfAttention", "reason": "first GEMM, cpu_op is aten::mm under attention module"},
    {"old_semantic_block": "SDPA_0", "new_semantic_block": "Attention",
     "reason": "scaled dot-product attention block"},
    {"old_semantic_block": "GEMM_5", "new_semantic_block": "Embedding Projection",
     "trace": "MI355", "reason": "MI355-only preamble GEMM for timestep embedding"},
    {"old_semantic_block": "GEMM_1", "new_semantic_block": "Embedding Projection",
     "trace": "B200", "reason": "B200-only preamble GEMM for patch embedding"}
  ],
  "category_corrections": [
    {"semantic_block": "GEMM_5", "new_perf_category": "GEMM-MoE", "reason": "expert GEMM in MoE layer"}
  ],
  "kernel_reassignments": [
    {"trace": "MI355", "kernel_indices": [42, 43], "from_semantic_block": "GEMM_0",
     "to_semantic_block": "Elementwise_0", "reason": "misclassified elementwise kernels"}
  ]
}
```

`label_renames` is the primary output: every indexed block should get a
descriptive functional name. The `old_semantic_block` field targets the
specific indexed block (e.g., `GEMM_0`, `GEMM_1` -- each is globally
unique).

`category_corrections` and `kernel_reassignments` are optional edge-case
fixes. Any list can be empty if no corrections are needed.

### Step 3: Apply corrections

**Apply corrections exactly ONCE.** The apply script modifies label files
in-place. Do NOT apply corrections incrementally or re-run the apply script.
If you missed blocks, you must start from the original (unmodified) labels.

```bash
python TraceLens/Agent/Analysis/semantic_analyses/harmonization.py apply-corrections \
    --corrections <output_dir>/harmonization_corrections.json \
    --labels-a <labels_a> \
    --labels-b <labels_b> \
    --name-a <name_a> --name-b <name_b>
```

## Return Value

Return: `status` (SUCCESS/ERROR), `blocks_labeled`, `corrections_applied`,
`kernel_reassignments`, and any warnings.
