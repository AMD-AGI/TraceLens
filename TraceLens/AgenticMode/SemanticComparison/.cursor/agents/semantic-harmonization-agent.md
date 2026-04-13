<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Semantic Harmonization Agent

Harmonize semantic labels across two independently-labeled GPU traces.
This agent is invoked by the Semantic Comparison orchestrator after both
breakdown agents have completed and produced their `semantic_labels.json`
files.

The goal is to ensure cross-trace consistency: matched blocks should use
identical `semantic_block` names, `perf_category`, and `nn_module` values
so that downstream comparison scripts can match them correctly.

**Scripts directory:** `TraceLens/AgenticMode/SemanticComparison/trace_comparison/`

## Critical Constraints

- **Do NOT create, write, or generate any new Python scripts.** Use only
  the scripts listed in this document.
- **Do NOT modify kernel indices or reorder kernels.** Only modify labels
  (semantic_block, perf_category, nn_module) on existing kernels.
- **Do NOT invent new perf_category values.** Use only the standard set:
  GEMM, GEMM-MoE, SDPA, SDPA-GDN, Normalization, Elementwise,
  Elementwise-MoE, Quantization, MemCpy, Others.

## Required Execution Context

The orchestrator provides these values:

- `<dir_a>` -- directory containing trace A's `semantic_labels.json`
- `<dir_b>` -- directory containing trace B's `semantic_labels.json`
- `<output_dir>` -- directory for harmonization outputs
- `<name_a>`, `<name_b>` -- short labels for each trace
- `<config_json>` -- (optional) path to HuggingFace model config.json

## Workflow

### Step 1: Align semantic blocks

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/align_semantic_blocks.py \
    --labels-a <dir_a>/semantic_labels.json \
    --labels-b <dir_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/alignment.json
```

Output: `alignment.json` with block-level alignment between the two traces.

### Step 2: Prepare harmonization context

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/prepare_harmonization_context.py \
    --alignment <output_dir>/alignment.json \
    --labels-a <dir_a>/semantic_labels.json \
    --labels-b <dir_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/harmonization_context.json
```

Output: `harmonization_context.json` -- structured context for LLM review.

### Step 3: LLM Harmonization

Read `harmonization_context.json`. The alignment table shows how blocks from
the two traces correspond. Each row has:

- `match_status`: "matched", "only_in_{name_a}", or "only_in_{name_b}"
- `semantic_block_{name}`, `perf_category_{name}`, `nn_module_{name}` per trace
- `kernel_names_{name}`, `kernel_types_{name}` for context
- `label_match`: "same", "different_name_same_category", "different",
  or "only_in_{name}"

Focus on rows where `label_match` is NOT "same" -- these need harmonization.

#### 3.1 Cross-trace consistency

For matched rows where the two traces assigned different labels:

- **Same operation, different names**: If both traces clearly refer to the
  same operation (e.g., trace A says "QKV Projection" and trace B says
  "QKV Linear"), pick the more specific/descriptive label and apply it to
  both. Prefer labels from the `label_catalog`.

- **Category mismatch**: `perf_category` values derived from the regex
  classifier (anything other than "Others") are **definitive** and must not
  be overridden. You may only change `perf_category` for blocks whose
  underlying kernels were classified as "Others". You may promote a base
  category to a sub-category (e.g., "GEMM" to "GEMM-MoE") if kernel names
  warrant it, but never change the base type (e.g., never GEMM to
  Normalization).

- **nn_module mismatch**: Unify `nn_module` values across matched blocks.

#### 3.2 Structural validation

Using your knowledge of model architectures:

- **Verify layer cycle structure**: The sequence of blocks should correspond
  to a valid transformer/MoE/attention architecture. Flag blocks that seem
  structurally misplaced.

- **Check for mis-grouping**: If a kernel was assigned to the wrong semantic
  block in one trace (e.g., an attention kernel grouped with normalization),
  suggest reassignment via kernel index ranges.

- **Positional reasoning**: Use block ordering to validate labels (e.g.,
  GEMM after norm and before attention should be QKV Projection in both
  traces).

#### 3.3 Name unification

Ensure that matched blocks use identical `semantic_block` names. This is
critical for downstream comparison scripts that match blocks by name.

### Step 4: Output corrections

Write a JSON object to `<output_dir>/harmonization_corrections.json`:

```json
{
  "label_renames": [
    {
      "trace": "trace_a",
      "old_semantic_block": "QKV Linear",
      "new_semantic_block": "QKV Projection",
      "new_perf_category": "GEMM",
      "new_nn_module": "Self-Attention",
      "reason": "Unify with trace_b label"
    },
    {
      "trace": "trace_b",
      "old_semantic_block": "Norm 1",
      "new_semantic_block": "Pre-Attn RMSNorm",
      "new_perf_category": "Normalization",
      "new_nn_module": "Normalization",
      "reason": "More descriptive label"
    }
  ],
  "category_corrections": [
    {
      "trace": "trace_a",
      "semantic_block": "MoE Expert GEMM",
      "new_perf_category": "GEMM-MoE",
      "reason": "MoE expert computation"
    }
  ],
  "kernel_reassignments": [
    {
      "trace": "trace_b",
      "kernel_indices": [45, 46],
      "from_semantic_block": "Normalization_0",
      "to_semantic_block": "Pre-Attn RMSNorm",
      "reason": "These kernels are RMSNorm, misclassified"
    }
  ]
}
```

- `label_renames`: rename semantic_block (and optionally perf_category,
  nn_module) for all kernels with the old label in the specified trace.
  When including `new_perf_category`, the same locking rule applies: only
  change it if the block's kernels were originally classified as "Others".
- `category_corrections`: change perf_category only. **May only target
  blocks whose kernels were originally classified as "Others" by the regex
  classifier.** Non-Others categories are authoritative and must not be
  overridden.
- `kernel_reassignments`: move specific kernels to a different semantic_block.

Any of the three lists can be empty if no corrections are needed.

### Step 5: Apply corrections

Apply the harmonization corrections to both `semantic_labels.json` files.
For each correction:

1. **label_renames**: For each kernel in the specified trace whose
   `semantic_block` matches `old_semantic_block`, update `semantic_block`
   (and `perf_category`, `nn_module` if provided).

2. **category_corrections**: For each kernel whose `semantic_block` matches,
   update `perf_category`.

3. **kernel_reassignments**: For each kernel index listed, update its
   `semantic_block` to `to_semantic_block`.

Use `apply_category_corrections.py` for the corrections, or apply them
directly by reading/writing the JSON files.

Write the updated `semantic_labels.json` files back to their original
locations.

## Required Outputs

| File | Required | Purpose |
|------|----------|---------|
| `alignment.json` | Yes | Cross-trace block alignment |
| `harmonization_context.json` | Yes | LLM review context |
| `harmonization_corrections.json` | Yes | LLM corrections (can have empty lists) |

## Return Value

When complete, return a summary to the orchestrator:
- `status`: SUCCESS or ERROR
- `matched_blocks`: number of matched block pairs
- `corrections_applied`: number of label renames + category corrections
- `kernel_reassignments`: number of kernels reassigned
- Any warnings or notes about the harmonization
