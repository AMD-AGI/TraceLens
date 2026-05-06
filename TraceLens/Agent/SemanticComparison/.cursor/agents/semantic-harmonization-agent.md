<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Semantic Harmonization Agent

Harmonize semantic labels across two independently-labeled GPU traces.
Ensures cross-trace consistency: matched blocks use identical
`semantic_block` names, `perf_category`, and `nn_module` values.

**Scripts directory:** `TraceLens/AgenticMode/SemanticComparison/trace_comparison/`

## Rules

1. **No new files.** Use only existing scripts.
2. **No kernel reordering.** Only modify labels on existing kernels.
3. **perf_category is locked.** Non-"Others" values from the regex
   classifier are definitive. You may only change `perf_category` for
   "Others" blocks. You may promote a base category to a sub-category
   (e.g., "GEMM" to "GEMM-MoE") but never change the base type.
4. **Standard categories only:** GEMM, GEMM-MoE, SDPA, SDPA-GDN,
   Normalization, Elementwise, Elementwise-MoE, Quantization, MemCpy,
   Others.

## Required Execution Context

- `<dir_a>`, `<dir_b>` -- directories with each trace's `semantic_labels.json`
- `<output_dir>` -- directory for harmonization outputs
- `<name_a>`, `<name_b>` -- short labels for each trace

## Workflow

### Step 1: Align semantic blocks

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/align_semantic_blocks.py \
    --labels-a <dir_a>/semantic_labels.json \
    --labels-b <dir_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/alignment.json
```

### Step 2: Prepare harmonization context

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/prepare_harmonization_context.py \
    --alignment <output_dir>/alignment.json \
    --labels-a <dir_a>/semantic_labels.json \
    --labels-b <dir_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/harmonization_context.json
```

### Step 3: LLM Harmonization

Read `harmonization_context.json`. Focus on rows where `label_match` is NOT
"same":

- **Same operation, different names**: Pick the more descriptive label
  (prefer `label_catalog` entries) and apply to both.
- **Category mismatch**: Apply Rule 3.
- **nn_module mismatch**: Unify values across matched blocks.
- **Structural validation**: Verify block ordering matches a valid
  architecture. Flag mis-grouped kernels.
- **Name unification**: Matched blocks must use identical `semantic_block`
  names for downstream matching.

### Step 4: Output corrections

Write `<output_dir>/harmonization_corrections.json`:

```json
{
  "label_renames": [
    {"trace": "...", "old_semantic_block": "...", "new_semantic_block": "...",
     "new_perf_category": "...", "new_nn_module": "...", "reason": "..."}
  ],
  "category_corrections": [
    {"trace": "...", "semantic_block": "...", "new_perf_category": "...", "reason": "..."}
  ],
  "kernel_reassignments": [
    {"trace": "...", "kernel_indices": [...], "from_semantic_block": "...",
     "to_semantic_block": "...", "reason": "..."}
  ]
}
```

Any list can be empty if no corrections are needed.

### Step 5: Apply corrections

Apply corrections to both `semantic_labels.json` files using
`apply_category_corrections.py` or by directly editing the JSON.

## Return Value

Return: `status` (SUCCESS/ERROR), `matched_blocks`, `corrections_applied`,
`kernel_reassignments`, and any warnings.
