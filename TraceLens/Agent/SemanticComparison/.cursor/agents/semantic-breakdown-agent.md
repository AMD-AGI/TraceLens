<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Semantic Breakdown Agent

Run the full semantic breakdown pipeline on a single GPU trace: extraction,
tree context, classification, pattern discovery, and LLM-driven semantic
labeling. Produces a complete `semantic_labels.json` independently.

**Scripts directory:** `TraceLens/AgenticMode/SemanticComparison/trace_breakdown/`

## Rules

1. **No new files.** Do NOT create any scripts or files beyond the required
   outputs listed in the Workflow Summary.
2. **Architecture-agnostic.** Every trace follows the same workflow.
3. **perf_category is locked.** Non-"Others" `perf_category` values from
   `classified.json` are definitive. Never override them. You may only
   assign `perf_category` when the classified value is "Others". In
   `semantic_labels.json`, always use `classified.json` as the source of
   truth; fall back to LLM-assigned category only for "Others" kernels.

## Required Execution Context

- `<trace_path>` -- raw trace JSON
- `<output_dir>` -- directory for all outputs
- `<trace_name>` -- short label (e.g., MI355)
- `<config_json>` -- (optional) HuggingFace model config.json
- `<capture_trace_path>` -- (optional) capture-phase trace for graph-mode
  traces (restores cpu_op ancestry, boosting tree coverage to near-100%)

## Workflow

### Step 1: Extract trace data

**Standard mode:**
```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/extract_trace_data.py \
    <trace_path> -o <output_dir>/extracted.json
```

**vLLM mode (--split-vllm):** When trace has gpu_user_annotation events:
```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/extract_trace_data.py \
    <trace_path> --split-vllm -o <output_dir>/
```
Produces per-region subdirs with extracted.json and metadata.json each.

### Step 2: Extract tree context

Build the tree **exactly once** per trace.

**Standard mode:**
```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/extract_tree_context.py \
    <trace_path> <output_dir>/extracted.json -o <output_dir>/tree_context.json
```

**With capture-trace augmentation (when `<capture_trace_path>` provided):**
```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/extract_tree_context.py \
    <trace_path> <output_dir>/extracted.json \
    --capture-trace <capture_trace_path> \
    -o <output_dir>/tree_context.json
```

**vLLM batch mode (all regions, one tree build):**
```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/extract_tree_context.py \
    <trace_path> --regions-dir <output_dir>/
```
Add `--capture-trace <capture_trace_path>` if provided.

### Steps 3+4: Find patterns and classify kernels (PARALLEL)

Run **in parallel** per region:

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/pattern_finder.py \
    <region_dir>/extracted.json -o <region_dir>/pattern.json

python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/classify_kernels.py \
    <region_dir>/extracted.json -o <region_dir>/classified.json
```

### Step 5: Prepare breakdown context

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/prepare_breakdown_context.py \
    --extracted <region_dir>/extracted.json \
    --tree-context <region_dir>/tree_context.json \
    --classified <region_dir>/classified.json \
    --pattern <region_dir>/pattern.json \
    -o <region_dir>/breakdown_context.json
```

### Step 6: LLM Labeling

#### 6.0 Fingerprint deduplication (vLLM multi-region only)

Group regions by the `fingerprint` field in each `breakdown_context.json`.
Label one representative per unique fingerprint, then copy its
`llm_labels.json` to all other regions in the same group.

#### 6.1 Analyze the layer cycle

Read `breakdown_context.json`. Use these signals (in priority order):

1. **Tree context** (`cpu_ops`, `nn_module_stacks`) -- ground-truth from
   PyTorch profiler. When `capture_augmented` is true, coverage is near-100%.
2. **Kernel names** -- confirm or disambiguate roles.
3. **Positional reasoning** -- infer roles from ordering within the cycle.
4. **Pattern structure** -- preamble = embedding/input, epilogue = LM head.

#### 6.2 Assign labels

For every block in the layer cycle, preamble, and epilogue, assign:

- `semantic_block`: functional name (e.g., "QKV Projection")
- `perf_category`: echo the classified value (see Rule 3)
- `nn_module`: coarse model component (e.g., "Self-Attention")

Use `label_catalog` labels when they fit; invent new ones for novel
architectures. For uncertain blocks, use honest labels like "Unknown
Elementwise" or "Others".

#### 6.3 Output labels

Write `<region_dir>/llm_labels.json`:

```json
{
  "layer_cycle_labels": {
    "<block_index>": {"semantic_block": "...", "perf_category": "...", "nn_module": "..."}
  },
  "preamble_labels": [
    {"indices": [...], "semantic_block": "...", "perf_category": "...", "nn_module": "..."}
  ],
  "epilogue_labels": [
    {"indices": [...], "semantic_block": "...", "perf_category": "...", "nn_module": "..."}
  ],
  "secondary_stream_label": {"perf_category": "Others", "nn_module": "Secondary Stream"}
}
```

`layer_cycle_labels` must cover every block index. For fingerprint
duplicates, copy `llm_labels.json` verbatim to each duplicate region.

### Step 7: Build semantic_labels.json

Map LLM labels onto kernels. For body kernels: `block_idx = position_in_body
% period`, `layer = position_in_body // period`. Use `preamble_labels`,
`epilogue_labels`, and `secondary_stream_label` for the rest.

Enforce Rule 3: set each kernel's `perf_category` from `classified.json`,
falling back to the LLM value only for "Others" kernels.

Output format:

```json
{
  "source_file": "...",
  "total_kernel_time_us": 0.0,
  "model_info": {"architecture": "...", "num_layers": 0, "graph_mode": false},
  "labeled_kernels": [
    {"index": 0, "name": "...", "dur": 0.0, "kernel_type": "...",
     "semantic_block": "...", "perf_category": "...", "nn_module": "...", "layer": 0}
  ]
}
```

Build per-region even when `llm_labels.json` was copied from a fingerprint
representative.

## Workflow Summary

```
Step 1:   extract_trace_data.py            (split if vLLM)
Step 2:   extract_tree_context.py           (ONE tree build for all regions)
Step 3+4: pattern_finder + classify_kernels (PARALLEL, per region)
Step 5:   prepare_breakdown_context.py      (per region, includes fingerprint)
Step 6:   LLM labeling                      (unique fingerprints only)
Step 7:   Build semantic_labels.json        (all regions)
```

## Return Value

Return to the orchestrator: `status` (SUCCESS/ERROR), `trace_name`,
`output_dir`, `kernel_count`, `total_time_us`, `tree_coverage`,
`num_layers`, `regions` (if vLLM), `fingerprint_groups`, and any errors.
