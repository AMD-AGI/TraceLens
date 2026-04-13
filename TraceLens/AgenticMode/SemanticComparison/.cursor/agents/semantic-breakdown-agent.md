<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Semantic Breakdown Agent

Run the full semantic breakdown pipeline on a single GPU trace: extraction,
tree context, classification, pattern discovery, and LLM-driven semantic
labeling. This agent produces a complete `semantic_labels.json` independently
-- no second trace is needed.

Two instances run in parallel (one per trace) when invoked by the
Semantic Comparison orchestrator.

**Scripts directory:** `TraceLens/AgenticMode/SemanticComparison/trace_breakdown/`

## Critical Constraints

- **Do NOT create, write, or generate any new Python scripts.** All required
  scripts already exist in `trace_breakdown/`. Use only the scripts listed in
  this document.
- **Do NOT create any new files** except the outputs explicitly required
  (extracted.json, tree_context.json, classified.json, pattern.json,
  breakdown_context.json, llm_labels.json, semantic_labels.json).
- **Do NOT route based on model type / architecture.** Every trace follows the
  same workflow regardless of what model produced it.

## Required Execution Context

The orchestrator provides these values when launching this agent:

- `<trace_path>` -- path to the raw trace JSON
- `<output_dir>` -- directory for all outputs
- `<trace_name>` -- short label for this trace (e.g., MI355)
- `<config_json>` -- (optional) path to HuggingFace model config.json

## Performance Notes

This workflow is optimized to avoid redundant work:

- **Tree built once**: `extract_tree_context.py` has a `--regions-dir` batch
  mode that builds the TraceToTree once and processes all regions against it.
- **Parallel script execution**: `pattern_finder.py` and `classify_kernels.py`
  are independent and run in parallel.
- **Fingerprint deduplication**: `breakdown_context.json` includes a
  `fingerprint` field. Regions with identical fingerprints share the same model
  structure, so the LLM labels only one and the results are copied to the rest.

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
Produces per-region subdirs (prefill_only_3072, decode_only_4, etc.) with
extracted.json and metadata.json each.

Output: `extracted.json` with ordered kernels, python call stack, graph mode
detection.

Verify: script exits 0, reports kernel count and total time.

### Step 2: Extract tree context

The trace tree is expensive to build. Use the appropriate mode to ensure it
is built **exactly once** per trace.

**Standard mode (single region):**
```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/extract_tree_context.py \
    <trace_path> <output_dir>/extracted.json -o <output_dir>/tree_context.json
```

**vLLM mode (batch -- all regions in one tree build):**
```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/extract_tree_context.py \
    <trace_path> --regions-dir <output_dir>/
```
Builds the tree once and writes `tree_context.json` into every region subdir
that contains an `extracted.json`.

Output: `tree_context.json` with per-kernel cpu_op ancestor names,
nn_module_stack, and callstack information from the TraceToTree hierarchy.

Reports coverage: the fraction of kernels that have cpu_op ancestors in the
trace tree. High coverage (near 100%) is typical for eager-mode regions
(e.g. prefill); low coverage is typical for graph-mode regions (e.g. decode).

Verify: script exits 0, reports coverage percentage.

### Steps 3+4: Find patterns and classify kernels (PARALLEL)

These two scripts are **independent** -- both only need `extracted.json` as
input. Run them **in parallel** (two shell commands launched simultaneously)
for each region.

**Step 3: Find repeating patterns**
```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/pattern_finder.py \
    <region_dir>/extracted.json -o <region_dir>/pattern.json
```

Output: `pattern.json` with discovered patterns, sequences, preamble/epilogue
indices, and stream information (primary_stream_id, secondary_stream_indices).

Verify: coverage > 80%, at least one pattern with >= 10 occurrences.

**Step 4: Classify kernel types**
```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/classify_kernels.py \
    <region_dir>/extracted.json -o <region_dir>/classified.json
```

Output: `classified.json` with per-kernel type and perf_category
(GEMM, SDPA, Normalization, Elementwise, MemCpy).

Verify: script exits 0, few or no "Unknown" kernels.

**For vLLM mode:** Run Steps 3+4 in parallel **per region**. All regions can
also be processed concurrently since they are independent.

### Step 5: Prepare breakdown context

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_breakdown/prepare_breakdown_context.py \
    --extracted <region_dir>/extracted.json \
    --tree-context <region_dir>/tree_context.json \
    --classified <region_dir>/classified.json \
    --pattern <region_dir>/pattern.json \
    -o <region_dir>/breakdown_context.json
```

Output: `breakdown_context.json` -- a compact JSON combining all prior
outputs into a single packet for LLM review. Contains:
- `model_info`: graph mode, python stack availability
- `pattern_info`: period, num_layers, preamble/epilogue/secondary counts
- `layer_cycle`: one super-cycle of the body with per-block kernel names,
  types, categories, and tree context (cpu_op, nn_module)
- `preamble_kernels`, `epilogue_kernels`: individual kernel details
- `tree_coverage`: fraction of kernels with tree context
- `fingerprint`: a hash of the layer cycle structure (used for deduplication)
- `label_catalog`: reference catalog of known functional labels

### Step 6: LLM Labeling (with fingerprint deduplication)

#### 6.0 Fingerprint deduplication (vLLM multi-region only)

When multiple regions exist, compare the `fingerprint` field in each region's
`breakdown_context.json`. Regions with **identical fingerprints** have the same
model structure and would receive identical labels.

1. Read the `fingerprint` from each region's `breakdown_context.json`
2. Group regions by fingerprint
3. For each unique fingerprint, pick one **representative region** to label
4. After labeling the representative (Steps 6.1-6.3 below), **copy** its
   `llm_labels.json` to all other regions in the same fingerprint group
5. Proceed to Step 7 for all regions

For single-region traces, skip this step.

#### 6.1 Analyze the layer cycle

Read `breakdown_context.json`. Using the kernel names, regex classifications,
tree context (cpu_op names, nn_module_stack), pattern structure, and your
knowledge of deep learning model architectures (transformers, MoE, linear
attention, state-space models, etc.), assign functional labels.

**perf_category is locked for non-Others blocks.** Every block has a
`perf_category` assigned by the regex classifier. If `perf_category` is
anything other than "Others", it is **definitive** and you MUST NOT change it.
You may only assign a new `perf_category` for blocks where the existing value
is "Others". Your primary creative scope is `semantic_block` (functional name)
and `nn_module` (model component).

Review the `layer_cycle` in `breakdown_context.json`. Each block has:
- `perf_category`: regex-derived category (GEMM, SDPA, Normalization, etc.)
  **Locked unless "Others".**
- `kernel_types`: specific kernel type classifications
- `unique_kernel_names`: the actual GPU kernel names
- `cpu_ops`: cpu_op ancestor names from the trace tree (when available)
- `nn_module_stacks`: nn.Module hierarchy from PyTorch profiler (when available)

Use all of these signals to determine what each block does:

1. **Tree context first**: When `cpu_ops` and `nn_module_stacks` are present,
   these are ground-truth signals from the PyTorch profiler. Use them as the
   primary source for understanding a block's role.
2. **Kernel names as hints**: Use GPU kernel names to confirm or disambiguate
   (e.g., `kernel_moe_gemm` -> MoE GEMM, `fmha` -> attention).
3. **Positional reasoning**: Use ordering within the cycle to infer roles
   (e.g., GEMM after norm and before attention -> QKV Projection).
4. **Pattern structure**: preamble = embedding/input, epilogue = LM head/final
   norm. Body repeats `num_layers` times.

#### 6.2 Assign labels

For every block in the layer cycle, assign:

- `semantic_block`: functional name (e.g., "QKV Projection", "MoE Expert
  W1 GEMM (Gate)", "Pre-Attn RMSNorm")
- `perf_category`: **echo the value from `breakdown_context.json`** for
  non-Others blocks. For "Others" blocks only, assign a category from the
  standard set (GEMM, GEMM-MoE, SDPA, SDPA-GDN, Normalization, Elementwise,
  Elementwise-MoE, Quantization, MemCpy, Others).
- `nn_module`: coarse-grained model sub-component (e.g., "Self-Attention",
  "MoE FFN", "Normalization", "Embedding")

Also assign labels for preamble and epilogue kernels (grouped or individual).
The same `perf_category` locking rule applies: only reassign "Others" kernels.

The `label_catalog` in the context provides a reference list of known labels.
Follow these rules:

1. **Prefer catalog labels** when they fit the block's role
2. **Invent new labels** when the model has components not in the catalog
3. **Naming style**: `nn_module` should be a noun phrase (component name),
   `semantic_block` should describe the specific operation
4. **Uncertain blocks**: Use honest labels like "Unknown Elementwise",
   "Misc Compute", or "Others". A wrong label is worse than an honest "Others".

#### 6.3 Output labels

Write a JSON object to `<region_dir>/llm_labels.json`:

```json
{
  "layer_cycle_labels": {
    "0": {"semantic_block": "Pre-Attn RMSNorm", "perf_category": "Normalization", "nn_module": "Normalization"},
    "1": {"semantic_block": "QKV Projection", "perf_category": "GEMM", "nn_module": "Self-Attention"},
    "2": {"semantic_block": "RoPE", "perf_category": "Elementwise", "nn_module": "Self-Attention"},
    "...": "..."
  },
  "preamble_labels": [
    {"indices": [0, 1, 2], "semantic_block": "Token Embedding", "perf_category": "Elementwise", "nn_module": "Embedding"},
    {"indices": [3], "semantic_block": "Input RMSNorm", "perf_category": "Normalization", "nn_module": "Normalization"}
  ],
  "epilogue_labels": [
    {"indices": [1820], "semantic_block": "Final RMSNorm", "perf_category": "Normalization", "nn_module": "Output Head"},
    {"indices": [1821, 1822], "semantic_block": "LM Head Projection", "perf_category": "GEMM", "nn_module": "Output Head"}
  ],
  "secondary_stream_label": {"perf_category": "Others", "nn_module": "Secondary Stream"}
}
```

- `layer_cycle_labels`: dict keyed by block_index (from `layer_cycle`).
  Must cover every block in the layer cycle.
- `preamble_labels`: list of label groups for preamble kernels. Each entry
  has `indices` (kernel indices) and label fields.
- `epilogue_labels`: same format for epilogue kernels.
- `secondary_stream_label`: default label for secondary-stream kernels.

**For fingerprint duplicates:** After labeling the representative region,
copy its `llm_labels.json` file verbatim to each duplicate region's directory.
The labels are identical because the layer cycle structure is identical.

### Step 7: Build semantic_labels.json

Apply the LLM labels to produce the final `semantic_labels.json`. Use the
existing `apply_category_corrections.py` script or construct the output
directly.

**Hard enforcement of perf_category:** For each kernel, set `perf_category`
to the value from `classified.json`, NOT from `llm_labels.json`. Only use the
LLM-assigned `perf_category` when the kernel's classified value is "Others".
This ensures the regex classifier's definitive categories are never overridden,
even if the LLM changed them in `llm_labels.json`.

The output format must match:

```json
{
  "source_file": "<trace_path>",
  "total_kernel_time_us": 12345.67,
  "model_info": {
    "architecture": "unknown",
    "num_layers": 28,
    "ffn_type": "moe",
    "graph_mode": true
  },
  "labeled_kernels": [
    {
      "index": 0,
      "name": "kernel_name",
      "dur": 1.23,
      "kernel_type": "GEMM",
      "semantic_block": "QKV Projection",
      "perf_category": "GEMM",
      "nn_module": "Self-Attention",
      "layer": 0
    }
  ]
}
```

To build this:

1. For each body kernel at index `i`:
   - Compute which block it belongs to: `block_idx = (i - preamble_size) % period`
     within the body (excluding preamble/epilogue/secondary)
   - Look up `layer_cycle_labels[block_idx]` for its labels
   - Compute `layer = (i - preamble_size) // period`
2. For preamble kernels: use `preamble_labels` (match by index)
3. For epilogue kernels: use `epilogue_labels` (match by index)
4. For secondary-stream kernels: use `secondary_stream_label`

Write the final `semantic_labels.json` to `<region_dir>/semantic_labels.json`.

**For vLLM mode:** Build `semantic_labels.json` for every region, including
fingerprint duplicates. Each region has its own kernel indices and counts,
so `semantic_labels.json` must be built per-region even when `llm_labels.json`
was copied.

## Required Outputs

The agent **must** produce these files in `<output_dir>/` (or per-region
subdirectory):

| File | Required | Purpose |
|------|----------|---------|
| `extracted.json` | Yes | Raw trace data |
| `tree_context.json` | Yes | Tree context (cpu_op, nn_module per kernel) |
| `pattern.json` | Yes | Repeating kernel patterns with stream info |
| `classified.json` | Yes | Per-kernel type classification |
| `breakdown_context.json` | Yes | Compact LLM context (includes fingerprint) |
| `llm_labels.json` | Yes | LLM-assigned functional labels |
| `semantic_labels.json` | Yes | Final labeled kernels |

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

When complete, return a summary to the orchestrator:
- `status`: SUCCESS or ERROR
- `trace_name`: the short label for this trace
- `output_dir`: path to the output directory
- `kernel_count`: number of kernels extracted
- `total_time_us`: total kernel time
- `tree_coverage`: fraction of kernels with tree context
- `num_layers`: detected number of model layers
- `regions`: list of region subdirectory names (if vLLM split was used)
- `fingerprint_groups`: number of unique fingerprints vs total regions
- Any errors or warnings encountered
