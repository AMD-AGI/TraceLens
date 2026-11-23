<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Inference Kernel Timeline Extension

A perf report extension that identifies the highest batch-size region of an
inference trace (the steady-state decode plateau) and selects **both** a
representative decode step and its immediately preceding prefill step from
that region.  It performs phase classification, adds a chronological kernel
timeline sheet enriched with layer decomposition, and automatically detects
and labels EAGLE/MTP speculative decoding sections.

**File:** `examples/custom_workflows/inference_kernel_timeline_breakdown/inference_kernel_timeline_breakdown.py`

> **Setup:** Install TraceLens (editable from a clone, or from GitHub), then pass
> the extension with `--extension_file`:
>
> ```bash
> pip install -e .   # from the TraceLens repo root
> # or: pip install git+https://github.com/AMD-AGI/TraceLens.git
> ```

---

## Quick Start

```bash
TraceLens_generate_perf_report_pytorch_inference \
    --profile_json_path trace.json.gz \
    --extension_file examples/custom_workflows/inference_kernel_timeline_breakdown/inference_kernel_timeline_breakdown.py \
    --group_by_parent_module \
    --output_xlsx_path report.xlsx
```

To output only the first N MoE layers (e.g. the first MoE layer in the
MTP verify section):

```bash
TraceLens_generate_perf_report_pytorch_inference \
    --profile_json_path trace.json.gz \
    --extension_file examples/custom_workflows/inference_kernel_timeline_breakdown/inference_kernel_timeline_breakdown.py \
    --group_by_parent_module \
    --output_xlsx_path report.xlsx \
    --moe_only --first_n_layers 1 --mtp_scope verify
```

> **Note:** `--group_by_parent_module` enables tree-based layer decomposition
> by building the tree with `add_python_func=True`.  Without it, the
> `decoder_layer` / `layer_kind` columns in the output will be empty.

The extension adds extra sheets to the report. When multiple architecture
types are detected (typical for Qwen 3.5, MiniMax, DeepSeek), it emits **one
tab per architecture per phase** instead of a single `kernel_timeline` sheet:

- **Per-architecture tabs** (e.g. `GatedDeltaNet_layer1_decode`,
  `attention_v1_layer3_decode`, `DeepseekV2_layer4_prefill`) -- full kernel
  rows for the representative layer of each detected architecture fingerprint.
- **`layer_summary`** -- per-representative-layer stats (phase, kind, template
  ID, kernel count, duration, compressed token sequence).
- **`mtp_section_summary`** -- (MTP traces only) kernel count and duration
  breakdown per EAGLE decode section (`draft`, `verify`, `draft_extend`).

If no architecture split is found, the extension falls back to per-phase sheets
(`prefill` / `decode`) or a single **`kernel_timeline`** sheet.

---

## What It Does

1. **Classifies run_batch phases** -- auto-detects MTP, non-MTP SGLang, or
   vLLM traces and labels each `run_batch` as `prefill` / `decode` /
   `decode_idle` / `prefilldecode`.
2. **Selects the best region** -- groups consecutive same-phase `run_batch`
   events into alternating runs, finds the decode run with the highest modal
   batch size (the "region"), selects a representative decode step from it,
   and a representative prefill step from the immediately preceding prefill
   run (if any).
3. **Prunes the trace tree** to the combined subtrees of the selected
   step(s), tagging every event with a `step_phase` column (`prefill` or
   `decode`).
4. **Detects EAGLE/MTP sections** and tags every event with a `decode_section`
   field.
5. **Decomposes decoder layers** -- finds `nn.Module: *DecoderLayer*` events,
   collects each layer's kernel subtree, tokenizes the sequence, classifies
   layers as MoE or dense, and clusters identical sequences into templates.
6. **Applies scope filters** -- optional `--moe_only`, `--first_n_layers`,
   and `--mtp_scope` CLI arguments narrow the output to specific layers or
   MTP sections.
7. **Detects multi-architecture layers** -- groups decoder layers by
   distinctive `cpu_op` fingerprint (excluding generic `aten::` and MoE
   routing ops), names each group from inner `nn.Module` children or kernel
   signatures, and picks `min(layer_idx)` as the representative per group.
8. **Builds additional Excel sheets** (per-architecture tabs when applicable,
   `layer_summary`, and optionally `mtp_section_summary`).

### Phase Classification

The extension auto-detects which framework/mode is in use and classifies
each `run_batch` event accordingly:

| Strategy | Detection | Labels |
|----------|-----------|--------|
| **MTP / EAGLE** | `forward_idle` or `eagle_worker: draft` children | `decode_idle`, `decode`, `prefill` |
| **Non-MTP SGLang** | `forward_extend` / `forward_decode` children | `prefill`, `decode`, `prefilldecode` |
| **vLLM** | `user_annotation` with context/generation counts | `prefill`, `decode`, `prefilldecode` |
| **Fallback** | No markers found | all `decode` |

Phase info is used during step selection to group `run_batch` events into
alternating runs and restrict decode candidates to the best decode region.

### Step-Selection Strategy

The extension uses a region-based strategy to find representative steps:

| Priority | Method | Description |
|----------|--------|-------------|
| **Primary** | Annotation-based | Matches vLLM/SGLang `user_annotation` events (e.g. `execute_context_N(...)_generation_N(...)`). The annotation with the highest `num_requests` is mapped to its containing `run_batch` event. (Decode step only; no region logic.) |
| **Fallback** | Region-based batch-size | Groups consecutive same-phase `run_batch` events into alternating runs. Finds the decode run with the highest modal batch size (the "region"). Selects a representative decode step from it. If the immediately preceding run is a prefill run, also selects a representative prefill step from it. Both steps use the same algorithm: modal first dim of `cpu_op` input shapes, with nearest-average-duration tie-breaking. |
| **Tie-breaker** | Nearest-avg GPU duration | When multiple steps share the same max score, picks the one whose total GPU kernel duration is closest to the average across tied candidates. |

When both a prefill and decode step are selected, the trace tree is pruned
to the **union** of both subtrees and every event is tagged with a
`step_phase` column (`prefill` or `decode`).

### MTP / EAGLE Section Detection

When the selected `run_batch` step contains EAGLE-style speculative decoding,
the extension automatically detects three section types from
`python_function` events:

| Section | Pattern matched |
|---------|-----------------|
| `draft` | `eagle_worker*.py(N): draft` |
| `verify` | `eagle_worker*.py(N): verify` |
| `draft_extend` | `eagle_worker*.py(N): forward_draft_extend_after_decode` or `_draft_extend_for_decode` |

Every event in the pruned tree is tagged with a `decode_section` field.
Events that fall outside any detected section receive an empty string.

### Layer Decomposition

When the tree is built with `--group_by_parent_module` (i.e.
`add_python_func=True`), the extension detects decoder-layer boundaries from
`nn.Module:` python_function events matching the pattern:

```
nn.Module: *DecoderLayer*_N  (or TransformerBlock, Block)
```

For each detected layer:
1. The layer's kernel subtree is collected via BFS tree traversal (including
   trailing GPU kernels reached through `gpu_events` links).
2. Each kernel is **tokenized** into a canonical label using the `TOKEN_RULES`
   regex table.
3. The layer is classified as **MoE** (if its tokens include any MoE-specific
   patterns like `TopK`, `MoESorting`, `fused_moe`, etc.) or **dense** (if
   `silu_and_mul` is present).
4. Layers with identical compressed token sequences share a **template ID**,
   assigned by frequency (most common = 0).

### Multi-Architecture Tabs

When a trace contains more than one decoder-layer fingerprint within a
`step_phase`, the extension exports one Excel tab per `(phase, arch_type)`
pair.  Tab naming follows `{arch_stem}_layer{N}_{decode|prefill}`, shortened
to fit Excel's 31-character sheet-name limit.

| Trace (example) | Decode tabs | Prefill tabs | Rep layers |
|-----------------|-------------|--------------|------------|
| Qwen 3.5 AMD | `GatedDeltaNet_layer1`, `attention_v1_layer3` | `GatedDeltaNet_layer1`, `and_cache_flash_layer3` | 1, 3 |
| MiniMax AMD | `aiter_rms_norm_layer0`, `attention_v1_layer1` | `aiter_rms_norm_layer0`, `and_cache_flash_layer1` | 0, 1 |
| DeepSeek R1 SGLang MTP | `DeepseekV2_layer4` | `DeepseekV2_layer4` | 4 (both phases) |

Fingerprint rules (aligned with the legacy KernelBreakdown export):

1. Exclude anomalous layers whose kernel count exceeds `10×` the per-phase
   median (minimum threshold 200).
2. Build a fingerprint from distinctive `cpu_op_name` values (skip `aten::`,
   `aiter::moe*`, `_moe_C`, etc.).
3. Within each fingerprint group, export kernels from `min(layer_idx)`.

**SGLang EAGLE/MTP auto-scope:** when EAGLE markers are detected, the
extension automatically narrows decode output to the verify section and picks
the first MoE decoder layer per phase:

- **Decode:** lowest layer index with `moe` in `Kernel name`.
- **Prefill:** layer of the first chronological `aiter::moe` / `_moe_C`
  parent `cpu_op` (vLLM routing parity); falls back to the lowest MoE layer
  by kernel name if no routing hit is found.

### Tokenization (`TOKEN_RULES`)

Kernel names are mapped to stable, human-readable tokens via an ordered list
of `(compiled_regex, token_label)` pairs.  First match wins.

To extend the vocabulary, add a new `(re.compile(r"..."), "TokenLabel")` entry
to the `TOKEN_RULES` list in the extension file.  Order matters -- more
specific patterns must precede general ones.

Current token categories include:

| Category | Example tokens |
|----------|---------------|
| Norms | `RMSNorm` |
| Attention | `fmha`, `mla_decode`, `RoPE`, `get_mla_metadata` |
| Communication | `all_reduce` |
| Activations | `silu_and_mul`, `quant` |
| MoE | `TopK`, `MoESorting`, `fused_moe`, `ck_moe_stage1/2`, `MoERouting`, `MoEFinalize` |
| Gemm | `Gemm`, `Gemm(mxfp4)`, `BatchedGemm(a16wfp4)` |
| Elementwise | `elementwise` (minor aten ops) |

### Output Sheets

#### `kernel_timeline`

One row per GPU kernel (`kernel`, `gpu_memcpy`, `gpu_memset`) in chronological
order:

| Column | Description |
|--------|-------------|
| `Kernel name` | GPU kernel function name |
| `Category` | Event category (`kernel`, `gpu_memcpy`, `gpu_memset`) |
| `Timestamp (us)` | Kernel start time in microseconds |
| `Duration (us)` | Kernel wall-clock duration |
| `Stream` | GPU stream ID |
| `Parent cpu_op` | Name of the nearest ancestor `cpu_op` |
| `Parent cpu_op UID` | UID of that parent |
| `Input Dims` | Input tensor shapes from the parent cpu_op |
| `nn.Module` | Nearest ancestor `nn.Module:` label (if any) |
| `step_phase` | `prefill` or `decode` (which selected step this kernel belongs to) |
| `decode_section` | MTP section (`draft`, `verify`, `draft_extend`, or empty) |
| `decoder_layer` | Decoder layer index (int, or empty if outside a layer) |
| `layer_kind` | `moe`, `dense`, or `unknown` |
| `template_id` | Template cluster ID for this layer |
| `arch_type` | Architecture fingerprint label (e.g. `paged_attention_v1`, `Qwen3_5GatedDeltaNet`) |
| `layer_token` | Canonical token label for this specific kernel |
| `UID` | Unique event ID |

Per-architecture tabs use the same columns as `kernel_timeline`.

#### `layer_summary`

One row per **exported representative** decoder layer (after scope filters):

| Column | Description |
|--------|-------------|
| `step_phase` | `prefill` or `decode` |
| `layer_idx` | Decoder layer index |
| `layer_name` | Full `nn.Module:` event name |
| `layer_kind` | `moe`, `dense`, or `unknown` |
| `template_id` | Template cluster ID |
| `kernel_count` | Number of GPU kernels in this layer |
| `total_dur_us` | Total kernel duration in microseconds |
| `pct_of_step` | Percentage of total step kernel duration |
| `token_sequence` | Compressed canonical token sequence (pipe-separated) |

#### `mtp_section_summary`

Only produced when MTP sections are detected. One row per section:

| Column | Description |
|--------|-------------|
| `Section` | Section name (`draft`, `verify`, `draft_extend`) |
| `Kernel count` | Number of GPU kernels in this section |
| `Total kernel duration (us)` | Summed kernel duration |
| `% of step duration` | Percentage of total step kernel time |

### Scope Filters (CLI Arguments)

The extension accepts additional command-line arguments (parsed from
`sys.argv` via `parse_known_args`).  They are passed alongside the normal
report generator flags and do not require any changes to the generator.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--moe_only` | flag | off | Keep only MoE-classified layers in `kernel_timeline` and `layer_summary`. |
| `--first_n_layers N` | int | `0` (all) | Keep only the first N decoder layers (applied after `--moe_only`). |
| `--mtp_scope SECTION` | str | `""` (all) | Keep only kernels whose `decode_section` matches (e.g. `verify`). |

Example with scope filters:

```bash
TraceLens_generate_perf_report_pytorch_inference \
    --profile_json_path trace.json.gz \
    --extension_file examples/custom_workflows/inference_kernel_timeline_breakdown/inference_kernel_timeline_breakdown.py \
    --group_by_parent_module \
    --output_xlsx_path report.xlsx \
    --moe_only --first_n_layers 1 --mtp_scope verify
```

### Extension Hooks Used

The extension implements two of the four available hooks:

- **`tree_postprocess_extension(tree)`** -- classifies run_batch phases,
  selects the representative decode and prefill steps from the highest
  batch-size region, prunes the tree to their combined subtrees, tags
  `step_phase` and MTP sections.
- **`get_additional_dataframes_extension(tree)`** -- runs layer decomposition,
  builds per-architecture tabs (or `kernel_timeline` / phase fallback),
  `layer_summary`, and `mtp_section_summary` DataFrames, applying scope
  filters.

### Compatible Report CLIs

The extension works with any report script that supports `--extension_file`:

| CLI / Module | Notes |
|--------------|-------|
| `TraceLens_generate_perf_report_pytorch` | General training/inference report |
| `TraceLens_generate_perf_report_pytorch_inference` | Inference-focused (supports `--capture_folder`) |
| `python -m TraceLens.Reporting.generate_perf_report_pytorch_vllm` | vLLM eager mode |
| `python -m TraceLens.Reporting.generate_perf_report_pytorch_vllm_graph` | vLLM graph mode |

---

**Last Updated:** November 2026
