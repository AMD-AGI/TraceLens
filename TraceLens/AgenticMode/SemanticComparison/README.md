<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceLens Agentic Mode: Semantic Trace Comparison

> **⚠️ Experimental**: This feature is under active development and may change.

TraceLens Agentic Mode for Semantic Comparison is a Cursor-based AI-powered cross-trace analysis tool that compares GPU kernel-level performance between two PyTorch profiler traces. It assigns semantic labels to every kernel (e.g., "QKV Projection", "MoE GateUp+SwiGLU", "Attention"), matches them across traces, computes roofline metrics from the model's HuggingFace config, and generates a prioritized gap analysis report. Typical use cases include comparing the same model across GPU platforms, software versions, or configuration changes.

---

## Prerequisites

### 1. Clone TraceLens-internal

```bash
git clone https://github.com/AMD-AGI/TraceLens-internal.git
cd TraceLens
```

### 2. Install Dependencies

The comparison scripts require Python 3.8+ with:

```bash
pip install pandas openpyxl
```

### 3. Prepare Inputs

You will need:

- **Two graph-mode Chrome trace JSON files** (`.json` or `.json.gz`) from the PyTorch profiler
- **HuggingFace `config.json`** for the model (download from the model's repo, e.g., `https://huggingface.co/<org>/<model>/raw/main/config.json`)
- **Number of active tokens** (`batch_size` for decode, `prompt_length * batch_size` for prefill)

---

## Quick Start - How to Use

### To run a semantic comparison:

1. **In a Cursor (v2.5+) chat with Claude-4.6-Opus-High, invoke:**
   ```
   Run semantic comparison on <path_to_trace_a.json> and <path_to_trace_b.json>
   ```

2. **Provide when prompted:**
   - Trace A and Trace B file paths
   - Short labels for each trace (e.g., MI355 / B200)
   - HuggingFace `config.json` path
   - Number of active tokens
   - KV cache context length (optional, for decode traces)
   - Output directory (optional)

3. **Get results:**
   - **Primary output**: `semantic_comparison_report.md` -- Stakeholder report with prioritized gap analysis
   - **Additional outputs:**
     - `comparison_report.xlsx` -- Multi-sheet Excel workbook
     - `comparison_report_csvs/` -- All CSV outputs (comparison, diff stats, kernel mapping)

---

### Output Files

```
<output_dir>/
├── semantic_comparison_report.md     # Stakeholder gap analysis report
├── comparison_report.xlsx            # Multi-sheet Excel workbook
├── comparison_report_csvs/           # All CSV outputs
│   ├── kernel_mapping.csv            #   Human-readable kernel-to-kernel mapping
│   ├── comparison.csv                #   Block-level timing + roofline metrics
│   ├── diff_stats.csv                #   Per-kernel TraceDiff rows
│   ├── diff_stats_summary.csv        #   Aggregated diff stats
│   ├── diff_stats_unique_args_summary.csv
│   ├── cpu_op_map.json               #   Kernel-to-op mapping (combined)
│   ├── cpu_op_map_trace1.json        #   Trace A kernel-to-op mapping
│   └── cpu_op_map_trace2.json        #   Trace B kernel-to-op mapping
├── comparison.csv                    # Block-level comparison (timing, roofline, ratio/gap)
└── priority.json                     # Ranked optimization targets
```

---

## Architecture

### Pipeline Overview

The analysis pipeline has two phases: **Semantic Breakdown** (per-trace, parallelized) and **Cross-Trace Comparison** (sequential, deterministic scripts + LLM report generation).

### Orchestrator

The **Semantic Comparison Orchestrator** skill coordinates the entire workflow. It queries user inputs, launches parallel breakdown subagents for both traces, harmonizes semantic vocabularies, runs the deterministic comparison pipeline, and generates the final gap analysis report.

### Workflow Steps

```
0.   Query User Inputs (Trace Paths, Names, Config, Tokens)
1.   Semantic Breakdown (PARALLEL subagents) → <name_a>/, <name_b>/
2.   Harmonize Semantic Vocabularies (LLM)
3.   Generate TraceDiff Output (script)
4.   Generate Comparison CSV (script)
5.   Compute Priority Ranking (script)
6.   Verify Outputs (script)
7.   Generate Comparison Report (script) → Excel + CSVs
8.   Write Gap Analysis Report (LLM)
9.   Validate Report
```

### Sub-Agents

| Agent | File | Purpose |
|-------|------|---------|
| Semantic Breakdown | `semantic-breakdown-agent.md` | Run the full breakdown pipeline on a single trace (extract, pattern, classify, LLM label, derive shapes, verify). Two instances run in parallel. |

### Comparison Scripts

| Script | Step | Purpose |
|--------|------|---------|
| `generate_semantic_diff.py` | 3 | Produce TraceDiff-style output (diff_stats.csv, diff_stats_summary.csv) |
| `match_and_compare.py` | 4 | Match blocks across traces, compute timing/roofline/ratio metrics |
| `compute_priority.py` | 5 | Rank optimization targets by `pct * (ratio - 1)` |
| `verify_comparison.py` | 6 | Verify comparison consistency (vocabulary, totals, ratios) |
| `generate_comparison_report.py` | 7 | Build multi-sheet Excel workbook and per-sheet CSVs |

### Breakdown Scripts

| Script | Step | Purpose |
|--------|------|---------|
| `extract_trace_data.py` | 1.1 | Parse Chrome trace JSON, extract ordered kernel list |
| `find_layer_pattern.py` | 1.2 | Detect repeating layer structure via autocorrelation |
| `classify_kernels.py` | 1.3 | Classify kernels by type (GEMM, RMSNorm, Attention, etc.) via regex |
| `category_mappings.py` | -- | Shared module: semantic block vocabulary, perf categories, shape formulas |
| `derive_shapes.py` | 1.5 | Compute theoretical FLOPS/bytes from HuggingFace config |
| `generate_breakdown_csv.py` | 1.6 | Convert semantic_labels.json to breakdown.csv |
| `verify_breakdown.py` | 1.6 | Check consistency of breakdown outputs |
| `augment_trace.py` | -- | Add Perfetto annotations for visualization (optional) |

---

## Semantic Categories

Every kernel is assigned a `semantic_block` label from a fixed vocabulary. Each block maps to a `semantic_group` (functional area) and a `perf_category` (roofline model type).

| perf_category | Description | Roofline Model |
|---------------|-------------|----------------|
| GEMM | All projection / matmul blocks | 2*M*N*K FLOPS |
| SDPA | Attention kernels (Flash, Paged) | Q*K^T + softmax + P*V |
| Normalization | RMSNorm, LayerNorm, GroupNorm | Element-wise with reduction |
| Elementwise | Residual adds, activations, routing | Element-wise read/write |

**Semantic block vocabulary** (abridged):

| Group | Blocks |
|-------|--------|
| Preamble | Embedding, Input Norm |
| Self-Attention | Pre-Attn Norm, QKV Projection, Q/KV Projection, Rotary Embedding, Attention, KV Cache Store, Output Projection, Attention Output Gate, Post-Attn Residual Add |
| MoE / FFN | Post-Attn Norm, Router Gate, MoE Routing, MoE GateUp+SwiGLU, MoE Quantize, MoE Down Projection, MoE Finalize, Shared Expert GateUp/Down, Post-MoE Residual Add |
| Dense FFN | FFN Norm, GateUp/Gate/Up Projection, Activation, Down Projection, Post-FFN Residual Add |
| Epilogue | Final Norm, LM Head |

The full vocabulary with perf_category mappings is defined in `trace_breakdown/category_mappings.py`.

---

## Extending Capability

### Custom Trace Preparation

The orchestrator assumes graph-mode Chrome trace JSON from the PyTorch profiler. For traces that require special handling, you can override the breakdown step:

```
Run semantic comparison on trace_a.json and trace_b.json

For trace A, the trace requires augmentation. Use this command for extraction:
python custom_extract.py trace_a.json -o extracted.json

Once extraction is complete, proceed with the normal flow.
```

### Adding New Semantic Blocks

To add a new `semantic_block` label:

1. Add the block to `SEMANTIC_BLOCK_TO_GROUP` and `SEMANTIC_BLOCK_TO_PERF_CATEGORY` in `category_mappings.py`
2. Add the shape derivation formula in `derive_block_shapes()` in the same file
3. Update the vocabulary list in `semantic-breakdown-agent.md`

### Single-Trace Breakdown

The breakdown pipeline can be run independently on a single trace using the **Trace Semantic Breakdown** skill:

```
Run semantic breakdown on <path_to_trace.json>
```

This produces `semantic_labels.json`, `derived_shapes.json`, and a semantic report without requiring a second trace for comparison.

---

## Example

**Comparing GPT-OSS-20B decode performance on MI355 vs B200:**

```
Run semantic comparison on traces/bs22_b200_graph.json and traces/bs22_mi355_graph.json

Model config: https://huggingface.co/openai/gpt-oss-20b/raw/main/config.json
Tokens: 22 (decode, batch_size=22)
Labels: B200 / MI355
```

This produces a gap analysis identifying:
- Which semantic blocks are faster/slower on each platform
- Roofline-informed analysis (compute-bound vs memory-bound gaps)
- Priority-ranked optimization targets
- Kernel-level mapping between platforms

---

## Bug Reporting

Please include the following details when reporting an issue. Please use the TraceLens-internal private repo to share sensitive data.

- Description
- Software Version (PyTorch, vLLM, SGLang)
- Hardware (e.g., GPU models for both traces)
- Issue Observed
- Expected Behavior
- Scripts/Commands Used
- Error/Unexpected Behavior
- Trace Files Used for Analysis

## 🗺️ Roadmap

TraceLens Semantic Comparison is currently an **experimental** feature.

### 🔄 In Progress

- Validation of semantic labeling accuracy across diverse model architectures (dense, MoE, multimodal).
- Integration tests for the full comparison pipeline.

### 🚀 Future Improvements

- Codify deterministic semantic labeling rules to reduce LLM dependence for common architectures.
- Support for prefill trace comparison (variable sequence lengths, different parallelism strategies).
- Automated detection of fusion differences and their performance implications.
- Integration with TraceLens perf report for combined single-trace + cross-trace analysis.
