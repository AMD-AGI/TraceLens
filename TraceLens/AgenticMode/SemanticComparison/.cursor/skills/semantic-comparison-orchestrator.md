---
name: Semantic Comparison Orchestrator
description: End-to-end semantic gap analysis for two GPU traces. Runs semantic breakdown on both traces in parallel, harmonizes vocabularies, generates TraceDiff and comparison outputs, and produces a stakeholder gap analysis report in the standalone visual language.
triggers:
  - semantic comparison
  - semantic gap analysis
  - compare two traces semantically
  - trace comparison
tools:
  - terminal
  - file_read
  - file_write
---

# Semantic Comparison Orchestrator

Orchestrate end-to-end semantic comparison of two GPU traces. The user provides
two raw trace files and the orchestrator handles everything: semantic breakdown
(in parallel), vocabulary harmonization, comparison pipeline, and final
stakeholder report.

**Role**: Run semantic breakdown on both traces as parallel subagents, harmonize
semantic vocabularies, execute comparison scripts, generate Excel/CSV reports,
and write a prioritized gap analysis report in the standalone visual language.

---

## Language Guidelines

Use vendor-agnostic terminology throughout such as GPU kernels, collective
communication, vendor GEMM library, DNN primitives, etc. Focus on operation
semantics, not vendor implementation details.

**Exception:** When quoting kernel names from traces, include the actual name.

---

## Workflow Steps

```
0.   Query User Inputs (Trace Paths, Names, Config, Tokens)
1.   Semantic Breakdown (PARALLEL subagents) -> trace_a/, trace_b/
2.   Harmonize Semantic Vocabularies (LLM)
3.   Generate TraceDiff Output (script)
4.   Generate Comparison CSV (script)
5.   Compute Priority Ranking (script)
6.   Verify Outputs (script)
7.   Generate Comparison Report (script) -> Excel + CSVs
8.   Write Gap Analysis Report (LLM)
9.   Validate Report
```

---

## Step 0: Query User Inputs

**When this skill is invoked, immediately ask the user for:**

### Required Information:

1. **Trace A Path**
   - Ask: "Please provide the full path to Trace A (.json or .json.gz)"

2. **Trace B Path**
   - Ask: "Please provide the full path to Trace B (.json or .json.gz)"

3. **Name A / Name B**
   - Ask: "What short labels should we use for each trace? (e.g., MI355 / B200)"

4. **Model config.json Path**
   - Ask: "Please provide the path to the HuggingFace config.json for the model"

5. **Number of Tokens**
   - Ask: "How many active tokens? (batch_size for decode, prompt_length * batch for prefill)"

### Optional Information:

6. **Context Length** (for decode traces)
   - Ask: "What is the KV cache length? (optional, defaults to num_tokens for prefill)"

7. **Output Directory**
   - Ask: "Where should we save results? (default: comparison_output/ in the current directory)"

---

## Step 1: Semantic Breakdown (PARALLEL subagents)

Run semantic breakdown on both traces simultaneously using two Task subagents.

### 1.1 Read the Agent Definition

Read the breakdown agent file:
`TraceLens/AgenticMode/SemanticComparison/.cursor/agents/semantic-breakdown-agent.md`

### 1.2 Launch Both Breakdown Subagents in PARALLEL

Launch **both** subagents simultaneously using the Task tool. Do NOT wait
between invocations.

**Task prompt structure for each subagent:**

```
---BEGIN AGENT INSTRUCTIONS---
<full contents of semantic-breakdown-agent.md>
---END AGENT INSTRUCTIONS---

**Execution Context:**
- Trace path: <trace_path>
- Trace name: <trace_name>
- Config JSON: <config_json>
- Number of tokens: <num_tokens>
- Context length: <context_length>
- Output directory: <output_dir>/<trace_name>/

Follow the agent instructions above to complete the semantic breakdown.
Return the summary described in the "Return Value" section.
```

**Output directories:**
- Trace A: `<output_dir>/<name_a>/`
- Trace B: `<output_dir>/<name_b>/`

### 1.3 Wait for Both Subagents to Complete

Both subagents must complete before proceeding to Step 2.

### 1.4 Verify Breakdown Outputs

After both subagents complete:

1. Check that each output directory contains `semantic_labels.json` and
   `derived_shapes.json`
2. Check each subagent's return status for SUCCESS/ERROR
3. If either failed, report the error and stop

```python
import os, json

for trace_dir in ['<output_dir>/<name_a>', '<output_dir>/<name_b>']:
    labels_path = os.path.join(trace_dir, 'semantic_labels.json')
    shapes_path = os.path.join(trace_dir, 'derived_shapes.json')

    assert os.path.exists(labels_path), f"Missing {labels_path}"
    assert os.path.exists(shapes_path), f"Missing {shapes_path}"

    with open(labels_path) as f:
        data = json.load(f)
    print(f"{trace_dir}: {len(data['labeled_kernels'])} kernels, "
          f"{data['total_kernel_time_us']:.1f} us")
```

---

## Step 2: Harmonize Semantic Vocabularies [LLM]

Read both `semantic_labels.json` files and ensure the `semantic_block` labels
use the **same vocabulary** (from `category_mappings.py`).

**Task**: Ensure:

1. The same set of `semantic_block` names appears in both files, drawn from the
   fixed vocabulary in `category_mappings.py`.
2. If one trace has an extra kernel (e.g., `Attention Reduce` on odd layers for
   one platform but not the other), decide:
   - If the kernel has a clear counterpart, label it the same.
   - If it's genuinely extra (no counterpart), keep it -- the comparison script
     handles blocks present in only one trace.
3. If fusion differences mean one trace has 1 kernel for a block and the other
   has 2, both should still use the same `semantic_block` name.
4. Every `semantic_block` label must map to a perf model category (GEMM, SDPA,
   Normalization, or Elementwise) -- no "Other" or unmodeled labels.

**Self-check**: After harmonization, list blocks that appear in only one trace.
Each should have a clear reason.

Update both `semantic_labels.json` files if needed. If labels change, re-run
`derive_shapes.py` to regenerate `derived_shapes.json`.

---

## Step 3: Generate TraceDiff Output [S]

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/generate_semantic_diff.py \
    <output_dir>/<name_a>/semantic_labels.json \
    <output_dir>/<name_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    --shapes-a <output_dir>/<name_a>/derived_shapes.json \
    --shapes-b <output_dir>/<name_b>/derived_shapes.json \
    -o <output_dir>/comparison_report_csvs/
```

Produces in `<output_dir>/comparison_report_csvs/`:
- `diff_stats.csv` -- per-kernel rows matching TraceDiff schema
- `diff_stats_unique_args_summary.csv` -- aggregated by semantic block
- `cpu_op_map.json`, `cpu_op_map_trace1.json`, `cpu_op_map_trace2.json`

---

## Step 4: Generate Comparison CSV [S]

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/match_and_compare.py \
    <output_dir>/<name_a>/semantic_labels.json \
    <output_dir>/<name_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    --shapes-a <output_dir>/<name_a>/derived_shapes.json \
    --shapes-b <output_dir>/<name_b>/derived_shapes.json \
    -o <output_dir>/comparison.csv
```

The comparison CSV includes:
- `semantic_block`, `semantic_group`, `perf_category`, `algorithm_order`
- Per trace: `kernel_names`, `kernel_count`, `total_us`, `avg_us`, `pct`
- `ratio` (trace_a / trace_b) and `gap_us` (absolute difference)
- Roofline columns: `theoretical_GFLOPS`, `theoretical_data_MB`,
  `FLOPS_per_Byte`, per-trace `TFLOPS_s`, `TB_s`

---

## Step 5: Compute Priority Ranking [S]

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/compute_priority.py \
    <output_dir>/comparison.csv \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/priority.json --top 5
```

Output: `priority.json` with top N optimization targets ranked by
`pct * (ratio - 1)`.

---

## Step 6: Verify Outputs [S]

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/verify_comparison.py \
    <output_dir>/<name_a>/semantic_labels.json \
    <output_dir>/<name_b>/semantic_labels.json \
    <output_dir>/comparison.csv \
    --name-a <name_a> --name-b <name_b>
```

Must exit 0. Fix any FAIL assertions before proceeding.

---

## Step 7: Generate Comparison Report [S]

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/generate_comparison_report.py \
    <output_dir>/<name_a>/semantic_labels.json \
    <output_dir>/<name_b>/semantic_labels.json \
    --comparison <output_dir>/comparison.csv \
    --diff-dir <output_dir>/ \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/comparison_report.xlsx \
    --output-csvs-dir <output_dir>/comparison_report_csvs
```

Produces:

- `comparison_report.xlsx` -- multi-sheet Excel workbook
- `comparison_report_csvs/` -- individual CSV files per sheet

**Excel sheets:**

| Sheet | Content |
|-------|---------|
| `kernel_mapping` | Human-readable kernel-to-kernel mapping |
| `comparison` | Block-level timing + roofline metrics |
| `diff_stats` | TraceDiff-compatible per-kernel diff |
| `diff_stats_summary` | Aggregated diff stats |

---

## Step 8: Write Gap Analysis Report [LLM]

Using `comparison.csv`, `priority.json`, and both `semantic_labels.json` files
(plus `derived_shapes.json` and TraceDiff outputs), produce
`<output_dir>/semantic_comparison_report.md`.

The report follows the **standalone analysis visual language**: P1/P2/P3
priorities with icons, Issue/Action/Impact structure, Executive Summary,
Detailed Analysis sections.

**Report template:**

```markdown
# [Name A] vs [Name B]: Semantic Comparison Analysis

## Executive Summary

[1 paragraph overview: model architecture, platform comparison, overall ratio,
which trace is faster and by how much]

| Metric | [Name A] | [Name B] |
|--------|----------|----------|
| Total Iteration Time | X us | Y us |
| Model | architecture | architecture |
| Kernel Count | N | M |
| Blocks Compared | K matched, J only in one trace |
| Overall Ratio (A/B) | Z | - |

## Priority Improvement Targets

<!-- Icon mapping by PRIORITY NUMBER: P1=🔴, P2=🟡, P3+=🟢 -->

### 🔴 P1: <semantic_block> -- <perf_category>

**Issue**: [Name A] is Xx slower than [Name B] for this block (A: Y us vs B:
Z us). [If roofline available: A achieves W TFLOPS/s vs B's V TFLOPS/s.]

**Action**: [Roofline-informed recommendation. For compute-bound blocks:
optimize kernel compute efficiency. For memory-bound blocks: reduce data
movement.]

**Impact**: Xus gap, Y% of total [Name A] runtime.

→ *See [Detailed Analysis: <semantic_group>](#detailed-analysis-kernel-matching)
for kernel-level details*

---

### 🟡 P2: <semantic_block> -- <perf_category>

**Issue**: [1-2 sentences]

**Action**: [1-2 sentences]

**Impact**: [gap_us savings, % of total]

---

### 🟢 P3: <semantic_block> -- <perf_category>

**Issue**: [1 sentence]

**Action**: [1-2 sentences]

**Impact**: [gap_us savings, % of total]

---

## Roofline Summary

[Only include when derived_shapes.json is available for both traces]

| perf_category | [Name A] TFLOPS/s | [Name B] TFLOPS/s | [Name A] TB/s | [Name B] TB/s | Gap |
|---|---|---|---|---|---|
| GEMM | X | Y | ... | ... | ... |
| SDPA | ... | ... | ... | ... | ... |
| Normalization | ... | ... | ... | ... | ... |
| Elementwise | ... | ... | ... | ... | ... |

## Detailed Analysis: Kernel Matching

[For each semantic_group, then for each semantic_block within the group:]

### Self-Attention

#### Pre-Attn Norm

- **[Name A]**: `kernel_name_a` (N kernels, X us)
- **[Name B]**: `kernel_name_b` (M kernels, Y us)
- **Ratio**: Z (A/B)
- **Throughput**: A: W TFLOPS/s, B: V TFLOPS/s [if roofline available]
- **Analysis**: [Why one is faster -- roofline position, architecture features,
  kernel quality, implementation differences]

[... continue for all semantic blocks ...]

## Blocks Where [Name A] is Competitive or Faster

| semantic_block | perf_category | [Name A] (us) | [Name B] (us) | Ratio | [Name A] TFLOPS/s | [Name B] TFLOPS/s |
|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... |

## Blocks Present in Only One Trace

| semantic_block | Present In | Reason |
|---|---|---|
| ... | [Name A/B] only | ... |

## Key Takeaways

[3-5 bullet points:
1. Overall performance comparison
2. Largest improvement opportunities
3. Roofline-informed insight (compute-bound vs memory-bound gaps)
4. Architecture/implementation differences driving the gap
5. Recommended next steps]
```

**For each semantic block, reason about:**

1. **Matching confidence**: Are these clearly the same operation?
2. **Fusion differences**: Does one platform fuse multiple ops into one kernel?
3. **Roofline position**: Is the block compute-bound or memory-bound?
4. **Implementation differences**: Different libraries?
5. **Architecture advantages**: Hardware features that explain gaps?

---

## Step 9: Validate Report

- Check that all `priority.json` entries appear in the P1/P2/P3 section
- Check that all semantic blocks from `comparison.csv` appear in the detailed
  analysis
- Verify no blocks are missing from the report

Prompt the user to review the report.

---

## Agent File Map

**Base path:** `TraceLens/AgenticMode/SemanticComparison/.cursor/agents/`

| Agent | File | Purpose |
|-------|------|---------|
| Semantic Breakdown | `semantic-breakdown-agent.md` | Run full breakdown pipeline on a single trace |

---

## Output File Structure

```
<output_dir>/
  comparison_report.xlsx                  # Multi-sheet Excel workbook
  comparison_report_csvs/                 # All CSV outputs
    kernel_mapping.csv
    comparison.csv
    diff_stats.csv
    diff_stats_summary.csv
    diff_stats_unique_args_summary.csv
    cpu_op_map.json
    cpu_op_map_trace1.json
    cpu_op_map_trace2.json
  comparison.csv                          # Block-level comparison
  priority.json                           # Ranked optimization targets
  semantic_comparison_report.md           # Final stakeholder report
```

---

## Error Handling

### Before Step 1 (Breakdown)
- Verify both trace files exist and are readable
- Verify config.json exists
- Create the output directory if it doesn't exist

### After Step 1 (Breakdown)
- Check each subagent's return status
- Verify `semantic_labels.json` and `derived_shapes.json` exist for both traces
- If either breakdown failed, report the error and stop

### After Step 6 (Verify)
- If verification fails, report which assertions failed
- Common fixes: re-harmonize vocabularies (Step 2), re-run derive_shapes.py

### In Final Report
- If any step failed, note it in the report
- Provide recommendations only for successfully compared blocks

---

## Key Principles

1. **Seamless flow** -- user provides two trace paths and the orchestrator
   handles everything from breakdown to final report
2. **Parallel breakdown** -- both traces are processed simultaneously as
   subagents for speed
3. **Vendor-agnostic language** -- use generic terms for all recommendations
4. **Roofline-informed** -- when shapes are available, distinguish
   compute-bound vs memory-bound gaps
5. **Standalone visual language** -- P1/P2/P3 icons, Issue/Action/Impact
   format, Executive Summary structure
6. **Complete coverage** -- every semantic block appears in the report
